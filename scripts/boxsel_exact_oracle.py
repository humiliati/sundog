#!/usr/bin/env python
"""BoxSEL Phase 2 - exact oracle for tiny role-free SEL fragments.

This is a local testbed oracle, not a scalable SEL reasoner. It handles role-free
concepts built from atomic names by conjunction only. Each Boolean type over the
atoms gets a nonnegative weight. SEL conditionals become linear constraints:

    l * weight(C) <= weight(D and C) <= u * weight(C)

To optimize a query P(D | C), we use the standard homogeneous trick: restrict to
models with weight(C) > 0, scale so weight(C) = 1, and minimize/maximize
weight(D and C). Extreme rational LP solutions correspond to finite models after
clearing denominators.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from random import Random
from typing import Iterable, Sequence


def _frac(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


@dataclass(frozen=True)
class Concept:
    """A role-free EL conjunction of atomic concept names.

    `Concept()` is top. There is no negation and no role constructor in this
    Phase 2 micro-oracle.
    """

    atoms: frozenset[str] = frozenset()

    def __and__(self, other: "Concept") -> "Concept":
        return Concept(frozenset(self.atoms | other.atoms))

    def __str__(self) -> str:
        return "TOP" if not self.atoms else "&".join(sorted(self.atoms))


def atom(name: str) -> Concept:
    return Concept(frozenset([name]))


TOP = Concept()


@dataclass(frozen=True)
class Conditional:
    consequent: Concept
    condition: Concept
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower < 0 or self.upper > 1 or self.lower > self.upper:
            raise ValueError(f"invalid interval [{self.lower}, {self.upper}]")


def conditional(consequent: Concept, condition: Concept, lower, upper=None) -> Conditional:
    if upper is None:
        upper = lower
    return Conditional(consequent, condition, _frac(lower), _frac(upper))


@dataclass
class OracleResult:
    lower: float
    upper: float
    lower_exact: Fraction
    upper_exact: Fraction
    feasible: bool
    atoms: tuple[str, ...]
    types: tuple[frozenset[str], ...]
    lower_weights: tuple[Fraction, ...] = ()
    upper_weights: tuple[Fraction, ...] = ()

    def interval(self) -> tuple[float, float]:
        return self.lower, self.upper

    def interval_exact(self) -> tuple[Fraction, Fraction]:
        return self.lower_exact, self.upper_exact


def concept_atoms(items: Iterable[Conditional | Concept]) -> tuple[str, ...]:
    atoms: set[str] = set()
    for item in items:
        if isinstance(item, Conditional):
            atoms.update(item.consequent.atoms)
            atoms.update(item.condition.atoms)
        else:
            atoms.update(item.atoms)
    return tuple(sorted(atoms))


def enumerate_types(atoms: Sequence[str]) -> tuple[frozenset[str], ...]:
    atoms = tuple(atoms)
    return tuple(
        frozenset(atom_name for atom_name, present in zip(atoms, mask) if present)
        for mask in product((False, True), repeat=len(atoms))
    )


def all_concepts(atoms: Sequence[str], max_size: int | None = None, include_top: bool = True) -> tuple[Concept, ...]:
    """All conjunction concepts up to `max_size` atoms."""
    atoms = tuple(sorted(atoms))
    if max_size is None:
        max_size = len(atoms)
    concepts: list[Concept] = []
    if include_top:
        concepts.append(TOP)
    for size in range(1, min(max_size, len(atoms)) + 1):
        for combo in combinations(atoms, size):
            concepts.append(Concept(frozenset(combo)))
    return tuple(concepts)


def extension_coeff(types: Sequence[frozenset[str]], concept: Concept) -> tuple[Fraction, ...]:
    return tuple(Fraction(1) if concept.atoms.issubset(t) else Fraction(0) for t in types)


def _lp_min_exact(
    cost: Sequence[Fraction], rows: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
):
    """Exact two-phase primal simplex (Bland's rule): minimize cost.x s.t. rows.x = rhs, x >= 0.

    All arithmetic is rational, so the optimum is certified exact (no floating point).
    Returns (status, value, x) with status in {"optimal", "infeasible", "unbounded"}.
    """

    m = len(rows)
    n = len(cost)
    if m == 0:
        return "optimal", Fraction(0), [Fraction(0)] * n

    a = [[_frac(v) for v in row] for row in rows]
    b = [_frac(v) for v in rhs]
    for i in range(m):
        if b[i] < 0:  # keep rhs nonnegative so the artificial basis is feasible
            a[i] = [-v for v in a[i]]
            b[i] = -b[i]

    # columns: n structural, then one artificial per row (identity). artificials never (re-)enter.
    tab = [a[i] + [Fraction(1) if j == i else Fraction(0) for j in range(m)] + [b[i]] for i in range(m)]
    basis = [n + i for i in range(m)]
    structural = range(n)

    def pivot(leave: int, enter: int) -> None:
        piv = tab[leave][enter]
        tab[leave] = [v / piv for v in tab[leave]]
        for i in range(m):
            if i != leave and tab[i][enter] != 0:
                f = tab[i][enter]
                tab[i] = [x - f * y for x, y in zip(tab[i], tab[leave])]
        basis[leave] = enter

    def optimize(obj: Sequence[Fraction]) -> str:
        while True:
            enter = None
            for j in structural:  # Bland: first improving column by index
                if j in basis:
                    continue
                rc = obj[j] - sum(obj[basis[i]] * tab[i][j] for i in range(m))
                if rc < 0:
                    enter = j
                    break
            if enter is None:
                return "optimal"
            leave = None
            best = None
            for i in range(m):
                if tab[i][enter] > 0:
                    ratio = tab[i][-1] / tab[i][enter]
                    if best is None or ratio < best or (ratio == best and basis[i] < basis[leave]):
                        best, leave = ratio, i
            if leave is None:
                return "unbounded"
            pivot(leave, enter)

    optimize([Fraction(0)] * n + [Fraction(1)] * m)  # phase 1: minimize sum of artificials
    if sum((Fraction(1) if basis[i] >= n else Fraction(0)) * tab[i][-1] for i in range(m)) != 0:
        return "infeasible", None, None
    for i in range(m):  # drive any artificial still basic (value 0) out where possible
        if basis[i] >= n:
            for j in range(n):
                if tab[i][j] != 0:
                    pivot(i, j)
                    break

    phase2 = [_frac(v) for v in cost] + [Fraction(0)] * m
    status = optimize(phase2)
    if status != "optimal":
        return status, None, None
    x = [Fraction(0)] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = tab[i][-1]
    value = sum(phase2[j] * x[j] for j in range(n))
    return "optimal", value, x


def exact_interval(
    ontology: Sequence[Conditional],
    consequent: Concept,
    condition: Concept,
    atoms: Sequence[str] | None = None,
) -> OracleResult:
    """Exact rational lower/upper bounds for P(consequent | condition).

    Type weights are rational; each SEL conditional is a homogeneous linear constraint; the
    query is optimized by the weight(condition)=1 trick and an exact rational simplex (no
    floating point). If no satisfying model has nonempty `condition`, the conditional is
    vacuous and the result is marked infeasible with [0, 1].
    """

    if atoms is None:
        atoms = concept_atoms([*ontology, consequent, condition])
    atoms = tuple(sorted(atoms))
    types = enumerate_types(atoms)
    n = len(types)

    query_den = extension_coeff(types, condition)
    query_num = extension_coeff(types, condition & consequent)

    ineq: list[list[Fraction]] = []  # homogeneous ontology inequalities A_ub . w <= 0
    for ax in ontology:
        den = extension_coeff(types, ax.condition)
        num = extension_coeff(types, ax.condition & ax.consequent)
        ineq.append([ax.lower * d - nu for d, nu in zip(den, num)])  # lower*den <= num
        ineq.append([nu - ax.upper * d for d, nu in zip(den, num)])  # num <= upper*den
    p = len(ineq)

    # standard form columns: w (n types) then one slack per inequality (p)
    rows: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    for k in range(p):
        rows.append(list(ineq[k]) + [Fraction(1) if j == k else Fraction(0) for j in range(p)])
        rhs.append(Fraction(0))
    rows.append(list(query_den) + [Fraction(0)] * p)  # weight(condition) = 1
    rhs.append(Fraction(1))

    obj = list(query_num) + [Fraction(0)] * p
    s_lo, v_lo, x_lo = _lp_min_exact(obj, rows, rhs)
    s_hi, v_hi, x_hi = _lp_min_exact([-v for v in obj], rows, rhs)
    if s_lo != "optimal" or s_hi != "optimal":
        return OracleResult(0.0, 1.0, Fraction(0), Fraction(1), False, atoms, types)

    lower, upper = v_lo, -v_hi
    return OracleResult(
        lower=float(lower),
        upper=float(upper),
        lower_exact=lower,
        upper_exact=upper,
        feasible=True,
        atoms=atoms,
        types=types,
        lower_weights=tuple(x_lo[:n]),
        upper_weights=tuple(x_hi[:n]),
    )


def pmp_ontology(q1, q2):
    """Phase-1 PMP hand-case ontology and query.

    Encodes:
        (A | Q1)[q1]
        (Q2 | A and Q1)[q2]
    Query:
        (Q2 | Q1)
    """

    q1_concept = atom("Q1")
    a = atom("A")
    q2_concept = atom("Q2")
    ontology = [
        conditional(a, q1_concept, q1),
        conditional(q2_concept, a & q1_concept, q2),
    ]
    return ontology, q2_concept, q1_concept


def exact_pmp_interval(q1, q2) -> OracleResult:
    ontology, consequent, condition = pmp_ontology(q1, q2)
    return exact_interval(ontology, consequent, condition)


def finite_cond(numerator: Iterable[int], denominator: Iterable[int]) -> Fraction | None:
    denominator = set(denominator)
    if not denominator:
        return None
    return Fraction(len(set(numerator) & denominator), len(denominator))


@dataclass(frozen=True)
class Alg2Counterexample:
    universe: frozenset[int]
    q1_set: frozenset[int]
    a_set: frozenset[int]
    q2_set: frozenset[int]
    q1: Fraction
    q2_marginal: Fraction
    q2_correct: Fraction
    shipped_upper: Fraction
    true_query: Fraction


@dataclass(frozen=True)
class WeightedTypeModel:
    """A finite-count/type-volume model over the Boolean atoms.

    Counts and type-volume weights are the same semantic object for this
    role-free oracle: nonnegative mass on each Boolean type. Integer weights can
    be read as a finite model; rational weights can be read as volumes of
    disjoint type cells.
    """

    atoms: tuple[str, ...]
    weights: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        expected = 1 << len(self.atoms)
        if len(self.weights) != expected:
            raise ValueError(f"expected {expected} weights for {len(self.atoms)} atoms")
        if any(w < 0 for w in self.weights):
            raise ValueError("type weights must be nonnegative")
        if sum(self.weights, Fraction(0)) <= 0:
            raise ValueError("at least one type must have positive weight")

    @property
    def types(self) -> tuple[frozenset[str], ...]:
        return enumerate_types(self.atoms)

    def measure(self, concept: Concept) -> Fraction:
        return sum(
            (weight for type_set, weight in zip(self.types, self.weights) if concept.atoms.issubset(type_set)),
            Fraction(0),
        )

    def conditional(self, consequent: Concept, condition: Concept) -> Fraction | None:
        den = self.measure(condition)
        if den == 0:
            return None
        return self.measure(condition & consequent) / den

    def satisfies(self, axiom: Conditional) -> bool:
        value = self.conditional(axiom.consequent, axiom.condition)
        return value is None or axiom.lower <= value <= axiom.upper


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    atoms: tuple[str, ...]
    ontology: tuple[Conditional, ...]
    query_consequent: Concept
    query_condition: Concept
    source_model: WeightedTypeModel
    source_query_value: Fraction

    def exact(self) -> OracleResult:
        return exact_interval(self.ontology, self.query_consequent, self.query_condition, atoms=self.atoms)

    def summary_key(self) -> tuple:
        return (
            self.case_id,
            self.atoms,
            tuple((str(ax.consequent), str(ax.condition), ax.lower, ax.upper) for ax in self.ontology),
            str(self.query_consequent),
            str(self.query_condition),
            self.source_query_value,
        )


def model_from_counts(atoms: Sequence[str], counts: Sequence[int | Fraction]) -> WeightedTypeModel:
    return WeightedTypeModel(tuple(sorted(atoms)), tuple(_frac(c) for c in counts))


def interval_around(value: Fraction, slack: Fraction) -> tuple[Fraction, Fraction]:
    return max(Fraction(0), value - slack), min(Fraction(1), value + slack)


def generate_source_model(atoms: Sequence[str], rng: Random, max_count: int = 5) -> WeightedTypeModel:
    atoms = tuple(sorted(atoms))
    count = 1 << len(atoms)
    weights = [Fraction(rng.randint(0, max_count)) for _ in range(count)]
    if sum(weights, Fraction(0)) == 0:
        weights[rng.randrange(count)] = Fraction(1)
    return WeightedTypeModel(atoms, tuple(weights))


def generate_corpus_case(
    case_index: int,
    rng: Random,
    atom_count: int = 3,
    axiom_count: int = 4,
    max_concept_size: int = 2,
    slack: Fraction = Fraction(1, 10),
) -> CorpusCase:
    """Generate one satisfiable tiny role-free SEL case from a source model."""

    atoms = tuple(chr(ord("A") + i) for i in range(atom_count))
    concepts = all_concepts(atoms, max_size=max_concept_size, include_top=True)

    for attempt in range(200):
        model = generate_source_model(atoms, rng)
        candidates: list[tuple[Concept, Concept, Fraction]] = []
        for condition in concepts:
            if model.measure(condition) == 0:
                continue
            for consequent in concepts:
                if consequent == TOP:
                    continue
                value = model.conditional(consequent, condition)
                if value is not None:
                    candidates.append((consequent, condition, value))
        if len(candidates) < axiom_count + 1:
            continue

        rng.shuffle(candidates)
        query_consequent, query_condition, query_value = candidates[0]
        axioms: list[Conditional] = []
        seen = {(query_consequent, query_condition)}
        for consequent, condition, value in candidates[1:]:
            if (consequent, condition) in seen:
                continue
            seen.add((consequent, condition))
            lo, hi = interval_around(value, slack)
            axioms.append(conditional(consequent, condition, lo, hi))
            if len(axioms) == axiom_count:
                break
        if len(axioms) != axiom_count:
            continue

        case = CorpusCase(
            case_id=f"tiny-{case_index:03d}",
            atoms=atoms,
            ontology=tuple(axioms),
            query_consequent=query_consequent,
            query_condition=query_condition,
            source_model=model,
            source_query_value=query_value,
        )
        exact = case.exact()
        if exact.feasible and exact.lower <= float(query_value) + 1e-8 and exact.upper + 1e-8 >= float(query_value):
            return case

    raise RuntimeError("failed to generate a satisfiable corpus case")


def generate_tiny_corpus(
    case_count: int = 8,
    seed: int = 240711821,
    atom_count: int = 3,
    axiom_count: int = 4,
    max_concept_size: int = 2,
    slack: Fraction = Fraction(1, 10),
) -> tuple[CorpusCase, ...]:
    rng = Random(seed)
    return tuple(
        generate_corpus_case(
            i,
            rng,
            atom_count=atom_count,
            axiom_count=axiom_count,
            max_concept_size=max_concept_size,
            slack=slack,
        )
        for i in range(case_count)
    )


def normalized_weights(weights: Sequence[Fraction]) -> tuple[Fraction, ...]:
    """Counts -> disjoint type-VOLUMES: divide by the total so the volumes sum to 1."""
    total = sum((_frac(w) for w in weights), Fraction(0))
    if total == 0:
        raise ValueError("cannot normalize an all-zero weight vector")
    return tuple(_frac(w) / total for w in weights)


def rescaled_weights(weights: Sequence[Fraction], factor: Fraction) -> tuple[Fraction, ...]:
    """An arbitrary positive rescaling of the type weights (conditionals are scale-invariant)."""
    if factor <= 0:
        raise ValueError("rescale factor must be positive")
    return tuple(_frac(w) * factor for w in weights)


def finite_type_volume_alignment(case: CorpusCase) -> bool:
    """Finite integer type COUNTS and rational type-VOLUMES induce the same SEL constraints.

    Conditionals are ratios, hence scale-invariant. A finite count model `w`, its normalized
    disjoint type-volume model `w / sum(w)` (volumes summing to 1), and an arbitrary positive
    rescaling `k*w` must agree on every axiom's conditional, on satisfaction, and on the query
    value. The volume and rescaled models carry DIFFERENT weight tuples than the counts, so this
    is a genuine invariance check, not a self-comparison. It does NOT claim single-box
    realizability -- that is the later I_box / representation-gap phase.
    """

    counts = case.source_model
    volume = WeightedTypeModel(case.atoms, normalized_weights(counts.weights))
    rescaled = WeightedTypeModel(case.atoms, rescaled_weights(counts.weights, Fraction(7, 3)))
    for axiom in case.ontology:
        base = counts.conditional(axiom.consequent, axiom.condition)
        if volume.conditional(axiom.consequent, axiom.condition) != base:
            return False
        if rescaled.conditional(axiom.consequent, axiom.condition) != base:
            return False
        if not (counts.satisfies(axiom) == volume.satisfies(axiom) == rescaled.satisfies(axiom)):
            return False
    qv = counts.conditional(case.query_consequent, case.query_condition)
    if volume.conditional(case.query_consequent, case.query_condition) != qv:
        return False
    if rescaled.conditional(case.query_consequent, case.query_condition) != qv:
        return False
    return qv == case.source_query_value


def alg2_shipped_upper(q1: Fraction, q2_marginal: Fraction) -> Fraction:
    return min(Fraction(1), q1 * q2_marginal + 1 - q2_marginal)


def find_alg2_shipped_counterexample(max_n: int = 5) -> Alg2Counterexample | None:
    """Bounded finite-model search for both printed Algorithm 2 artifacts together.

    We require the premise drift to be live:
        P(Q2 | A) != P(Q2 | A and Q1)
    and the as-printed shipped upper to sit below the true P(Q2 | Q1).
    """

    for n in range(1, max_n + 1):
        universe = tuple(range(1, n + 1))
        subsets = []
        for mask in range(1 << n):
            subsets.append(frozenset(universe[i] for i in range(n) if mask & (1 << i)))
        for q1_set in subsets:
            if not q1_set:
                continue
            for a_set in subsets:
                if not a_set or not (a_set & q1_set):
                    continue
                q1 = finite_cond(a_set, q1_set)
                if q1 is None:
                    continue
                for q2_set in subsets:
                    q2_marginal = finite_cond(q2_set, a_set)
                    q2_correct = finite_cond(q2_set, a_set & q1_set)
                    true_query = finite_cond(q2_set, q1_set)
                    if q2_marginal is None or q2_correct is None or true_query is None:
                        continue
                    if q2_marginal == q2_correct:
                        continue
                    shipped = alg2_shipped_upper(q1, q2_marginal)
                    if shipped < true_query:
                        return Alg2Counterexample(
                            universe=frozenset(universe),
                            q1_set=q1_set,
                            a_set=a_set,
                            q2_set=q2_set,
                            q1=q1,
                            q2_marginal=q2_marginal,
                            q2_correct=q2_correct,
                            shipped_upper=shipped,
                            true_query=true_query,
                        )
    return None


if __name__ == "__main__":
    result = exact_pmp_interval(Fraction(1, 5), Fraction(4, 5))
    print("PMP toy exact interval:", result.interval())
    ce = find_alg2_shipped_counterexample()
    if ce:
        print("Alg2 shipped counterexample:")
        print("  universe =", sorted(ce.universe))
        print("  Q1 =", sorted(ce.q1_set), "A =", sorted(ce.a_set), "Q2 =", sorted(ce.q2_set))
        print("  q1 =", ce.q1, "q2_marginal =", ce.q2_marginal, "q2_correct =", ce.q2_correct)
        print("  shipped_upper =", ce.shipped_upper, "true =", ce.true_query)
