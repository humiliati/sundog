#!/usr/bin/env python
"""BoxSEL Phase-1 — minimal probabilistic-modus-ponens (PMP) calculator + the replication gate.

Phase 1 of docs/SUNDOG_V_BOXSEL.md. Implements the PMP bound from the anchor paper (Zhu, Potyka,
Xiong, Tran, Nayyeri, Kharlamov, Staab, "Approximating Probabilistic Inference in Statistical EL
with Knowledge Graph Embeddings", UAI 2026) plus the two CONFIRMED Appendix Algorithm 2 artifacts,
so published intervals are not trusted as ground truth until they are checked by hand.

Premise chain (Proposition 2, "Probabilistic Modus Ponens (PMP)"):
    T |= (D | C)[l1, u1]
    T |= (E | C and D)[l2, u2]          <-- the SECOND premise is conditioned on C and D
    ==> T |= (E | C)[l3, u3],   l3 = l1*l2,   u3 = min(1, u1*u2 + 1 - l1)   (paper's printed form)

With C = Q1, D = A, E = Q2 the second premise is (Q2 | A and Q1); for POINT premises (l = u = q)
this collapses to the gate form
    lower = q1*q2,   upper = min(1, q1*q2 + 1 - q1).
The "1 - q1" slack is the probability mass in Q1 outside A. Sanity: q1 = 1 -> [q2, q2].

Two CONFIRMED Appendix Algorithm 2 artifacts (verified 2026-06-20 vs arXiv HTML 2407.11821v2;
line 16 / line 13):
  (a) UPPER-SLACK TYPO    -- Algorithm 2 prints  min(1, q1*q2 + 1 - q2)  (1 - q2, not 1 - q1).
  (b) PREMISE-SHAPE DRIFT -- Algorithm 2 inputs (Q2 | A)[q2]; it drops the "and Q1", so it plugs
      P(Q2 | A) into the slot Proposition 2 reserves for P(Q2 | A and Q1). HARMLESS only when
      A subset Q1 (then A and Q1 = A); otherwise it can push the "upper bound" BELOW the true
      conditional probability -- a soundness break. See premise_shape_counterexample().

SHARPNESS NOTE: the paper's u1*u2 + 1 - l1 is SOUND but not TIGHT; the sharp max over all models
is l1*u2 + 1 - l1 (pmp_upper_sharp). They coincide for point premises, so the Phase-1 gate is
unaffected; an interval-premise oracle should prefer the sharp form or knowingly match the paper.

NOT an attack on the paper (falsifier BOXSEL-PMP-TYPO-AS-ATTACK). Phase 1 locks the body formula
and characterizes the artifacts; the effect on the paper's REPORTED metrics is unresolved here
(it needs the authors' evaluation code).
"""
from fractions import Fraction


# --- the bounds (floats; the body/point gate) ---

def pmp_interval(l1, u1, l2, u2):
    """Proposition 2 bound (paper form) on (E|C) from (D|C)[l1,u1] and (E | C and D)[l2,u2]."""
    return (l1 * l2, min(1.0, u1 * u2 + 1.0 - l1))


def pmp_point(q1, q2):
    """Point-premise gate form: (A|Q1)[q1], (Q2 | A and Q1)[q2] -> (Q2|Q1)."""
    return pmp_interval(q1, q1, q2, q2)


def pmp_upper_sharp(l1, u1, l2, u2):
    """The TIGHT upper bound (sharp max over models): min(1, l1*u2 + 1 - l1). Equals paper at points."""
    return min(1.0, l1 * u2 + 1.0 - l1)


def pmp_upper_alg2(q1, q2):
    """The PRINTED Appendix Algorithm 2 upper (the confirmed slack typo): min(1, q1*q2 + 1 - q2)."""
    return min(1.0, q1 * q2 + 1.0 - q2)


# --- finite-model helpers for the premise-shape audit (exact rational counting) ---

def cond(num, den):
    """P(num | den) = |num and den| / |den| over a finite universe. None if den is empty."""
    den = set(den)
    if not den:
        return None  # SEL: a conditional is vacuously satisfied when its conditioning class is empty
    return Fraction(len(set(num) & den), len(den))


def premise_shape_harmless(Q1, A, Q2):
    """True iff dropping 'and Q1' is harmless for THIS model: P(Q2|A) == P(Q2 | A and Q1)."""
    A, Q1 = set(A), set(Q1)
    return cond(Q2, A) == cond(Q2, A & Q1)


def pmp_point_from_model(Q1, A, Q2, marginal_premise):
    """Prop-2 point UPPER bound + the TRUE value, computed from a finite model (exact Fractions).

    marginal_premise=False uses the CORRECT premise P(Q2 | A and Q1) (Proposition 2);
    marginal_premise=True  uses P(Q2 | A) -- what Algorithm 2 plugs in. The slack term is the
    CORRECT 1 - q1 in both cases, to isolate the premise-shape effect from the slack typo.
    Returns (q1, q2_used, computed_upper, true_value).
    """
    Q1, A, Q2 = set(Q1), set(A), set(Q2)
    q1 = cond(A, Q1)
    q2 = cond(Q2, A) if marginal_premise else cond(Q2, A & Q1)
    computed_upper = min(Fraction(1), q1 * q2 + 1 - q1)
    return q1, q2, computed_upper, cond(Q2, Q1)


def premise_shape_counterexample():
    """A finite model where dropping 'and Q1' breaks soundness (the load-bearing audit result).

    Q1 = {1..5}; A spills outside Q1 (A not subset Q1); within A and Q1 every individual is in Q2.
      P(A|Q1) = 4/5,  P(Q2 | A and Q1) = 1,  P(Q2 | A) = 4/10,  true P(Q2|Q1) = 4/5.
    Plugging the marginal P(Q2|A) gives upper 0.52 < true 0.80 -> the "upper bound" is violated.
    """
    return ({1, 2, 3, 4, 5}, {1, 2, 3, 4, 6, 7, 8, 9, 10, 11}, {1, 2, 3, 4})


if __name__ == "__main__":  # tiny demo
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("toy (q1=0.2, q2=0.8): body", pmp_point(0.2, 0.8), "| Alg2 upper", pmp_upper_alg2(0.2, 0.8))
    Q1, A, Q2 = premise_shape_counterexample()
    print("premise drift:", pmp_point_from_model(Q1, A, Q2, marginal_premise=True),
          "harmless?", premise_shape_harmless(Q1, A, Q2))
