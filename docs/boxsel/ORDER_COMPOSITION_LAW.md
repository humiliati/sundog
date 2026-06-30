# The Composition Law — "order is vector-valued; scalar = join"

**Status:** first move LANDED (machine-checked on the cohomological axis); general statement
sketched with a falsifiable boundary. Expedition off the Order-Relative Resolution Law
([ORDER_RELATIVE_LAW.md](ORDER_RELATIVE_LAW.md)). Internal; frozen-as-portfolio.

The three mode-vectors in the order-relative ledger — `√2` (search-reach `⊤` vs radical 2),
`(9+√17)/32` (algebraic-degree 2 vs denominator `⊤`), and the cohomological `(1,1) ∈ ℤ × ZMod m`
(free `⊤` vs torsion `m`) — were not isolated curiosities. They are evidence that **"order" is
natively vector-valued, and the scalar order is a lossy projection of that vector.** This note makes
the claim precise, proves it on the one axis that carries a clean product, and marks where it stops.

---

## 1. The law, made precise (PROVEN)

On the cohomological axis (`order = additive/torsion order of a (co)homology class`), the scalar
order of a **product** class is exactly the **lcm** of the coordinate orders:

> `j • (s, t) = 0  ↔  ord(s) ∣ j  ∧  ord(t) ∣ j  ↔  lcm(ord s, ord t) ∣ j`.

Machine-checked in `Sundogcert/OrderRelativeCompose.lean`:

- `composeProblem a b` — the product class `(1,1) ∈ ZMod a × ZMod b`; its `resolves_iff` **proves**
  the scalar order is `lcm a b`.
- `compose_order_eq_lcm` — the named law: `ord(product) = lcm a b`.
- `compose_lcm_not_max` — the **sharp witness**: at `a=4, b=6` the composite order is **`12 = lcm`**,
  while the ≤-join (`max`) is only **`6`**.

So "scalar = join" holds — but with the **divisibility-lattice join (lcm)**, *not* the naive ≤-max.

### The two orders on `ℕ∞`

The precision that makes this a real finding: there are **two distinct orders** on `ℕ∞` at work, and
they must not be conflated.

| order | role | law |
|---|---|---|
| the **≤-order** | budget | `Resolves k t ↔ ord t ≤ k` |
| the **divisibility order** | composition | `ord(s ⊗ t) = lcm(ord s, ord t)` |

They are compatible (`a ∣ b → a ≤ b`) but not equal (`lcm(4,6)=12 > max(4,6)=6`). The free/`⊤` case
is simply where `lcm` absorbs `⊤` — which is *why* a mixed class collapses to a resist pole under the
scalar order. The mode-vector is the un-collapsed vector; the scalar is its join projection.

---

## 2. The general statement (SKETCH)

**Claim (composition / vector-valuedness).** For an axis whose targets carry a product `⊗`, the order
is a **lattice-valued grading homomorphism**

> `ord : (targets, ⊗) ⟶ (L, ⊔)`,   `ord(s ⊗ t) = ord(s) ⊔_L ord(t)`,

for an axis-specific lattice `L`. The order is faithfully the **vector** of per-irreducible-component
orders; the **scalar** is the lattice join, lossy exactly when the components are `L`-incomparable.
Together with the budget law `Resolves k ↔ ord ≤ k`, this makes `ord` a *grading* — a structure map,
not a bare number.

**Proven instance.** Cohomological axis: `L =` divisibility lattice on `ℕ∞`, `⊗ = ⊕`, `ord = lcm`.

---

## 3. The boundary — it is NOT universal (break-first)

The law needs `ord` to actually *be* a join-homomorphism. That is a real constraint, and it fails on
axes whose product has **cancellation**. A three-instance classification:

| axis | product `⊗` | order lattice `L` | join-homomorphic? | character |
|---|---|---|---|---|
| **cohomological** | `⊕` (direct sum) | divisibility (`lcm`) | **yes** | rich — full lattice |
| **moment / spectral** | convolution (`X+Y`, indep.) | 2-chain `{1, ⊤}` | **yes** | degenerate — binary |
| **search-reach** | `×` (rational mult.) | — | **no** | cancellation breaks it |

- **Search-reach breaks it (the negative).** Denominators under multiplication:
  `denom(½ · ½) = denom(¼) = 4`, but `lcm(2,2) = 2`. So `denom(xy) ≠ join(denom x, denom y)` — the
  product has cancellation, and there is no clean composition law. Search-reach is *not*
  join-homomorphic.

- **The moment axis holds it, degenerately (the second positive).** Order = `1` if a finite mean
  exists, `⊤` if not; product = convolution (`X + Y` for independent `X, Y`). Then
  `ord(X+Y) = max(ord X, ord Y)`:
  - both finite mean ⇒ `E|X+Y| ≤ E|X| + E|Y| < ∞` ⇒ order `1` (`Integrable.add`);
  - either has no mean ⇒ for fixed `y`, `E|X+y| ≥ E|X| − |y| = ∞`, so `E|X+Y| = ∞` ⇒ order `⊤`.

  On the 2-chain `{1, ⊤}` the divisibility join *is* the max, so the law holds — but trivially, because
  the order is binary (the same "structural, not deep" character as the moment axis itself). Lean:
  the easy half is `Integrable.add`; the converse (sum-integrable ⇒ both, for independent variables)
  is the harder direction, deferred.

**Honest verdict.** *"Order is vector-valued, and scalar = join" is an **axis-internal** law, holding
exactly for join-homomorphic axes — cohomological (rich), moment (degenerate) — and failing on axes
with a cancelling product (search-reach). It is not a universal cross-axis identity.* That is a real,
falsifiable line, and it reframes the mode-vectors as evidence of a **grading**, not a coincidence.

---

## 4. Open probes (where the expedition goes next)

1. **Classify the join-homomorphic axes.** The pattern so far: join-homomorphism ⟺ the product is
   *cancellation-free / order-monotone* (`⊕`, independent sum) and fails under cancellation (`×`). Is
   that the right characterization? Prove or break it.
2. **Algebraic-degree under multiplication.** `deg(αβ) ≤ deg(α)·deg(β)` (sub-multiplicative, with
   strict drops from cancellation, e.g. `√2 · √2 = 2`). So algebraic-degree is *bounded* but not a
   clean join — another likely negative, worth pinning.
3. **The radical axis** has `⊕`-like structure on its exponents — a candidate third positive.
4. **Lean the moment converse** (independent sum-integrable ⇒ both) to upgrade the degenerate positive
   from analysis to machine-checked.

---

*Sundog Research Lab — the composition law off the Order-Relative Resolution Law. Proven on the
cohomological axis (`compose_order_eq_lcm`, `compose_lcm_not_max`); general statement = `ord` is a
lattice-valued grading homomorphism, axis-internal not universal (search-reach is the negative,
moment the degenerate positive). Internal; frozen-as-portfolio.*
