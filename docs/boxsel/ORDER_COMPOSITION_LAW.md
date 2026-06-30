# The Composition Law — "order is vector-valued; scalar = join"

**Status:** central conjecture PROVEN in its provable form — the cancellation-free (coproduct) join
law machine-checked in general (`annihilates_prod`, for all elements), the boundary pinned with a
**2-positive / 2-negative** classification (cohomological + moment positive; search-reach + algebraic-
degree negative, the latter two machine-checked). Expedition off the Order-Relative Resolution Law
([ORDER_RELATIVE_LAW.md](ORDER_RELATIVE_LAW.md)). Lean: `Sundogcert/OrderRelativeCompose.lean` +
`OrderRelativeComposeLaw.lean`. Internal; frozen-as-portfolio.

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
axes whose product has **cancellation**. A **2-positive / 2-negative** classification:

| axis | product `⊗` | order lattice `L` | join-homomorphic? | character |
|---|---|---|---|---|
| **cohomological** | `⊕` (direct sum) | divisibility (`lcm`) | **yes** ✓Lean | rich — full lattice |
| **moment / spectral** | convolution (`X+Y`, indep.) | 2-chain `{1, ⊤}` | **yes** | degenerate — binary (analysis) |
| **search-reach** | `×` (rational mult.) | — | **no** | cancellation *inflates* the order |
| **algebraic-degree** | `×` (real mult.) | — | **no** ✓Lean | cancellation *drops* the order |

The two negatives fail in **opposite directions** — cancellation can throw the order either way off
the join:

- **Search-reach inflates (negative).** Denominators under multiplication: `denom(½ · ½) = 4`, but
  `lcm(2,2) = 2`. So `ord(xy) = 4 > 2 = join` — the product order *exceeds* the join. Not
  join-homomorphic.

- **Algebraic-degree drops (negative, machine-checked — `algDeg_not_join_under_mul`).** `√2` has
  algebraic degree 2 (irrational), but `√2 · √2 = 2` has degree 1 (rational). So
  `ord(√2·√2) = 1 < 2 = join` — the product order falls *below* the join: the irrationalities cancel.
  Not join-homomorphic.

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
exactly for join-homomorphic axes — cohomological (rich) and moment (degenerate) — and failing on
cancelling products (search-reach *inflates*, algebraic-degree *drops*). Not a universal cross-axis
identity.* A real, falsifiable line, reframing the mode-vectors as evidence of a **grading**.

---

## 4. The conjecture, proven (provable form)

The conjecture `join-homomorphic ⟺ cancellation-free` is not a single cross-axis theorem
("cancellation-free" is instance-specific). But its two halves are now established — the substantive
positive in **general** form, the negatives by **machine-checked witnesses**:

- **Cancellation-free ⇒ join (general, `annihilates_prod`).** For the **coproduct** (direct sum), a
  budget `j` annihilates `(s, t)` iff it annihilates *both* coordinates: `j • (s,t) = 0 ↔ j•s = 0 ∧
  j•t = 0`, for **all** `s, t` in any two additive groups. The coordinates are independent — no
  interaction, no cancellation — so the order is the join (lcm) of the coordinate orders. This is the
  general law that `compose_order_eq_lcm` was a special case of. *(Axiom-clean with only
  `[propext, Quot.sound]` — it does not even need choice.)*

- **Cancellation ⇒ not join (witnesses).** Any product with a cancelling pair breaks the join, in
  either direction: `within_group_cancels` — addition *inside* a group is not the coproduct (in
  `ZMod 2`, `1 + 1 = 0`, so budget 1 annihilates the sum but not the operand, the order drops);
  `algDeg_not_join_under_mul` (`√2·√2 = 2`, drop); search-reach (`denom` inflates).

So the precise statement: **the order is join-homomorphic exactly on the cancellation-free coproduct
— independent coordinates — and any cancelling product (within-group `+`, real/rational `×`) breaks
it.** "Cancellation-free" *means* the coproduct; that is where, and only where, the grading composes.

---

## 5. Open probes (where the expedition goes next)

1. **The radical axis** has `⊕`-like structure on its exponents — a candidate *third positive*.
2. **Lean the moment converse** (independent sum-integrable ⇒ both) to upgrade the degenerate positive
   from analysis to machine-checked.
3. **Abstract the coproduct law** — state `annihilates_prod` as: `ord` is a monoid homomorphism
   `(M, ⊕) → (L, ⊔)` on any coproduct, and ask which categorical products are cancellation-free.

---

*Sundog Research Lab — the composition law off the Order-Relative Resolution Law. The cancellation-
free (coproduct) join law is machine-checked in general (`annihilates_prod`); the boundary is a
2-positive / 2-negative classification (cohomological + moment positive; search-reach + algebraic-
degree negative, the latter two in Lean: `compose_lcm_not_max`, `algDeg_not_join_under_mul`,
`within_group_cancels`). `ord` is a lattice-valued grading homomorphism exactly on the cancellation-
free coproduct — axis-internal, not universal. Internal; frozen-as-portfolio.*
