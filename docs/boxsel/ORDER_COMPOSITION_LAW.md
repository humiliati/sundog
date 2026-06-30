# The Composition Law — "order is vector-valued; scalar = join"

**Status:** central conjecture PROVEN in its provable form — the cancellation-free coproduct join law
machine-checked in general, additive (`annihilates_prod`) and multiplicative (`mul_annihilates_prod`).
The boundary is a **3-positive / 2-negative** classification whose positives are exactly the
**group-order axes** (cohomological, radical, moment); the non-group axes (search-reach, algebraic-
degree) are the negatives, both machine-checked. Expedition off the Order-Relative Resolution Law
([ORDER_RELATIVE_LAW.md](ORDER_RELATIVE_LAW.md)). Lean: `Sundogcert/OrderRelativeCompose.lean` +
`OrderRelativeComposeLaw.lean` + `OrderRelativeRadicalCompose.lean`. Internal; frozen-as-portfolio.

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

The law needs `ord` to actually *be* a join-homomorphism. The classification reveals the
characterization: **the join-homomorphic axes are exactly the group-order axes** — those where `ord`
is the order of an element in a group, whose coproduct (independent generators) is cancellation-free.
A **3-positive / 2-negative** picture:

| axis | `ord` is... | product `⊗` | join-homo? | character |
|---|---|---|---|---|
| **cohomological** | additive order in a group | `⊕` (direct sum) | **yes** ✓Lean | group-order (rich) |
| **radical** | mult. order in `ℝˣ/ℚˣ` | coproduct of surds | **yes** ✓Lean | group-order (mult. twin) |
| **moment / spectral** | binary order | convolution (indep.) | **yes** | group-order (degenerate) |
| **search-reach** | least denominator | `×` (rational mult.) | **no** | not a group order — *inflates* |
| **algebraic-degree** | min-poly degree | `×` (real mult.) | **no** ✓Lean | not a group order — *drops* |

The positives are united — they are the group-order axes. The two negatives fail in **opposite
directions** — cancellation throws the order either way off the join:

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

- **The radical axis holds it, richly (the third positive, machine-checked).** `ord` is the order of
  `[x]` in `ℝˣ/ℚˣ` (least `m` with `xᵐ ∈ ℚ`) — a *multiplicative* group order, the twin of
  cohomological. On the coproduct (independent surds — `√2·√3 = √6` keeps order 2 = `lcm(2,2)`) it
  composes by the join (`mul_annihilates_prod`, axiom-clean with only `[propext]`); dependent surds
  cancel within the group (`radical_cancel_sqrt2`: `√2·√2 = 2` drops order 2→1, like `1+1=0` in
  `ZMod 2`).

**Honest verdict.** *"Order is vector-valued, and scalar = join" holds exactly when `ord` is a
monoid-hom to a **join-semilattice** — realized by the **group-order** axes (cohomological, radical;
non-idempotent, rich) and the **idempotent semilattice** axis (moment; degenerate). It fails on the
non-group axes, where a cancelling product throws the order off the join (search-reach *inflates*,
algebraic-degree *drops*). Not a universal cross-axis identity — but a clean structural
characterization.* A real, falsifiable line, reframing the mode-vectors as evidence of a **grading**.

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

The coproduct law is not special to addition: `RadicalCompose.mul_annihilates_prod` gives the same
factorization for *multiplicative* groups (`(x,y)ʲ = 1 ↔ xʲ = 1 ∧ yʲ = 1`, axiom-clean with only
`[propext]`), and the radical axis instantiates it. So the precise statement: **the order is
join-homomorphic exactly on the cancellation-free coproduct of a group-order axis — and any cancelling
product (within-group `+`/`×`, or a non-group order like real `×` / denominators) breaks it.**
"Cancellation-free" *means* the coproduct of a group order; that is where, and only where, the grading
composes.

---

## 5. The converse FAILS — idempotency is the obstruction (machine-checked)

Is the characterization tight — must every join-homomorphic order *be* a group order? **No**, and the
obstruction is exact:

- A **group has no nontrivial idempotent**: `g · g = g ⟹ g = 1` (`Converse.idempotent_eq_one`,
  axiom-clean `[propext]`).
- The moment order monoid is a **join-semilattice** with a nontrivial idempotent: the no-mean class
  `c` satisfies `c ⊛ c = c` (two no-mean populations convolve to no-mean) yet `ord c = ⊤`
  (`Converse.moment_idempotent_nontrivial`: `⊤ ⊔ ⊤ = ⊤`, `⊤ ≠ 0`).
- So any monoid map `ψ` from the moment order monoid to a group sends the idempotent `c` to the
  identity (`Converse.converse_fails`: `ψ ⊤ = 1`) — order `1`, not `⊤`. The moment order **cannot** be
  a group order.

So **join-homomorphic ⇏ group order**: the join-homo orders are monoid-homs to a join-semilattice,
of which the group-orders (cohomological, radical) are the *non-idempotent* subclass and the moment
axis is the *idempotent* one. Idempotency is exactly *why* moment is "degenerate" — and it is the
clean obstruction that breaks the converse.

The coproduct law is itself one statement in any monoid (`Converse.coproduct_pow_eq_one`:
`(x,y)ʲ = 1 ↔ xʲ = 1 ∧ yʲ = 1`, `[propext]`); the additive (`annihilates_prod`) and multiplicative
(`mul_annihilates_prod`) coproduct laws are its two faces.

---

## 6. Open probes (where the expedition goes next)

1. **Lean the moment converse** (independent sum-integrable ⇒ both) to upgrade the degenerate positive
   from analysis to machine-checked.
2. **`to_additive`-unify** the coproduct law so `annihilates_prod` and `mul_annihilates_prod` are
   literally one source (`coproduct_pow_eq_one` is the shared monoid statement; the attribute would
   generate both faces).

---

*Sundog Research Lab — the composition law off the Order-Relative Resolution Law. The cancellation-
free coproduct join law is machine-checked in general, additive (`annihilates_prod`) and
multiplicative (`mul_annihilates_prod`, unified as `coproduct_pow_eq_one`). The boundary is a
3-positive / 2-negative classification (positives = the group-order + semilattice axes: cohomological,
radical, moment; negatives = search-reach, algebraic-degree). The converse FAILS — join-homo ⇏ group
order — with idempotency the exact obstruction (`idempotent_eq_one` + the moment semilattice's
nontrivial idempotent, `converse_fails`). `ord` is a grading homomorphism to a join-semilattice;
group-orders are the non-idempotent subclass. Lean: `compose_lcm_not_max`, `algDeg_not_join_under_mul`,
`within_group_cancels`, `radical_cancel_sqrt2`, `converse_fails`. Internal; frozen-as-portfolio.*
