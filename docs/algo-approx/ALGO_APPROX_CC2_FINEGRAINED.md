# C-C2 — fine-grained cornerstones as one tropical cost-ledger family

> **Slate hook:** [`ALGO_APPROX_CONJECTURE_SLATE.md`](ALGO_APPROX_CONJECTURE_SLATE.md) C-C2.
> **Verdict (2026-06-27): `UNIFIES_AS_TROPICAL_COST_FAMILY` (typed-positive, bounded).**
> The falsifier `REDUCTIONS_NOT_COST_PRESERVING` does **not** fire: the canonical
> fine-grained cornerstones live in the `CircuitNet` tropical fragment, their cost-ledger
> gate counts equal their DP/brute-force sizes, and the standard reductions among them are
> gate-count-preserving up to the fine-grained factor. **Bounded** by: the fine-grained
> *hardness conjectures* (the lower bounds) stay imported — the cost ledger measures
> construction *upper* bounds — and one cornerstone (3SUM) is genuinely **out of fragment**,
> marking the boundary. Analytical tabulation, no Lean.

## The two semirings, both inside the `CircuitNet` fragment

`CircuitNet` compiles the **piecewise-linear / tropical** fragment (`+`, scale, `max`,
hence `min`, `abs`) exactly into ReLU DAGs. The fine-grained landscape runs on two
semirings, *both* of which land in that fragment:

1. **`(min,+)` tropical** — `min`/`max` and real addition. *Native* `CircuitNet`: a
   min-plus DP is a tropical circuit, compiled exactly (this is the `ShortestPathCert` /
   `bellmanStep` story).
2. **Boolean `(OR,AND)` on `{0,1}`** — on the Boolean cube, `OR(a,b)=max(a,b)`,
   `AND(a,b)=min(a,b)`, `NOT(x)=1−x` (affine). So a Boolean circuit is a `(max,min,affine)`
   tropical circuit *agreeing with the Boolean function on `{0,1}`* (the ReLU net is its PL
   extension off the cube — irrelevant, the problem only queries `{0,1}`).

So both the `(min,+)` and the Boolean cornerstones are tropical, and the shared `costOf`
ledger reads their gate count uniformly.

## Tabulation

| problem | semiring | tropical form | `CircuitNet` gate count (construction = DP/brute-force size) | imported fine-grained conjecture |
|---|---|---|---|---|
| **APSP** | `(min,+)` | min-plus matrix powering (Bellman–Ford–Moore) | `Θ(n³)` (the `n³` relaxations) | no `O(n^{3−ε})` (APSP conjecture) |
| **min-plus matmul** | `(min,+)` | `C_ij = min_k(A_ik + B_kj)` | `Θ(n³)` | ≡ APSP (sub-cubic equivalent) |
| **negative-weight triangle** | `(min,+)` | `min_{i,j,k}(A_ij+A_jk+A_ki)` | `Θ(n³)` | ≡ APSP (Williams–Williams 2010) |
| **edit distance / LCS / alignment** | `(min,+)` | min-plus DP over an `n×n` grid | `Θ(n²)` cells | no `O(n^{2−ε})` under SETH (Backurs–Indyk 2015) |
| **Fréchet distance** | `(min,+)`-ish (min over a reachability DP) | min/max DP over the grid | `Θ(n²)` | no truly subquadratic under SETH (Bringmann 2014) |
| **Boolean matmul (BMM)** | `(OR,AND)` = `(max,min)` on `{0,1}` | `C_ij = max_k min(A_ik, B_kj)` | `Θ(n³)` combinatorial | no truly subcubic *combinatorial* BMM (conjecture) |
| **Orthogonal Vectors (OV)** | Boolean `(OR,AND)` on `{0,1}` | `max_{(i,j)} min_k (1 − min(a_k^{(i)}, b_k^{(j)}))` | `Θ(n²d)` | no `O(n^{2−ε})` under SETH (Williams 2005) |
| **3SUM** | **none** (additive combinatorics) | — *not* `(min,+)` or `(OR,AND)`; needs exact additive cancellation | — out of fragment | no `O(n^{2−ε})` (3SUM conjecture) |

## Worked example — Orthogonal Vectors is tropical

A pair `(a, b) ∈ {0,1}^d × {0,1}^d` is orthogonal iff `a_k · b_k = 0` for all `k`, i.e.
`AND_k ¬(a_k ∧ b_k)`. On `{0,1}`: `a_k ∧ b_k = min(a_k,b_k)`, `¬x = 1−x`, `AND = min`. So

> `orth(a,b) = min_k (1 − min(a_k, b_k))`  (= 1 iff orthogonal, else 0),

and the OV decision is `OV = max_{(i,j)} orth(a^{(i)}, b^{(j)})` — a `max`/`min`/affine
tropical circuit of `O(n²d)` gates that, by `compile_eval`, compiles to a ReLU DAG of the
same size computing OV exactly on the Boolean cube. The brute-force `O(n²d)` *is* the
tropical gate count.

## Reductions are cost-ledger-preserving (the falsifier check)

The canonical fine-grained reductions preserve the construction gate count up to the
factor that defines the fine-grained class, so the cost ledger transports hardness:

- **APSP ≡ min-plus matmul ≡ negative triangle** (Williams–Williams 2010) — tight,
  gate-count-preserving up to `O(1)`/log factors.
- **OV → edit distance / LCS / Fréchet** (Backurs–Indyk; Bringmann–Künnemann; Abboud et
  al.) — near-linear blowup, *subquadratic-preserving*: a sub-`n²` circuit for the target
  would give a sub-`n²` circuit for OV (the cost ledger carries the SETH lower bound).
- **SETH → OV** (Williams 2005) — the source of the quadratic wall, also a circuit-size
  argument.

None blows the gate count past the conjectured factor, so `REDUCTIONS_NOT_COST_PRESERVING`
does not fire on the tropical/Boolean cornerstones.

## The boundary — what does NOT unify

- **3SUM is out of fragment.** `a + b + c = 0` needs *exact additive cancellation* over
  the integers/reals, not `min`/`max`/Boolean structure; it is the additive-combinatorial
  cornerstone, and the tropical fragment (which has no two-variable product or exact
  equality test as a cheap gate) does not capture it. 3SUM-hard geometric problems inherit
  this. This is the honest edge of the unification.
- **Algebraic (`n^ω`) algorithms are off-ledger too.** Strassen-style fast matmul (and the
  `n^ω` BMM route) uses real *cancellation* (subtraction of products) — exactly the
  tropical-rational vs tropical-polynomial divide of [C-B1](ALGO_APPROX_CONJECTURE_SLATE.md).
  The tropical/combinatorial gate count is the `n³` route; the `n^ω` route lives on the
  cancellation side, outside the monotone tropical fragment.

## Honest verdict

The `(min,+)` and Boolean `(OR,AND)`-on-`{0,1}` fine-grained cornerstones form **one
tropical cost-ledger family**: each compiles exactly to a ReLU DAG whose gate count is its
DP/brute-force size, the `costOf` ledger reads them uniformly, and the standard reductions
preserve the measure. **What stays imported:** the fine-grained *hardness conjectures* —
the cost ledger gives construction *upper* bounds and the find-vs-check gap is the
conjecture (APSP / SETH / 3SUM), never proved here. **The boundary is named:** 3SUM
(additive cancellation) and the `n^ω` algebraic route sit *outside* the monotone tropical
fragment — the same cancellation wall C-B1 located. No promotion; this records a real but
bounded unification.
