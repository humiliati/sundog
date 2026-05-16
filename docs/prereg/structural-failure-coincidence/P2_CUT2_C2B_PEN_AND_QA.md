# Structural Failure Coincidence — Cut 2 C2-B Resolution: `pen(q)` and the admissible `q_a` range

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Parent condition: [`P2_CUT2_C2_NUISANCE_AND_BRIDGE.md`](P2_CUT2_C2_NUISANCE_AND_BRIDGE.md) (C2-B)
Consumers unblocked: [`P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md`](P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md) (C3-T, C3-B(ii), C3-D) · [`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md) (C4-C / D1)
Filed: **2026-05-16 (PT)**. Status: **C2-B FILED FOR AUDIT — HOLD FOR
EXECUTION**. Resolves the free-`q_a` degeneracy by design only. Cut-2
execution remains **HELD** on the C2-B calibration blocker below, the
still-open C2-A/C/D, C3-A/B/C/D, C4-A/B/C/D, and C5, and a fresh
P2-spec admission re-check. No harness written; nothing run.

## Purpose

C2-B is the cascade hub. The C2 route ridge
`I_route(q;bundle) = exp(−[ f_par_obs − R22/cos(q_h) − q_a ]² / 2σ²)`
with `q_a` **unconstrained** has a continuum of exact maxima: for any
`q_h`, set `q_a = f_par_obs − R22/cos(q_h)` ⇒ bracket `= 0` ⇒
`I_route = 1`. So `argmax_q I_route` is not unique, `π_route` is
undefined, C2 P-A is undefined, and — via C4-C — D1 has no
construction-level object to audit. This resolution makes the route
optimum unique and well-posed, which unblocks C3-T's `π_route` baseline,
C3-B(ii), C3-D, and C4-C/D1.

## 1. The resolution — receipt-grounded convex anchor-prior penalty

The reason `q_a` exists is to let the controller hypothesise the
unknown anchor error `ε`. C2 already declares that error **bounded and
zero-centred**: `ε ~ U[−A,+A]`, `A = ρ·R22`. So:

- **Admissible range:** `q_a ∈ [−A, +A]`, `A = ρ·R22`. This is the
  receipt-grounded anchor-error support. `A` is a constant times the
  scale-lock `R22`; **no `h`** (A1-safe).
- **Penalty:** a strictly convex, zero-centred prior on the anchor
  correction (Occam: the smallest anchor correction consistent with the
  signal):

  ```
  pen(q) = λ · (q_a / A)² ,   λ > 0   (pre-registered, A3 tolerance)
  O(q)   = I_route(q;bundle) − pen(q)        ← what the controller climbs
  ```

### Unique argmax (degeneracy broken)

On the zero-bracket manifold `q_a*(q_h) = f_par_obs − R22/cos(q_h)`,
`O = 1 − λ(q_a*(q_h)/A)²`, uniquely maximised where `|q_a*|` is
**minimised**, i.e. `q_a* = 0` ⟺ `R22/cos(q_h) = f_par_obs` ⟺
`q_h = arccos(R22/f_par_obs)`. Off the manifold `I_route < 1` and
`pen ≥ 0`, so `O < 1`. Hence the **unique global maximum** is

```
(q_h*, q_a*) = ( arccos(R22 / f_par_obs) , 0 ) ,   O* = 1
```

`q_h*` is exactly `q_naive(h,ε) = arccos(R22/f_par_obs)` — the **biased
naive inverse**, not true `h` (bias = the C2 anchor-noise bias, which
blows up at the L1 low-leverage band). Therefore:

- **`π_route` is well-defined** (climb `O`; converges to `q_naive`).
- **C2 P-A holds and is now computable:** `argmax_q I_route(−pen) =
  q_naive ≠ h` on the must-differ band — the route construction is an
  honest-but-anchor-biased inverse, never a readout of the hidden cause.
- **P-B unaffected:** `O` is smooth; climbability is the separate
  conditioning question pinned numerically with C2-A.

## 2. C2-B load-bearing calibration window (surfaced adversarially)

`λ` must thread a window — the now-standard pattern (cf. C3-B, C4-B):

- **`λ` too small ⇒ degeneracy not broken in practice.** The
  along-manifold curvature of `O` near `q_h*` scales like
  `2λ·(R22 sin q_h*/cos²q_h* / A)²`; as `λ→0` it →0, `argmax` is
  numerically non-unique, `π_route` non-reproducible.
- **`λ` too large ⇒ it moves the optimum.** A steep `q_a`-penalty makes
  accepting a small `r≠0` (exp cost `~r²/2σ²`) worthwhile to shrink
  `q_a`, pulling `q_h*` off `arccos(R22/f_par_obs)` — P-A would then be
  a λ-artifact, not the biased naive inverse.

C2-B is not closed until a **pre-run numeric demonstration** shows the
frozen `λ` satisfies **both**:

- **C2-B(i):** the post-penalty argmax is unique with along-manifold
  conditioning ≥ a pre-registered floor on the eligible/biased band
  (degeneracy genuinely broken; `π_route` reproducible).
- **C2-B(ii):** the located `q_h*` deviates from
  `arccos(R22/f_par_obs)` by < a pre-registered negligible tolerance
  over the C2 bias grid (the penalty breaks the tie **without moving
  the optimum** — P-A stays exactly the biased naive inverse).

## 3. Honest couplings and cross-condition findings (recorded, not papered over)

- **C2-B(ii) ↔ C2-A.** The (ii) demonstration is computed on the C2
  bias grid; `λ`, the conditioning floor, and the P-A tolerance
  **fold into the C2-A numeric freeze** (one set of frozen numbers).
- **Exposes C2-D, does not absorb it.** When `f_par_obs < R22` there is
  **no** zero-bracket manifold (`R22/cos(q_h) ≥ R22 > f_par_obs` ∀
  `q_h∈[0,80]`); the unique `O`-max then sits at the penalty-best-fit
  domain edge. C2-B makes this geometry explicit; classifying it
  (abstain / invalid) is **C2-D's** job and is deferred there, not
  silently resolved here.
- **C4-C / D1 clarification (genuine tension, flagged for the C4
  reviewer).** Once C2-B makes `argmax I_route(−pen) ≡
  arccos(R22/f_par_obs)` *by construction*, C4-C's stated D1 — "compare
  `argmax I_route` against `arccos(R22/f_par_obs)`" — becomes
  **degenerate** (the two are identical, so a literal reading makes D1
  trivially fail). D1's anti-Cut-1 intent must therefore be the **P-A
  form**: `argmax I_route (= q_naive) ≠ **true h** on the must-differ
  low-leverage band` — the route construction is not a clean readout of
  the *hidden cause* (the actual Cut-1 sin), not "differs from its own
  closed form." This is raised for the C4 reviewer in the
  `P2_SPEC_ADMISSION` filing log; the frozen C4 body is **not** edited
  here.

## 4. Cut-2 C2-B binding rules

1. `q_a` is hard-clamped to `[−A,+A]`, `A = ρ·R22`; an unbounded `q_a`
   in any Cut-2 run is **void**.
2. `pen(q) = λ(q_a/A)²` reads only `q_a` and the constant `A`; any
   `h`-dependence in `pen` or the `q_a` range ⇒ run **VOID** (A1).
3. `λ`, the C2-B(i) conditioning floor, and the C2-B(ii) tolerance are
   pre-registered with the C2-A freeze and **never** edited
   post-results (A3). Immutable geometry/receipt boundaries unchanged.
4. The C2-B(i)/(ii) artifacts are produced and frozen **before** any
   controller instantiation.

## Explicit non-bindings (cannot satisfy C2-B)

- An unbounded or `h`-dependent `q_a` / `pen` (degeneracy or A1 leak).
- A non-convex / non-zero-centred `pen` that does not single out the
  minimal anchor correction (degeneracy not provably broken).
- A `λ` outside the C2-B(i)/(ii) calibrated window.
- Quietly handling `f_par_obs < R22` here instead of in C2-D.
- Tuning `λ`/floor/tolerance after seeing any controller result.

## Open items

C2-B files the `pen(q)` form, the admissible `q_a` range, the unique-
argmax derivation, and the C2-B(i)/(ii) calibration obligation for
audit. Still open before any Cut-2 run:

- **C2-A** absorbs the C2-B numerics (`λ`, conditioning floor, P-A
  tolerance) alongside `ρ, A, σ, seed`, the grid and `q_h/q_a` domains.
- **C2-C** (leverage-confidence function) and **C2-D** (`f_par_obs<R22`
  invalid/abstain handling — geometry now made explicit here) remain
  open.
- **C4-C/D1** comparison target needs the P-A reformulation above
  (flagged, reviewer to rule).
- Still-open siblings: **C3-A/B/C/D**, **C4-A/B/C/D**, **C5**.

After C2-A/B/C/D, C3-A/B/C/D, C4-A/B/C/D, and C5 are all filed, the
P2-spec admission check **must be re-run** as one audit of the whole
discriminating cut; only on **ADMIT** may a Cut-2 harness be built or
run. Public-Language Constraint remains fully in force: no `CONFIRMED` /
traceability-success / theorem language anywhere (including the rail).

## Honest prior (unchanged)

A well-posed honest route (biased naive inverse, C2-B) climbed by a
real inverse-free ESC controller (C1) against a biased signal (C2) with
a tempting reachable decoy (C3) keeps the likely honest outcome at
**D / BOUNDARY FOUND** — the Proxy-Collapse confirmation avenue
(`debunked.md`, P1 §C). **B** is earned *only* by a measured refusal of
the tempting decoy at the quantified in-sample cost **and** emergent
failure coincident with L1/L2/L3. Either is a clean result; the
in-between is not.

## Audit Notes

*(reviewer space — append-only below)*
