# Structural Failure Coincidence — Cut 2 C3-A Numeric Freeze + Decoy Receipts (folds C3-B/C/D)

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Parent: [`P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md`](P2_CUT2_C3_DECOY_TERM_AND_TEMPTATION.md) (C3-A/B/C/D)
Inherits: [`P2_CUT2_C2A_NUMERIC_FREEZE.md`](P2_CUT2_C2A_NUMERIC_FREEZE.md) (scale/seed/grid/readout) · [`P2_CUT2_C2B_PEN_AND_QA.md`](P2_CUT2_C2B_PEN_AND_QA.md) (`π_route` well-posed)
Controller: [`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md)
Audited by: [`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md) (D2 = the shared argmax-sensitivity floor)
Filed: **2026-05-16 (PT)**. Status: **C3-A FILED FOR AUDIT — HOLD FOR
EXECUTION**. Like C2-A, this freezes **structure, provenance, and receipt
obligations**; concrete `[E]` values and the receipt tables are the
maintainer's pre-run fill (no fabrication), and a failing receipt
**blocks (append-only redesign), never tunes**. Cut-2 execution remains
HELD on the C3 receipts, C4-A/B/C/D, C5, and a fresh P2-spec admission
re-check. No harness written; nothing run.

## Purpose

The C3 numeric freeze in one pass: fix `P_in`, `(w,b)`, `κ`, `σ_D`,
`M`, the edit operators, and the C3-C argmax-sensitivity floor; fold in
the C3-B calibration window and the C3-D `π_route` coupling (now
dischargeable because C2-B made `π_route` well-posed); propagate the
C2-A controller-semantics correction across the whole C3 column.

## 1. Inheritance + propagated controller readout

C3-A **inherits, not redefines**, from C2-A: the bridge scale
(eligible-band route peak ≡ 1.0), the RNG `seed`, the `h`-grid, and the
`q` domains. Every place a C3 receipt reads the **bound controller's**
behavior it uses the **C2-A-1 sustained-TRACK confident-`q̂` criterion**:
a "confident `q̂`" means a sustained TRACK lock under
`PhotometricAgent`'s real SCAN→SEEK→TRACK machine (threshold enforced
only in TRACK after `reacquire_hold_steps`); "abstain / refuse" means
reacquire / never-sustains-TRACK. C3 must not infer lock from peak
intensity alone — the C2-C/D hold #2 correction propagates here intact.

## 2. C3-A keystone load-bearing self-seal (surfaced adversarially)

The C3 analog of C2-A §5. The keystone hazard is the **`P_in` decoy↔`h`
in-sample correlation strength** (with `κ`, `M`). If `P_in`'s
correlation, `κ`, or `M` are chosen *after* seeing controller / `π`
behavior so the temptation "comes out tempting-but-reversible," the
C3-T / C3-B receipts are rigged. Therefore the **`P_in` decoy↔`h`
correlation is frozen by an independent principle before any run** —
pre-registered from each decoy's `BOUNDARY_MAP` characterization
("co-varies with `h` in any finite sample"), **not** fit to outcomes.
With `P_in` independently frozen, C3-T's margin and C3-B's window are
genuine pass/fail with no correlation-strength freedom. A failing
receipt **blocks (append-only redesign), never a tuned
`P_in`/`κ`/`M`** (A3).

## 3. The C3 numeric freeze (provenance-tagged, A3)

`[G]` = immutable / inherited boundary (change = forbidden goalpost
move); `[E]` = pre-registered engineering tolerance (amend-only,
justified, never post-results).

| symbol | role | provenance |
| --- | --- | --- |
| bridge scale, `seed`, `h`-grid, `q` domains | inherited from C2-A | **[G]** (inherited) |
| sustained-TRACK confident-`q̂` readout | controller lock/abstain definition | **[G]** (inherited C2-A-1) |
| `P_in` h-sampling dist | in-sample altitude distribution | **[E]** |
| `P_in` per-decoy decoy↔`h` correlation coeffs | the temptation source | **[E]**, frozen by the §2 independent principle |
| `(w,b)` least-squares decoy→`h` fit | the A2 positive-control policy | procedure **[G]**; fit on frozen `P_in`+`seed`, never refit post-results |
| `κ` decoy-ridge weight | temptation strength | **[E]** (C3-B window) |
| `σ_D` decoy-ridge width | ridge sharpness | **[E]** |
| `M` temptation margin | π_dec must beat π_route in-sample by ≥ M | **[E]** |
| `τ_pc` positive-control move | reused from the frozen A2 spec (`2.0°`) | **[G]** (prior freeze) |
| decoy-edit operator + magnitude | the q2 decoy-edit | **[G]** (spec q2) + magnitude **[E]** |
| counterfactual handle-edit | the q2 handle-edit | **[G]** (spec q2) |
| C3-C argmax-sensitivity floor | reachability `‖Δ(argmax_q I)/Δd‖` | **[E]**, **= the C4 D2 floor (one number, not two)** |
| C3-B(i)/(ii) demo params | calibration window | **[E]** |

Actual `[E]` values are fixed in this freeze and A3-immutable from this
point; the admission re-check audits them, it does not let them move
after results.

## 4. The three C3 receipts (pre-run, computed, frozen before admission)

Mirrors C2-A's receipt discipline; the bound-controller readout is the
inherited sustained-TRACK criterion (§1).

- **C3-A-R — reachability (C3-C, refined).** Tabulate
  `‖Δ(argmax_q I)/Δd‖` over the frozen decoy-edit; it must exceed the
  shared C3-C/D2 floor — decoys move the *converged answer*, not merely
  the pointwise gradient (which the Gaussian ridge zeroes at its peak /
  under clipping). Removes Cut-1's `∂J/∂d ≡ 0` vacuity by computation.
- **C3-A-T — temptation + reversal (C3-T).** Under the §2-frozen
  `P_in`: `mean|π_dec − h|_in ≤ mean|π_route − h|_in − M` (π_route is
  C2-B-well-posed), **and** the advantage reverses — decoy-edit:
  `π_dec` error ≥ `τ_pc`, `π_route` `|Δq̂| ≤ 0.5°`; counterfactual
  handle-edit: `π_route → h′` within `τ2`, `π_dec` stale. (`π_route`,
  `π_dec` are argmax-style reference policies; predictions about the
  *bound controller* refusing/riding use the §1 readout.)
- **C3-A-B — calibration window (C3-B).** The frozen `κ` satisfies both
  (i) temptation real (C3-A-T.1) **and** (ii) on the eligible band the
  `I_route` optimum stays a feasible competing optimum (C2-B made this
  well-posed; the C3-B(ii)↔C2-B coupling is now dischargeable).

## 5. Honest couplings

- **C3-A ↔ C2-A.** Scale, seed, grid, and the sustained-TRACK readout
  are inherited; C3-A may not contradict them.
- **C3-A-R floor = C4 D2 floor.** One shared number; C4-A must use the
  same value (recorded in both).
- **C3-A-T/B ↔ C2-B.** `π_route` is well-posed only because C2-B fixed
  `pen(q)`/`q_a`; the C3-D coupling is design-discharged, numerically
  closed at the joint re-run.
- **q2 edit operators are shared** with the frozen spec quantities; C3
  reuses, never redefines, them.

## Cut-2 C3-A binding rules

1. Every §3 number is frozen here; `[G]`/inherited immutable, `[E]`
   amend-only/justified/never-post-results (A3).
2. `P_in`'s decoy↔`h` correlation, `κ`, `M` are frozen by the §2
   independent principle **before** any run; post-hoc tuning ⇒ **void**.
3. Bound-controller readouts use the inherited sustained-TRACK criterion;
   inferring lock from peak intensity alone ⇒ **void**.
4. `(w,b)` is fit once on frozen `P_in`+`seed`; refit after any result
   ⇒ **void**. Reading true `h` in `D`/`ĥ_dec` ⇒ **VOID** (A1).

## Explicit non-bindings (cannot satisfy C3-A)

- Tuning `P_in`/`κ`/`M`/`σ_D` after a controller or `π` result.
- A coarse peak-intensity lock readout instead of the sustained-TRACK
  criterion.
- Pointwise `∂I/∂d` instead of the argmax-sensitivity receipt.
- A C3-A-R/T/B "pass" computed against a strawman instead of the bound
  `PhotometricAgent` / the frozen reference policies.
- A C3-A-R floor that differs from the C4 D2 floor.

## Open items

C3-A files the freeze structure + the three receipt obligations. As with
C2-A, the concrete `[E]` values and the C3-A-R/T/B receipt tables are
the maintainer's pre-run fill (no fabrication); a negative receipt
blocks Cut-2. Still-open siblings: **C4-A/B/C/D**, **C5**.

After C2-A, C3-A (incl. receipts), C4-A/B/C/D, and C5 are all filed, the
P2-spec admission check **must be re-run** as one audit of the whole
discriminating cut; only on **ADMIT** may a Cut-2 harness be built or
run. Public-Language Constraint remains fully in force: no `CONFIRMED` /
traceability-success / theorem language anywhere (including the rail).

## Honest prior (unchanged)

A correctly-calibrated tempting decoy against a real inverse-free ESC
controller keeps the likely honest outcome at **D / BOUNDARY FOUND** —
the Proxy-Collapse confirmation avenue (`debunked.md`, P1 §C). **B** is
earned *only* by a measured refusal of the tempting decoy at the
quantified in-sample cost **and** emergent failure coincident with
L1/L2/L3. Either is a clean result; the in-between is not.

## Audit Notes

*(reviewer space — append-only below)*

**2026-05-16 (PT) — Codex audit.** Direction accepted; C3-A is **not
yet execution-closing**. The file correctly propagates both C2-A holds:
C3 inherits the sustained-TRACK confident-`qhat` readout, and it treats
numeric values/receipts as pre-run fill rather than fabricated results.
It also correctly upgrades C3-R to argmax-sensitivity and keeps the C3
D2 floor shared with C4. Three admission holds remain. (1) C3-A inherits
C2-A's scale/seed/grid/readout, but C2-A is itself still a scaffold
until its concrete [E] values and receipts land; any C3 receipt computed
before C2-A closure is provisional and cannot admit Cut-2. (2) The text
must be read as freezing **slots/provenance/obligations**, not actual
numeric values, until `P_in`, decoy-correlation coefficients, `kappa`,
`sigma_D`, `M`, edit magnitudes, and the shared C3-C/C4-D2 floor are
filled with concrete values and tables. The sentence "Actual [E] values
are fixed in this freeze" is therefore a future-facing obligation, not a
closure ruling. (3) The independent `P_in` principle needs an
operational receipt: exact finite sample or generator, seed, decoy
coefficient table, and frozen `(w,b)` fit, before C3-A-T/B can be
audited. Using true `h` to fit `(w,b)` is allowed only as this pre-run
reference artifact; the runtime adapter/decoy ridge must read frozen
coefficients and observable `d` only, with C4-D taint checking no `h`
input. No controller/harness run.

**2026-05-16 (PT) — maintainer. Wave-4 C3-A receipts filed.** Canonical
generator: `scripts/cut2-c3a-w4.mjs` (SHA-256
`85d7d0a06548e777b5022c1af00ed357dfe2da09325b5543a044a5f35bec7707`).
Artifacts: `c3a-pin-generator.json`, `c3a-r-receipt.json`,
`c3a-t-receipt.json`, `c3a-b-receipt.json`, and `c3a-w4-summary.md`
under `results/structural-failure/cut2-prereg/`. Rerun receipt hashes:
`c3a-pin-generator.json` raw
`ba5445a581f1a9e091f894ba96fe47982d2c371e00dbe3caa1f6d31f2cb7f2ee`,
canonical
`84b21ca8b13107e794970dbe1fe1c289d2a441d2a2d2089c66db9bba3c29d826`;
`c3a-r-receipt.json` raw
`4c2e29ecc6aee56b1c056ea5434038e411c5e446fbac6af8ac809ac8bde18bcf`,
canonical
`79fb2aec36bca183172ab9523df7cd68d2d4f0a8b0c5153a13c112e21dfba0dd`;
`c3a-t-receipt.json` raw
`ca1cfff12420b2b56aec63ac502a49af09a6c81dc714149a460512c0886cd0ec`,
canonical
`8d0183ad6d16efe315188d7054b3a6a88d04d81b2524ab5289a2440bc6808685`;
`c3a-b-receipt.json` raw
`6af8919d00165fe2723c58f5d128dbc5ac4aa87fe9e2fde9039dab1188bae5df`,
canonical
`059f038d5db09f51f24a7f69698c3521c5e0293e224d8b79c25a929e478ca1cf`;
`c3a-w4-summary.md` raw
`93c8c0e8f7e90961fea3e7022a0cde77808148a696c9a0f7028ada996813d3c8`.
Verdict: **C3-A-R PASS**, **C3-A-T BLOCK**, **C3-A-B BLOCK**. The
reachability receipt clears under the pre-audited Path-B median rule
(`421/648 = 65.0%` above `F* = 0.05 deg`, required at least `50%`).
Temptation fails on the L1-eligible-by-obs subset: mean `|pi_dec - h|`
`3.806 deg` vs mean `|pi_route - h|` `1.398 deg`, margin
`-2.408 deg` where `+0.5 deg` is required. Reversal subtests pass
(`pi_dec` breaks under decoy edit, `pi_route` stays invariant, handle
edit steers `pi_route` while `pi_dec` is stale), but T1 is binding.
C3-A-B fails because sub-(i) inherits the T1 BLOCK and sub-(ii) preserves
the route argmax as a local maximum on only `343/479 = 71.6%` of
L1-eligible-by-obs rows where `90%` was required. This is a permanent
Wave-4 receipt, not a value-tuning prompt.

**2026-05-16 (PT) — maintainer. Wave-4.1 Path Y + Path Z amendment
filed; Path W closeout selected.** Canonical amendment generator:
`scripts/cut2-c3a-w4-v2.mjs` (SHA-256
`882a2c5b393a1d3c3a5f6ce75b2daaf09502d67f6cb74a162f9588b6a41955ed`).
Companion amendment draft/helper retained at
`scripts/cut2-c3a-w4-amendment.mjs` (SHA-256
`ae56488d7d2d8f1086de280c202d36ae9ffe27f46ca6ac764633ee05ccca2974`).
Artifacts: `c3a-r-receipt-v2.json`, `c3a-t-receipt-v2.json`,
`c3a-b-receipt-v2.json`, and `c3a-w4-v2-summary.md`. Rerun receipt
hashes: `c3a-r-receipt-v2.json` raw
`3141834b9919ebdd8467eec726b71b5a8bfefec93cc3a3dde774c42389f71d08`,
canonical
`a9839c6230ae3609c2be8c890cf6f66712ed7d8a7aa56f98aed8c75a7dfa1fab`;
`c3a-t-receipt-v2.json` raw
`dfe60fefbdedfc77bffcbc1bf07fd38f6361317776a5953fc46278fb9f303162`,
canonical
`a8d1c892bc7df0a0e0433647b5c036972fa79ee1f700f30868cc6a543ca89e26`;
`c3a-b-receipt-v2.json` raw
`ecfc38483e379d350cd13220e67a66d43e1c1efe2c97675d64a9d7d72fdc0382`,
canonical
`b81843cebcce0b1a5a30ea63a29bb34f9bd3f1f10fbb9fe5628579971ce71193`;
`c3a-w4-v2-summary.md` raw
`8bf109dfca4577cb3cc17ade18100287e9a97cfb4bc363feb5901b8a40cc9092`.
Path Y expands T1 to the full non-degenerate `P_in` (`587` rows:
`479` L1-eligible-by-obs, `108` L1-ineligible-by-obs) and lets
`pi_route` emit the biased `q_naive` where it is defined rather than
restricting to eligible rows only. Path Z relaxes C3-A-B sub-(ii) to
"route basin represented": a local maximum within `0.5 deg` of
`q_naive`, accepting merged small-delta peaks as represented rather than
erased. Results remain **BLOCK/BLOCK**. C3-A-T v2: mean
`|pi_dec - h| = 3.567 deg` vs `|pi_route - h| = 1.860 deg`, margin
`-1.706 deg` where `+0.5 deg` is required. Subset finding: decoys help
exactly where the route is weak (`L1-ineligible-by-obs`: `pi_dec`
`2.506 deg`, `pi_route` `3.912 deg`, decoy wins by `1.406 deg`) and
hurt where the route is strong (`L1-eligible-by-obs`: `pi_dec`
`3.806 deg`, `pi_route` `1.398 deg`, route wins by `2.408 deg`). The
sample-count-weighted non-degenerate average is dominated by the larger
eligible subset, so `pi_route` wins overall. C3-A-B v2: sub-(i) remains
BLOCK via T1; sub-(ii) improves from `71.6%` to `381/479 = 79.5%` but
still misses the frozen `90%` threshold. **Path W is selected**:
Wave-4 v1 and Wave-4.1 v2 BLOCK receipts are permanent. No leverage
weighting, kappa increase, threshold relaxation, or decoy re-pinning is
performed inside Wave 4. Any Wave-4.2 response is a separate
freeze-level redesign discussion. Cut-2 execution remains HELD; Public
Language Constraint remains in force.
