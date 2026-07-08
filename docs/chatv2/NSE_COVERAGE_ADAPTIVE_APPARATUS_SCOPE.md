# Coverage-Adaptive Fiber Apparatus — Scope

> 2026-07-07. Successor apparatus scope after H3 closed at
> `NSE-H3-INCONCLUSIVE_COVERAGE` (`NSE_H3_ADMISSION_RECEIPT.md`, v1.2 receipt:
> both adjudicators deferred on the registered coverage gates, and the deferral
> was N-flat — candidate coverage 0.4588 → 0.4692 across a 4× sample lift). **This
> is an apparatus-generality question, not an H3 rescue.** H3 is closed; the
> anchor witness needs nothing from this. If a coverage-adaptive apparatus is
> registered, validated, and it reads the G=675 cell, H3 *reopens* under the new
> apparatus — and only if the apparatus first passes a regression gate that
> proves it measures the same thing the banked cells were measured with.
> Non-promotional; finite-Galerkin, proxy-relative; nothing here promotes C1.

## 0. The wall, receipts-true

At (k_f=3, G=675) the objective is powered and portable (held-out damp ~0.307 at
both N), the flow is chaotic, and the fiber pairs that *exist* behave anchor-like
(disagree ~0.033). The apparatus deferred because the **frozen-ε_K fiber ball
covers a shrinking fraction of a wider attractor:**

| cell | signature energy range | rule ε_K | twin candidate coverage (gate ≥ 0.50) | kNN sweep fit points (need ≥ 2) |
| --- | --- | --- | --- | --- |
| G=200 anchor | [0.715, 0.735] (1.03:1) | 0.060598 | 1.000 | full |
| G=300 anchor | (compact) | 0.066422 | 1.000 | full |
| G=675 v1.1 (50k) | [0.386, 1.352] (3.5:1) | 0.058934 | 0.4588 | 0 |
| G=675 v1.2 (200k) | same | 0.058934 | 0.4692 | 0 |

The +0.010 coverage move per 4× samples is the diagnosis: the shortfall is
attractor geometry (large local effective dimension over most of the support),
not sampling. More N will not clear it at house scale.

## 1. What the coverage gate is (so the fix is exact)

Both adjudicators use one absolute radius `ε_K = 0.05·√(2·E_max)` (global E_max):

- **kNN-sweep:** `fidelity_coverage` = fraction of samples whose k-th neighbor
  distance `r_k ≤ ε_K`. A sweep point enters the fit only if its coverage is
  sufficient; `fit_point_count < 2` ⇒ `INCONCLUSIVE_CONVERGENCE`.
- **twin-state:** `candidate_sample_fraction` = fraction of samples with ≥ 1
  signature-near pair within `ε_K`; `< s_pos = 0.50` ⇒
  `TWIN_STATE_DEFERRED_COVERAGE`.

The single lever is: **a fixed absolute radius against a spread-out
distribution.** Every honest fix must widen coverage without loosening the
fidelity meaning of "near" — or a positive can be manufactured.

## 2. Two candidate apparatus (the heart; they differ in what they touch)

### Approach A — fixed-radius, density-stratified (adaptive-k, ε_K UNCHANGED)

Keep `ε_K` frozen by the same global rule. **Invert the query:** instead of
"is the k-th neighbor within ε_K" (fails when the k-th neighbor is far), do a
fixed-radius query — count neighbors *within* ε_K, and admit a fiber wherever the
count ≥ `k_min` (= the banked 50). Read the witness on the union of admitted
(dense) fibers, and report the **attractor-fraction covered** as a first-class
scope limit.

- **Provably does not touch fidelity:** no pair farther than ε_K is ever
  compared. A changes only *where* the test is honestly applicable and forces
  the covered fraction into the claim.
- **Honest failure mode:** if the attractor is uniformly sparse at ε_K (no
  k_min-dense balls anywhere), A defers again — but with a sharper type ("no
  ε_K-dense neighborhoods at k_min", i.e. a genuine geometric fact about the
  cell), not a bookkeeping deferral.
- **Claim shape if it lands:** "control-sufficient on the ε_K-dense fibers,
  which are `f` of the sampled attractor" — a scoped positive, never a global
  one. `f` is reported, never gated away.

### Approach B — regime-conditioned ε_K (relative fidelity; more power, real risk)

Replace the global radius with a pointwise `ε_K(u) = 0.05·√(2·E_local(u))` —
the same 5%-relative-resolution criterion applied to the local signature scale
rather than the peak. Coverage recovers because the ball grows proportionally in
high-energy regions.

- **This changes the fidelity rule** — and is the one move in this lane that can
  manufacture a false positive (an inflated ε_K(u) sweeping in genuinely
  different states). It is admissible **only** behind the §3 regression gate as a
  hard precondition, and only as the registered escalation if A defers.
- **Built-in cross-check:** the witness disagree-fraction is scale-free, so
  adaptive-B disagree at G=675 must be read against the banked anchor disagree
  (0.033); a B apparatus that lands a "positive" by inflating ε_K would show its
  hand as an anomalous disagree signature.

## 3. The regression gate (PRECONDITION — this is what makes it honest)

Before either apparatus is permitted to read G=675, it must **reproduce the
banked frozen-ε_K verdicts on the compact cells:**

- Re-run G=200 and G=300 with the new adjudicator. On a compact attractor,
  Approach A reduces to the frozen adjudicator by construction (every ε_K-ball is
  k_min-dense) and Approach B has ε_K(u) ≈ global ε_K (energy near-constant), so
  **both must return the banked verdicts** — G=200/G=300 `TWIN_STATE_CERTIFIED`,
  G=200 `STRICTNESS_WITNESS_POSITIVE` — with disagree fractions matching the
  banked 0.0367 / 0.0382 to a tight registered tolerance (e.g. ±0.005).
- **Fail ⇒ `NSE-H3-APPARATUS-REJECTED`, final:** the apparatus is not measuring
  what the banked cells were measured with; no G=675 read is taken. The apparatus
  itself is the finding.

This is the load-bearing new element versus every prior H3 rung: a coverage fix
is only admissible if it is verifiably the *same test* on the cells where the
test already worked.

## 4. Recommendation + kill conditions

**Primary: Approach A.** It cannot manufacture a positive — it changes coverage
bookkeeping, not the fidelity criterion, and it converts the deferral into either
a scoped positive (with a covered-fraction fence) or a sharper geometric
deferral. **Escalation: Approach B, only if A defers again**, and only after B
clears §3. Kill A/B immediately if the §3 regression fails, or if A's covered
fraction is so small (< a pre-registered floor, e.g. 0.10) that the scoped claim
is vacuous — a sliver-core positive is not worth a new apparatus.

## 5. Rungs and honest cost (samples are NOT banked)

The banked runs persisted only summaries + witness CSVs, so **every rung
re-integrates** (this is real compute, not post-processing). The new adjudicator
should additionally persist the raw sample arrays (signatures + high modes) so
any *third* apparatus iteration is pure post-processing.

```text
R0  regression: G=200 + G=300 with the new adjudicator (twin), N=50k   ~35 min each, agent-runnable
    [gate §3: reproduce banked verdicts + disagree within tolerance]
R1  application: G=675 fallback cell (N=200k) with the new adjudicator   ~2 h, owner-run
    [only if R0 passes]
```

R0 is cheap enough to run agent-side; R1 is the owner-run lock. No G=675
re-integration happens unless the apparatus has earned it on the compact cells.

## 6. Branch table

| outcome | verdict |
| --- | --- |
| R0 regression fails (adaptive ≠ banked on G=200/G=300) | `NSE-H3-APPARATUS-REJECTED` (final; apparatus is the finding) |
| R0 passes; R1 covered fraction < floor | `NSE-H3-COVERAGE-SLIVER` (apparatus valid but the cell has no substantial resolvable fiber structure) |
| R0 passes; R1 powered + covered, witness constant | `NSE-H3-FORCING-GENERAL` (scoped to covered fraction `f`; the reopened positive) |
| R0 passes; R1 powered + covered, witness fails | `NSE-H3-GRASHOF-LOCAL` (the informative failure, now reachable) |
| R0 passes; R1 still deferred (A: no k_min-dense balls) | `NSE-H3-COVERAGE-WALL-CONFIRMED` (a geometric fact about the cell, sharper than the N=200k deferral) |

## 7. The one owner decision

Two coupled choices, both needed before build:

1. **Which apparatus is primary** — A (recommended) or B, with B otherwise the
   registered escalation.
2. **Additive harness sign-off** — a new `--adjudicator` mode (e.g.
   `twin-state-adaptive`) that computes the adaptive read *alongside* the frozen
   one in the same pass (so R0 emits both verdicts on identical samples), plus the
   raw-sample export. Additive-only; no existing adjudicator path changes;
   validated by the existing self-test + a non-verdict smoke + the §3 regression
   before any G=675 read.

On sign-off I freeze the apparatus spec (the A/B math, k_min, tolerances, the
covered-fraction floor), build the adjudicator, and run R0 — the regression gate
is agent-runnable, so the apparatus either earns its G=675 read or is rejected
without owner compute.

## 8. Does not claim

No promotion, no infinite-dimensional statement, no change to H1's proxy-relative
typing or the closed slate ledger. A reopened `FORCING-GENERAL` here is scoped to
a covered fraction of one matched-Re cell under a new apparatus — one point, one
axis, fenced. The apparatus does not retroactively touch any banked verdict; the
G-axis cells stand exactly as receipted. `docs/chatv2/` stays no-publish.

Cross-refs: `NSE_H3_ADMISSION_RECEIPT.md` (the three walls),
`NSE_H3_KF3_SCOPE.md`, `PDE_C1_REGIME_GENERALITY_v1.md` (§3 ε_K rule,
adjudicator gates), `PDE_C1_TWIN_STATE_CERTIFICATE.md` /
`PDE_C1_KNN_ADJUDICATION_DESIGN.md` (the frozen adjudicators this extends),
`results/proof/c1-h3-kf3-g675-fb-{knn-sweep,twin}/` +
`c1-paired-fiber-g{200,300}/` (the banked comparators),
`NSE_STATIONARITY_GATE_CHECKLIST.md` (H4 discipline carried).
