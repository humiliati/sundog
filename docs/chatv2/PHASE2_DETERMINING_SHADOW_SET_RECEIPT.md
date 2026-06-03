# Chat-v2 Phase 2 — Determining-Shadow-Set Receipt

> 2026-06-03. Run + adjudication of the frozen spec
> `PHASE2_DETERMINING_SHADOW_SET_SPEC.md` (+ Amendment 1). **Status: internal
> receipt, not a promotion.** Synthetic toy residual stream; no real-LLM, NSE,
> consciousness-field, or side-channel claim. The test is partial observability on
> saved chatv2 bodies.

## Provenance

- Spec: `docs/chatv2/PHASE2_DETERMINING_SHADOW_SET_SPEC.md` (frozen + Amendment 1).
- Runner: `scripts/chatv2_phase2_shadowset.py`. Outputs:
  `results/chatv2/phase2-determining-shadow-set/` (7 files).
- git `d3be41a1` (working tree dirty: 8 files — this lane's edits, uncommitted).
- Wall-clock **498.5 s** (inline). PROBE_SEED 0; 1500/1500 split; pooled
  selection-corrected null `R=30` on seed 0.

## Probe 1 — same-seed determining-set control: **PREDICTED NULL, all 3 seeds**

Read against the **selection-corrected null** (Amendment A4), not the absolute
`0.70`. Null floor (random-direction best-of-254, 95th pct): **`func=0.730`,
`state=0.453`**.

| seed | l\* | gen max_func | twin max_func | det_func | det_state | branch |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3 | 0.578 | 0.594 | False | False | `det_shadow_predicted_null` |
| 1 | 3 | 0.562 | 0.583 | False | False | `det_shadow_predicted_null` |
| 2 | 3 | 0.615 | 0.618 | False | False | `det_shadow_predicted_null` |

Every seed's best-of-254 gen metric sits **below** the random-direction selection
floor → no real determination. **gen ≈ twin** on all three → no gen-specific
determining structure. Paired-fiber (Probe 1b) is `paired_fiber_conflict` on all
three: shadow-near pairs disagree on the omitted functional → `S` is not
functionally sufficient — the null-consistent local reading. The independence
prediction is confirmed.

> **Methods catch (verify-before-file).** The first run used the frozen absolute
> `0.70` threshold and flagged seed 2 `det_shadow_void` because a single
> random-direction control draw tripped `0.70`. Diagnosis (10 draws/seed) showed
> the trip is **selection optimism**: best-of-254 reaches max-func ~0.60–0.73 for
> *random* directions; seed 0 actually trips more often. Amendment A4 replaced the
> absolute threshold with the random-direction max distribution — under which all
> three seeds are cleanly null.

## Probe 2 — cross-seed transplant (HEADLINE): **`cross_seed_no_transfer`**

| pair | Tier1 direct | Tier2 calib | Tier3 refit | Tier3b subspace | subspace_overlap |
| --- | --- | --- | --- | --- | --- |
| 0→1 | 0.497 | 0.502 | 0.535 | 0.576 | 0.045 |
| 0→2 | 0.498 | 0.495 | 0.511 | 0.523 | 0.048 |
| 1→0 | 0.497 | 0.493 | 0.506 | 0.546 | 0.045 |
| 1→2 | 0.502 | 0.495 | 0.503 | 0.532 | 0.014 |
| 2→0 | 0.506 | 0.480 | 0.511 | 0.577 | 0.048 |
| 2→1 | 0.501 | 0.501 | 0.515 | 0.577 | 0.014 |
| **mean** | **0.50** | **0.49** | **0.51** | **0.555** | **0.036** |

Pass-counts (Tier1, 2, 3) = **(0, 0, 0)**. No tier reaches `0.70` on any pair.

**A3 verdict: `different_encoding_subspaces` — the *stronger* negative.** The two
seeds' 8-dim readout frames are **near-orthogonal**: `subspace_overlap = 0.036`,
essentially the random-subspace baseline `k/d = 8/192 ≈ 0.042`. Tier 3b (a's whole
subspace + b-label refit) only reaches 0.555 → a's subspace does not carry b's
latents. Independently-trained seeds encode the same 8 latents in **near-random,
uncorrelated subspaces** of the 192-dim residual stream. The shared-geometry /
"allelopathy" hypothesis is **falsified on the toy** — and specifically *not*
because of reparameterization fragility.

## A1/A2 — selection-spectral-gap reliability (isotrophy v0.20 transplant)

| quantity | value |
| --- | --- |
| `frame_spread` (within-seed direction stability) | median **0.119**, q10 0.087 — **LOW (stable, not fragile)** |
| `eigengap` (residual-stream covariance top gap) | q10 0.511 |
| `rho(eigengap, frame_spread)` | **0.065 — no link** |

**chatv2 is a contrast case for the spectral-gap bridge, not a 4th confirmation.**
The isotrophy fragility principle is *"small gap → reparameterization-fragile →
non-transfer."* Here the readout directions are **stable within-seed** (low
`frame_spread`) and the residual-stream eigengap does **not** predict that stability
(`rho 0.065`) — so the covariance eigengap is not the right discriminative
conditioning on this substrate (the A1 caveat, realized). Per Amendment A2, a
**non-fragile regime with no transfer is the stronger negative**: the non-transfer
is genuine near-random subspace placement, not fragility. chatv2 marks a **boundary**
of the v0.20 spectral-gap reliability law, reported honestly — not evidence for it.

## Synthesis

- **Same-seed determining-set: predicted null, confirmed** (independent pair-XOR
  latents do not determine each other; the control banked its prediction).
- **Cross-seed allelopathy: falsified**, with the mechanism pinned by the
  Procrustes/subspace refinement — **near-random orthogonal encoding subspaces**,
  not a fragile shared frame.
- **Cross-substrate:** chatv2 *contrasts* the isotrophy spectral-gap fragility
  bridge rather than confirming it (stable directions, no gap→spread link).

Three verify-before-file catches in this lane: the Tier-1/2 source-layer rigor leak
(owner), the selection-optimism threshold (diagnosed → A4 null), and the
subspace-vs-fragility ambiguity (isotrophy transplant → A3, decisive).

## Allowed / forbidden language (per roadmap)

Allowed (earned by this receipt):
> The saved pair-XOR bodies confirmed the independence prediction (same-seed
> determining-set null on all three seeds). Independently trained seeds did **not**
> share a readable coordinate geometry: their per-latent readout subspaces are
> near-orthogonal (overlap 0.036 ≈ the random baseline), so a shadow basis learned
> in one seed does not transplant to another — and *not* for a fragility reason
> (within-seed directions are stable). Toy observability result only.

Forbidden (unchanged): no "AI models communicate through a field," no "EM foam is
the body," no "proves hidden collusion," no "explains why LLMs work," no
consciousness-field reading, no label-aware cross-seed pass called allelopathy.

## Tier + next

Toy-demonstrated; **unpromoted**; promote-gate **R1 not met** (this is a
within-portfolio observability probe, not a real-LLM claim). The coupled-substrate
`k_func << k_state` closure (roadmap Phase 7) and the high-fidelity NSE sidecar
(Phase 6) remain the only places a *positive* closure could live. **Phase 4 (SVG)**
data contract = `cross_seed_transfer.csv` (the near-orthogonal-subspace branch) +
`same_seed_kfunc_kstate_summary.csv` (the three-seed null); a null/negative branch
must render as such.
