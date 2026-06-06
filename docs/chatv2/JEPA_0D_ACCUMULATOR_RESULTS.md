# JEPA-0D Accumulator Results

**Status:** SHELVED (2026-06-05) — `blocked_by_unfaithful_jepa`. The accumulator substrate did
**not** rescue JEPA functional retention. On a designed count substrate, a from-scratch JEPA failed
to keep the running count `u_t` that a matched generative body keeps trivially, robust across three
independent JEPA framings. Banked as a toy-tier (R1) honest negative per the 2026-06-04 pivot
(generality nulls = portfolio).

Executed spec: `docs/chatv2/JEPA_0D_ACCUMULATOR_SPEC.md`. Runner: `scripts/jepa_0d_accumulator.py`.
Substrate/preflight: `scripts/jepa_0d_accumulator_preflight.py`.

## TL;DR

The decisive within-experiment contrast is at the early checkpoint, where the count is easily
learnable and the GEN positive control is live:

```text
tick 4:   GEN u_det = 0.81-0.90   (keeps the count)
          JEPA u_det = 0.00       (does not, at any read surface)
```

A generative body is *forced* to reconstruct the count-bearing tokens, so it builds `u_t`. JEPA's
embedding-prediction can satisfy its loss without ever representing `u_t`, so it doesn't. This is
the parity Phase-0 failure mode (`blocked_by_unfaithful_jepa`) reproduced on a substrate built
specifically to prevent it — and now mechanistically explained.

## Gate 1-2 + support + mask-necessity — model-free preflight PASSED

Receipt: `results/chatv2/jepa-0d-accumulator-preflight/preflight.json` (+ `adversarial_verification.json`).

| read | value | bar | status |
| --- | ---: | --- | --- |
| de-confound raw linear `u_det` (max) | −0.050 | ≤ 0.10 | PASS |
| oracle event-route `u` recovery | 0.9941 | ≥ 0.95 | PASS |
| support (class/flip starvation) | none; flip-min 250 | none | PASS |
| mask-necessity (whole-ckpt shortcut / event-path) | 0.131 / 0.99 | ≤ 0.15 / ≥ 0.70 | PASS |

5-agent adversarial verification confirmed the de-confound (robust across seeds, p_noise, n up to
16k), the honest oracle, and the position-prior neutralisation; four findings were fixed before the
spec (mask shortcut → whole-checkpoint mask; nan-drop; metric pin; K≥8 crash). **The substrate is
sound — the functional is present, input-undecodable, and recoverable by a sufficient parser.**

## Gate 3-4 — smoke FAILED, then diagnosed

Receipt: `results/chatv2/jepa-0d-accumulator-smoke/smoke.json` (~21 min, GPU, d=128, 1 seed).

| read | GEN | JEPA |
| --- | ---: | ---: |
| `u_det` (median over checkpoints) | 0.492 | **0.021** |
| eff-rank (read body) | 1.9 | **1.0 (collapsed)** |
| `z_flip_acc` | 0.395 | 0.364 |

JEPA `u_det = 0.021 ≪ 0.70` and the read body collapsed (effR 1.0). The GEN positive control was
also weak (median 0.492 < 0.70), so the headline was `blocked_by_gen_control` (instrument not
established) — not yet a clean JEPA verdict. A multi-surface diagnostic was run to localise the cause
rather than misattribute it.

## Diagnostic — `u_det` by read surface (three controlled variants)

Probed `u_t` at multiple read surfaces for both bodies (final / event-integration position /
readout position; for JEPA also the masked-checkpoint summary). Receipts:
`results/chatv2/jepa-0d-accumulator-diag{,2,-pa}/diag.json`.

**GEN (consistent across all runs) — the count is present but *decays over ticks*, on every surface:**

| surface | tick4 | tick8 | tick12 |
| --- | ---: | ---: | ---: |
| event | 0.814 | 0.492 | 0.252 |
| readout | **0.905** | 0.541 | 0.280 |

Not a read-placement artifact (all surfaces agree). A 3-layer TinyGPT holds a ~4-tick count but not
a 12-tick one → the GEN positive control fully lives only at the early checkpoint; the median is
dragged below 0.70 by the deep late count. (Secondary finding: T=12 is too deep for this body.)

**JEPA — pinned at the floor at every surface and checkpoint, across three framings:**

| JEPA variant | best surface×ckpt `u_det` | note |
| --- | ---: | --- |
| whole-checkpoint mask (λ_cov=1) | 0.021 | `final_maskC` effR 0.0; `readout_vis` effR 49 but no count |
| + standardized targets + λ_cov=10 | 0.077 | rank improved, count still absent |
| predict-ahead (mask last ckpt, causal) + fix | 0.059 | summary position carries no count |

## Verdict: `blocked_by_unfaithful_jepa` (SHELVE)

At tick 4 the GEN positive control is live (0.90) and JEPA is 0.00 — a clean within-experiment
failure. We ruled out every explanation reachable with our resources:

- ❌ **read-surface bug** — probed final / event / readout / masked-summary; none carry `u_c`.
- ❌ **position-dominated embedding target** — standardized targets per masked-checkpoint group; floor.
- ❌ **low-rank collapse** — raised λ_cov 1→10; floor.
- ❌ **bidirectional-mask artifact** — causal predict-ahead; floor.

**Mechanism.** Generative reconstruction must carry the count-bearing tokens, so it builds `u_t`.
JEPA's EMA-target embedding-prediction never *bootstraps* a count representation: the target encoder
doesn't encode `u_c`, so there's no `u_c` signal to predict, so the context encoder never learns to
integrate events. The accumulator made the masked target *depend* on `u_c`, but embedding-prediction
still found a shortcut (match the position/local structure, eat the residual loss, keep VICReg happy
via per-dim std). The substrate change did not rescue the parity failure mode.

## What this answers / shapes

1. **Our own question (`SUNDOG_V_JEPA` §4) — answered, negatively, at toy tier.** A JEPA body does
   *not* automatically "train the closure" (keep the functional, discard the state). The
   closure-keeping is not free from the objective; it depends on whether the prediction target
   *forces* functional representation. On a substrate where the functional must be sequentially
   built, the generative objective was the better functional-builder. This complicates the clean
   LeCun-flavoured framing — useful as an honest internal data point.
2. **Transferable hypothesis for the mech-interp / compander program** (see
   `internal/feedback/Human/REDDIT_ImOutOfIceCream_UNIT-DISTANCE.md`). The compander /
   categorical-centroids ⊥ generator-algebras / so(3) finding is on **autoregressive** transformers.
   This toy is a (small, R1) data point that the **JEPA objective behaves differently**: the
   useful low-rank, functional-bearing structure that generative reconstruction produces is *not*
   reproduced by embedding-prediction here. Prediction for the program's "look at JEPA next" step:
   expect **partial** cross-architecture transfer — the categorical-centroid half is plausibly a
   discrete-token-vocabulary consequence (JEPA predicts continuous embeddings, no output vocab) and
   may be absent; the so(3)/generator half, if it is normalization-geometry, is architecture-general
   and may survive. Offered as the §9e "thinking-partner on JEPA experiment design" contribution if
   the program re-engages.

## Relation to the COMPANDER_PAPER_HOOK citation rail (unchanged)

The 9 pre-staged `COMPANDER_PAPER_HOOK` anchors wait on the **autoregressive** compander finding —
the model class our 0/5,670 chat result actually runs on. They are **decoupled from this lane**: the
rail was never sourced from JEPA-0D, and shelving JEPA-0D does not touch it. The staged §9 copy is
already correctly hedged ("the mod has not yet looked at JEPA … open cross-cut"), so this negative
breaks no shipped or staged claim. **Keep the rail open at zero cost.**

## What we are NOT leaving on the table

Given our approach (R1 toy, from-scratch JEPA) and resources (one GTX 1080, no probe access to
pretrained models), this is a complete close: preflight verified + adversarially hardened; smoke;
multi-surface diagnostic; three controlled objective/mask fixes; mechanism understood. The question
that would feed the citation rail — compander geometry in *real* models, including JEPA — was never
reachable with our instrument; it is the mech-interp program's to answer. Remaining "more" is out of
scope (real-model probes) or against the kill-gate (endless objective-tweaking, risks JEPA→not-JEPA).

## Artifacts

- `scripts/jepa_0d_accumulator_preflight.py`, `scripts/jepa_0d_accumulator.py`
- `results/chatv2/jepa-0d-accumulator-preflight/` (preflight + adversarial verification)
- `results/chatv2/jepa-0d-accumulator-smoke/smoke.json`
- `results/chatv2/jepa-0d-accumulator-diag/`, `-diag2/`, `-diag-pa/` (the three diagnostic variants)

---

*Sundog Research Lab — JEPA-0D accumulator. SHELVED `blocked_by_unfaithful_jepa`. R1 toy; banked
negative. The accumulator did not rescue functional retention; embedding-prediction did not bootstrap
the count a generative body builds for free. Citation rail (autoregressive compander) stays open.*
