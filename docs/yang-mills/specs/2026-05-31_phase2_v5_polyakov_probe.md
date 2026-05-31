# Yang-Mills Phase 2 v5 Polyakov Powered-Target Probe Spec

Filed: **2026-05-31 (PT)**

Author triggers:
- v4 powered-target audit
  [`2026-05-31_SU2_3D_phase2_v4_underpowered.md`](../receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md)
  → `YM-P2-UNDERPOWERED` (no Wilson-loop re-summary is both powered and disjoint
  at 12³ — the power-vs-disjointness squeeze).
- P0 amendment 2
  [`../../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md`](../../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md)
  admitting the Polyakov loop as held-out target vocab v4.

Status: **probe spec**, not binding. Selects the v4-pre-stated "Polyakov target +
P0 amendment 2" fallback, locks the symmetric-cell Polyakov candidate pool, and
pre-states the v6 finite-T fallback. Binding artifact:
[`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v5.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v5.md).

## Motivation — Polyakov dodges the squeeze; power is the question

v4 proved the 12³ Wilson-loop envelope has no powered+disjoint held-out target:
the candidates were either noise (high-area `W33`/`σ²_W33`, ICC ≈ 0) or leaky
(low-area `mean_W14`, CV-R² 0.576 at β=2.8). The Polyakov loop escapes the
**disjointness** side by construction — a global temporal-wrap shares no sub-loop
with the small Wilson signature `{W11,W12,W13,W22}`. The open question is
**power**: on the symmetric 12³ lattice at confined β, `⟨P⟩ ≈ 0` by centre
symmetry, but the **magnitude** and **susceptibility** are positive-definite and
may carry per-config signal. This probe audits that cheaply on the existing v0
ensembles before any finite-T investment — the same audit-first move that made
v4 an evidence-based trigger rather than a hunch.

## Target Selection — the symmetric Polyakov pool (held-out vocab v4)

Candidate pool (per-config summaries, frozen at amendment 2; each averaged over
all 3 periodic wrap directions on the symmetric lattice):

| Candidate | Definition | Why |
| --- | --- | --- |
| `abs_mean_P` | `\|mean_{x_⊥} (½)Tr P(x_⊥;μ)\|` | the centre-symmetry order-parameter magnitude; non-zero despite `⟨P⟩≈0` |
| `mean_abs_P` | `mean_{x_⊥} \|(½)Tr P(x_⊥;μ)\|` | per-site magnitude (less cancellation than the global mean) |
| `chi_P` | spatial variance of `(½)Tr P(x_⊥;μ)` | Polyakov susceptibility — config-to-config inhomogeneity, often the most powered |

Prior-target re-audit (reported, not admissible): `γ_held` — **must FAIL the power
gate** (the built-in self-validation, carried over from v4; if it passes the
probe voids).

## v5 Design — same discipline as v4, new target class

Identical to the v4 powered-target probe
([`2026-05-31_phase2_v4_powered_target_probe.md`](2026-05-31_phase2_v4_powered_target_probe.md))
except the candidate pool is the Polyakov summaries above. Verbatim gates:

- **Stage 1 — power gate (pre-scoring):** split each config's Polyakov values by
  transverse-**site parity** into two halves; `ICC := corr(T1,T2) ≥ 0.50` AND
  tertile-agreement `≥ 0.50`, in **all three β**.
- **Stage 1 — leakage gate:** `CV-R²(target | 8-dim v1 signature) ≤ 0.25`
  (the same 5-fold-OLS estimator as `scripts/yang-mills-q1q5-controls.py`).
- **Gauge-invariance health (new, target side):** `CTRL_GAUGE_RAND` must leave
  each Polyakov summary invariant to ≤ 1e-12 (amendment 2 health gate).
- **Admission:** powered ∧ disjoint in all three β; primary = highest mean-β ICC
  (tie-break lowest leakage), frozen in `admitted_target.json` before scoring.
- **Stage 2 (only if admitted):** v0-identical relative-locality test — 8-dim v1
  signature L2-NN, k=5 primary, 6 controls, **unchanged v0 promotion gates**
  (purity@5 ≥ 0.5; margin ≥ 0.10 over RAND/META/RAW; across-β ≥ 0.05 over
  RAND_STRAT; PERM within 0.05 of 1/3; GAUGE_RAND ≤ 1e-12). **If it lands, this is
  the first powered Stage-2 read on the lane.**

## Branches

| Branch | Trigger | Verdict |
| --- | --- | --- |
| UNDERPOWERED | no Polyakov candidate powered+disjoint in all three β | `YM-P2-UNDERPOWERED no_powered_target_in_envelope` → **v6 finite-T** (pre-stated) |
| A | admitted primary clears all promotion gates | `P2-A bounded_positive` (→ Phase 3) |
| C | primary fails RAND/RAND_STRAT margin | `YM-P2-NEG-A` — **now informative** (signature vacuous on a powered, disjoint target) |
| B / G / D / Z | as parent P0 branch table | as P0 |

## v6 Fallback (pre-stated, fully spec'd in parallel)

If v5 lands `YM-P2-UNDERPOWERED` (symmetric-cell Polyakov too weak — the confined
`⟨P⟩≈0` regime), the binding consumer is
[`../../prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6.md`](../../prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6.md):
a finite-temperature `12²×4` asymmetric ensemble straddling the deconfinement
crossover, where `⟨|P|⟩` is a genuine powered order parameter. If v5 lands `P2-A`
or an informative `NEG-A`, v6 is not invoked.

## Anti-Scope-Creep

The candidate pool (3 summaries), the wrap-direction averaging, the gates, and the
v0-identical Stage-2 are frozen here. The held-out class is Polyakov v4; the
signature is v1; the ensembles are the v0 12³ ensembles bit-for-bit. Anything else
is a future dated probe spec.

## Filing List

1. [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v5.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v5.md)
   — binding v5 spec.
2. New runner `scripts/yang-mills-phase2-v5-su2-3d-aggregate.mjs` + npm
   `yang-mills:phase2:v5:su2-3d:aggregate`, filed only after the binding spec is
   signed off. Reuses `scripts/lib/yang-mills-su2-3d-core.mjs` (+ a new
   `polyakovLoop` function); reads the v0 `configs/su2_links.jsonl`. No new
   ensembles.
