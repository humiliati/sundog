# Yang-Mills cheap Category-A controls (Q1 + Q5) — and what they reveal

## Header

- Receipt id: `2026-05-30_cheap_a_controls_target_validity`
- Purpose: run the two cheap external-review Category-A controls in-house —
  **Q5** (is ⟨W11⟩ in family with standard SU(2) values?) and **Q1**
  (signature/target disjointness) — on the registered SU(2) 3D Phase-2 data.
  Going one step past the packet's framing, Q1 also tests **target validity**.
- Script: `scripts/yang-mills-q1q5-controls.py` →
  `results/yang-mills/phase2/SU2_3D/cheap_a_controls/q1q5_summary.json`
- Type: bounded-null **diagnostic**. Re-characterizes the Phase-2 null; promotes
  nothing. Not a Yang-Mills / confinement / mass-gap / Clay claim.

## Q5 — ⟨W11⟩ is in family (lattice is healthy)

The exact 2D SU(2) plaquette is `I_2(β)/I_1(β)` (modified-Bessel ratio; holds on
any finite periodic 2D lattice). In 3D the plaquette sits **above** the 2D value
(more neighbouring plaquettes → more order), and the gap should grow toward weak
coupling. Observed:

| β | observed ⟨W11⟩ | 2D-exact `I_2/I_1` | observed − benchmark |
| --- | --- | --- | --- |
| 2.0 | 0.4571 | 0.4331 | +0.0240 |
| 2.4 | 0.5288 | 0.4935 | +0.0352 |
| 2.8 | 0.5967 | 0.5451 | +0.0516 |

All three sit just above the 2D-exact value, monotone, with the gap widening as
β rises — textbook. (The script's binary `in_family` flag tripped to `false` at
β=2.8 only because of a too-strict constant 0.05 threshold; the *direction and
growth* of the gap are exactly correct.) External anchor: recent HMC work on 8×8
2D SU(2) reproduces the Bessel-exact plaquette at β=2.0 to |Δ|≤0.001
([arXiv:2602.09045](https://arxiv.org/abs/2602.09045); cf. Phys. Rev. D 28, 2076).
**Verdict: the ensemble is standard. The null is not a broken-lattice artifact.**

## Q1 — disjointness holds; but the target is noise/clamp-dominated

**Leakage (the packet's actual question): NONE.** The signature
`{W11,W12,W13,W22} mean/var` cannot predict the held-out target `gamma_held`:

| β | max \|corr\|(sig, γ) | CV-R²(γ \| sig) | adj-R²(γ \| sig) |
| --- | --- | --- | --- |
| 2.0 | 0.251 | −0.858 | −0.094 |
| 2.4 | 0.384 | −0.428 | +0.092 |
| 2.8 | 0.290 | −0.850 | −0.156 |
| pooled (β-demeaned, n=96) | — | −0.208 | −0.028 |

So a positive rank-locality read would **not** have been trivial
signature-into-target leakage. Q1, as posed, is answered: the test was valid in
that narrow sense.

**Target validity (the real finding): `gamma_held` is essentially an ε-floor
clamp on a noise-level loop.**

| β | clamp frac | R²(γ \| clamp) | R²(γ \| W33) | corr(γ,W33) | W33 per-config SNR | clamp-frac in top γ-tertile |
| --- | --- | --- | --- | --- | --- | --- |
| 2.0 | 0.344 | **0.992** | 0.613 | −0.78 | **0.384** | **1.00** |
| 2.4 | 0.219 | **0.982** | 0.540 | −0.74 | **0.792** | 0.636 |
| 2.8 | 0.062 | **0.991** | 0.440 | −0.66 | 1.994 | 0.182 |

`gamma_held` (the LS area-law slope) is **~99% determined by the binary clamp
indicator**, which fires when the 3×3 Wilson loop `W33` dips toward/below zero.
And `W33` at this cell is a **per-config noise-level observable** (SNR 0.38 / 0.79
at β=2.0 / 2.4; only ~2.0 by β=2.8). At β=2.0 the **top `gamma` tertile is 100%
the clamp set** — so the tertile *labels* the three γ-probes (v0/v1/v2) tried to
predict were effectively "did `W33` noise-dip below zero," a coin flip. The fourth
probe (v3) used `σ²_W33`, a variance of the **same** 3×3 noise loop.

## What this means (honest)

- The four Phase-2 `YM-P2-NEG-A` scores landing at chance (0.310 / 0.294 / 0.308 /
  0.329 vs 1/3) was **structurally guaranteed**: three probes asked the signature
  to predict a noise-driven clamp, the fourth a variance of the same noise loop.
  No signature predicts noise → chance.
- The null is therefore **real but largely uninformative about the Sundog
  apparatus**. It is a statement that the *chosen held-out targets had no
  per-config signal at this (β, 12³) cell*, **not** that the small-loop signature
  lane lacks relative-locality structure. The lattice is in family (Q5); the
  signature is disjoint and well-measured; the **targets were the problem**.
- This does **NOT** show Sundog works on Yang-Mills. It removes the cleanest
  pessimistic reading of the bounded null and relocates the open question to
  target design.

## Disposition / reopen gate (not p-hunting)

The bounded-null synthesis should be re-characterized: "no signal in
noise-dominated targets at this cell," with the per-config target SNR now
measured. A principled reopen is **not** target-shopping; it requires a
**pre-registered target whose per-config SNR > 1 is demonstrated before any
rank-locality scoring** — e.g. a weaker-coupling / larger-volume cell, or a
held-out observable that is not a large-loop quantity sitting in the noise floor.
The W33-SNR diagnostic here is the objective admission gate. Absent such a target,
the lane stays a bounded null; with one, the test would for the first time be
*powered to detect structure if it exists*.

## Cross-references

- `2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md` — the null this
  re-characterizes (does not overturn; sharpens *why*).
- `EXTERNAL_REVIEW_PACKET.md` — Q1 (disjointness) and Q5 (lattice in family) now
  answered in-house; the leakage question is settled, the deeper target-validity
  point is new.
