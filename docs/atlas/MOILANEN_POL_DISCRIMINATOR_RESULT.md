# RESULT — S3-A3: Moilanen-arc mechanism-class polarization discriminator

**Registered run 2026-06-11 · VERDICT: PASS (conditional on contrast-model slice, as pre-registered).**
Prereg: `MOILANEN_POL_DISCRIMINATOR_PREREG.md` (frozen before run; one pre-run amendment A1, estimator
intensity mask, frozen before any registered Stage-3 number existed). Script
`scripts/moilanen_pol_discriminator.py`; frozen test `scripts/test_moilanen_pol_discriminator.py`
(35/35, incl. post-run verdict pins; deterministic, no RNG). Slate: S3-A3 of
`internal/slates/HYP_SLATE_3_EXTERNAL_ANCHORS_2026-06-11.md`. All four pre-registered kills were live;
none fired.

## Headline

**Polarimetry CAN adjudicate the Moilanen-arc mechanism classes.** On the frozen Können-validated
Mueller chain, refraction-class and reflection-class candidate paths are separable by ≥ 0.5 % signed
DoLP for ANY optically active internal bounce — the separation threshold is **b\* = 3.0°** of bounce
incidence (claim bound was ≤ 5°), with **2.01 % worst-case margin** over the active range (≥ 2× floor →
PASS, not MARGINAL) and sign opposition (refraction radial, reflection tangential, up to |q| ≈ 80 %).
The result is smear-invariant across the entire in-box grid (flat-DoP rows) and holds on every
contrast-model slice in which the arc is photometrically visible at all.

## Per-slice verdict (prereg §5)

C0 = 0.30 slices (k = 2/3/5): **PASS** (9/6/3 visible in-box cells). C0 ∈ {0.05, 0.10} slices:
NO-VISIBLE-CELLS (named outcome — an arc at sub-solar-disk contrast is not photometrically detectable
at the pinned 1 % RMS; consistent with observed Moilanen arcs being conspicuous features).

## The table's load-bearing rows

| row | value | meaning |
|---|---|---|
| Apex-60 gate | 21.839° / 3.653 % / 0.1061° | Können anchors reproduced (CALIBRATION kill silent) |
| Floor law | 440/440 incidences, worst margin +8.5e-9 | q ≥ DoP(D) numerically everywhere (FLOOR-LAW kill silent) |
| W-34 refraction | q = +0.930 % radial at D = 11.04°; doublet 0.0508° (Ih) / none (iso); V ≡ 0 | the rigid-wedge candidate |
| **LEF (computed FIRST)** | D track 10.92/13.09/15.02/18.20° at e = 0/10/15/20 (published ~11/13/15/18); TIR cutoff 26.29° (published “> 26”); **q = +1.83 → +16.45 % rising toward cutoff, q/floor 2.0–3.4** | the published composite path, source-pinned |
| REFL-P(b) | b → 0 convergence exact (1.9e-16); q(5°) = −1.08 %, q(20°) = −34.5 %, max sep 100.9 % | b-resolved envelope; b\* = 3.0° |
| REFL-TIR | max\|V/I\| = 0.236 % (backbone-pinned, max-over-azimuth) | see honesty note 3 |
| Ih/iso ledge pair | excess 17.7 → 3.8 % across the in-box grid (mask-0.3 column 11.8 → 2.9 %); **12/12 cells SEP** | PARTIAL-tier surprise — see note 2 |
| W-34 track | q(D) single-valued: 2.55/4.28/7.02 % at 13/15/18° (branch band collapses by path reversibility) | floor inequality holds at every point |

## Findings beyond the claim

1. **The LEF path acquired its own falsifiable polarization signature.** Source-pinning the published
   geometry (30° prism, vertical entry, θ_i1 = −e) reproduces Lefaudeux's entire published track to
   ≤ 0.2° AND predicts a strongly elevation-dependent DoLP (1.8 % → 16.5 %, 2–3.4× the radius floor)
   terminating at the 26.3° TIR cutoff — while the rigid-34°-wedge class stays pinned near its DoP(D)
   floor track. The two REFRACTION candidates are therefore polarimetrically distinguishable from each
   other by track SHAPE, not just from reflection paths. (Low-sun snapshot alone: 0.897 % — in the
   pre-named marginal band; the discrimination lives in the elevation dependence.)
2. **The W-Ih vs W-iso doublet pair did NOT die under smear** (pre-flagged as the fragile pair): the
   o-edge shoulder retains 3.8–17.7 % DoP excess across the whole in-box grid (still 1.45 % at the
   out-of-box 2° wobble diagnostic). The non-Ih-material hypothesis class is testable after all —
   at the modeled edge-shoulder observable (note 2 below).
3. **Classification resolution held:** the published Lefaudeux composite path contains no reflection
   bounce — it instantiates the refraction class (prereg §2.2). The reflection class is carried by the
   labeled envelope columns, with the b → 0 convergence anchor reproduced to machine precision.

## Honesty notes / boundaries

1. **Conditional on the pinned contrast model.** PASS holds on the C0 = 0.30 slices; the low-contrast
   slices have no visible cells by construction. The conservative span was pre-registered without
   photo-derived calibration; a photo-calibrated C0 from published Moilanen images upgrades this
   (cheap, separate ticket-level task; an eye-visible arc implies post-smear contrast ≳ 3 σ, i.e. the
   C0 = 0.30 regime).
2. **The Ih/iso ledge excess is mask-definition-dependent** (17.7 % at the Ic ≥ 0.1 cut vs 11.8 % at
   0.3): it is the polarized SHOULDER of the rising o-edge, a low-flux feature. Both columns reported;
   any manuscript states the observable exactly as pinned (A1).
3. **The backbone-pinned TIR V envelope (0.236 %) sits BELOW the 0.5 % live-leg floor** — the
   V-detection kill channel (any measured |V/I| > 0.5 % kills the pure-refraction classes) is only
   LIVE for reflection paths with stronger pre-TIR linear polarization than the min-deviation backbone
   provides (e.g. grazing entries). V remains excluded from the separation matrix (prereg §2.3); the
   forbidden-signature logic is unchanged.
4. **In-plane leg only** (V-arc apex region); 2-D arm smear, oriented-population statistics, and the
   2.66× equality-level ratio test are the polyhedron ticket's scope. NOT triggered by the primary
   pair (margin 2.01 % ≥ 1.0 %); the only marginal-band entry is the non-load-bearing low-sun
   LEF-vs-W34 snapshot.
5. **Stage-4 D = 11.04 row reports no crossing** because 11.04 sits marginally below the true minimum
   11.0401 — that row IS the min-deviation point (q = 0.930 %); a tangency, not a gap.
6. Geometric-optics tier; diffraction bracketing per prereg control C7 unchanged (HS13 parked).

## What this unlocks (ALL owner-gated; nothing sent)

Per prereg §1 PASS rule: the pre-registered Applied Optics-style note + discrimination table
("Polarization signatures of candidate Moilanen-arc mechanisms") is now WRITABLE before next
diamond-dust season, with the pre-manuscript absence re-verification list binding (JQSRT 2022 full
text, Können 1998 target list, Tape & Moilanen 2006, post-1998 Können reviews).
**UPDATE 2026-06-11 (same day): BOTH DONE.** The absence gate was executed against FULL TEXTS of all
four sources — verdict: claim survives, scoped (Können 1998 = the umbrella identification program;
the M-arc absent from its target list; Tape & Moilanen 2006 endorses the technique in App. C while
leaving the M-arc "an open problem" in Ch. 18). Receipt:
`internal/manuscripts/moilanen_pol_note/ABSENCE_REVERIFICATION_2026-06-11.md`. The manuscript draft
(v0.2, 6 pp., 3 pgfplots figures regenerated from the frozen apparatus, verified 14-ref bibliography)
is at `internal/manuscripts/moilanen_pol_note/moilanen_pol_note.tex|.pdf` — NOT submitted;
author-contact block, submission, and the optional Können/Moilanen courtesy email all owner-gated.
Bonus catch: arXiv:2405.07864 is Stockmans et al., not "Maciejewski et al." — corrected here in
`ATLAS_HALO_POL_CAMPAIGN.md` ref 7. Measurement ask:
single calibrated DoFP camera on one display (community: Riikonen/Moilanen orbit, AKM/Hinz;
Gritsevich senior-author shape per the scout dossier). Citation discipline: Können 1983 + 1998
ancestors named.

*Provenance: prereg frozen → frozen test (27 pre-run pins) → registered run (exit 0, zero check
failures) → post-run pins appended (35/35). One pre-run amendment (A1, estimator intensity mask)
triggered by a frozen-test sanity check, not by any registered result.*
