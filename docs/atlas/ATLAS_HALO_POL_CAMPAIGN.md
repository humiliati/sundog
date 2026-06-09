# Measured-sky LINEAR-polarization halo campaign — prediction package (in-house prep)

> **2026-06-09.** Promotes the Atlas DoP(R) observable (`ATLAS_HALO_POLARIZATION_OBSERVABLE.md`) toward
> a measured-sky campaign. **This is in-house campaign PREP** — the prioritized target list, feasibility,
> and falsification design. The **actual measurement** (collaboration + instrument + a cirrus season) and
> **any outreach** stay **owner-gated / external** (the V-campaign scope-and-hold of
> `S2_MEASURED_SKY_SCOPE.md`). **NOT public-eligible.** Grounded by a lit/instrument workflow (2026-06-09;
> citations below). The DoP(R) law is a **synthesis** of Können's accepted mechanism, not new physics.

## Why a LINEAR campaign now (vs the hard V campaign)
The circular-V campaign needs a bolt-on quarter-wave retarder + linear→circular crosstalk calibration
below ~0.1–0.3% (a hard systematic). The DoP(R) law is a **linear** observable, and **DoFP imaging
polarimeters measure linear Stokes natively — no retarder, no crosstalk**. So the linear DoP(R) ladder
is the **near-term, high-feasibility track**, and a signal may already be **latent in existing archives**.

## The gap = the novelty (grounded)
- **Measured today: only the 22° halo / 22° parhelion** linear pol — radial, U=0, ~3.7% Fresnel floor /
  birefringence-enhanced peak (Können & Tinbergen 1991; Können–Wessels–Tinbergen 2003; confirmed all-sky
  by Pust & Shaw 2008).
- **Unmeasured (the targets):** the **46° halo** (Forster & Mayer 2022's 22°/46° imaging is radiance-only)
  and the **entire pyramidal odd-radius family** (9/18/20/23/24/35°) — *no* polarization measurement of
  any kind.
- **The DoP(R) closed-form ladder is unpublished.** Können 1998 names the odd-radius family but computes
  only pol *direction/visibility* (theory, as a crystal-axis diagnostic) — **not** a monotone DoP-vs-radius
  law. The falsifiable monotone ladder is the novel contribution.

## The predicted observable (the FLOOR/ORDERING law)
`DoP(R) = (1 − cos⁴(R/2)) / (1 + cos⁴(R/2))`, radial (Q in the scattering plane), **U = 0**, no net V.
**Pre-register:** this is the **geometric-optics floor / ordering**. Real peaks ride *above* it via
birefringence-diffraction (Können's intrinsic 22° peak ~8.7% vs the 3.7% Fresnel floor), size-dependent.
So the robust, falsifiable claim is the **monotone ordering vs radius**, not an absolute peak value. And
it is the **inner-edge (min-deviation)** value — ring-averages run lower (our raytracer: 46° 13% vs 16.4%).

## Prioritized target list (`atlas_halo_pol_campaign.py` scorecard)
| halo | R | predicted DoP | nn-sep | isolated? | tier |
|---|---|---|---|---|---|
| 9° pyramidal | 8.96° | 0.61% | 9.0° | **yes** | archival/stack |
| 18° pyramidal | 18° | 2.48% | 2.0° | no | degenerate |
| 20° pyramidal | 20° | 3.06% | 1.8° | no | degenerate |
| 22° (prism 60°) | 21.84° | 3.65% | 1.2° | no | degenerate (measured) |
| 23° pyramidal | 23° | 4.05% | 0.8° | no | degenerate |
| 24° pyramidal | 23.82° | 4.35% | 0.8° | no | degenerate |
| **35° pyramidal** | 35° | **9.45%** | 10.7° | **yes** | **easy** |
| **46° (prism+basal 90°)** | 45.73° | **16.23%** | 10.7° | **yes** | **easy** |

## The sharpened falsification design (adversarially reviewed)
**Do NOT run the full 8-rung ladder.** The 22/23/24° rings are within **1°** and **0.35 pp** of each
other — below every instrument floor and blended by the always-brighter regular 22° halo. They are a
**degenerate trap**. The clean test is the **3 angularly-isolated radii spanning the dynamic range**:

> **9° (0.61%) → 35° (9.45%) → 46°-inner-edge (16.23%)** — separations 9–11°, a ~26× span.

- **Primary falsifier — the monotone DoP *ratio* test (confound-robust).** Report **ratios** between
  rungs, not absolute DoP: `DoP(46)/DoP(35) ≈ 1.72`, `DoP(35)/DoP(9) ≈ 15.4`. Common-mode calibration
  error (gain non-uniformity, finite extinction ratio, micropolarizer misalignment), the sky pedestal,
  and multiple-scattering depolarization all act ~multiplicatively or as an additive pedestal — they
  **shift the whole ladder together but preserve the ordering and ratios**. **KILL the radius-only law
  if** a measured ring breaks monotonicity (e.g. 46° ≤ 35°) beyond the differential error bar — a null
  cannot fake a monotone radius-ordered staircase.
- **Secondary veto — the U=0 / radial-sign-of-Q gate (every frame, free).** The halo is radial (**+Q**);
  the Rayleigh background is tangential (**−Q**). A pre-registered pass/fail: the ring excess must be
  **+Q with U≈0** in the scattering-plane frame. (Note: U=0 does *not* reject the background — both halo
  and sky have U=0; it is a **null/sign veto** separating a real halo signal from a mis-fit background
  residual.)
- **Tertiary (weak) — absolute value at a clean rung.** Bounded between the Fresnel floor and ~2× it
  (birefringence up, multiple scattering down); a consistency check, never the headline.

## Confounds + isolation (the real bottleneck)
- **Background is spatial, not orthogonal.** Halo (+Q) and sky (−Q) share the U=0 channel; orthogonality
  buys nothing. What isolates the halo is its **spatial narrowness** (~0.5–1.5° ring) on a broad smooth
  sky gradient → **Können's 5-parameter annular background fit** (A, C, Q_B, U_B, I_B) is mandatory;
  work in **radial-Q**, not total intensity.
- **Multiple scattering** (optically-thick cirrus) is the dominant DoP killer (τ-dependent, downward,
  non-uniform) → restrict to **high-sun, optically-thin** displays.
- Aerosol path pedestal; low-sun structured sky-pol → use **high-sun**, azimuth sectors away from the
  90°-scattering sky-pol maximum.

## Instruments + warm-data paths
| instrument | linear floor | covers | archive / access |
|---|---|---|---|
| Commodity DoFP (Sony IMX250MZR) | 0.8% single → 0.08% @100-stack | 35/46 single-shot; all 8 stacked | DIY; native linear (no V crosstalk) |
| specMACS (LMU, LUCID DoFP) | ~5% rel | only 35/46 | aircraft cirrus I,Q,U; **public CC-BY** (PANGAEA) but no halo frame in release |
| Shaw all-sky (LCVR, full Stokes) | **~0.5% abs** | all 8 (9° at floor, stack) | **multi-year rooftop archive**, collaboration-gated; also bounds V≈0 |

**Two warm-data paths to try before any field campaign (owner-gated outreach):** (i) mine **Shaw's
all-sky archive** — the single most likely place a >22° halo linear-pol signal is *already latent*, and
it bounds the V≈0 prediction too; (ii) latent halo frames in **specMACS-class DoFP** cirrus data.

## Honest caveats (pre-register)
- DoP(R) is the **inner-edge / min-deviation floor**, not the ring-average (the 46° 13-vs-16% gap) and
  not the peak (birefringence rides above). Test **ordering/ratios**, restrict absolute checks to the edge.
- Restrict the U=0 / DoP(R) test to **randomly-oriented (circular-halo) displays**; oriented (plate/Parry)
  crystals populate only arcs where U need not be 0.
- The pyramidal ladder needs a **rare pyramidal {10-11} display**; 35°/46° also arise in commoner displays,
  so the **9°/35°/46° isolated-radius test is the bankable one**; the full pyramidal ladder is aspirational.

## Owner-gated next steps (NOT done here)
Outreach to **Joseph Shaw** (C1 archive data-mine, ~$0) and **LMU specMACS** (Forster/Mayer/Weber);
**G.P. Können** as falsification-design referee/author (not a data source). The field campaign / any
publication are strategic, owner-gated decisions. This package is the campaign-ready plan.

## Files
- `scripts/s2_optics.py` `halo_pol_dop` — the law; `scripts/atlas_halo_polarization.py` — the Atlas table.
- `scripts/atlas_halo_pol_campaign.py` (+ `test_…`) — the feasibility + falsification scorecard.
- `scripts/s2_halo_raytracer.py` — the engine that confirms the law (9°/22° to ~0.1%).

## Citations (grounded 2026-06-09)
1. Können & Tinbergen 1991, *Appl. Opt.* 30, 3382 — 22° halo polarimetry (the floor anchor).
2. Können, Wessels & Tinbergen 2003, *Appl. Opt.* 42, 309 — 22° parhelia pol + sampled crystals.
3. Können 1998, *Appl. Opt.* 37, 1450 — odd-radius family inner-edge pol (direction-only, theory).
4. Pust & Shaw 2008, *Appl. Opt.* 47, H190 — all-sky imaging polarimetry; only imaging halo (22°) detection.
5. Forster & Mayer 2022, *ACP* 22, 15179 — 22°/46° imaging, radiance-only (proves the pol gap).
6. Weber et al. 2024, *AMT* 17, 1419 + PANGAEA 10.1594/PANGAEA.965546 — specMACS DoFP linear-Stokes (public).
7. Maciejewski et al. 2024, arXiv:2405.07864 — sub-percent IMX250MZR DoFP characterization (floors).
