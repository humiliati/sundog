# S2 — Measured-Sky Polarimetry: Scope

> **Date:** 2026-06-07 · **Status:** SCOPE (forward-looking; nothing here is executed or public). The S2
> handedness leg is currently forward-model tier (`ATLAS_PHASE5_CROSS_SUBSTRATE.md` §3.13). This
> memo scopes the jump to *observed* (or clean falsification). Grounded in a 5-agent web-confirmed recon
> (2026-06-07). Lane remains **NOT public-eligible**.

## Headline: the handedness claim splits in two, with opposite fates
The recon's make-or-break finding (Area 3): the forward model's "handedness" is actually **two distinct
claims**, and they must be separated.

- **Claim A — PER-FEATURE V (from TIR + birefringence). PHYSICALLY SOUND, MEASURABLE.** Total internal
  reflection genuinely converts linear→circular (the *rainbow* is the web-confirmed precedent — TIR in
  nonspherical scatterers produces a small Stokes-V; conversion can reach tens of % outside Snell's
  window). Ice Ih birefringence adds a second retardance source (Können's own halo work). So a faint
  (~1%) Stokes-V **on TIR-rich features** (parhelic circle, subhelic/anthelic arcs, 46° grazing paths)
  is physically *expected*, and a spatially-resolved measurement would likely **confirm** it.
- **Claim B — NET-V = "population handedness imbalance". PHYSICALLY DISFAVORED, LIKELY-NULL.** No
  mechanism breaks ice-crystal handedness symmetry: ice Ih is achiral, growth is non-enantioselective,
  and E/B-field alignment is not a pseudoscalar (it reshuffles *which* features are bright, not the
  ±V balance). **Können's own proof that `U=0` for a randomly-oriented halo is the direct linear analog
  of net-V cancellation.** A display-integrated net-V measurement will most likely return ~0 — which
  would **falsify** the (uncited, novel) population-handedness framing.

**Consequence:** the lit-pass's flagged landmine ("net-V = population handedness") was an *overreach*;
the defensible, measurable claim is the **per-feature ±V**. The cleanest pre-registerable target is an
**azimuthally-antisymmetric `±V(θ,φ)` pattern that integrates to ~0** — one observation that
*simultaneously* confirms the per-feature mechanism and demonstrates net cancellation. ⇒ **Doc-
reconciliation action:** sharpen the §3.12/§3.13 + `S2_LITPASS_E_G.md` handedness framing from
"net-V = handedness (novel)" to "per-feature V (defensible) vs net-V (disfavored/quarantined)".

## The ladder (cheapest → hardest)

### Stage A — Archival LINEAR-pol validation (cheap, in-house, ~1–3 days) ✅ actionable now
A real measured-sky dataset already exists: **Können & Tinbergen, *Polarimetry of a 22° halo*, Appl.
Opt. 30:3382 (1991)** — linear Stokes (I,Q,U) at 7 wavelengths vs scattering angle, with hard numbers:
- peak intrinsic linear DoP **~9%** (diffuse 22° halo), **~40%** (parhelion);
- **0.11° birefringence double-image split** (ordinary+extraordinary images);
- classical **Fresnel floor ~3.7%** (`(1−F)/(1+F)`, `F=cos⁴(θ/2)=0.929`); birefringence ~doubles it;
- **U=0** by mirror symmetry (the net-V-cancellation analog);
- independent hold-out: **Können, Wessels & Tinbergen, Appl. Opt. 42:309 (2003)** (Antarctic parhelia,
  DoP ~0.10–0.16).

**Action:** extend `scripts/s2_optics.py` to emit `(I,Q,U)` for randomly-oriented hex columns with the
**birefringent two-image split** + diffraction/solar-disk convolution, and check it reproduces Können's
profile (radial E / +Q, U≈0, ~9% peak, 0.11° split, ~3.7% Fresnel floor). **This is a model-credibility
GATE** — *if the same Mueller code that predicts V can't reproduce measured Q, its V is untrustworthy*
(cheap falsification route). On pass, the **linear-pol physics moves to "validated against observation"**
— but **V/handedness stays quarantined** (Können measured *no* V; his U=0 cancellation, if anything,
reinforces the net-V-cancels prior). **Honest risk:** the current single-prism Mueller chain may *not*
match ~9% without explicitly adding the two-image birefringence split — a genuine test that can fail.

### Stage B — Reframe + pre-register the V observable (cheap, doc-only) ✅ actionable now
Split Claim A from Claim B in the docs (above). Pre-register the falsifiable target as the
**antisymmetric `±V(θ,φ)`** pattern (NOT integrated net-V — that's a trivially-expected null). State the
null up front: aperture-integrated full-display V consistent with zero (within instrumental ~0.1–0.5%)
falsifies "population handedness imbalance." This corrects the lane's biggest honesty landmine *before*
any outreach.

### Stage C′ — In-house forward ±V(φ) map (cheap, in-house) ✅ DONE 2026-06-09
The pre-registered ±V map (the "ideal honest result" named below) **computed in-house** — the one
deliverable that did not need the measured sky. Receipt: `S2_HANDEDNESS_MAP_RESULT.md`; pre-reg:
`S2_HANDEDNESS_MAP_PREREG.md`; code: `scripts/s2_handedness_map.py` (+ 16/16 frozen test).
- **Found + fixed the blind spot:** the transmission-only chain returned **V≡0 on every TIR ray** (it
  never modeled the **Fresnel-rhomb TIR phase** — the *primary* linear→circular mechanism on the
  TIR-rich features). `s2_optics.tir_retardance` / `mueller_tir` add it; validated to the analytic
  anchors (ice δ_max=30.57°@59.1°; glass 45° rhomb pair 48.6°/54.6°).
- **Claim A confirmed:** per-feature ±V is real — per-ray `|V/I|` 3.8% (pure TIR) → 11.8% (with
  birefringence); flux-averaged feature observable 0.8–2.6% (robust over sun-elev 10–35°).
- **Claim B confirmed (structural null):** the achiral-ice display is azimuthally **antisymmetric**,
  `∮V→0` (net 0.00%, residual 0.00%) — the **V-analog of Können's U=0** — while a single chiral
  sub-path keeps net handedness 100% (so the cancellation is mirror-partner physics, not averaging).
- **Still owed to Stage C (unchanged):** the measured-sky detection itself + a full per-habit raytrace;
  V stays forward-model, linear pol remains the observed-tier anchor.

### Stage C — Measured-sky V campaign (the real jump; collaboration-gated; months)
The genuinely-novel observation: **no published visible-halo Stokes-V measurement exists** (confirmed —
not even an upper bound). The ~1% target is at the *high* end of atmospheric circular polarization
(aerosol DCP 10⁻²–10⁻⁵) and above good-instrument noise floors, so **magnitude is not the bottleneck**.

- **The binding systematic = linear→circular crosstalk.** The halo's strong *linear* pol (~4–9%) leaks
  into V via QWP retardance error; uncalibrated systems leak ~1.4% (would *fake* the whole signal). A
  credible detection needs per-wavelength Mueller-matrix crosstalk calibration **below ~0.1–0.3%**, and
  the observable **must be spatially resolved** (imaging V-map), not integrated.
- **Instrument:** commodity DoFP linear-polarization cameras (Sony IMX250MZR) are *blind* to V; the cheap
  fix is a **bolt-on quarter-wave retarder ahead of the micropolarizer array** (published technique) →
  full-Stokes. Prosumer tier ~$5–15k; research-grade full-Stokes (LCVR/PEM) ~$30–100k+ (overkill).
- **Collaboration tiers (skip the from-scratch build):**
  - **C1 — data-mine (cheapest, ~$0):** **Joseph Shaw (Montana State)** runs a full-Stokes (LCVR)
    all-sky imager with a multi-year archive; V may be latent in raw frames on halo days. Likely a
    credible **upper bound**, not a detection.
  - **C2 — bolt-on (best risk-adjusted, ~$hundreds + partner, 6–12 mo):** **LMU Munich (Forster /
    Mayer / Weber)** — the *only* group routinely imaging halos with a calibrated DoFP polarimeter
    (specMACS). Marginal ask: one retarder + recalibration. **Best path to a genuine detection.**
  - **Referee/credibility:** **G.P. Können** — falsification-design partner (what population asymmetry
    could survive cancellation) + authorship, *not* a data source (he'll predict net-V≈0).
  - **Field-trigger / crystal-sampling / credibility layer:** amateur halo networks (German AKM, Finnish
    Ursa / Marko Riikonen, atoptics/Cowley) — no V data (a Finnish observer's casual circular-polarizer
    try saw "no notable advantage," consistent with ~1% V being below eyeball/JPEG detection).
- **De-risk fallback:** a **lab ice-analogue / glass-crystal halo rig** (Herts-style) to demonstrate
  per-feature V *on demand*, decoupling the physics validation from rare-sky opportunity.
- **Pre-registered prediction:** per-feature `±V ~1%` antisymmetric around the ring, integrating to ~0.
  Outcomes: **confirm** per-feature V (novel first) · **null** integrated net-V (falsifies Claim B,
  expected) · a clean ±V map that does both = the ideal honest result.

## Decision points
1. **Stage A (archival validation): do now?** Cheap, in-house, high-value — moves the *linear*-pol
   physics to observed-tier for free and stress-tests the Mueller chain that also generates V. Strong
   recommend. (Can fail — that's informative.)
2. **Stage B (reframe): do now?** Cheap doc-correction of the lane's biggest honesty landmine. Strong
   recommend (pairs with A).
3. **Stage C (measured-sky campaign): commit, or scope-and-hold?** This is the months-long external
   commitment (collaboration + instrumentation + a cirrus season). Given the lane is frozen-as-portfolio
   and not-public-eligible, the likely call is **scope-and-hold** — keep this memo as the campaign-ready
   plan, optionally open a low-cost C1 data-mining conversation with Shaw, and trigger C2 only on a
   strategic decision to chase a genuine first-of-kind measurement.

## Honest status
- Stage A/B are cheap and move/clarify real tiers; Stage C is the genuine forward-model→observed jump.
- Even the *best* outcome is a **per-feature V detection + a net-V null** — which confirms the mechanism
  and *retires* the population-handedness overreach. That is a clean, honest, publishable first; it does
  **not** resurrect "net-V = population handedness."
- Nothing here changes the public-eligibility gate (`SHADOW_INVERTIBILITY_PHASE5_HANDOFF_2026-06-07.md`)
  until an actual measurement lands.
