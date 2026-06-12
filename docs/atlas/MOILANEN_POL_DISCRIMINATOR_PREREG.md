# PREREG — S3-A3: Moilanen-arc mechanism-class polarization discriminator (validity-domain form)

**Status: FROZEN 2026-06-11, before first registered run.** Executes S3-A3 of
`internal/slates/HYP_SLATE_3_EXTERNAL_ANCHORS_2026-06-11.md` with every refuter fix and both audit fixes
(PARTIAL-DOMAIN verdict; solar-disk smear alignment) binding. Apparatus: the FROZEN `scripts/s2_optics.py`
Mueller chain ONLY (untouched), composed in the new `scripts/moilanen_pol_discriminator.py`.

**Standing discipline (inherited verbatim):** pre-registered KILL criterion — a clean null is a SUCCESS;
forward-generate only (no inversion fitting); deterministic runs + frozen tests; cheap headless CPU first
leg (no GPU, no external data, no spend); name the nearest prior, state the delta.

---

## §1 Claim

Using ONLY the frozen s2_optics Mueller chain plus an apex-angle parameterization of the existing
two-refraction chain (calibration-gated at apex=60° against the Können anchors before any 34° number is
read), over the pre-registered instrument-plausible smear box (§3.1) and pinned arc-contrast model (§3.2):

**CLAIM** — the refraction-vs-REFL class pair retains ≥ 0.5 % signed-DoLP separation against every
*optically active* reflection bounce (b > b_inert = 5°, §3.4) at EVERY visible in-box smear cell, i.e.
the separation-threshold bounce incidence satisfies **b\*(cell) ≤ 5° on every visible in-box cell**, for
every contrast-model slice (or reported as conditional on slice). The W-Ih vs W-iso pair (hanging on the
0.0508° Ih o/e doublet) is held to the pre-named PARTIAL tier and does not bind the claim.

**KILL (PRIMARY)** — the ≥ 0.5 % validity domain is EMPTY inside the box: even the most polarizing
bounce (b → b_max, the sub-critical maximum) fails to separate from the refraction row at some visible
in-box cell. Discriminator DEAD; banked bounded null ("polarimetry cannot adjudicate Moilanen mechanism
classes at DoFP-class floors"); no Applied Optics note, no contact.

**PARTIAL-DOMAIN (pre-named)** — domain non-empty but b\*(cell) > 5° on some visible in-box cells →
CONDITIONAL discriminator, never pass language; deliverable = the b\*(cell) validity-domain map; outreach
only if documented Moilanen display conditions fall inside the recovered domain (this rule is the pinned
outreach decision rule).

**MARGINAL (pre-named)** — worst-case in-box load-bearing separation lands in [0.5 %, 1.0 %) →
"marginal — polyhedron-gated"; the polyhedron leg is a SEPARATE slate ticket, never silent scope
expansion here.

No magnitude is pre-committed for any reflection path: every REFL number is "as computed for the named
path" (refuter fix #2). The near-normal/near-Brewster-bounce branch is pre-registered as exactly the
scenario in which the REFL row converges toward the refraction row and the PRIMARY kill becomes live.

## §2 Mechanism classes, observables, conventions

Sign convention: signed DoLP q = −Q/I in the chain s/p frame, so **q > 0 = radial** (E in the plane of
incidence, p-dominant transmission) and **q < 0 = tangential** (s-dominant). In-plane chains keep U ≡ 0
(mirror symmetry; the Können U≈0 anchor) — nonzero U = frame bug, quarantine (control C2).

| class | path (pinned) | DoLP | doublet | V |
|---|---|---|---|---|
| **W-Ih** | generic 34° wedge (the atoptics/Tape simulation wedge), two refractions, min-deviation edge, ice Ih | radial, floor DoP(D) | o/e split `halo_min_deviation(34, n_e) − halo_min_deviation(34, n_o)` ≈ 0.0508° (n_o = 1.3090, n_e = 1.3104, the s2_konnen_validate constants) | ≡ 0 |
| **W-iso** | same wedge, optically isotropic non-Ih medium (dn = 0) | identical radial track | NONE (the pair's only separator → PARTIAL tier) | ≡ 0 |
| **LEF** | the published Lefaudeux composite path, source-pinned §2.1 | radial, elevation-dependent (rises toward the TIR cutoff) | N/A (two-crystal composite; column marked not-modeled) | ≡ 0 |
| **REFL-P(b)** | one internal partial bounce at incidence b inserted into the W-34 min-deviation backbone (entry Fresnel × `mueller_fresnel_reflect(b)` × exit Fresnel) — b an explicit reported axis, 0–49.5° | b-resolved signed q; → refraction row as b → 0 (the convergence anchor, must reproduce numerically) | — | ≡ 0 (sub-critical reflection is real-coefficient) |
| **REFL-TIR(b)** | one TIR bounce, b ∈ (critical, 90°), `mueller_tir` fast-axis azimuth φ swept | \|Q\| range reported | — | max-over-azimuth \|V/I\| ENVELOPE, labeled as such |

**§2.1 LEF source pin (verified live 2026-06-11, Ursa post "Possible explanation of Moilanen arc",
Lefaudeux 2010):** composite crystal = horizontal main plate + plate/column stuck on its inferior base,
stuck-crystal bases ⊥ two side faces of the main plate; ray "entering a vertical side face of the main
plate and exiting the upper lateral side face of the stuck crystal" = "a 30° prism … with the entrance
face vertical"; published distance track ≈ 11° (low sun), 13° @ 10°, 15° @ 15°, 18° @ 20°; "total
internal reflection condition is met when the solar elevation is higher than 26°"; invalidated by
Moilanen's lunar observation at 26.2° elevation (arc 15.4° above moon). Chain: signed entry incidence
θ_i1 = −e on the vertical face, prism constraint θ_i2 = A − θ_t1 (signed), A = 30°. **Pre-run analytic
check (recorded here at freeze time): this chain reproduces D(0) ≈ 10.9°, D(10) ≈ 13.1°, D(15) ≈ 15.0°,
D(20) ≈ 18.1°, TIR cutoff e_c ≈ 26.3° — the published track to ~0.1°.** Stage 2 must reproduce these
numerically (LEF-track consistency gate); discrepancy → reported as a LEF-geometry caveat, never
silently re-derived.

**§2.2 Classification resolution (binding, source-grounded):** the slate's "reflection row computed for
the named published Lefaudeux composite path first" resolves at prereg time: the published path contains
NO reflection bounce (§2.1) — LEF instantiates the REFRACTION class with its own geometry and track. The
reflection class is therefore carried by the named envelope columns REFL-P(b)/REFL-TIR(b), with b an
explicit axis including the near-normal branch. LEF is still computed FIRST in Stage 2 (binding order),
and the LEF-vs-W-34 row (two competing *refraction* candidates: elevation-rising DoLP vs the DoP(D)
floor track) is reported as a named bonus pair, not load-bearing for the claim.

**§2.3 Separation metric:** intrinsic smeared signed-DoLP difference |Δq| at the smeared arc peak,
background-subtracted convention (the campaign convention: DoP of the arc feature itself, not the
contrast-diluted ratio — `ATLAS_HALO_POL_CAMPAIGN.md`; the field confound note in the slate's open
risks is thereby pinned: published-table values are arc-minus-local-background differentials).
**V is EXCLUDED from the separation matrix** (envelope-only quantity): the V row enters solely as the
refraction-class forbidden signature — any live-leg measured |V/I| > 0.5 % kills the pure two-refraction
classes (detection-kills, direction fixed per refuter fix #3). The doublet observable enters only the
W-Ih/W-iso PARTIAL-tier pair, as the smeared ledge-peak DoP excess (konnen_validate profile recipe).

## §3 Pinned grids and models

**§3.1 Smear box (audit fix #2 applied):** a fixed solar-disk convolution — Gaussian, FWHM 0.53°
(pinned simplification; s2_konnen_validate precedent uses a Gaussian for the disk+diffraction width) —
applied to EVERY cell, plus wobble σ ∈ {0.1, 0.3, 0.5, 1.0}° × instrument-PSF FWHM ∈ {0.02, 0.05, 0.1}°
as additional smear on top (quadrature). In-box = that 4×3 grid. Out-of-box diagnostics (wobble 2.0°,
3.0°; PSF 0.3°) are computed and reported as bounded validity limits, never claim violations.
Justification for the box: diamond-dust Moilanen displays are sharp (snow-gun crystals "near optically
perfect" — atoptics, verified live); wobble beyond ~1° would wash the arc's V-shape itself.

**§3.2 Arc-contrast / visibility model (refuter fix #4 applied):** no photo-derived calibration is
available at freeze time, so the stated CONSERVATIVE span is swept as an explicit grid axis: intrinsic
arc ridge = Gaussian, FWHM w₀ = 0.1° (pinned); smeared peak dilution = w₀/√(w₀² + 0.53² + (2.355·σ_w)² +
PSF²) (FWHM quadrature); unsmeared peak/background contrast C₀ ∈ {0.05, 0.1, 0.3}; background photometric
RMS σ_rel = 0.01; **visible(cell, C₀, k) ⟺ C₀ · dilution ≥ k · σ_rel**, k ∈ {2, 3, 5} (3 primary; 2/5 =
threshold-robustness columns, control C3). A (C₀, k) slice with zero visible in-box cells reports
"arc not visible under model — no verdict for this slice" (named outcome, not a kill).

**§3.3 Track evaluation points (forward-only):** observed arc distances 11.04/13/15/18° at sun elevation
0/10/15/20° (Ursa-documented) enter SOLELY as forward evaluation points. Stage 4 evaluates each class
along its OWN predicted track: W-34 = the two-branch (steep/shallow incidence) in-plane band [q_low,
q_high] at each D, plus the floor inequality q ≥ DoP(D); LEF = its single-valued q(e) along its own D(e)
track. The equality-level 2.66× ratio test is DEFERRED to the polyhedron ticket (refuter fix #3).

**§3.4 Bounce-inertness bound (a-priori):** b_inert = 5°, fixed at freeze time before any b\* is
computed. Rationale: below ~5° the bounce Mueller factor is within ~2 % diattenuation of identity and the
two mechanisms become PHYSICALLY degenerate in polarization — the validity statement is then "polarimetry
adjudicates the bounce's optical role, not its geometric presence" (a scope boundary, stated in any note,
not an instrument failure). Stage 0 records d_b(5°) as computed (no tuning).

**§3.5 b grids:** REFL-P: b ∈ {0.5, 1, 2, …, 49.5}° (0.5° steps ≤ 5°, 1° steps above; critical angle =
arcsin(1/1.31) ≈ 49.76°); REFL-TIR: b ∈ {50, 52, …, 88}°, φ ∈ 64 points over [0, π).

## §4 Stages (binding order)

- **Stage 0 — gate.** `wedge_chain(apex_deg, theta_i1_deg, n)` composed from frozen
  `so.mueller_fresnel` (+ `mueller_fresnel_reflect`/`mueller_tir` for envelopes); s2_optics.py untouched.
  CALIBRATION GATE at apex=60: must reproduce 21.84 ± 0.01°, Fresnel floor 3.71 ± 0.10 abs%, o/e split
  0.106 ± 0.005°. Gate failure → CALIBRATION KILL (fix or withdraw; nothing downstream runs).
- **Stage 1 — floor law.** Deviation-vs-incidence sweep at apex=34 (both branches); numeric check
  q(θ_i1) ≥ DoP(D(θ_i1)) − 10⁻⁹ everywhere. Violation → FLOOR-LAW KILL (retract the analytic floor
  claim; rebuild the refraction row numerically; verdicts then rest on numeric rows only).
- **Stage 2 — class table, LEF FIRST.** LEF chain + track-consistency gate (§2.1), then W-Ih, W-iso,
  then REFL-P(b) (convergence anchor at b → 0 must reproduce the W-34 row to < 0.05 %), then REFL-TIR(b).
  Per-class observables: q, |U/I| (must be < 10⁻¹²), |V/I|, edge profile with/without doublet.
- **Stage 3 — kill-gate smear sweep.** Low-sun configuration (the Stage-3 verdict configuration:
  e ≈ 0, D ≈ 11°). Solar-disk + box convolution per §3.1; visibility per §3.2; b\*(cell) = min{b :
  |q_REFL(b′) − q_refr| ≥ 0.5 % ∀ b′ ≥ b} per visible in-box cell; Ih/iso smeared ledge excess per cell;
  verdict per §1 per (C₀, k) slice.
- **Stage 4 — track co-predictions.** Floor inequality + two-branch band at the four evaluation points
  per class on its own track (§3.3); LEF q(e) column with the TIR-cutoff endpoint.

## §5 Verdict table (exhaustive, disjoint)

CALIBRATION KILL · FLOOR-LAW KILL · PRIMARY KILL (empty domain) · **PASS** (b\* ≤ 5° on all visible
in-box cells, all slices or conditional-on-slice) · PARTIAL-DOMAIN · MARGINAL ([0.5, 1.0)% worst-case
in-box margin) · NO-VISIBLE-CELLS (per slice) · DETERMINISM ABORT (reruns not byte-identical — abort,
not result). The W-Ih/W-iso outcome is reported on its own PARTIAL tier whatever it is. V-DETECTION is a
live-leg downstream kill (no internal branch).

## §6 Controls

C1 apex-60 Können gate (Stage 0). C2 U ≡ 0 mirror/anti-bug (quarantine on trip). C3 visibility-threshold
robustness (k = 2/3/5; verdict robust-across or reported conditional). C4 reflection-class scope: REFL
verdicts per named path + b-resolved envelope, never "all reflection paths"; structural statements
(partial reflection s-dominant/tangential; TIR generates V; sub-critical reflection cannot generate V)
carry the class level. C5 forward-only: no parameter fitted to any observation. C6 determinism: pure
deterministic numpy (no RNG anywhere); frozen test requires byte-identical rerun. C7 diffraction
robustness column (bounded, not load-bearing): one fold-Airy dressing column via frozen
`so.fold_airy_profile` at two κ values brackets the geometric-optics validity domain; HS13 stays parked.

## §7 Citations + absence re-verification list (binding on any manuscript)

Können 1983 JOSA (refraction-halo polarization theory — the ancestor); Können & Tinbergen 1991 Appl. Opt.
(the validated 22° anchors); **Können et al. 1998 Appl. Opt.** (inner-edge polarization identification —
the direct intellectual ancestor of the doublet column); Lefaudeux 2010 (Ursa post, §2.1); Moilanen &
Gritsevich 2022 JQSRT (inventory; arc unexplained); atoptics.co.uk/blog/moilanen-arc (34° wedge,
crystallographic improbability, zero polarization mentions — verified live 2026-06-11 by writer AND
refuter). Pre-manuscript absence re-verification (refuter fix #6): JQSRT 2022 full text, Können 1998
target list, **Tape & Moilanen 2006 (Atmospheric Halos and the Search for Angle x)**, and Können's
post-1998 halo-polarization reviews.

## §8 Out of scope (this leg)

The 34°-wedge polyhedron + oriented-sampler raytracer leg (separate ticket; triggered by MARGINAL);
2-D V-arm smear (in-plane apex region only here); any outreach (owner-gated, PASS/PARTIAL-rule only);
HS13 full diffraction dressing; field background-subtraction pipeline (the observable convention is
pinned in §2.3 so the table is measurable as stated).

**AMENDMENT A1 (pre-run, 2026-06-11, BEFORE the registered run; trigger = frozen-test sanity
check, not any registered result):** the Ih/iso ledge-peak estimator takes its max over points
with convolved intensity ≥ 0.1 × plateau (primary; 0.3 reported as a robustness column). Without
the mask, the dark-side Gaussian-tail intensity RATIO produces a spurious ~100 %-polarized region
at e⁻⁸-scale flux — an unmeasurable artifact, not a feature. No registered Stage-3 number had been
computed when this was frozen.

**Frozen-test contract:** `scripts/test_moilanen_pol_discriminator.py` pins, BEFORE the registered run:
the three apex-60 gate anchors, dmin(34°) = 11.040°, the 0.0508° split, the DoP(D) floor track
{0.930, 1.290, 1.718, 2.477}% at {11.04, 13, 15, 18}°, the LEF track {≈10.9, 13.1, 15.0, 18.1}° + TIR
cutoff ≈ 26.3°, and the REFL-P convergence anchor. After the registered run, the headline verdict
numbers are appended to the same test (standard lane pattern) and the run is re-executed byte-stably.
