# Sundog Atlas — the Halo Possibility-Space as a Classified Bifurcation Diagram

> **STATUS: DRAFT SCAFFOLD, unpromoted.** Opened 2026-06-05, enriched 2026-06-05 with the
> caustics / catastrophe-theory framing. The atlas was the lab's FIRST page and first discovery;
> this roadmap returns to it with the cross-substrate thesis grown up. Governs the
> generative/structural atlas + the determinacy map + the thesis embodiment; it USES the geometry
> workbench (`SUNDOG_V_GEOMETRY.md`) and the HaloSim apparatus (`SUNDOG_HALOSIM_CINEMATIC_SIDECAR.md`)
> as instruments. **We stand on the shoulders of giants** (§0.1) — no claim is defensible until the
> lit-pass (Phase 0.5) lands and attribution is prominent. The public atlas already lives at
> `/sundog` (`atlas.html` is a legacy redirect). No new public surface or claim beyond established
> atmospheric / catastrophe optics without owner sign-off + evidence-tier review.
>
> **Update 2026-06-07 — the determining-shadow-tower thread (§1.2–1.3, Phases 8.5–8.7) advanced from
> roadmap to a banked forward-model result.** The lossiness-crossover ran (synthetic
> `operator_confirmed_synthetic` + an S2 **partial physical leg** on real halo optics); the diffraction
> + Stokes/Mueller layers are built (`scripts/s2_optics.py`, standalone — HaloSim can't host them) and
> the polarization model is archival-validated against Können 1991. The structural **CORE** (bifurcation
> diagram, Phases 6.5/11) is unchanged. See §1.2–1.3, the Phase-4 table, `proof/PHASE5_CROSS_SUBSTRATE.md`
> §3.11–3.14, `atlas/S2_MEASURED_SKY_SCOPE.md`, and the public-eligibility gate in
> `atlas/SHADOW_INVERTIBILITY_PHASE5_HANDOFF_2026-06-07.md`. **Still unpromoted, NOT public-eligible.**

Working hook:

> The atlas is where we first learned the shadow can't rebuild the body. So we stopped inferring the
> halos and started **generating** them — forward, from the geometry, and letting the sky testify.
> The transitions between halos are not edges of a catalog; they are the **bifurcation set**.

Short version:

> The original objective was to infer the hypothetical *invisible* halos from the measurable ones.
> We can't — you cannot infer the next halo from the previous (the founding instance of the program's
> non-invertibility thesis). The resolution: the apparatus (HaloSim) is the forward geometry→halo map,
> so we **generate** the complete atlas — invisible halos included — by sweeping the geometry, and we
> organize it by the global structure we *can* pin down. That structure is a **classified bifurcation
> diagram**: each halo is a caustic, and the merge/appear/disappear transitions are the bifurcation
> set, split into (A) caustic catastrophes and (B) ray-admissibility boundaries. Phase 11 is the
> capstone: Kepler's platonic-solid dream of a cosmos from a few perfect forms — realized on the
> object where it is actually true.

---

## 0. The founding loop (and the honest boundary)

**What the atlas discovered first.** Building the original atlas, the lab could pin down *global*
relationships — chiefly the **sun-elevation relationship** (how halos morph and merge as the sun
rises) — but **could not infer the next halo from the previous**. That negative is not a failure. It
is the **first measurement of the program's load-bearing thesis** — the shadow does not rebuild the
body — observed in optics, before mesa, Navier–Stokes, chatv2, or the determining-shadow-set
instrument existed. (The same non-invertibility is already written in halo form in
`docs/calibration/SPECULATIVE_HALO_PROOFS.md`: the 6-fold crystal symmetry means a single ring
"cannot be isolated" — the shadow can't single out the wedge.)

### 0.1 We stand on the shoulders of giants (attribution is load-bearing)

This lane invents **no** physics. Its foundations are entirely borrowed and must be credited
prominently (the public workbench attribution gap is already a flagged blocker —
`docs/calibration/PHASE11_OUTREACH_SYNTHETIC_MEMO.md` W2):

- **The apparatus** (HaloSim) is **Les Cowley & Michael Schroeder's**. We run it; we did not build it.
- **The halo geometry**: Greenler; Tape (and Tape & Moilanen); Können; Cowley (Atmospheric Optics).
- **The catastrophe optics**: Thom and Arnold (the elementary catastrophes / caustic singularities);
  Berry & Upstill and Nye (catastrophe optics — caustics and their diffraction dressing).

**Sundog's contribution is the synthesis, not the foundations** (§2). The formal lit-pass +
cite-list (Phase 0.5 / `docs/atlas/ATLAS_LITPASS_MEMO.md`) is a hard precondition for any claim or
public surface, exactly as in the P-vs-NP and JEPA lanes.

### 0.2 Boundary (read before any claim)
- **Not discovering halo or catastrophe physics.** Both are established (the giants, §0.1). Sundog's
  contribution is the generative-structural *organization*, the *determinacy map*, the
  forward-generated *invisible-halo predictions*, the *bifurcation-set classification* of the
  generated atlas, and the atlas's place as a physical instance of the cross-substrate thesis.
- **The halo display is a READ-OFF body, not a CONTROL body.** Most legible physical instance of "the
  shadow resists the body" (sibling to cap-set / unit-distance in `CROSS_SUBSTRATE_NOTES.md`); it does
  **not** touch the founding *control*-regime-2 lacuna.
- **We generate; we do not invert.** The apparatus produces halos forward; photographs verify. No
  claim infers crystal populations backward from an observed display.
- **The classification lives in the ray-optics limit.** Real halos are the caustic *smeared* by
  orientation-averaging, the sun's angular size, and diffraction. The sharp bifurcation set is the
  skeleton; observed transitions are its smoothed image. The smearing is itself catastrophe-theoretic
  (the diffraction catastrophes, §3), so the skeleton survives the dressing — but every "the
  transition is at X°" statement is the smeared image of a ray-optics prediction, not a sharp edge.

---

## 1. The thesis the atlas embodies — and its bifurcation structure

The atlas is the **finding-vs-checking / certificate** structure made physical (the shape the
P-vs-NP, Ramanujan, and IUT lanes circle): **forward (geometry → halo) is cheap; inverse (halo →
crystal population) is hard and non-unique** — capacity-relative one-wayness, in ice. The apparatus is
the *finder*; the photographed sky is the *verifier*.

**Each halo is a caustic.** Minimum deviation — the lab's existing generator vocabulary
(`HALO_PHENOMENA_ACCOUNTING.md`) — *is* the **fold caustic** (the ray density diverges where
`dδ/dθ = 0`). So the atlas is already a catastrophe-theoretic object; it just hadn't said so.

**The transitions are the bifurcation set**, the low-dimensional stratified object in control space
(sun elevation × crystal habit × orientation symmetry) where the caustic structure reorganizes. It is
the **determining shadow of the atlas's topology**: you can read the inter-halo relationship structure
off it, but you **cannot** rebuild the full intensity display from it (the founding non-invertibility).

### 1.1 The two-component wall taxonomy (a structural result to verify)

Grounding the actual transitions, they split cleanly into two mechanisms — and every observed wall is
one or the other:

- **(A) Caustic catastrophes** — caustics *coalescing / changing type* (the *merges*). Folds meeting
  is a **cusp (A₃)**; higher coincidences give swallowtails (A₄), butterflies, umbilics (D₄).
  *Candidate:* the **UTA + LTA → circumscribed-halo merge near elevation ≈ 29°** (Tape AH-CH06 p62) —
  two caustics coalescing.
- **(B) Ray-admissibility boundaries** — a face-pair ray path *appearing/disappearing* because of TIR,
  grazing incidence, or geometric admissibility (the *domain walls* of the catastrophe, where the
  generating family ceases to be defined). *Candidates:* the **circumzenithal arc disappearing above
  ≈ 32°** (internal reflection goes total) and the **circumhorizontal arc appearing only above ≈ 58°**
  — the same 90° plate prism on opposite sides of two admissibility thresholds.

Pinning each transition to (A) or (B) — and, for (A), to a Thom catastrophe type — is the Phase-6.5
receipt, not an armchair claim.

### 1.2 The determining-shadow tower (invertibility tracks the hidden-variable KIND)

The atlas does not have one shadow — it has a **tower**, each layer reading a different hidden
variable, and the **invertibility changes as you climb because the *kind* of variable changes**:

- **Shadow 1 — geometry (caustic position).** Reads `{habit, orientation symmetry, elevation}`.
  Continuous, scale-free. **RESISTS** — many crystal populations give the same display (the founding
  non-invertibility).
- **Shadow 2 — diffraction (the dressing of the caustic).** Reads crystal **size** — a continuous
  scalar the geometric shadow is structurally blind to. Conditionally legible (monodispersity gate),
  richest at the cusps (Pearcey). **RESISTS — but only PARTIALLY** (2026-06-07 forward-model finding,
  `proof/PHASE5_CROSS_SUBSTRATE.md` §3.13): size is *magnitude*-encoded, so ensemble-averaging preserves
  the mean and the diffraction envelope leaks a rough size (`cont` 0.97→0.45, not →0); only *phase-
  offset* continuous variables wash fully, a scale/size does not. *(Also corrected: the size dressing is
  **not** on the 22° refraction-halo edge — that edge is a zero-contrast step, Berry 1994 — but on the
  **corona / parhelion-fold** Airy structure; see §3, Phase 8.5.)*
- **Shadow 3 — polarization (the Stokes structure).** Reads a **discrete/structural** variable: the
  **handedness/parity** of the ray-path-plus-crystal configuration (Stokes `V` sign), and the c-axis
  class via ice **birefringence**. A different *kind* of hidden variable — a sign, not a magnitude.
  **This is the layer where the shadow INVERTS** — confirmed in the forward model (2026-06-07): a
  discrete `Z₂` (ice-phase halo-radius, handedness `V`-sign) is *determined* `disc=1.000` flat across
  the whole lossiness grid, robust to the distribution-smearing that defeats Shadows 1–2 — the optical
  analog of the program's one *exact* regime-2 (Aharonov–Bohm). The **linear**-pol half of the Mueller
  model is archival-validated against Können 1991; the **circular** `V`/handedness is reframed into
  *per-feature V* (defensible, measurable) vs *net-V = population handedness* (disfavored, quarantined),
  and stays forward-model pending a sky measurement (§3.14, `atlas/S2_MEASURED_SKY_SCOPE.md`).

**The structural finding: invertibility tracks the hidden-variable's kind** — now forward-model-tested
(2026-06-07) and *graded*. Continuous magnitudes (position, size) → *resisting* shadows, but the
resistance is **graded by encoding**: phase-offset continuous variables wash to ~0, while a
scale/magnitude (size) resists only *partially* (its mean survives averaging). Discrete parities /
topological invariants (handedness, ice-phase) → *determined* shadows, exactly (`disc=1.000`), the
AB-exact analog. The tower climbs from the partial-resisting (continuous magnitude) to the
cleanly-determined (discrete, topological); the polarization layer is where the flip happens. So the atlas embodies not just "the shadow resists the
body" but the full **boundary** of *when* it resists (continuous bodies) and *when* it determines
(discrete/topological variables) — the lab's mature thesis, physical and photographable.

### 1.3 The atlas instantiates the Shadow-Invertibility Law (and is its cleanest test bed)

The tower of §1.2 is the optical instance of a candidate **cross-substrate operator** — the
**Shadow-Invertibility Law**: *a lossy averaged shadow determines the structurally-stable
(discrete/topological) part of a hidden state and resists the continuous-magnitude part; the lossiness
is essential.* Full statement, the portfolio instantiation table (AB / syndrome / mesa-marginal /
Gate-0-null / the halo layers), and the alignment-side correction it implies live in
[`proof/PHASE5_CROSS_SUBSTRATE.md`](proof/PHASE5_CROSS_SUBSTRATE.md) — the candidate operator for the
coarse-graining roadmap's long-missing Phase 5.

**Why the atlas is the test bed:** the optics gives discrete variables (handedness, ice phase, halo
radius, optical-vortex index) and continuous variables (position, size, `n`, `C_n²`) **side by side,
photographable, forward-generable, with the lossiness (population spread) tunable** — which mesa and
turbulence do not. The **lossiness-crossover experiment** (sweep the population spread; watch the
continuous coordinates wash out while the discrete ones hold exact, the *same* crossover across
substrates) **has now been RUN** (§3.11/§3.13/§3.14 of `proof/PHASE5_CROSS_SUBSTRATE.md`):
`operator_confirmed_synthetic` on two synthetic substrates (S0+S1) + an **S2 partial physical leg** on
real halo optics — discrete-determines confirmed (`disc=1.000`), continuous-resists *partial* (the
magnitude scale-leak). That converts the candidate operator into a *measured* (forward-model-tier)
cross-substrate identity on the synthetic side and a *partial* one on the physical side; a full physical
discharge (a clean continuous washout + measured-sky circular-`V`) is scoped (`atlas/S2_MEASURED_SKY_
SCOPE.md`) and owed. The prettiest page in the lab is, at last, where its deepest theorem **began to get
measured** — at forward-model tier, not yet public.

## 2. What is honest vs. what is reach

**Honest:** a forward-generated complete atlas (invisible halos as falsifiable forward predictions); a
determining-shadow-set map of the halo geometry; the bifurcation-set classification of the *generated*
atlas (computed, then verified against the documented transitions and the apparatus); the atlas's
substrate-rhyme placement; a structural generative model (Phase 11) whose forward predictions the
apparatus and the sky can check.

**Reach; do not claim:** "Sundog discovered/explains halo optics or catastrophe optics"; "we inferred
the crystal population from a photograph"; "the atlas is a control-regime-2 receipt"; any **priority
claim over Greenler / Tape / Können / Cowley / Berry / Nye / Thom / Arnold** or over the
catastrophe-optics literature; asserting a specific catastrophe type without the Phase-6.5 computation.

## 3. Core definitions

- **Forward apparatus.** HaloSim3 (Cowley & Schroeder) — a Monte-Carlo halo ray-tracer (geometry →
  halo image), HS-0 proven; ~1M B&W rays / geometry-confirmation render; sun-altitude sweeps native.
- **Caustic / fold.** The minimum-deviation locus where ray density diverges (`dδ/dθ = 0`) — an A₂
  fold; the bright edge of a halo.
- **Bifurcation set.** The stratified subset of control space (elevation × habit × orientation) where
  the caustic structure is degenerate — the transitions. Two components: **(A)** caustic catastrophes
  (cusp A₃ / swallowtail A₄ / butterfly / umbilic D₄), **(B)** ray-admissibility boundaries.
- **Diffraction catastrophe.** The wave dressing of a geometric caustic (Berry–Upstill): the fold
  dressed by the **Airy** function, the cusp by **Pearcey**. A *finer* shadow (§ Phase 8.5).
  **Caveat (2026-06-07 lit-pass, `atlas/S2_LITPASS_E_G.md`):** the 22° refraction-halo edge is **not** a
  fold but a *step* (zero-contrast diffraction shoulders, no supernumeraries — Berry, Appl. Opt. 33:4563,
  1994), so the size-bearing Airy dressing lives on the **corona** (pure diffraction, `θ∝λ/a`) and the
  **parhelion fold** (faint supernumeraries, contrast 0.178), not the 22° ring.
- **Determining shadow (of the atlas topology).** The bifurcation set: low-dim, determining for the
  transition structure, non-invertible to the full display.
- **Invisible-halo prediction.** A generated rare halo at a higher-codimension stratum — a falsifiable
  forward claim ("this arc appears under *this* habit at *this* elevation").
- **Read-off resister.** A body whose shadow cannot reconstruct it, read off rather than controlled —
  the atlas's class.
- **Stokes vector / polarization shadow.** The 4-component `(I, Q, U, V)` description of the light at
  each halo point — a 4× richer shadow than intensity. `Q, U` = linear polarization (reads the
  refraction geometry / Fresnel angles); `V` = circular polarization (a *signed parity* reading
  handedness).
- **Handedness (the discrete hidden variable).** The parity of a chiral ray-path-plus-crystal
  configuration (left vs right), read off the sign of Stokes `V` — a `Z₂` variable, the optical analog
  of a topological invariant. **Honest caveat:** ice Ih is **not** optically active like quartz; the
  handedness read here is ray-path parity + the c-axis (optic-axis) class via ice **birefringence**,
  not bulk molecular chirality.
- **The determining-shadow tower.** The stratified decomposition (geometry → diffraction →
  polarization) whose layers read hidden variables of different kinds (continuous magnitude → discrete
  parity), with invertibility tracking the kind (§1.2).

## 4. The phase arc (built to the capstone)

| phase | goal | status |
| ---: | --- | --- |
| 0 | Scope + the founding-loop reframe (this doc) | open |
| **0.5** | **Lit-pass + cite list** — formal `ATLAS_LITPASS_MEMO.md`; ground every claim against the giants; prominent attribution (the apparatus = Cowley/Schroeder; geometry = Greenler/Tape/Können/Cowley; catastrophe optics = Thom/Arnold/Berry/Nye) | **Tracks E+G READ** (2026-06-07, `atlas/S2_LITPASS_E_G.md`); A–D/F still open; attribution gate still a hard precondition |
| 1 | The elevation relationship — the global handle that organizes the display | grounded (Tape) |
| 2 | The forward apparatus — HaloSim geometry→halo, sun-altitude sweeps | DONE (HS-0 proven) |
| 3 | The known-phenomena catalog — verification anchors + documented merge/morph relationships | grounded (`HALO_PHENOMENA_ACCOUNTING.md`) |
| 4 | The parametric workbench — "draws the parhelion from the math"; the rendering *is* the proof | grounded (`SUNDOG_V_GEOMETRY.md`) |
| 5 | Determining-shadow-set on the optics — what the display determines vs. resists; bank the halo as a physical read-off resister | advanced — lossiness-crossover RUN (`proof/PHASE5_CROSS_SUBSTRATE.md` §3.11–3.14): discrete *determines*, continuous magnitude *resists partially* |
| 6 | The global-invariant search — the determining invariants beyond elevation (habit class, orientation symmetry, optical-path topology) | open |
| **6.5** | **Compute the bifurcation set** — from the ray-optics deviation map: caustics (`∂δ=0`), their coalescences (cusp `∂²δ=0`, swallowtail `∂³δ=0`) = component (A); TIR/grazing admissibility walls = component (B). Predict the transition elevations + types; verify 29° (A-cusp), 32°/58° (B-walls) fall out vs. the apparatus | **open — the key receipt** |
| 7 | The forward sweep — generate the complete atlas; the higher-codim strata are the directed search for the invisible halos | open |
| 8 | Invisible-halo predictions — each generated rare halo (higher catastrophe) as a falsifiable forward claim | open |
| **8.5** | **Shadow 2 — diffraction → crystal SIZE** — read size off the Airy dressing, calibrated by the corona. **Dressing layer BUILT** (`scripts/s2_optics.py`, standalone — HaloSim can't host it). **Corrected:** size lives on the corona / parhelion-fold, NOT the 22° step edge (Berry 1994); and size **resists only PARTIALLY** (magnitude scale-leak, §3.13). | built — partial-resister, not a clean washout |
| **8.6** | **Shadow 3 — polarization → crystal HANDEDNESS** — Stokes `V` sign + c-axis class via birefringence; the cleanly-determined shadow (discrete `Z₂`, AB-exact analog). **Mueller layer BUILT** (`scripts/s2_optics.py`); **linear-pol archival-validated vs Können 1991** (`scripts/s2_konnen_validate.py`); handedness *determines* in forward-model (`disc=1.000`). Circular-`V` reframed (per-feature defensible vs net-V quarantined), sky-measurement scope-and-hold. | built + linear-validated; circular-V forward-model, sky owed |
| **8.7** | **Net circular polarization → population handedness imbalance** — **DEMOTED (2026-06-07):** net-`V` is **~0 by symmetry** — no mechanism breaks ice-crystal handedness (ice achiral; E/B not pseudoscalar; Können's `U=0` is the exact net-V-cancellation analog). The measurable signal is a **per-feature antisymmetric `±V(θ,φ)` integrating to ~0**, not net-CP. | demoted — likely-null; per-feature V is the real target |
| 9 | The verification layer — photographed displays as verifier; geometry-confirmation receipts; named falsifiers | open |
| 10 | Substrate-rhyme / certificate placement — the atlas as the physical forward-easy/inverse-hard card | open |
| **11** | **THE CAPSTONE — the classified bifurcation diagram (platonic-solid model, except holograms)** | aspirational |

Phases 1–4 are grounded/done; Phase 0.5 (lit-pass) gates everything outward-facing; the new structural
arc is 5–11, anchored on the 6.5 bifurcation-set computation. **Status note (2026-06-07):** the
**determining-shadow-tower sub-thread (8.5–8.7)** advanced substantially — the diffraction + Stokes/
Mueller layers are built (`scripts/s2_optics.py`), the polarization model is archival-validated against
Können 1991, and the lossiness-crossover ran (forward-model tier). The atlas's **structural CORE — the
bifurcation-set computation (6.5) and the capstone (11)** — remains entirely open and is untouched by
this work.

## 5. Phase 11 — the platonic-solid model, except holograms (the capstone)

Kepler's *Mysterium Cosmographicum* derived the cosmos from a small set of nested platonic solids — a
beautiful structural-generative model that was **wrong**, because the planets are not generated by the
solids. The halo atlas is **that instinct on the object where it is true**: the display *is* generated
by a small discrete set of geometric forms — the **ice-crystal habits and their symmetry groups**
(hexagonal plates and columns, pyramidal crystals, bullet rosettes) — plus the continuous global
invariants (elevation, orientation). These crystal forms are the "platonic solids" of the halo world;
their optical-path families are the generators.

The capstone is the **classified bifurcation diagram**: the whole atlas — every halo and every
inter-halo relationship — **derived** as the caustic family of these forms over the control space, with
its transition walls *classified* (component A by Thom catastrophe type; component B by admissibility
mechanism) and its rare phases (the invisible halos) located at the higher-codimension strata. The
relationships the original atlas could only *list* become **deduced consequences of the geometry**: the
atlas is a *phase diagram*, the merge transitions its phase boundaries.

The discipline that makes it not-Kepler's-error: **the apparatus verifies it forward.** Every
structural prediction is renderable (HaloSim) and, where the sky cooperates, photographable. Beautiful
*and* checkable — the platonic-solid dream without the platonic-solid mistake. **Exit:** a
small-parameter structural model whose forward predictions (transition elevations, catastrophe types,
the higher-stratum invisible halos) match the apparatus and the photographed catalog, with named
failure boundaries where the reduction breaks (multiple-scattering displays, rare habits, the
ray-optics-limit caveat of §0.2).

## 6. Falsification surface / scope gates

| gate | violation | disposition |
| --- | --- | --- |
| **uncited claim** | any claim made before Phase 0.5 lands / without prominent attribution | hold — the lit-pass is a hard precondition |
| **inversion** | inferring crystal geometry backward from an observed display | scope violation — forward-generate-and-verify only |
| **physics/priority** | claims to have discovered halo or catastrophe optics, or priority over the giants | over-claim — credit the literature |
| **armchair catastrophe** | asserting a transition's catastrophe type without the Phase-6.5 computation | demote to conjecture |
| **ray-optics overreach** | stating a sharp transition edge without the smearing caveat | demote to "smoothed image of the ray-optics prediction" |
| **control conflation** | treating the read-off atlas as a control-regime-2 receipt | category error |
| **public over-reach** | public surface beyond established science without sign-off + evidence tiers | hold — internal only |

## 7. Promotion criteria

- **To Active:** Phase 0.5 (lit-pass) + Phase 5 (determinacy map) + Phase 6.5 (bifurcation set) land,
  the documented 29°/32°/58° transitions fall out of the computation, and the halo is banked as a
  measured read-off resister with a reproducible forward catalog.
- **To Hero/public:** the forward-generated atlas + invisible-halo predictions are apparatus-verified,
  attribution is prominent, and owner-signed for public copy (evidence-tier reviewed).
- **To Capstone (Phase 11):** the classified bifurcation diagram's forward predictions match the
  apparatus across the swept space with named failure boundaries.

## 8. Cross-references

**Lane instruments / docs:**
- `docs/atlas/ATLAS_LITPASS_MEMO.md` — the formal lit-pass + citation spine (Phase 0.5; to be written).
- `docs/SUNDOG_V_GEOMETRY.md` — the parametric geometry workbench (Phase 4 grounding).
- `docs/SUNDOG_HALOSIM_CINEMATIC_SIDECAR.md` / `docs/calibration/HALOSIM_VALIDATION_PROTOCOL.md` — the
  HaloSim apparatus (Cowley & Schroeder).
- `docs/calibration/HALO_PHENOMENA_ACCOUNTING.md` — known-phenomena catalog + transitions (Tape).
- `docs/calibration/SPECULATIVE_HALO_PROOFS.md` — the 6-fold-symmetry non-isolation (halo-form
  non-invertibility).
- `docs/CROSS_SUBSTRATE_NOTES.md` — the body-resistance / determining-shadow-set thesis + the
  Aharonov–Bohm holonomy lesson the atlas embodies physically.
- `docs/SUNDOG_V_P_V_NP.md` / `docs/pvnp/SUNDOG_CERTIFICATE_PROBLEM.md` — the forward-easy/inverse-hard
  certificate structure.

**The giants (citation spine — verify/expand in Phase 0.5):**
- *Halo geometry:* Greenler, *Rainbows, Halos, and Glories*; Tape, *Atmospheric Halos* (+ Tape &
  Moilanen); Können, *Polarized Light in Nature*; Cowley, Atmospheric Optics.
- *Catastrophe optics:* Thom, *Structural Stability and Morphogenesis*; Arnold (caustic/wavefront
  singularities); Berry & Upstill, "Catastrophe optics"; Nye, *Natural Focusing and Fine Structure of
  Light*.

## 9. Forbidden language

- "Sundog discovered / explains halo optics or catastrophe optics."
- "We inferred the crystals from the photo." / any inversion claim.
- "The atlas proves body-resistance on a trained system."
- Any priority claim over Greenler / Tape / Können / Cowley / Berry / Nye / Thom / Arnold.

## 10. One-paragraph public summary (draft, DO NOT DEPLOY)

The Sundog Atlas is a map of *every possible halo* — not inferred from the ones we can photograph (you
cannot infer the next halo from the previous; that was our first discovery), but **generated forward**
from the geometry of ice crystals and the angle of the sun, on an apparatus built by Les Cowley and
Michael Schroeder, and verified against the sky. Each halo is a caustic, and the transitions between
them — the merges, the appearances, the disappearances — are a **bifurcation set**: the skeleton of
the whole display, the same mathematics Thom, Arnold, Berry, and Nye gave to caustics, applied to the
geometry Greenler, Tape, and Cowley gave to halos. Its capstone derives the entire atlas, invisible
halos included, from a handful of crystal forms: Kepler's dream of generating a cosmos from a few
perfect shapes, finally pointed at the place where it comes true.

---

*Sundog Research Lab — SUNDOG_V_ATLAS scaffold. The founding page, returned to with the mature thesis:
do not rebuild the body from the shadow — generate the body forward and let the shadow testify. A
physical read-off instance of the cross-substrate thesis, organized as a classified bifurcation
diagram. We stand on the shoulders of giants; the lit-pass gates the claims. Internal; unpromoted; no
public surface beyond established science without owner sign-off.*
