# Sundog Atlas — project folder

The **Sundog Atlas** treats the halo possibility-space as a **classified bifurcation diagram**:
forward-generate halos from the crystal geometry (you can't invert halo→crystal — the founding
non-invertibility), and organize the transitions as a bifurcation set = (A) caustic catastrophes ∪ (B)
ray-admissibility walls. This folder contains the atlas's banked receipts and supporting memos.

> **Frozen-as-portfolio per the 2026-06-04 pivot; NOT public-eligible.** The Phase-0.5 lit-pass +
> prominent attribution (Cowley & Schroeder for the HaloSim apparatus; Greenler/Tape/Können for the
> geometry; Thom/Arnold/Berry/Nye for the catastrophe optics) gate any outward claim. The public page
> `/sundog` is the separate geometry *workbench* (`../SUNDOG_V_GEOMETRY.md`), untouched by this lane.

**Roadmap:** [`../SUNDOG_V_ATLAS.md`](../SUNDOG_V_ATLAS.md) — the top-level atlas roadmap (the thesis, the
two-component wall taxonomy, the phase table, the §6 falsification gates).

> **Note on file naming.** The atlas runs its **own** phase scheme (0.5, 5, 6.5, 7, 8, 8.5–8.7, 11), distinct
> from the geometry-workbench phases (`calibration/PHASE10_*`, `PHASE11_OUTREACH_*`, …). Atlas receipts carry
> the **`ATLAS_PHASE…`** prefix and live here, so `ATLAS_PHASE11_CAPSTONE` is never confused with the
> geometry workbench's `calibration/PHASE11_OUTREACH_SYNTHETIC_MEMO`.

## The structural core (the classified bifurcation diagram) — the arc 6.5 → 8 → 7 → 11
- [`ATLAS_PHASE65_BIFURCATION_SET.md`](ATLAS_PHASE65_BIFURCATION_SET.md) — **Phase 6.5:** the bifurcation set
  computed — the documented transition elevations (22/29/32/46/58°) fall out of one computation from
  `{n, crystal geometry, sun elevation}`, none hardcoded (CZA 32.196°, CHA 57.804°, the 29.7° merge).
- [`ATLAS_PHASE8_STRATA.md`](ATLAS_PHASE8_STRATA.md) — **Phase 8:** the catastrophe-stratum classifier — the
  full 2-DOF single-crystal sweep (column 60°/90°, Wegener, Lowitz, pyramidal). **All confirm Berry 1994:
  no A₄ swallowtail, no D₄ umbilic;** the one higher catastrophe = the Lowitz A₃-lips (8-C; jet-verified).
- [`ATLAS_PHASE7_PHASE_DIAGRAM.md`](ATLAS_PHASE7_PHASE_DIAGRAM.md) — **Phase 7:** the forward sweep — the
  classified (elevation × habit) **phase diagram** (6 habits, 21 cells, 8 derived phase-boundary
  elevations), each transition classified component-A / component-B / occlusion.
- [`ATLAS_PHASE11_CAPSTONE.md`](ATLAS_PHASE11_CAPSTONE.md) — **Phase 11 (the capstone):** the small-parameter
  structural (platonic-solid) model — the whole classified atlas forward-generated from ~1 free continuous
  physical parameter (`n`) + the fixed ice lattice + the ~7 discrete habits, with the recompute-from-n
  demonstration, the match scorecard, and the named failure boundaries.

## The determining-shadow tower (the S2 / cross-substrate thread — Phases 5, 8.5–8.7)
- [`ATLAS_PHASE5_CROSS_SUBSTRATE.md`](ATLAS_PHASE5_CROSS_SUBSTRATE.md) — the Shadow-Invertibility candidate
  operator (a lossy averaged shadow *determines* discrete/topological hidden variables and *resists*
  continuous-magnitude ones); the lossiness-crossover result + the S2 **partial physical leg** on real halo
  optics. (Sibling lane; see also the proof-track Shadow-Invertibility ledger.)
- [`S2_LITPASS_E_G.md`](S2_LITPASS_E_G.md) — the S2 lit-pass (Tracks E/G: crystal size from diffraction;
  polarization / handedness).
- [`S2_MEASURED_SKY_SCOPE.md`](S2_MEASURED_SKY_SCOPE.md) — the scope for moving S2 from forward-model to
  measured-sky polarimetry (the Können-validated Mueller chain; per-feature-V vs net-V).
- [`SHADOW_INVERTIBILITY_PHASE5_HANDOFF_2026-06-07.md`](SHADOW_INVERTIBILITY_PHASE5_HANDOFF_2026-06-07.md) —
  the webdev/promo hand-off (SAFE-TO-SAY / DO-NOT-SAY, attribution checklist, doc-index registration).

## Attribution / lit-pass (a hard precondition)
- [`ATLAS_LITPASS_MEMO.md`](ATLAS_LITPASS_MEMO.md) — the citation spine + gap map (Track A geometry, Track B
  catastrophe optics incl. the decisive Berry 1994 prior-art, Tracks E/G). The Phase-0.5 reading gates
  every outward claim.

## Code (in `../../scripts/`) + tests (all pass)
`atlas_bifurcation_set.py` · `atlas_caustic_map.py` · `atlas_strata_map.py` · `atlas_jet_classify.py` ·
`atlas_forward_sweep.py` · `atlas_model.py` (+ `cza_formula.py`, `s2_optics.py`). Frozen tests:
`test_atlas_{bifurcation_set,caustic_map,strata_map,jet_classify,forward_sweep,model}.py` — 6 suites green.

**Shared halo catalog** (stays in `../calibration/`, used by both the atlas and the geometry workbench):
`HALO_PHENOMENA_ACCOUNTING.md` (the Gate-2 catalog), `SPECULATIVE_HALO_PROOFS.md`.
