# Pass C7 — Column-orientation-only render at h=18.6° (definitive measurement)

**Goal:** isolate the **upper tangent arc + 22° halo** at h=18.6° from the parhelia (plate-orientation) and the 46°-halo/supralateral (also random-orientation) clutter that conflate the multi-orientation render.

The preliminary measurement on `halosim_tangent_p2_h18.6_25000mr.bmp` (which used 4 crystal blocks simultaneously) showed bright features near the C5 hand-anchor predicted positions but with R/L asymmetry and dim apex — noise + overlapping concurrent features. A column-only render at higher ray count will give a definitive verdict.

---

## Crystal-block configuration in HaloSim

Currently the autosav1.sim (your last h=18.6° run) has **4 crystal blocks** enabled:

| Block # | Orientation file | Role | Action for C7 |
| ---: | --- | --- | --- |
| 1 | `Horiz column .1 deg disp.xng` | Column → upper/lower tangent arc + parhelic circle | **KEEP enabled** |
| 2 | `random.xng` | Random → 22° halo + 46° halo + supralateral | **KEEP enabled** *(for the 22° halo as scale reference)* — see note below |
| 3 | `plate 8 deg disp.xng` | Plate → parhelia | **DISABLE** |
| 4 | `plate 5 deg disp.xng` | Plate → parhelia | **DISABLE** |

To disable blocks 3 + 4 in HaloSim's Control Panel: either set their "weight" to 0, or change their orientation file to "no selection" via the GUI.

**Why keep `random` (Block 2):** the 22° halo from random crystals gives us a clean scale reference (R22 = 207 px from your h=0° calibration). Without it, we couldn't verify the scale is still consistent at h=18.6°. The 46° halo / supralateral *also* come from random crystals, so they'll still be present — but at h=18.6° they sit much further from the upper tangent arc than the parhelia did, so they're less likely to conflate the wing measurement.

**Alternative (cleanest possible):** disable Block 2 too. Then the render shows ONLY the upper + lower tangent arcs from columns. No 22° halo reference visible. We'd have to trust R22=207 from the prior calibration without in-render verification. Marginally cleaner, marginally riskier.

---

## What you do

1. Open HaloSim (it should still have the autosav1.sim state from your last h=18.6° render).
2. Open the Control Panel / Orientation selector.
3. Disable Block 3 (`plate 8 deg disp`) and Block 4 (`plate 5 deg disp`) — either set their weight to 0 or change their orientation file to "no selection".
4. Confirm sun elevation is still set to **18.6°**.
5. Confirm projection is still "Camera View" (Type9 in the .sim, default for HaloSim).
6. Set ray count to **4-6 million rays** (matching the h=0° calibration quality). The b&w h=0° render used 4.5M rays.
7. Render. Wait for it to complete (4-6M rays takes longer than 25K but should still finish in a few minutes).
8. Save as `halosim_tangent_p2_h18.6_columnonly_4.5mr.bmp` (or `.png` if HaloSim supports it) into:
   ```
   C:\Users\hughe\Dev\sundog\docs\calibration\halosim_outputs\
   ```

Optional but recommended: do BOTH a b&w version (matching the h=0° calibration style) and a colored version. The b&w gives cleanest arc geometry; the color confirms the scene matches what HaloSim's standard rendering shows.

---

## What I'll measure

Once the file lands:

1. Sample brightness along the predicted upper-tangent-arc locus (azimuth scan from -45° to +45° from zenith, radial following the C5-hand-anchor-fit curve).
2. With no parhelia / plate features confounding, the upper tangent arc wings should now be the ONLY bright signal along the predicted locus.
3. Find the azimuth at which the wing brightness drops to background → measured wing extent.
4. Compare to C5 hand-anchor measurement (wings at ±25.5° azimuth from zenith).

**Verdict logic:**
- If HaloSim wings at ≈±25° (within ±3°): **C5 hand-anchor is confirmed under the canonical handle.** Tangent route moves from "C5↔C6 substrate tension" → "passes canonical-handle reformulation." Adds a third promoted inverse route on the geometry side.
- If HaloSim wings at significantly different azimuth (±5° or more from C5's ±25.5°): **C5's circle fit was geometrically inconsistent with the canonical handle.** Tangent route remains unpromoted including under the canonical-handle test; the C5 recovery would re-classify as symmetry-bias artifact.
- If wings extend off-image-top (i.e. C5 wings are NARROWER than HaloSim's canonical): same as above ("C5's circle fit was too narrow").

---

## QC criteria

The column-only render at h=18.6° should show:
- A 22° halo (faint, from random Block 2) — sun-centered circle, **radius 207 px** (this confirms scale invariance)
- An **upper tangent arc**: bright "eyelid" on top of the halo, apex on halo crown (at sun_y - 207), wings extending up-and-out
- A **lower tangent arc**: mirror eyelid on bottom of halo (apex on halo bottom)
- A **parhelic circle**: horizontal bright band passing through the sun
- **NO parhelia** (plate orientation disabled) — the bright spots that were at ~±23° horizontal from sun should be gone
- Horizon line: white horizontal line below sun (at sun_y + 18.6 × 9.41 = sun_y + 175)

If parhelia are still visible, Block 3 or 4 didn't get disabled properly.

---

## If you can't disable plate blocks via GUI

I can edit the autosav1.sim file directly to zero out Blocks 3 + 4, then you'd open it in HaloSim and just click Start. Let me know if the GUI disable path doesn't work and I'll provide a pre-edited .sim.
