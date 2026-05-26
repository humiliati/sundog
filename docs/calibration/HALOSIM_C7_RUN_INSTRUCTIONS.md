# Pass C7 — HaloSim run instructions (upper tangent arc at p2 + p13)

**Goal:** produce two HaloSim simulation images of the **upper tangent arc** at the sun elevations the calibration set anchors on:

- `h = 18.6°` (corresponds to photo **p2**)
- `h = 6.83°` (corresponds to photo **p13**)

These are well below the h ≈ 29° tangent-arc → circumscribed-halo merger Tape Ch 6 names, so both should show a clearly separated upper tangent arc with visible wings.

The Pass C7 question is: **what opening angle do the wings make?** That's the canonical Tape inverse handle (per W4) we use to re-evaluate C5's hand-anchor circle fit on p2.

---

## What you do

1. **Launch HaloSim** — double-click `%USERPROFILE%\HalSim361.exe`.
2. **Load the baseline simulation:** `File → Open` (or equivalent), choose `%USERPROFILE%\Circumscribed halo.sim`. The bundled description says: *"When the sun is lower, the circumscribed halo 'splits' into upper and lower tangent arcs. Try simulations at other solar elevations."* That's exactly what we want — drop the sun, get tangent arcs.
3. **Change the sun elevation to 18.6°.** The control is probably labeled "Sun elevation" or "Sun altitude" in the parameters panel. The default in this `.sim` file is 45°; we want 18.6°.
4. **Render the simulation** — click Run / Simulate / Render (whatever HaloSim calls its render button). Wait for the dot-density to build up enough that the upper tangent arc is clearly visible.
5. **Save the output image** — `File → Save image as` (or screenshot). Save as PNG to:
   ```
   docs\calibration\halosim_outputs\halosim_tangent_p2_h18.6.png
   ```
6. **Change sun elevation to 6.83°**, re-render, save as:
   ```
   docs\calibration\halosim_outputs\halosim_tangent_p13_h6.83.png
   ```

That's it. Drop me a line when both PNGs are on disk and I'll pick up from there (measurement of the opening angle θ at each altitude).

---

## QC criteria — what each image should look like

If any of these are wrong, the run is misconfigured and we'll iterate.

**At h = 18.6° (p2):**
- A clear circular 22° halo (from the random-orientation crystal contribution Circumscribed halo.sim includes).
- An **upper tangent arc** sitting *on top of* the 22° halo, brighter than the halo, with a recognisable apex directly above the sun and wings curving down-and-outward.
- The arc is *open* — wings clearly diverge from the apex, not yet closed into a circumscribed halo. (Tape Ch 6 confirms the merger doesn't happen until h ≈ 29°.)
- No lower tangent arc unless the rendering includes the area below the sun (it should — `Circumscribed halo.sim` simulates both).

**At h = 6.83° (p13):**
- Similar 22° halo + upper tangent arc.
- The upper tangent arc should be **more open** than at h = 18.6° — wider V-shape, less hugging the 22° halo crown.
- Lower tangent arc visible below the sun as a separate feature.

**If you see something off** — circumscribed halo instead of split tangent arcs, no tangent arc at all, weird arc shapes — most likely the column-orientation crystal block in the `.sim` got modified accidentally. The baseline `Circumscribed halo.sim` should already have the right crystal population (`Horiz column .3 deg disp.xng` orientation file); we're only changing sun elevation.

---

## What I'll measure from the PNGs

Once both PNGs are on disk:

1. **Identify the upper tangent arc apex** (directly above the sun, sitting on the 22° halo).
2. **Identify the wing endpoints** — where the arc fades into background brightness or merges into the parhelic circle.
3. **Measure the opening angle θ** — the angle subtended at the apex by the two wing endpoints.
4. **Compare predicted θ(h=18.6°) to measured θ from p2's existing hand-anchor circle fit** in `p2-anchor.json` `upper_tangent_manual_samples.points`.
5. **Verdict:** does the canonical-handle θ(h=18.6°) agree with the C5 hand-anchor circle? If yes, Pass C7 confirms C5 under the canonical handle → tangent route moves from "C5↔C6 substrate tension" → "passes under canonical-handle reformulation." If no, C5's circle fit was geometrically inconsistent with the canonical handle.

This is the Step 2 cross-check from the Pass C7 workflow.

---

## Optional: HaloSim render-quality settings

If the image is too sparse / noisy to identify wing endpoints clearly:

- Increase the **ray count** (e.g. from 60,000 to 200,000+ — the number in line 135 of the `.sim` file is the initial setting).
- Increase the **image resolution** if HaloSim has that option in its GUI.
- Render for longer if HaloSim has a "continue simulation" mode.

Higher ray count makes the arc better defined — more dots, sharper apex/wings. Cost is wall-clock time; for a single run we don't need millions of rays, just enough to see the wing endpoints clearly.

---

## If anything blocks you

If a step doesn't work (e.g., HaloSim doesn't recognise the `.sim` file, the sun elevation control isn't where I described, the render hangs), drop a screenshot or paste the error and I'll guide the next step. The workflow above is my best guess based on the `.sim` file content and HaloSim's general design; it isn't tested against your specific install.
