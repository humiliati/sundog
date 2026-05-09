# Sundog Generator Spec

Working thesis:

> The Sundog Generator converts natural-language optical descriptions into
> reproducible parametric poses, then renders those poses as SVG. AI may
> assist prompt interpretation and atmospheric styling, but the optical
> skeleton is deterministic.

That sentence protects the claim. The Sundog Generator is not a magic art
model. It is a constrained composition engine where the geometry is primary
and any AI is optional. The proof travels with every output because every
output has a parameter trace.

## Why a Spec, Not a Manifesto

A prompt-only image generator can produce plausible sundogs. It cannot
prove that the parhelion description was responsible. The strongest claim
the project can defend in public is reproducibility:

> Here is the parameter state. Here is the geometry the description derives
> from those parameters. Here is the SVG render. Here is the same render
> overlaid on a real photographed sundog.

A spec — a callable surface with a fixed pose schema — makes that claim
legible to other tools and to technical readers. It also makes the
generator innocuous in third-party contexts (a callable visual utility,
not a research artifact in tension with whichever LLM is hosting it).

## Related Documents

- [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md) — the workbench roadmap
  this spec serves. The workbench is the human-tunable face of the
  generator; the spec is the programmatic face.
- [`SUNDOG_OVERLAY_PROTOCOL.md`](SUNDOG_OVERLAY_PROTOCOL.md) — the
  photo-overlay test methodology. Mode two of the generator depends on it.
- [`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) — the
  site-wide visual frame. Owns theme-token reservation; any new tokens the
  generator introduces go via `sundog-theme.css`.
- `public/poses/*.json` — the named-pose library. The canonical pose is
  the truth-object reference.

## Architecture

The pipeline is a single direction with optional terminal beautification:

```
Prompt
   ↓
Optical intent parser  (rules-based; LLM optional)
   ↓
Sundog parameter JSON  (the truth object)
   ↓
Geometry solver        (parhelion description → derived geometry)
   ↓
SVG / vector render    (deterministic)
   ↓
[optional] raster / postprocess
   ↓
[optional] AI beautification
```

The arrows are one-way. The renderer never reads from the AI side. The
parameter JSON is the only surface that can be persisted and replayed.

### Truth object (pose JSON)

```json
{
  "phenomenon": "parhelion_display",
  "sunAltitudeDeg": 18,
  "halo22Intensity": 0.85,
  "halo46Intensity": 0.25,
  "czaIntensity": 0.65,
  "czaCurvature": 0.85,
  "czaBloom": 1.4,
  "parhelicCircleIntensity": 0.55,
  "parhelicCurvature": 0.66,
  "parheliaIntensity": 0.9,
  "parheliaDaggerLength": 1.0,
  "sunPillarIntensity": 0.80,
  "sunPillarLength": 0.65,
  "compassRayLength": 1.0,
  "dispersionWidth": 0.35,
  "rainbowSaturation": 0.85,
  "secondarySunsStrength": 0.0,
  "ringOverlapBias": 0.5,
  "idleScintillationAmplitude": 0.0,
  "skyPalette": "polar_twilight",
  "compositionMode": "canonical"
}
```

Field tiers (inherited from
[`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md) Actionability Audit):

- **math-derived**: `sunAltitudeDeg` (planned), `parhelicCurvature` (planned)
- **free visual**: most intensity / length / width / saturation knobs
- **composition fiction**: `secondarySunsStrength`, `ringOverlapBias`
- **palette enum**: `skyPalette` ∈ {`polar_twilight`, `winter_blue`,
  `dawn_warm`, `dusk_violet`, `night`} (extensible)
- **composition enum**: `compositionMode` ∈ {`canonical`, `educational`,
  `nine_halo_eye`, `poster`}

Unknown fields are ignored. Missing fields fall back to canonical defaults.

### Geometry solver

A pure function: `solveGeometry(pose) → derivedGeometry`. Returns:

```json
{
  "parhelionOffsetDeg": 22.8,
  "parhelionPositionsSvg": [{"x": 280, "y": 480}, {"x": 720, "y": 480}],
  "czaVisible": true,
  "czaApex": {"x": 500, "y": 80},
  "parhelicArcPath": "M 0 368 Q 500 632 1000 368",
  "compassRayLength": 36,
  "halo22RadiusSvg": 220
}
```

This is the layer that encodes the parhelion description. It is the
testable core: given a pose, did the solver produce the geometry the
description requires? Phase 3 of the geometry roadmap binds the
math-derived fields here.

### Renderer

A pure function: `render(pose, derivedGeometry) → svgString`. Returns
clean SVG with no embedded scripts. Every rendered element has a stable
`data-sd-component` attribute (`sun-core`, `halo-22`, `cza-primary`,
`parhelion-left`, etc.) so downstream tools (overlay-fit measurer, AI
beautification ControlNet) can reason about the structure without
re-parsing arbitrary SVG.

## The Three Modes

### Mode 1 — Render from Math (the proof surface)

The user moves sliders or types a prompt. The system renders only the
mathematically derived SVG. No AI beautification. No raster
postprocessing. Sharp vector lines, prismatic gradients, declarative
clarity. This mode is the proof.

Default landing surface for `sundog-workbench.html` and the eventual
hero. A first-time visitor should see this mode.

### Mode 2 — Compare to Sky (the proof amplifier)

The user loads a reference photograph. The app overlays the generated
geometry at adjustable scale / rotation / opacity and reports fit
measurements: 22° halo radius error, parhelia azimuth error, CZA apex
error, pillar alignment, parhelic-arc curvature delta.

This is where the claim becomes legible to technical viewers.
[`SUNDOG_OVERLAY_PROTOCOL.md`](SUNDOG_OVERLAY_PROTOCOL.md) is the
canonical procedure.

### Mode 3 — Make It Beautiful (the promotion surface)

The app takes the validated vector render and uses it as a guide /
control layer for atmospheric glow, sky texture, clouds, film grain,
poster art, or hero-section imagery. AI beautification (ControlNet,
img2img with the SVG as init, depth-based diffusion, etc.) belongs here
and only here.

The promotion surface must visibly derive from the proof surface — every
"pretty" render should be one-click reachable from its underlying
parametric pose, and that pose should be downloadable as JSON.

## Tool Schema (External GPT Apps)

Other GPT apps do not need to know the parhelion theorem. They need a
callable tool with a stable shape. The tool is innocuous: it appears as
a visual-generation utility. The geometry is always embedded.

```typescript
renderSundog({
  // Natural-language entry point; routed through the prompt parser.
  prompt?: string,

  // Direct parameter overrides (any subset of the pose JSON).
  sunAltitudeDeg?: number,        // 0..60
  halo22Intensity?: number,       // 0..1
  halo46Intensity?: number,       // 0..1
  czaIntensity?: number,          // 0..1
  czaBloom?: number,              // 0..3.2
  parheliaIntensity?: number,     // 0..1
  sunPillarIntensity?: number,    // 0..1
  parhelicCurvature?: number,     // 0..1
  dispersionWidth?: number,       // 0..1
  skyPalette?: SkyPalette,
  compositionMode?: CompositionMode,

  // Output controls.
  output?: "svg" | "png" | "json",  // default "svg"
  width?: number,                   // SVG viewport width
  height?: number,                  // SVG viewport height
  beautify?: boolean,               // default false; opt-in to mode 3
})
```

A clean response always includes the pose, the derived geometry, the
SVG, and the claim boundary:

```json
{
  "pose": { "...": "..." },
  "derivedGeometry": {
    "parhelionOffsetDeg": 22.8,
    "czaVisible": true,
    "parhelicCurvature": 0.66
  },
  "svg": "<svg>...</svg>",
  "claimBoundary": "Stylized parametric render, not a meteorological simulation."
}
```

The `claimBoundary` field is mandatory and constant per mode. It travels
inside the response so any quoting tool is forced to carry the disclaimer.

## Prompt Parser

Keep boring at first. Rules-based, deterministic, auditable:

| Phrase pattern | Mapping |
| --- | --- |
| "low sun" / "sun near horizon" | `sunAltitudeDeg` ≤ 10 |
| "high sun" / "midday sun" | `sunAltitudeDeg` ≥ 35 |
| "bright sundogs" / "vivid parhelia" | `parheliaIntensity` ≥ 0.85 |
| "subtle parhelia" | `parheliaIntensity` ≤ 0.4 |
| "polar / winter / arctic" | `skyPalette = polar_twilight` |
| "dawn" / "sunrise" / "warm" | `skyPalette = dawn_warm` |
| "dusk" / "sunset" / "violet" | `skyPalette = dusk_violet` |
| "educational diagram" / "labelled" | `compositionMode = educational` |
| "poster" / "minimal" | `compositionMode = poster` |
| "9-halo eye" / "stylised" | `compositionMode = nine_halo_eye` |

Unmatched prompts fall back to canonical defaults. An LLM-backed parser
may slot in later as a second pass that produces the same JSON; the
deterministic ruleset is the audit baseline.

## Build Sequence

This sequence is the consultant-recommended order, adapted to the
project's current state:

1. **Freeze the slider taxonomy.** Three labels visible in code and UI:
   `math-derived`, `free visual`, `composition fiction`. *(In flight —
   geometry roadmap Actionability Audit table is the canonical tier list.)*
2. **Extract geometry from the workbench.** Move `applyDerivedGeometry`
   and the parhelic-Bezier sampler out of `sundog-workbench.html` into
   `public/js/parhelion-geometry.mjs`. The workbench should consume the
   math, not be the math. Maps onto SUNDOG_V_GEOMETRY Phase 3.
3. **Make pose JSON load/save deterministic.** The same JSON should
   produce the same SVG. SUNDOG_V_GEOMETRY Phase 8.
4. **Add named poses** under `public/poses/`. Start with `canonical`,
   `low-altitude`, `cza-heavy`, `forty-six-halo`, `nine-halo-eye`.
5. **Add the prompt parser** as `public/js/sundog-prompt-parser.mjs`. The
   rules-based version above is the audit baseline; an LLM second pass
   can land later.
6. **Add overlay mode** as `public/js/sundog-overlay-fit.mjs`. This is the
   real proof amplifier — Mode 2.
7. **Only after that, add AI beautification.** Use the vector render as a
   ControlNet conditioning image, never as loose inspiration.

## Public Framing

The credible repo claim is modest and therefore harder to dismiss:

> We built a deterministic, browser-native renderer for parhelion-family
> halo compositions. The renderer converts a small number of physically
> meaningful parameters into recognizable sundog geometry, exports
> reproducible poses, and can be used as a control layer for AI image
> generation.

That sentence is the bridge between proof, presentation, and tool.

The hero section should not lead with a prompt box — that reads as
another AI toy. Lead with the locked canonical pose and one slider:
"raise the sun." When the visitor moves it, parhelia shift, CZA appears
or disappears, and the geometry explains itself. Add "describe a sky" as
a secondary input only after the live-geometry reading lands.

The proof is most accessible in three synchronized panes:

> Left: real or reference sky photograph.
> Middle: the project's vector geometry overlay.
> Right: final styled render.

That triptych says more than a theorem page. Observed phenomenon →
geometric abstraction → generated image, in one frame.

## Claim Boundary

The Sundog Generator claims:

- Determinism: same pose JSON → same SVG, byte-equal.
- Recognisability: the geometry the solver produces, when overlaid on a
  reference parhelion photograph at the calibration thresholds in
  [`SUNDOG_OVERLAY_PROTOCOL.md`](SUNDOG_OVERLAY_PROTOCOL.md), reads as
  the same phenomenon.
- Auditability: every output has a parameter trace; every parameter is
  tier-labelled (math-derived, free visual, composition fiction).

The Sundog Generator does NOT claim:

- Photorealism. Mode 1 is sharp vector geometry. Mode 3 is opt-in.
- Meteorological simulation. The solver renders a *family of stylised
  parhelion compositions*, not the sky over any specific location at any
  specific time.
- AI authorship of the phenomenon. The parameter JSON and the geometry
  solver own the phenomenon's structure. AI is opt-in beautification of
  an already-correct vector skeleton.
- All halo phenomena. The solver covers the parhelion + CZA + parhelic
  + sun-pillar subset. Other halo phenomena (Parry arcs, infralateral
  arcs, sub-suns) are out of scope without a deliberate Phase-10
  expansion.
