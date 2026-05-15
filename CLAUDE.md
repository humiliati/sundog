# Sundog — Claude Code Guide

The canonical agent guide for this repository is **[AGENTS.md](AGENTS.md)**.
Read it first — it is the single source of truth and is kept up to date;
this file is a pointer so the two do not drift.

AGENTS.md covers:

- Website tooling, build, and Cloudflare Pages deploy (`npm run dev` /
  `build` / `deploy`, scoped Cloudflare tokens, publishing shape).
- **HaloSim halo-rendering toolset** — cinematic + geometry-confirmation
  rendering. No CLI; the proven HS-0 zero-click mechanism
  (Startup.sim swap → Reset/Start → poll `autosave.bmp`) and the
  **ray-count guidance** (B&W ~300k or colour ~3M for geometry confirm;
  ~10M colour for spectacular thumbnail / logo candidates) are in
  AGENTS.md ▸ "HaloSim Halo Rendering".

## Key docs under `docs/`

- `docs/SUNDOG_V_GEOMETRY.md` — the geometry-workbench roadmap. It has a
  **Document Map at the top** (line-numbered section index) — use it to
  navigate; the file is long.
- `docs/SUNDOG_HALOSIM_CINEMATIC_SIDECAR.md` — the HaloSim HS-0…HS-7
  generative pipeline. **HS-0 is proven**; HS-1…HS-7 are unblocked.
- `docs/calibration/HALOSIM_VALIDATION_PROTOCOL.md` — the
  validation-direction HaloSim procedure (distinct from the generative
  sidecar; do not conflate the two).

## Conventions

- `npm run sundog:check` validates the public page contract + Phase 6
  drag constraints; run it after touching `sundog.html` or the geometry
  module.
- Only `dist/` is published — never deploy the repo root.
- Roadmap line-number indexes are dated snapshots; if a number looks
  stale, search the heading text instead.
