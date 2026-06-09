# H7 — curved-Earth mirage ray-tracer (deepening the H5 mirage ladder) + product assessment

> **2026-06-08.** Deepening of H5 (the flat-Earth mirage (Δn,s) ladder). RESEARCH MODEL + a deferred
> product/SEO/workbench assessment (per the session decision: build + validate the physics now, defer the
> public page/workbench). The research model is internal; a *public* mirage explainer/interactive would be
> a separate, public-eligible artifact (standard textbook optics — distinct from the frozen H5 cusp-ladder).
> Attribution: standard atmospheric refraction / mirage theory (Lehn; Können; Bruton; A. Young's mirage pages).

## The model (`scripts/mirage_curved.py`, frozen test 7/7)

Over a spherical Earth the surface curves away from a straight ray, adding the curvature term `1/R_E` in
the height-above-surface frame: `dh/dx = ψ`, `dψ/dx = 1/R_E + n'(h)` (rays curve toward higher `n`; `h<0`
= ray hit the Earth). One extra term over the H5 flat tracer — and it unlocks the curved-Earth physics:

| validation | result |
|---|---|
| **k-factor** (effective Earth radius) `k = 1/(1+R_E·dn/dh)` | **1.333** (textbook 4/3; R_eff = 8492 km) ✓ |
| **horizon** extended by refraction (~√k) | traced ratio 1.149 ≈ √k 1.155 ✓ |
| **DUCTING threshold** `dn/dh = −1/R_E` (= −1.57e-7/m) | at threshold a horizontal ray stays **level** (50 m over 300 km) → reaches **12× the geometric horizon** = **Novaya-Zemlya**; sub-critical rises, super-critical dives ✓ |
| **superior mirage / looming** (elevated duct) | a moderate elevated inversion TRAPS rays (600/600 stay <600 m over 300 km vs 0/600 in standard air) = looming / Fata Morgana ✓ |
| **inferior mirage** (hot ground) | transfer folds at short range — curvature-insensitive, matches the flat H5 model ✓ |

**The physical heart:** the single dimensionless number `R_E·dn/dh` governs everything — `k≈4/3` standard,
and the **ducting threshold at `dn/dh = −1/R_E`** (a temperature inversion of ~0.11 K/m) is exactly where
ray curvature matches Earth curvature, so light follows the surface and the geometric horizon dissolves
(Novaya-Zemlya, looming, Fata Morgana, the green flash). The H5 catastrophe ladder (1→3→5 images) now
lives over a *real horizon*, with ducting as the curved-Earth regime the flat model couldn't express.

## Product / SEO / workbench opportunity — ASSESSMENT (build deferred)

**Verdict: a strong, low-risk product opportunity — recommend building it as a follow-up.**

- **Public-eligibility: clean.** Mirages/refraction/ducting/green-flash/Fata-Morgana are standard
  textbook atmospheric optics — **no proprietary lab IP** (unlike the internal H5 cusp-ladder, which
  stays frozen). Same public tier as the existing halo-geometry content (`geometry.html`, `sundog.html`,
  `sundog-workbench.html`). A public mirage page/interactive would expose *only* standard physics.
- **SEO fit: excellent.** Atmospheric optics is evergreen, visual, and searchable; high-interest terms
  ("green flash", "Fata Morgana", "Novaya Zemlya effect", "why is the horizon farther than it looks",
  "desert/road mirage", "inferior vs superior mirage") are under-served by *interactive* content. It is a
  natural sibling to the site's existing halo theme (halos → mirages), extending topical authority.
- **Workbench fit: direct.** This H7 ray-tracer is the engine for an interactive **"temperature profile →
  mirage" simulator**: the user sets a profile (or picks a preset — hot road / cold sea / strong
  inversion), and sees the ray-traced transfer curve + a rendered mirage (inverted / multiple / looming /
  ducted-over-the-horizon). It slots beside the existing `parhelion-geometry.mjs` workbench.
- **Build sketch (deferred, ~moderate):** port `mirage_curved.py` → a `mirage.mjs` module (the RK4 tracer
  is ~30 lines); a `<canvas>` renderer (ray paths + the synthesized image, like a horizon-camera view);
  3–5 preset profiles; an SEO explainer page (`mirages.html` / `atmospheric-optics.html`) wired into the
  sitemap. The physics is done + validated; remaining work is the JS port, the UI, and the explainer copy.
- **Honest caveats:** the tracer is paraxial / 1-D horizontally-stratified (right for an explainer +
  interactive, not a research-grade full-atmosphere model); SEO payoff scales with overall site
  authority; the build is real front-end work, not free.

## Files
- `scripts/mirage_curved.py` (+ `test_mirage_curved.py`, 7/7) — the curved-Earth ray-tracer + validation.
- `docs/atlas/H5_MIRAGE_LADDER_RESULT.md` — the flat-Earth catastrophe ladder this deepens.
- (deferred) `public/js/mirage.mjs` + a `mirages.html` explainer — the public product play.
