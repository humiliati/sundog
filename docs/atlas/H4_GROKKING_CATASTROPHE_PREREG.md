# H4 pre-registration — grokking as a Whitney fold+cusp (point the calibrated jet classifier at it)

> **LOCKED 2026-06-08, before analyzing the sweep.** Hypothesis #4 of slate `ww6koomb1`. Tests whether
> grokking's memorize↔generalize transition, over a 2-control plane (train-fraction × weight-decay), is a
> **Whitney fold+cusp catastrophe** — and points the Atlas's CALIBRATED jet classifier
> (`atlas_jet_classify.py`) at the resulting chart. NOT public-eligible. Attribution: Whitney 1955;
> Thom/Zeeman (cusp catastrophe); Power et al. 2022 (grokking); the Atlas jet classifier.

## Stage 1 (DONE) — the classifier reads a Whitney cusp
`scripts/grokking_catastrophe.py`: the canonical Whitney cusp map `F(x,y)=(x, y³+xy)` is read as **A₃**
(exactly one cusp, **|c3|=6.000** bounded, corank-1), STABLE across translations, distinct from the A₄
swallowtail (|c3| collapses 6.18→1.39 as h→0) and the D₄ umbilic (corank-2). The A₂/A₃/A₄/D₄
synthetic-control ladder is closed; the cusp template is established.

## The hypothesis (Stage 2)
The cusp catastrophe is the canonical 2-control model of a transition that is **continuous** on one side
of a point and **discontinuous (a jump, with a bistable/hysteretic wedge)** on the other. Grokking's
generalization order parameter `z` (0=memorize, 1=generalize), as a function of train-fraction (the
"normal"/threshold factor) × weight-decay (the "splitting"/sharpness factor — the pilot scan showed a wd
WINDOW where grokking occurs, opening as frac rises), is conjectured to be the equilibrium set of an
effective potential with a **cusp** bifurcation set.

## The measurement (`grokking_catastrophe_sweep.py`)
Modular addition mod p=23, tiny embed-MLP, AdamW, 25000 steps. Grid: frac∈[0.40,0.66]×7, wd∈[0.5,4.0]×8
(geom), **K=3 seeds/cell** (the seed ensemble exposes bistability as bimodality). Per seed: final test
acc `z`, weight norm, grok step. (Pilot: groks at frac≈0.6, wd≈1–2 @ ~15k steps; fails below frac≈0.5
and outside the wd window — a real 2-control boundary.)

## Why a seed ensemble (the load-bearing design choice)
A catastrophe (fold/cusp) lives in a **multivalued** equilibrium surface: `z*` must take TWO stable
values over a **bistable region**. A single-init sweep gives a single-valued GRAPH — no fold for the
classifier to read. The seed ensemble exposes bistability as **bimodality**: in a truly bistable wedge,
random seeds fall into different basins (some grok, some don't); in a smooth crossover they cluster. The
bimodal region is the cusp's wedge. (Caveat pre-committed: seed-bimodality is a *proxy* for the
deterministic init-dependence/hysteresis of a true cusp — defensible but not identical.)

## The chart + classifier (data-adaptive construction, LOCKED verdict criteria)
- **Bistability map** `g(frac,wd)` = fraction of seeds that grok (`z>0.5`). Bistable region = `{0<g<1}`.
- **Chart for the classifier:** the equilibrium catastrophe map reconstructed from the (multivalued)
  surface — the bistable branches (memorize `z≈0`, generalize `z≈1`) define the fold lines; the chart
  `χ(z, wd) = (frac*(z,wd), wd)` projects the equilibrium manifold to the control plane. Point
  `jet_from_chart`/`cusp_c3`/`corank_from_chart` at it.
- If there is **no** bistable region, the equilibrium chart is single-valued (no fold) → there is no
  catastrophe to classify (a NULL, see below).

## Pre-registered verdict criteria (LOCKED)
- **CUSP (A₃) — the hypothesis confirmed:** a bistable region exists AND is **wedge-shaped** (its width
  narrows toward a point as wd decreases), AND the reconstructed equilibrium chart classifies as a cusp
  (a cusp present, `|c3|` bounded away from 0, corank-1 — matched to the Stage-1 template), AND the
  cusp-normal-form fit to the surface beats a no-catastrophe (smooth sigmoid) fit.
- **FOLD (A₂) — partial:** a sharp/bistable boundary exists but is a **uniform strip** (not a wedge, no
  cusp point in range); the chart classifies as a fold without a cusp.
- **NULL (smooth crossover) — a clean, legitimate result:** `z*(frac,wd)` is single-valued and smooth
  (no bimodality, `g∈{0,1}` essentially everywhere or a monotone ramp), i.e. grokking here is a
  **delayed crossover** (a dynamical phenomenon), not a steady-state bistable catastrophe.

## Kill criteria
- KILL the **cusp** claim if: no bimodality anywhere (no bistability → no fold → no cusp); OR the
  bistable region is a uniform strip (fold, not cusp); OR the reconstructed chart classifies as fold-only
  / `|c3|`→0 (higher) / corank-2; OR the cusp fit does not beat the smooth fit.
- A NULL is **not** a failure — it is the honest finding that the cusp framing of grokking does not hold
  at this scale (and would itself be a bounded, citable result).

## Honest boundaries (pre-committed)
- Grokking is fundamentally a **training-dynamics** phenomenon; whether its long-time *steady state* is a
  bistable catastrophe (vs a delayed-but-single-valued crossover) is exactly the open question — a NULL
  is a real possibility and a real result.
- Small scale (p=23, tiny model, 3 seeds, coarse grid); the surface is interpolated for the jet
  computation (the classifier was built for ng~320 charts), so the cusp-vs-fold call rests on the
  **topology** (bistable wedge + cusp presence), not the absolute `|c3|` magnitude.
- Forward-only; no inversion. Bands not points (report the seed spread).

## Files
- `scripts/grokking_catastrophe.py` — Stage 1 (cusp template, DONE).
- `scripts/grokking_catastrophe_sweep.py` → `results/atlas/h4/grok_sweep.json` — the measurement.
- `scripts/grokking_catastrophe_analyze.py` (to write) — bistability map, chart, classifier, verdict.
- `docs/atlas/H4_GROKKING_CATASTROPHE_RESULT.md` — the receipt (after, against this pre-reg).
