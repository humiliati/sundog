# H4 Result — grokking as a Whitney fold+cusp (the calibrated jet classifier pointed at it)

> **Run against the LOCKED pre-registration** (`H4_GROKKING_CATASTROPHE_PREREG.md`). Hypothesis #4 of
> slate `ww6koomb1`. NOT public-eligible. Attribution: Whitney 1955; Thom/Zeeman (cusp catastrophe);
> Power et al. 2022 (grokking); the Atlas jet classifier.

## Headline verdict

**CLEAN NULL on the cusp hypothesis + a TOOL-VALIDATION WIN.**
- **Stage 1 (positive):** the Atlas's calibrated jet classifier provably reads the canonical **Whitney
  fold+cusp** as an A₃ cusp (`|c3|=6.000` exactly, corank-1), cleanly separated from the A₄ swallowtail
  (`|c3|` collapses 6.18→1.39 as h→0) and the D₄ umbilic (corank-2, s1_min_rel=0.0036). The synthetic-
  control ladder **A₂/A₃/A₄/D₄** is now closed.
- **Stage 2 (null):** grokking's memorize↔generalize transition over a (train-fraction × weight-decay)
  control plane is **NOT a Whitney bistable cusp** at this scale. The transition IS **init-dependent
  multistable** — but with a **continuum of stable partial-generalization plateaus** (inits settle at
  0.50/0.53/0.66/0.96, flat over 40k steps), **NOT the two discrete basins** a cusp requires. So it fails
  the cusp test for the opposite-of-trivial reason: too MANY stable levels, not too few. No clean cusp in
  the chart; the apparent bistability of the raw sweep was a seed=split confound.

## Stage 1 — the classifier reads a Whitney cusp (`scripts/grokking_catastrophe.py`)

| chart | classifier reads | gate |
|---|---|---|
| Whitney cusp `F(x,y)=(x, y³+xy)` | **1 cusp, \|c3\|=6.000, corank-1** (A₃), stable across translations | PASS |
| A₄ swallowtail `(x, y⁴+hy²+xy)` | \|c3\| collapses 6.18→1.39 as h→0 (A₄) | separated |
| D₄ hyperbolic umbilic | corank-2 (s1_min_rel=0.0036) | separated |

## Stage 2 — the grokking sweep + analysis

Modular addition mod 23, tiny embed-MLP, AdamW, 25k steps; grid frac∈[0.40,0.66]×7 × wd∈[0.5,4.0]×8 ×
**3 seeds** (`scripts/grokking_catastrophe_sweep.py` → `results/atlas/h4/grok_sweep.json`).

### Order parameter z* = mean test accuracy (rows=frac, cols=wd) — a smooth grok-tongue
```
frac\wd  0.50  0.67  0.91  1.22  1.64  2.21  2.97  4.00
0.400    0.00  0.00  0.00  0.01  0.01  0.01  0.01  0.01
0.487    0.03  0.09  0.11  0.31  0.25  0.22  0.25  0.14
0.530    0.27  0.39  0.59  0.75  0.71  0.58  0.51  0.45
0.573    0.60  0.85  0.94  0.87  0.88  0.79  0.71  0.46
0.660    0.99  1.00  0.99  0.99  0.99  0.99  0.96  0.81
```
A smooth ramp in frac, with a wd window (peak ~1.2, high wd=4.0 underperforms). NOT an obviously folded
surface.

### Three independent reasons the cusp hypothesis fails

1. **No genuine bistability — the decoupled probe (`grokking_bistability_probe.py`) is decisive.** The
   sweep confounded the data split with the init (one `seed` set both); its apparent "bistability" was
   **one split (seed 1) grokking everywhere = a split-QUALITY effect**, not init-dependence (tally
   `{seed 1: 5}` across all partial cells). Decoupling them (fix split, vary init, K=8) at the transition:
   the spread over inits is a **smooth continuum, not two basins** — e.g. frac=0.53 split=0:
   `[0.26,0.33,0.35,0.37,0.39,0.44,0.49,0.52]`. Only **1 of 18** (frac,wd,split) groups trips the strict
   bimodal flag, and that one is also a continuous ramp. A cusp REQUIRES two discrete stable states.

2. **The stable states are a CONTINUUM, not two basins (the 80k-step tiebreaker).** At frac=0.53, 34/48
   of the 25k-step runs sit in the mid-range 0.3–0.7. Trained to **80k steps** (fixed split=2, varied
   init), those inits do NOT all converge to one attractor — they settle at **distinct, stable plateaus**:
   `init0→0.66, init1→0.96, init2→0.53, init3→0.50` (flat over 40k–79k steps). So the transition is
   genuinely **init-dependent multistable**, but with **≥4 distinct stable levels over 4 inits, none at
   the memorize basin (≈0)** — a continuum of partial-generalization plateaus, NOT the two discrete stable
   states (memorize / generalize) a Whitney cusp predicts. A cusp's cubic potential has exactly two stable
   roots; a continuum of plateaus is a glassy/rough landscape, not a cusp.

3. **The jet classifier finds a fold but no clean cusp.** On the control→(test-acc, −log-wnorm) chart
   (`grokking_catastrophe_analyze.py`, interpolated 160×160): a caustic (det DF sign-change) IS present —
   the transition is a fold-like ridge — but the "cusps" have `|c3|≈29000` (vs the Stage-1 real cusp's
   6.000), i.e. **interpolation noise** from blowing up a coarse 7×8 seed-averaged surface, NOT a genuine
   A₃. corank-1 (no D₄).

### Pre-registered verdict
- **CUSP (A₃):** ✗ — no two-basin bistability (a cusp needs exactly two stable states; the transition is
  init-multistable with a *continuum* of stable plateaus), no clean cusp in the chart.
- **FOLD (A₂):** ✗ — the mean order parameter has a sharp ramp, but it is not the multivalued fold of a
  catastrophe.
- **NULL (not a catastrophe):** ✓ — **the confirmed result**: grokking here is an **init-dependent
  multistable / glassy** transition with a continuum of stable partial-generalization plateaus — neither
  a two-basin Whitney cusp nor a clean single-attractor crossover. NOT the catastrophe the hypothesis
  conjectured.

## Confounds caught (the methodological value)
- **seed = split + init** in the sweep → apparent bistability was split-quality; fixed by the decoupled
  probe.
- **interpolation noise** in the chart classifier (coarse grid → spurious high-|c3| cusps); the cusp call
  must rest on bistability topology, not interpolated |c3|.
- **25k-step snapshot ambiguity** → at 25k the intermediate values *could* be transient; the 80k-step
  tiebreaker resolved it — they are STABLE init-dependent plateaus (a continuum), which is what kills the
  two-basin cusp reading (and corrects an initial "delayed-transient" mis-read).

## Incidental finding (not the headline, tested thin)
The 80k-step tiebreaker surfaced **init-dependent multistability**: at a fixed transition cell, different
inits converge to **distinct stable partial-generalization plateaus** (0.50 / 0.53 / 0.66 / 0.96). That is
a *glassy/rough-landscape* signature — interesting, and genuinely different from both a clean cusp and a
single attractor — but it is **NOT** a two-basin cusp, and it is tested with only **4 inits × 80k steps**,
so it is a flagged observation, not a claim.

## Honest boundaries
- Small scale (p=23, tiny model, 3 seeds, coarse grid; tiebreaker 4 inits). A cusp could in principle
  appear at other scales/architectures; this is a null **at this scale/budget**, not a universal claim.
- This tests the *steady-state* bistability (catastrophe) reading of grokking; it does not deny grokking's
  well-known *dynamical* delayed-generalization. Forward-only; the cusp call rests on bistability topology
  (continuum vs two basins), not on interpolated |c3|.

## Files
- `scripts/grokking_catastrophe.py` — Stage 1 (Whitney-cusp validation; the tool win).
- `scripts/grokking_catastrophe_sweep.py` → `results/atlas/h4/grok_sweep.json` — the (frac×wd×seed) sweep.
- `scripts/grokking_bistability_probe.py` → `results/atlas/h4/grok_bistability.json` — the decoupled
  (fixed-split, varied-init) bistability probe (the decider).
- `scripts/grokking_catastrophe_analyze.py` — surface, grok-fraction map, chart + classifier.
- `docs/atlas/H4_GROKKING_CATASTROPHE_PREREG.md` — the locked pre-registration.
