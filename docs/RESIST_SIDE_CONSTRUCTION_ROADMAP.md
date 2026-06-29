# Resist-Side Construction - Roadmap (PROPOSED / pre-registration)

- Status: **PROPOSED 2026-06-28; Pass 0 RUN 2026-06-28 - exit gate MET.** A
  pre-registration of a falsifiable research program. NOT an owner-ratified plan;
  Passes 1-4 not yet run. Each pass freezes its prediction / falsifier / receipt
  before execution.
- Motivation: Ghost lane **H3** (internal, unratified) -
  `internal/slates/GHOST_HYP_INTERNAL_2026-06-28.md`; the recoverability axis in
  [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md) sections 6.3 + 10;
  [`SUNDOG_V_GHOST.md`](SUNDOG_V_GHOST.md) Q2.
- Lead testbed: [`SUNDOG_V_NAVIERSTOKES.md`](SUNDOG_V_NAVIERSTOKES.md) (the
  strongest marginal / the review-packet asset); second testbed Mesa `net.7`.

> Thesis under test: the regime-2 generality nulls came in **marginal because the
> bodies were recoverable**, not because they were under-resourced. Dialing
> **constructed resistance** (not dimension/compute) should sharpen the split -
> and the portfolio already has one cheap constructed sharp split (the syndrome
> certificate) to anchor the ladder.

## 0. Why this exists (the H3 reframe, fenced)

The 2026-06-04 regroup filed the body-resistance strikeout as needing "a body
high-dimensional by construction, **unreachable at accessible resources**" - a
compute / dimension-bound waiting game. H3 challenges the axis: the marginal
substrates are marginal because **recoverable** (NSE C1: low-Fourier shadow
reconstructs the body at FVE ~ 0.99; Mesa `net.7` FVE ~ 0.97-0.99), and the
portfolio's own **syndrome / SIS certificate** is a sharp regime-2 separation
reached **cheaply by construction** (secret lost algebraically, deviation lost
computationally), not by scale. So resistance is constructible; this program tests
that constructively and tries to convert marginals to sharp, measured splits.

## Claim Boundary

This program does **not** claim:

- that natural 2D Navier-Stokes (or any natural marginal substrate) secretly
  resists its shadow - those are recoverable, and that stays a known boundary;
- that compute is useless (it is genuinely owed for *measuring* a hard substrate -
  the NSE C2 / Sabra tooling wall - just not for *manufacturing* resistance);
- a new complexity-theoretic or physics result;
- that constructed resistance is *proved* - it is **imported** (syndrome decoding
  hardness; Aharonov-Bohm topology; cap-set dimension).

It **does** stage: a graded construction ladder, measured thresholds, and an
honest test of whether resistance (not scale) is the regime-2 axis.

## The measurable

For each task family, at **fixed compute and dimension**, measure two gaps:

- **state-reconstruction gap** - how poorly the shadow reconstructs the body
  (`1 - FVE`, or the analogous decoding-failure rate);
- **control-sufficiency gap** - whether a controller using only the shadow still
  achieves the task.

**Regime-2 sharp** = state-insufficient (body NOT recoverable from the shadow) yet
control-sufficient. **Sharpness** = the separation between the two gaps. The
load-bearing control: sharpness is plotted against a single **resistance dial rho**
with compute and dimension **held fixed**. The whole program turns on resistance
moving the needle while compute does not.

## Passes (pre-registered)

### Pass 0 - Recoverability audit (cheap, no construction)

Place every relevant substrate on ONE recoverability scale: the marginals (NSE C1,
Mesa `net.7`) and the existing wins (cap-set = dimensional, Aharonov-Bohm =
topological, syndrome = computational). Measure shadow->body reconstruction for each.

- Exit gate: marginals land **recoverable** (high FVE), wins land **resistant** -
  the H3 receipt formalized on a single axis.
- Falsifier: a marginal that is actually non-recoverable (low FVE) yet still gave a
  soft split => recoverability is not the axis; reframe or close.
- Cost: ~zero (re-reads existing receipts + one reconstruction metric).

#### Pass 0 - RESULT (run 2026-06-28): exit gate MET

Recoverability axis = "does the shadow determine the body?", read as a
shadow->body reconstruction figure (FVE for the continuous substrates; the
lossiness mechanism + magnitude for the constructed wins). The metric is native
per substrate; the axis is the common ordinal.

| substrate | role | shadow->body recoverability | mechanism | source |
| --- | --- | --- | --- | --- |
| NSE C1 (2D Kolmogorov) | marginal | **FVE ~0.99** (both physical norms) | low-Fourier shadow reconstructs the field | NSE ledger / failure map |
| Mesa `net.7` (5D) | marginal | **FVE ~0.97-0.99** | shadow reconstructs the hidden subspace | P-vs-NP entry / Mesa |
| Syndrome / SIS (computational) | win | **chance** (0.513 measured, 0.5 exact) | `z = eH^T` is independent of `s`; 2^64 preimages/syndrome; decoding NP-hard | Lean lossiness + Pass-0 anchor `scripts/resist-pass0-anchor.mjs` |
| Aharonov-Bohm (topological) | win | **0, exact** | flux `Phi=oint A` is one int per `H^1` class; interior `B(x)` not rebuildable; many bodies -> one shadow | CSN section 8.2 (B7-topology receipt) |
| chatv2 `d_dec` 7->14 (dimensional) | win | resistant, at-the-bar SHARP (toy) | de-confounded high-dim body resists + scales; contrast 0.205+/-0.022, 3 seeds | CSN section 8.2 / chatv2 |
| (NSE C2 / Sabra) | - | **unmeasured** (tooling wall) | shadow unmeasurable with current integrator | NSE ledger |

**Exit gate MET.** The two marginals land at the recoverable end (FVE 0.97-0.99);
all three constructed wins land at the resistant end (chance / exact-lossy). Clean
separation on one axis. The Pass-0 anchor makes the resistant end concrete on the
reconstruction-accuracy version of the metric: the same secret reads ~0.94 from
the body but ~chance (0.513; exactly 0.5 in theory) from the lossy shadow - the
same axis where NSE/Mesa read 0.99.

**Falsifier did NOT fire.** No marginal is non-recoverable: both gave soft splits
*and* are high-FVE, consistent with recoverability being the axis.

**The H3 sharpener, confirmed (load-bearing).** Recoverability is **orthogonal to
dimension/compute**: Aharonov-Bohm resists *exactly* with a **tiny** body (one
integer per `H^1` generator); the syndrome win resists at small/cheap scale; the
dimensional win is `d_dec < 20` (toy). Conversely NSE C1 is a PDE field
(effectively high-dimensional) yet **recoverable**. So resistance is neither bought
with scale nor needs it - directly against the "high-dimensional body, unreachable
at accessible resources" diagnosis. Pass 0 does not *prove* H3, but it removes its
cheapest disconfirmation and clears the path to Pass 1.

**Honest caveats.** The axis is ordinal across incommensurable native metrics (FVE
vs algebraic/topological lossiness); the anchor computation only operationalizes
the resistant end. Sabra stays an unmeasured tooling wall, not a recoverability
data point. Constructed-win resistances are imported (decoding hardness; AB
topology).

### Pass 1 - Minimal computational-resistance injection (core)

Build a task family = recoverable shadow + a hidden body channel constructed to
resist on the **computational** axis (syndrome / SIS-style; cheapest, already
Lean-backed for soundness/lossiness). Resistance dial `rho` = secret entropy / code
parameters. Hold compute and dimension fixed; sweep `rho`; measure sharpness(`rho`).

- Pre-registered prediction: sharpness rises with `rho`, with a measured threshold
  `rho*` where the split goes sharp - and does **not** rise when compute is raised
  at fixed `rho`.
- Receipt: the measured threshold `rho*` (the syndrome cert's "measured capacity
  threshold" pattern, now as a sweep).
- Falsifier: sharpness flat in `rho`, or only rises with compute/dimension =>
  **H3 dies**, program closes with a recorded null (the original compute-bound
  diagnosis stands).

### Pass 2 - Axis generalization

Repeat the `rho`-sweep on the **topological** (Aharonov-Bohm-style) and
**dimensional** (cap-set-style) flavors.

- Exit gate: >= 2 of 3 axes give a sharp split at fixed compute => resistance (any
  genuine flavor), not compute, is the axis.
- Falsifier: only one specific construction works => it is a trick, not the axis;
  downgrade the claim accordingly.

### Pass 3 - "Recover a marginal" (NSE lead testbed)

Take NSE C1's control question (or a faithful reduced surrogate) and inject the
minimal `rho >= rho*` from Pass 1 as a hidden resistant channel coupled to the
control task, **without raising resolution or compute**. Measure whether the
previously-marginal split becomes sharp. Then repeat on Mesa `net.7`.

- **Honest fence:** this builds a **hybrid** body (the natural recoverable shadow +
  a constructed resistant channel). It answers "can a resistant body on this
  substrate's shadow give a sharp split," NOT "the natural substrate secretly
  resists" - the natural recoverability from Pass 0 is preserved as a stated
  boundary.
- Exit gate: a sharp, measured split on the hybrid at fixed compute = the marginal
  "recovered" in the constructive sense.
- Receipt: the NSE-hybrid threshold + a one-page note (clearly labelled
  *constructed*) for the review packet.

### Pass 4 (optional, must NOT gate) - Natural-resistance search

Cheap pre-checks only: recoverability-audit candidate natural regimes for genuine
non-recoverability - explicitly NOT the compute-bound Sabra measurement wall.

- Exit: either flag a natural resistant candidate for a later (compute-gated)
  measurement, or record that natural resistance stays unreached and **constructed
  resistance is the demonstrated path**. Either outcome is informative.

## Pre-registration discipline

- Freeze each pass's prediction, falsifier, and receipt before running.
- **Compute and dimension held fixed across every `rho`-sweep** - this is the
  control that makes the result mean "resistance, not scale."
- Privileged-truth anchors: NSE C1 FVE ~ 0.99 (known); syndrome `2^64` exact
  algebraic lossiness (Lean, axiom-clean); AB unconditional topological lossiness;
  cap-set dimensional bound.
- Imported walls named where used (syndrome decoding hardness; AB topology).
- Not a complexity or physics claim; constructed resistance imported; the H3
  motivation is internal and unratified.

## Whole-program exit / kill

- **WIN:** >= 2 axes show sharpness rising with `rho` at fixed compute, and the NSE
  hybrid (Pass 3) converts marginal -> sharp. Reading: the strikeout was
  recoverability-bound, not compute-bound; constructed resistance recovers the
  thesis cheaply.
- **KILL:** sharpness tracks compute/dimension and not `rho`. Reading: H3 is false,
  the original compute-bound diagnosis stands; record the null and close.

Either way the program is cheap (constructed resistance, fixed compute) and
honest (a clean null is a publishable outcome, per the lane's own discipline).

## Backlinks

- Motivation / receipt: `internal/slates/GHOST_HYP_INTERNAL_2026-06-28.md` (H3);
  [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md) sections 6.3, 10 (the
  recoverability axis + the three constructed resistance flavors).
- Lead testbed + privileged truth: [`SUNDOG_V_NAVIERSTOKES.md`](SUNDOG_V_NAVIERSTOKES.md)
  (C1 FVE; the Sabra tooling-wall boundary).
- The constructed-resistance anchor: the P-vs-NP syndrome/SIS certificate and its
  Lean cores (`sundogcert`).
