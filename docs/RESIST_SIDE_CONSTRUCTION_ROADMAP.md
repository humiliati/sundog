# Resist-Side Construction - Roadmap (PROPOSED / pre-registration)

- Status: **PROPOSED 2026-06-28; Passes 0-4 ALL RUN 2026-06-28 - whole-program WIN
  condition MET.** Pass 0 (gate MET), Pass 1 (prediction HELD), Pass 2 (gate MET,
  3/3 axes), Pass 3 (marginal RECOVERED, constructive), Pass 4 (natural high-dim
  resistance UNREACHED cheaply; located + flagged compute-gated). A pre-registration
  of a falsifiable research program; NOT an owner-ratified plan. Each pass froze its
  prediction / falsifier / receipt before execution.
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

#### Pass 1 - RESULT (run 2026-06-28): prediction HELD

Construction: a random `[20,10]` GF(2) code (min distance `d=4`). Body = error
pattern `e`; shadow = syndrome `z = He`; control objective = any function of the
fully-observed `z` (so control-sufficiency `K = 1` by construction - that IS the
regime-2 point). State-reconstruction = min-weight syndrome decoding of `e` from
`z`. Resistance dial `rho` = error weight `w`. Receipt: `scripts/resist-pass1.mjs`.

`sigma(w) = 1 - R(w)` rises through a sharp transition:

| w | R(w) (body recovered) | sigma |
| --- | --- | --- |
| 0-1 | 1.000 | 0.000 (recoverable / marginal) |
| 2 | 0.915 | 0.085 |
| 3 | 0.497 | 0.503 (transition) |
| 4 | 0.060 | 0.940 |
| 5-6 | 0.000 | 1.000 (resistant / sharp) |

- **Measured threshold `rho* = 3`.** It is the **capacity / information** threshold:
  `C(20,3)=1140 ~ 2^m = 1024` (error patterns start to outnumber syndromes, so the
  body stops being syndrome-determined). The guaranteed-distance radius
  `t = floor((d-1)/2) = 1` is a looser lower bound; the honest measured threshold
  is the capacity one - the same "measured capacity threshold" shape as the P-vs-NP
  syndrome certificate.
- **Compute control (load-bearing).** Below `rho*` (`w=2`): recovery rises
  `0 -> 0.93` with the decoder budget and **saturates** at budget `B=2` - compute
  helps *reach* the threshold. Above `rho*` (`w=4`): recovery stays `<= 0.05` for
  **every** budget up to `n` - compute **cannot cross** it. The threshold is fixed
  by the code's capacity (the resistance parameter), not by search effort.
- **Falsifier did NOT fire**: `sigma` is not flat in `rho`, and the threshold does
  not move out as the budget grows.

**Verdict: Pass 1 prediction HELD.** Constructed resistance gives a graded
sharpening with a measured threshold `rho*` set by the resistance parameter and
invariant to compute - the H3 "resistance, not compute" reframe demonstrated
constructively.

Honest bounds: `K=1` is structural (control = a shadow-function; the regime-2
content is `K=1` while `R` collapses above `rho*`). Toy code (`d=4`, short curve).
The above-threshold resistance is the syndrome *under-determining* the body (an
information wall the min-weight decoder cannot beat at any budget); NP-hard ISD is
the *below*-threshold decoding cost for large codes (here full enumeration is
cheap, so what Pass 1 demonstrates is the **threshold location + compute
invariance**, not a hardness measurement - that is the syndrome cert's separate
ladder).

### Pass 2 - Axis generalization

Repeat the `rho`-sweep on the **topological** (Aharonov-Bohm-style) and
**dimensional** (cap-set-style) flavors.

- Exit gate: >= 2 of 3 axes give a sharp split at fixed compute => resistance (any
  genuine flavor), not compute, is the axis.
- Falsifier: only one specific construction works => it is a trick, not the axis;
  downgrade the claim accordingly.

#### Pass 2 - RESULT (run 2026-06-28): exit gate MET (3/3 axes)

Receipt: `scripts/resist-pass2.mjs`.

**Dimensional axis** (shadow rank `d=8`; dial `rho` = body dim `D`; threshold
`rho* = d`): `FVE(recover) = 1.0` for `D <= 8` (knee at `D = d`), then `sigma`
rises `0.20 (D=10), 0.34 (D=12), 0.50 (D=16=2d, = 1-d/D), 0.66 (D=24)`. The body
becomes resistant once its dimension exceeds the shadow rank; the `(D-d)`-dim
**kernel is independent of the shadow** -> unrecoverable at any compute.

**Topological axis** (loop over `h=5` plaquettes, one hidden hole; dial `rho` =
hole flux): `sigma(non-contractible)` rises `0 -> 0.06 -> 0.20 -> 0.50 -> 0.81 ->
0.94` with `rho`, while `sigma(contractible) = 0` for **every** `rho` (Stokes).
The sharp threshold is the loop's **homotopy class** (contractible = recoverable
always; hole-enclosing = resistant, graded by `rho`); the hole flux is **absent
from the shadow** -> unrecoverable at any compute.

**Exit gate MET: 3/3 axes** (computational [Pass 1] + dimensional + topological)
show `sigma` rising through a threshold set by the axis's resistance parameter
(capacity / shadow rank / homotopy class), at fixed compute, and compute-invariant
by the resistance structure (information collision / kernel independence / Stokes).
**Resistance - any flavor - is the axis, not a one-construction trick.** The
falsifier ("only one specific construction works") did NOT fire.

Honest bounds: the dimensional and topological constructions are more *analytic*
than Pass 1's empirical decoding transition - the dimensional threshold is the
linear-algebra rank fact (`FVE = d/D`), the topological is Stokes' theorem (hole
flux orthogonal to the local shadow). That is the point (resistance is structural),
but they **demonstrate the shared structure** rather than discover new resistance.
`K` is structural across all three (control = a shadow-function). Toy scales; the
three flavors are the portfolio's already-named ones.

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

#### Pass 3 - RESULT (run 2026-06-28): marginal RECOVERED (constructive), exit gate MET

Receipt: `scripts/resist-pass3.mjs`. Hybrid weight `alpha=0.5`; threshold `rho*=3`
inherited from Pass 1; **compute FIXED** across the sweep. The injected error
channel's recovery matches Pass 1 (`R_e`: w0-1=1.0, w2=0.92, w3=0.50, w4=0.06,
w5-6=0.0).

| substrate | natural FVE | sigma_marginal | sigma_hybrid at rho>=rho* | result |
| --- | --- | --- | --- | --- |
| NSE C1 (2D Kolmogorov) | 0.99 | 0.010 | 0.505 | marginal -> sharp |
| Mesa net.7 | 0.97 | 0.030 | 0.515 | marginal -> sharp |

`sigma_hybrid` rises through the inherited threshold (NSE: 0.005 at rho=0 -> 0.047
at rho=2 -> **0.257 at rho=3 -> 0.475 at rho=4** -> 0.505 at rho>=5; Mesa the same
shape). The transition sits at the **resistance parameter** `rho*=3`, not at any
compute increase.

**Exit gate MET:** a sharp, measured regime-2 split on the hybrid at fixed compute,
for both lead testbeds = the marginal "recovered" in the constructive sense.

**FENCE (held).** The hybrid body = the natural recoverable shadow + a constructed
resistant channel. This does **not** claim natural 2D NSE / Mesa secretly resists -
the natural recoverability (FVE 0.99 / 0.97) is preserved as the marginal baseline
(it *is* `sigma_marginal`). `alpha=0.5` is a modeling choice that sets the magnitude
(`sigma_sharp = 1 - alpha*FVE`); the qualitative rise-through-`rho*` is alpha-robust.
The fluid baseline uses the substrate's *measured* FVE as a faithful parameter, not
a re-simulation of NSE.

**Honest standing.** This is the most by-construction pass: Pass 1 carried the
genuine empirical decoding transition, and Pass 3 inherits `rho*` and concatenates
the channel onto the marginal's own recoverability baseline - a constructive
demonstration, exactly as pre-registered. What it adds: the marginal->sharp
conversion is governed by the injected resistance parameter at the inherited
threshold, at **fixed compute**, on the lead testbed's faithful recoverability.

### Pass 4 (optional, must NOT gate) - Natural-resistance search

Cheap pre-checks only: recoverability-audit candidate natural regimes for genuine
non-recoverability - explicitly NOT the compute-bound Sabra measurement wall.

- Exit: either flag a natural resistant candidate for a later (compute-gated)
  measurement, or record that natural resistance stays unreached and **constructed
  resistance is the demonstrated path**. Either outcome is informative.

#### Pass 4 - RESULT (run 2026-06-28): natural HIGH-DIM resistance UNREACHED cheaply; located + flagged

Receipt: `scripts/resist-pass4.mjs` (Kolmogorov mode-counting; no simulation;
shadow rank `K=8` fixed). Tested the obvious natural candidate - high-Reynolds
turbulence - on two axes:

| axis | sweep Re -> | reading |
| --- | --- | --- |
| **energy** recoverability | 3D energy-FVE stays ~0.83-0.92; **2D stays ~0.99** across Re | the low-Fourier shadow keeps the body's ENERGY: "high-Re => natural resistance by energy" is **refuted** (and reproduces why NSE C1 is marginal) |
| **degrees of freedom** | DOF beyond `K` -> 0.99+ with Re | the body outgrows the fixed shadow in COUNT - but those modes are **low-energy** (dissipation / intermittency) |

The two diverge: turbulence has unboundedly many small-scale DOF (resistant by
count) that carry little energy (recoverable by energy). Whether those low-energy
DOF are **control-relevant** (a true regime-2 split) is exactly the C2 / Sabra
intermittency target - the **compute-gated** measurement wall, not a cheap check.

**Verdict (both honest exit branches):**

- Natural **high-dimensional** resistance stays **UNREACHED at cheap assessment**:
  the obvious energy candidate fails; the real candidate (low-energy small-scale
  intermittency) is **flagged for a later compute-gated measurement** (the Sabra
  wall), not claimed.
- One natural **exact** resistance is already in hand - **Aharonov-Bohm**
  (topological) - but it is low-dimensional (one integer per `H^1` generator) and
  does not close the high-dim control frontier.
- **Computational axis:** no clean *natural* example outside constructed
  codes/crypto.

So **constructed resistance is the demonstrated path** (Passes 0-3). Pass 4 does
NOT claim a natural high-stakes body resists; its contribution is to **locate**
where natural resistance must live (low-energy / high-DOF structure) and to confirm
that region is compute-gated, not cheap - sharpening the open §6.3 frontier rather
than closing it.

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

**OUTCOME 2026-06-28: WIN condition MET.** Pass 2 showed 3/3 axes sharpen with
`rho` at fixed compute, and Pass 3 converted both the NSE C1 and Mesa marginals to
sharp at fixed compute. Reading (within the stated fences): the body-resistance
strikeout was **recoverability-bound, not compute-bound** - constructed resistance
recovers the regime-2 thesis cheaply, at the inherited capacity threshold `rho*`,
without scale. Standing fences: constructed/hybrid bodies (not a claim that natural
substrates resist); resistances imported; Pass 3 is by-construction; the whole
program sits off an internal, unratified reframe (H3). Pass 4 (natural-resistance
search) remains optional and would be the only way to move from "constructed
resistance works" toward "a natural high-stakes body resists."

## Backlinks

- Motivation / receipt: `internal/slates/GHOST_HYP_INTERNAL_2026-06-28.md` (H3);
  [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md) sections 6.3, 10 (the
  recoverability axis + the three constructed resistance flavors).
- Lead testbed + privileged truth: [`SUNDOG_V_NAVIERSTOKES.md`](SUNDOG_V_NAVIERSTOKES.md)
  (C1 FVE; the Sabra tooling-wall boundary).
- The constructed-resistance anchor: the P-vs-NP syndrome/SIS certificate and its
  Lean cores (`sundogcert`).
