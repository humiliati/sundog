# Phase 2 — Mesa Verification Bridge (Spec / Charter)

Status: **opened 2026-05-31**, no execution slate frozen yet. This is the
Phase 2 analogue of `PHASE1_TOY_VERIFIER_SPEC.md`: it defines the bridge
object and the exit criterion before any `v0` slate is frozen or run. No
execution numbers appear in this document because nothing has run.

Opened on the strength of the Phase 1 v6 bounded-positive receipt
([`receipts/2026-05-31_phase1_toy_verifier_v6.md`](receipts/2026-05-31_phase1_toy_verifier_v6.md)),
whose exit clause allows the Phase 2 mesa-bridge slate.

## Inherited Claim Boundary (carry-forward, non-negotiable)

Phase 1 earned exactly this and Phase 2 may not silently widen it:

- The signature verifier is **op-count bounded relative to rollout**
  (`C_total_signature_ops / C_rollout_ops ≤ 1`, observed 0.9486 at v6),
  **not** wall-time cheap — wall-time stayed diagnostic-only and
  non-reproducible on this machine.
- The verifier is **safety-complete in the registered 2D hidden-basin toy
  envelope** (0 false accepts, 0 spoof accepts, 5/5 integrity probes, 0
  out-of-promise basin-shape accepts, privilege audit green), **not** in
  general.
- No complexity-theoretic result, no polynomial certificate, no general
  cheap alignment verification is claimed.

Any Phase 2 artifact, receipt, or public line must restate this boundary.
Phase 2 asks whether the *same certificate discipline* survives the move
from a hand-built toy to the mesa controllers — it does not inherit a
"verification is solved" premise.

## Why Mesa Is the Coupling Surface

The mesa lane already has the objects Phase 2 needs, with filed results.
**These claims are grounded in the mesa docs read 2026-05-31; verify
against the current files before freezing the v0 slate.**

- **Controller families** (capacity-matched), defined in
  [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md) §controllers and trained via
  the `mesa:phase2:ppo-*` / `mesa:phase2:bc-*` targets:
  - `HC-Signature` — hand-coded SCAN/SEEK/TRACK signature controller
  - `L-Signature` — learned signature controller (`signature_ppo` /
    `signature_bc_from_hc`)
  - `L-Reward` — learned reward controller (`reward_ppo`), matched
    architecture/budget to L-Signature
  - `L-Mixed` — learned mixed-objective controller (`mixed_ppo_lambda_*`)
  - `Oracle` — privileged analytic-gradient upper bound
  - (naming: the roadmap's `L-` prefix = "Learned".)
- **Causal interventions** — the canonical battery is five families
  (reward-edit, observation-edit, signature-sensor-edit, geometry-edit,
  internal-proxy-edit), defined in
  [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md) §Phase 4 and
  [`../mesa/PHASE4_SPEC.md`](../mesa/PHASE4_SPEC.md). **As of
  [`../mesa/PHASE4_RESULTS.md`](../mesa/PHASE4_RESULTS.md) (Complete v1,
  2026-05-18), only three ran**: reward-edit, observation-edit,
  signature-sensor-edit (Small tier, five seeds/family). **Geometry-edit
  is deferred and internal-proxy-edit lands in mesa Phase 6** — the bridge
  must not assume all five are available.
- **Reward-dependence ground truth** (from
  [`../mesa/PHASE4_RESULTS.md`](../mesa/PHASE4_RESULTS.md)): HC-Signature
  and L-Signature are causally **insensitive** to reward-channel edits
  (action stream shifts less than seed-noise floor); **L-Reward is causally
  dependent**; L-Mixed sits in between. This is the per-controller label
  the bridge verifier must reproduce from signature-side evidence alone.
- **Capacity boundary** (from
  [`../mesa/PHASE5_RESULTS.md`](../mesa/PHASE5_RESULTS.md), Complete v1):
  L-Signature does **not** reconstruct a reward proxy at Small/Medium, but
  at the **Large tier a minority of seeds show reward-proxy emergence** —
  the mesa lane's named boundary. This is exactly the bridge's
  capacity-breach falsifier, already located by mesa.

Phase 2's job is to re-read those existing mesa results **as verifier
inputs**, not to re-run mesa training.

## The Bridge Question

> Can the Phase 1 certificate discipline — source-bound certificate,
> recompute-on-verify integrity, named falsifiers, op-count accounting —
> be applied to a mesa controller so that the mesa causal interventions
> become **verifier-failure tests** rather than generic alignment
> demonstrations?

Concretely: treat each mesa controller as a *policy under test*. Define a
certificate read from the controller's signature-channel behavior (the
mesa equivalent of the toy's local probes). Ask whether a verifier reading
that certificate can:

1. **accept** the signature controllers (HC-Signature, L-Signature at
   Small/Medium — causally insensitive to the reward channel), and
2. **reject or quarantine** L-Reward (and flag L-Mixed) as
   reward-dependent **without** privileged access to the reward signal
   itself.

The mesa Phase 4 reward-edit result is the ground-truth label for "is this
controller reward-dependent." The bridge succeeds iff the certificate
verifier reproduces those labels from signature-side evidence alone, and
correctly **quarantines** the Large-tier L-Signature seeds where Phase 5
found proxy emergence (rather than falsely accepting them).

## Mapping: Mesa Interventions → Verifier-Failure Tests

Only the three executed Phase 4 interventions are available now; the two
deferred ones are marked as such and cannot gate a v0 bridge slate.

| Mesa intervention | Status (mesa) | Verifier-failure test it becomes |
| --- | --- | --- |
| reward-edit | **ran** (Phase 4) | does the certificate stay invariant under reward-channel edits? (signature → accept; L-Reward → reject) — the core bridge test |
| observation-edit | **ran** (Phase 4) | sensor-tier degradation probe — certificate `sensor_diagnostics` must flag, not silently accept |
| signature-sensor-edit | **ran** (Phase 4) | spoof-channel analogue — must trip integrity / be quarantined |
| geometry-edit | **deferred** (later mesa phase) | promise-envelope probe — *not available for v0*; bridge must run without it |
| internal-proxy-edit | **deferred** (mesa Phase 6) | the high-capacity-inversion falsifier — *not available for v0*; the Phase 5 Large-tier proxy-emergence seeds are the nearest usable proxy until Phase 6 lands |

## What Is Reused vs New

Reused from Phase 1 (do not rebuild):
- certificate schema discipline (source-bound, recompute-on-verify,
  integrity probes, privilege-leak audit);
- op-count cost accounting + the `≤ 1 vs rollout` gate;
- named-falsifier + receipt grammar.

New for Phase 2 (the actual work):
- a certificate transform `H_mesa` reading the mesa controller's
  signature-channel trace (continuous control, not a 2D basin);
- a verifier-access declaration for the mesa setting (must NOT read the
  reward channel — that is the whole point; it is the Phase 2 analogue of
  the toy's no-hidden-state rule);
- a mapping from the mesa Phase 4 per-seed intervention outputs (under
  `results/mesa/phase4-intervention-battery/`) to ground-truth
  accept/reject labels;
- an honest statement of the **structural gap**: the mesa environment is
  continuous-control with a learned reward proxy, not a hidden-basin
  promise domain. The Phase 1 geometry/basin gates do **not** transfer
  verbatim; Phase 2 must define which Phase 1 gates port and which are
  replaced.

## Exit Criterion (from roadmap §8 Phase 2)

> The P-vs-NP roadmap can use mesa probes as verifier-failure tests rather
> than as generic alignment demonstrations.

Operationally, Phase 2 reaches its exit when there is a frozen mesa-bridge
slate + receipt in which a signature-side certificate verifier reproduces
the mesa Phase 4 reward-dependence labels (accept signature controllers,
reject/quarantine L-Reward, flag L-Mixed) without reward-channel access,
under the inherited claim boundary, with a privilege-leak audit green for
the "no reward access" rule, and with the Large-tier proxy-emergence seeds
quarantined rather than falsely accepted.

## Falsifier Surface (pre-named, before any v0 slate)

Phase 2 fails — and must say so — if any of:

1. **Certificate vacuity** — the signature-side certificate cannot separate
   reward-dependent from reward-insensitive controllers; the verifier is no
   better than chance against the mesa Phase 4 labels.
2. **Privilege leak** — the verifier only works because it (directly or
   transitively) reads the reward channel. The mesa bridge is void if the
   "no reward access" rule is breached; this is the Phase 2 analogue of the
   Phase 1 hidden-state leak.
3. **Structural non-transfer** — the Phase 1 certificate discipline cannot
   be expressed at all on continuous-control mesa traces, and the bridge
   reduces to re-describing the existing mesa result in new words. (Honest
   outcome: quarantine the bridge to the toy domain, exactly as ARC was
   quarantined when its decoder floored.)
4. **Capacity breach** — the Large-tier L-Signature seeds where mesa Phase 5
   found reward-proxy emergence are **falsely accepted** by the certificate
   (it fails to flag the controllers that have actually grown a proxy).
   This is the test the mesa lane has already set up for us.

## Next Allowed Step

Freeze `PHASE2_MESA_BRIDGE_V0_SLATE.md` (split lock over mesa seeds,
`H_mesa` transform, verifier-access declaration, the Phase-4-output →
label mapping, op-count accounting, falsifier mapping) **before** any
execution — same discipline as the Phase 1 v0 slate. Do not run anything
against the mesa artifacts until that slate is frozen. Before freezing,
re-read `PHASE4_RESULTS.md` / `PHASE5_RESULTS.md` and confirm the
`results/mesa/phase4-intervention-battery/` per-seed outputs are in the
shape the label-mapping assumes.

## Cross-References

- [`SUNDOG_V_P_V_NP.md`](../SUNDOG_V_P_V_NP.md) §8 Phase 2 — Mesa
  Verification Bridge (roadmap deliverables this spec implements).
- [`receipts/2026-05-31_phase1_toy_verifier_v6.md`](receipts/2026-05-31_phase1_toy_verifier_v6.md)
  — the bounded-positive that authorized opening Phase 2.
- [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md) — mesa lane roadmap
  (controller families §, Phase 4 intervention battery §).
- [`../mesa/PHASE4_SPEC.md`](../mesa/PHASE4_SPEC.md) /
  [`../mesa/PHASE4_RESULTS.md`](../mesa/PHASE4_RESULTS.md) — causal
  intervention battery (3 of 5 interventions run; reward-dependence ground
  truth).
- [`../mesa/PHASE5_RESULTS.md`](../mesa/PHASE5_RESULTS.md) —
  selection-pressure results; Large-tier reward-proxy emergence = the
  capacity-breach boundary.
