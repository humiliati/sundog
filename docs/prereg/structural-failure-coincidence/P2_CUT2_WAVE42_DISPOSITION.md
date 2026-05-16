# Structural Failure Coincidence — Wave-4.2 Disposition: C3-A BLOCK recorded (γ) + Cut-3 escalation (α)

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Receipts disposed: [`P2_CUT2_C3A_NUMERIC_FREEZE.md`](P2_CUT2_C3A_NUMERIC_FREEZE.md) (Wave-4 v1 + Wave-4.1 v2)
Cut-3 staging origin: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md) staged-cut amendment ("Cut 3 — rendered-signal escalation, conditional")
Filed: **2026-05-16 (PT)**. Status: **WAVE-4.2 DISPOSITION — FILED.**
This is the named freeze-level redesign decision (user-selected **α+γ**),
not a patch: it changes **no** C3 frozen value and reopens **no** Wave-4
receipt. It records the closed-form result as permanent and escalates to
Cut-3 per the pre-registered staged rule. Cut-3 is **not started** —
it carries its own admission gate. Nothing run; Public-Language
Constraint in force.

## What this disposes

Wave-4 (v1) and Wave-4.1 (Path Y/Z, v2) produced, under SHA-256-pinned
reproducible generators, **C3-A-R PASS · C3-A-T BLOCK · C3-A-B BLOCK**,
held permanent under Path W with **no** leverage-weighting / κ-increase /
threshold-relaxation / decoy-re-pinning. C2-A is already closed
(Wave-3: C2-A-1/2/3 PASS). The remaining closed-form blocker is
specifically the C3-A **temptation** receipt. The user selected the
**α+γ** Wave-4.2 path. This document files both halves.

## γ — the permanent honest finding (no pass language)

The Wave-4 receipts are recorded as a **permanent honest result**, not a
prompt to retune:

- **C3-A-R PASS.** Decoys provably move the argmax
  (`421/648 = 65.0% ≥ 50%`). Cut-1's `∂J/∂d ≡ 0` vacuity is removed *by
  computation*. This is a genuine partial positive and it stands.
- **C3-A-T / C3-A-B BLOCK (permanent).** The binding fact is **T1**
  (temptation must be real), not T2 (reversal — which *held*). The
  closed-form decoy is **worse** than the documented route wherever the
  route is eligible (`|π_dec−h| 3.806°` vs `|π_route−h| 1.398°`, v1;
  `3.567°` vs `1.860°`, v2), and only "wins" in the L1-ineligible /
  low-leverage band where the route is **already abstaining by design**
  (C2-D). The sample-weighted non-degenerate average (eligible subset
  larger) favours the route; the subset where the decoy wins was **not**
  cherry-picked.

**Stated precisely (this is a statement about the signal geometry, not
about any agent — no controller was run):** *in the closed-form
parhelion bundle the documented inverse strictly dominates the
closed-form correlate wherever the inverse is eligible; the correlate
only substitutes in the inverse's abstain region (regime-separability).*
This is **traceability-favorable** in spirit — it matches the README's
frozen traceable-agent prediction ("succeeds on strict eligible cases
and reports low leverage or ineligibility outside them") — but it is
**explicitly NOT a traceability pass and NOT a controller result.**
Per the Public-Language Constraint: no `CONFIRMED`, no "traceability
harness passes", no theorem, no "the controller is traceable". The
allowed framing is exactly: *closed-form domain regime-separability
finding; the closed-form Cut-2 cannot stage the Proxy-Collapse
temptation discriminator.*

Outcome-table mapping: this is **not** a convergence null (D) and **not**
a B pass. It is the prereg's *"Cut-2 ambiguous / the discriminator
cannot be constructed crisply here"* condition — which the spec
pre-registered as the **Cut-3 escalation trigger**, not a halt.

## α — Cut-3 escalation (why it is the right move, and what gates it)

The C3-A-T BLOCK is **structural, not numeric**: the only ways to make a
scalar closed-form decoy beat the high-leverage parhelion signal in the
eligible band are leverage-weighting / κ-inflation / threshold-
relaxation — the explicitly forbidden self-seals. A *principled*
closed-form eligible-band-competing decoy was considered (Path β) and is
recorded as **most-likely-vacuous / a tuning hiding place**; it is **not
pursued**.

Cut-3 (the pre-registered rendered-signal escalation) is where the
discriminator becomes constructable: a **learned** correlate on a
rendered signal can latch image-style / metadata / halo-prominence
features that co-vary with `h` *even in the eligible band* — exactly the
README L1 *mere-correlate* prediction for image inputs. So an eligible-
band temptation genuinely exists in the rendered domain. The
regime-separability finding (γ) is therefore not just a null — it is the
*reason* Cut-3 is the correct, non-arbitrary escalation: the closed-form
scalar decoys are structurally too weak to compete where the route is
strong; a learned image correlate is not.

**Cut-3 is NOT started by this disposition.** It inherits its own gate,
already pre-registered in the `P2_RUN_SPEC.md` staged-cut amendment:
Cut-3 admission **must first show the px↔° centring/scale hazard
(exhausted in Phase 15) is resolved (e.g. HaloSim-native Scale) or Cut-3
is itself blocked, not forced.** Artifact-before-agent applies one level
up: a Cut-3 run spec + Cut-3 admission must exist and clear the px↔°
hazard before any Cut-3 controller run. C1 (the bound controller) and
the staged rule's "trained learning agent and/or rendered signal"
scope, the prereg Outcome Branching table, and the Public-Language
Constraint all carry forward into Cut-3 unchanged.

## What is permanent / frozen / unchanged

- Wave-4 v1 and Wave-4.1 v2 BLOCK receipts are **permanent** (SHA-pinned
  generators; independently re-runnable). Cut-3 does **not** reopen or
  retune them — it is a *different cut*, not a re-run of closed-form
  C3-A.
- **No** C3 frozen `[E]` value, geometry/receipt boundary, threshold,
  adapter rule, or outcome mapping is changed by this disposition
  (append-only; no body rewrite).
- C2-A (closed, PASS), C4-A, C5 closed-form artifacts retain their
  status; they now serve to **record the closed-form γ finding**, not to
  enable an admittable closed-form Proxy-Collapse *pass* (which the
  C3-A-T BLOCK shows is unconstructable in closed form without forbidden
  tuning). The single joint admission re-run, for the closed-form line,
  can therefore only ever certify *γ + the escalation* — never a
  traceability pass. State this so the program state is unambiguous.

## Honest disposition (the B-or-D resolution at this layer)

This is neither the demoralising in-between nor a premature B. It is a
**clean structural finding** (the closed-form domain is traceability-
favorable and does not *offer* an eligible-band Proxy-Collapse shortcut)
**plus** the prereg's own staging rule firing exactly as designed (the
real correlate-competition test moves to the rendered Cut-3). The
discipline held under maximum pressure: a failed receipt with an obvious
tuning escape was refused, recorded permanent, and escalated by the
pre-registered route — which is the entire point of the apparatus.

## Audit Notes

*(reviewer space — append-only below)*

**2026-05-16 (PT) — maintainer. Cut-3 spec/admission opened.**
[`P2_CUT3_RUN_SPEC.md`](P2_CUT3_RUN_SPEC.md) and
[`P2_CUT3_ADMISSION.md`](P2_CUT3_ADMISSION.md) now instantiate the
gated escalation named above. The px↔° / Phase-15 hazard is handled as
H0 angular calibration: per-frame sun-centered angular map, valid span,
anchor residuals, and hashes before any rendered-signal run. Admission
is **HOLD**, not ADMIT: the protocol shape is accepted, but no concrete
render corpus, H0 residual table, agent path, baselines, or edit
operators exist yet. Cut-3 remains opened but not started.
