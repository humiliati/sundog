# H8 v3 pre-registration — a **LOAD-BEARING** RD determine/resist crossover (substrate S3τ)

> **DESIGN LOCKED 2026-06-09, before any v3 run.** Successor to the **H8 v2 SCOPED bounded-positive**
> (`H8V2_RD_PHASE_DEFECT_RESULT.md`). v2 demonstrated a genuine charFun crossover on CGL spirals **but an
> ablation proved the reaction–diffusion dynamics were NOT load-bearing** — a bare analytic vortex
> reproduced every gate, because v2's latents were *symmetry-orbit coordinates* (rotational phase = the
> SO(2) orbit; chirality = the reflection) of one frozen template. v3's single defining requirement: the
> RD dynamics must be **load-bearing**, enforced by a **pre-registered ablation battery** (bare-vortex /
> rigid-rotation / **time-warp** surrogates MUST FAIL). NOT public-eligible; frozen-as-portfolio; a clean
> NULL is a success (and here the most likely + most valuable one). Attribution: Aranson & Kramer (CGL);
> the Shadow-Invertibility / charFun laws; Debye/Waller; S0/S1/S2.

## The obstacle, and the escape (the design's whole logic)
A charFun-**resisting** latent is a **phase**; a phase is a **one-parameter symmetry-orbit coordinate**
(rotation φ, translation), so `field(phase) = g_φ · field(0)` for a group element `g_φ` — and a non-RD
surrogate that can apply `g_φ` to one static template reproduces every phase. **That is why v2 was S1.**
The escape (the only one the design-recon found): a phase whose orbit map `g_φ` is the **PDE flow**, not a
spatial symmetry — i.e. the spiral's **TEMPORAL phase** `τ` (integration time mod the rotation period),
read from **real CGL time-integration snapshots**. Time-advance of a CGL spiral is *not* a rigid rotation
(v2 residual 0.71); a static template has no `τ` at all. **v3 stands or falls on whether `τ` genuinely
escapes the symmetry-orbit trap — measured by the time-warp ablation.**

## The body & shadow — substrate S3τ (real CGL time-integration)
The v2 CGL integrator, **recording frames**: one long trajectory `A(x,t)` per base, snapshots stored at
stride → a **time-indexed library** (the faithful analogue of v2's rotation library; ~0.2 min to build).
- **Continuous latent (resist):** `xc = τ`, the snapshot integration time (mod rotation period). The
  **shadow** is the mean over `K` subunits at **jittered integration TIMES** `t_i = t(τ) + λ·ξ_i` (the
  lossiness population is now jitter-in-TIME). Recovery uses a **temporal-phase-sensitive** read-out over a
  short window of frames (the temporal Fourier phase at the spiral's intrinsic ω) — NOT a single
  snapshot's spatial orientation (which would be the rotation angle = the v2 trap).
- **Discrete latent (determine):** **Milestone-1** — the chirality (as v2), *carried along but reported
  honestly*: LB-disc is expected to FAIL (a rotation+mirror surrogate reproduces it), so the
  load-bearing-ness burden is on the resist. **Milestone-2 (stretch)** — an **emergent annihilation-parity**
  determine (below), the genuinely load-bearing determine.

> **⚠ CORRECTION (post-review):** the ablation battery below was operationalized with a **cross-test**
> (train a recoverer on real frames, test on surrogate frames) that turned out to be **VACUOUS** — it
> measures feature-distribution overlap, not mechanism necessity (a real-RD field with byte-identical
> dynamics but column-shuffled features "fails" it: cross R²=0.000, own R²=0.864). The **anti-vacuity guard**
> below ("own-preflight-pass ⇒ cross-fail is meaningful") is **logically broken** — the real-RD shuffle
> control passes its own preflight and still cross-fails. So LB-cont as implemented is invalid; v3 is a NULL
> (KILL-LB + KILL-PERIODIC). See `H8V3_RD_LOADBEARING_RESULT.md`. The gate text is retained as the record.

## THE LOAD-BEARING GATE (the centerpiece v2 lacked) — a pre-registered ablation battery
**Binding rule:** `RD-load-bearing := real_RD.PASS  AND  (∀ surrogate S) S.FAIL(gate_S)`, evaluated
**PER-HALF** (LB-cont and LB-disc reported separately — substrate-agnosticism can live in either latent).
**Anti-vacuity guard (the inverse of v2's vacuous PASS):** each surrogate must PASS its **own proper
preflight** (the latent it *can* carry, at the *same* n/seed/noise/probe/thresholds as real-RD) — so its
crossover-FAILURE is a genuine absence of the mechanism, not a dead/under-powered probe.

| Surrogate | What it is | MUST fail |
|---|---|---|
| A1 bare static vortex | `tanh(r/r₀)e^{iqθ}`, zero PDE | LB-cont **preflight** (no τ channel → cont₀<0.30) |
| A2 rigid rotation+mirror of one frame | the v2 construction | LB-cont preflight **AND** the v2-catcher: if it reproduces the crossover, KILL |
| **A3 TIME-WARP (decisive)** | resample **one** real trajectory in time, no re-integration | LB-cont — if τ is a mere reparametrization coordinate, A3 reproduces it → **NULL** |
| A4 matched-power phase-randomized | same S(k), random phase | LB-disc / the C2 strong null (disc<0.95) |
| A5 diffusion-only (no reaction) | linearized, no limit cycle → no selected ω | LB-cont (proves the **reaction** term specifically is load-bearing) |
| A6 non-defect CGL regime | stable bulk oscillation, no spiral | LB-disc (no topological charge) |

**FALSIFIABLE KILL:** if any pure-template surrogate (A1/A2/A3) reproduces the FULL crossover within
δ=0.10 (cont) / 0.03 (disc) of real-RD, the latent→field factored through a group action → **v3 is
S1-again → bank as NULL** (the boundary "even temporal phases are reparametrization coordinates").

## Pre-registered crossover gates (reuse the frozen apparatus; the v2 integrity lessons baked in)
`CONT0_MIN=0.70, DISC0_MIN=0.95, CONT_MAX_MAX=0.10, DISC_MIN_MIN=0.95`; `LAMBDAS` as frozen.
- **G1–G4:** cont₀≥0.70 (τ recoverable), cont washes to ≤0.10 (in-grid half-life); disc₀≥0.95, min disc≥0.95
  censored.
- **G-KINV (BOTH forms, no substitution this time):** report the half-life-vs-K table **AND** the fixed-λ
  `cont(λ_test)` vs K, **AND a finite-mean LLN control side-by-side** — pre-registered together; the verdict
  states which discriminates and why (the v2 lesson: a failing half-life form must be disclosed, not swapped).
- **NON-RIGIDITY PREFLIGHT (the load-bearing anchor):** the **equilibrated** (transient-dropped) residual
  between time-advance and best rigid-symmetry alignment must be **> 0.4**. If it collapses on the limit
  cycle (the spiral is asymptotically rigidly rotating), τ is a symmetry coordinate → expect A2/A3 to pass →
  NULL. *Pre-registered tension: the transient gives non-rigidity but charFun purity wants the attractor.*
- **TEMPORAL-charFun-DECAYS check:** verify `‖charFun(jitter-in-time)‖→0` (a periodic single-frequency
  temporal phase is the *lattice SURVIVE-case* → would DETERMINE not resist — itself a finding).
- **C-NONTRIVIAL** (handedness/parity-blind probe ≤0.60) and **C-CHANNEL** (phase↔chirality leak) —
  C-CHANNEL is a **hard gate** this time (v2 let it fail silently); expect it may still fail (temporal phase
  and local defect structure are dynamically coupled) and report it.

## Milestone-2 (stretch) — the emergent annihilation-parity determine
`xd =` the **parity of the net topological charge after a fixed integration time** in a multi-defect field
seeded from a supercritical random IC with controlled imbalance — set by the **irreversible annihilation
cascade**, not a countable IC feature. **Load-bearing determine gate (shared-IC, Q-withheld):** feed the
surrogate the *same* IC; without integrating it counts pre-annihilation charge → gets parity wrong →
surrogate disc(Q) at chance while real-RD disc stays high. Lossiness = defect contamination/annihilation
noise → per-subunit Q corrupts, ensemble-majority parity **survives** (S2-grade). **Honest prior the
emergent determine is load-bearing: ~45%** (the read-out is substrate-blind by symmetry; emergence must
live in *how Q is set*, not in reading it; parity must also clear C-NONTRIVIAL vs a gradient-energy leak).

## Honest prior (pre-committed, 3-way)
- **~40% clean dual-pass** — load-bearing resist AND determine.
- **~35% load-bearing RESIST + scoped/orthogonal DETERMINE** — the single most likely; bankable as the
  first genuinely RD-load-bearing *resist*, with the determine honestly scoped (the burden falls on the
  resist, as all four designers predicted).
- **~25% full NULL** — A3 (time-warp) reproduces τ, OR the equilibrated phase is a single-frequency
  determine; banks the sharp boundary "no charFun-resisting latent of an RD field is load-bearing — even
  temporal phases are reparametrization coordinates." The most valuable null.

## Kill criteria (each a bankable result)
- **KILL-LB:** A1/A2/A3 reproduces the crossover (δ-close) → S1-again → NULL.
- **KILL-RIGID:** equilibrated non-rigidity residual ≤ 0.4 → τ is a symmetry coordinate → NULL/expected-A3-pass.
- **KILL-PERIODIC:** τ DETERMINES not resists (G-KINV finite-mean / temporal charFun does not decay) → the
  temporal phase is a lattice-survive case → crossover collapses to determine/determine → NULL.
- **KILL-DISC:** emergent parity uncontrollable (disc<0.95 / C5 fails) or leaks (C-NONTRIVIAL fails) →
  Milestone-2 scoped null (Milestone-1 may still stand).

## Honest boundaries (pre-committed)
- **A time-series read-out is required** (the design-recon's core verdict): a single frame cannot, even in
  principle, distinguish RD-produced from template-produced structure (the v1 single-snapshot wall). This
  changes the apparatus shape from S0/S1/S2's single feature vector to a short temporal window — disclosed
  up front; it is still the same lossy-ensemble Shadow operator, with the population over *time*.
- Compute: library-amortized like v2 for the single-spiral resist (~v2 cost); the multi-defect emergent
  determine needs fresh dynamical ensembles (~20–50× v2) — grid-48/64 rescue pre-registered.
- Forward-only; deterministic re-run (pin seeds; record the determinism check).

## Files (to be produced)
- `scripts/reaction_diffusion_temporal_shadow.py` — the S3τ probe (CGL time-series library; temporal-phase
  resist; the ablation battery A1–A6 as first-class gates; both G-KINV forms + finite-mean control).
- `scripts/test_reaction_diffusion_temporal_shadow.py` — frozen test locking the outcome (crossover-or-null
  + the load-bearing battery verdict + determinism).
- `results/atlas/h8v3/` + `docs/atlas/H8V3_RD_LOADBEARING_RESULT.md` — the receipt (post-run, against THIS pre-reg).
