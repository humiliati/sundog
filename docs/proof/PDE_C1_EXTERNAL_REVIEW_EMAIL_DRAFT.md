# PDE C1 External Review Email Draft

> Reviewer-facing sanity-check request for the C1 result. Goal: a
> bounded, honest framing check from a PDE / data-assimilation analyst —
> not a referee report, not an endorsement. The whole virtue of this ask
> is brevity; keep it that way. Placeholders in `[brackets]`.

## Send Prep (staged 2026-06-01)

**Status: send-ready once the two owner items below are filled.** The sender
signature is pre-filled as **Jeffery Hughes Jr.** across all three versions —
confirm, or adjust to the form you prefer (e.g. add "Stellar Aqua LLC / Sundog
Research Lab").

**Which version to send:** the **Recommended Outreach Version** for a general PDE
analyst; switch to the **Technical Short Version** if the reviewer is a
determining-modes / data-assimilation specialist (it leads with the
determining-modes framing they will recognize fastest).

**Owner-fill before send — the only blockers:**

1. **`[Name]` — the reviewer.** One named person; this is a 1:1 ask. Best fit: a
   PDE analyst fluent in 2D NSE / determining modes / data assimilation / observer
   theory (full fit list under "Reviewer Fit" below).
2. **`[link]` — the packet.** Point to the four files (all verified present), as a
   GitHub folder link or a single PDF bundle:
   - `docs/proof/PDE_C1_REGIME_GENERALITY_v1.md`
   - `docs/proof/PDE_C1_KNN_CONVERGENCE_CHECK.md`
   - `docs/proof/PDE_C1_TWIN_STATE_CERTIFICATE.md`
   - `docs/SUNDOG_V_NAVIERSTOKES.md`
   Optional add if the reviewer asks "isn't this just near-determination?":
   `docs/proof/PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md` (the adversarial
   near-determination pre-empt).
3. **Tailor the specialty phrase** `[PDE / determining modes / data assimilation /
   numerical NSE]` in the opening line to the chosen reviewer (delete the others).
4. **`[1-2 weeks]`** — timeline; the default is fine.

**Do not** soften the scope clauses (two-point Grashof / `k_f`-fixed /
single-objective-family / finite-Galerkin) to make the result sound bigger: per
the sender notes below, a reviewer flagging a missing caveat is a worse outcome
than one confirming a conservative framing. **A negative reply is the most
valuable outcome.**

## Recommended Outreach Version

**Subject:** Technical framing check request: finite-Galerkin NSE signature/control result

Hi [Name],

I am Jeffery Hughes Jr., working on Sundog, an independent research project about
coarse-grained signatures for control. We have been testing whether a compact
observation can be useful for a registered control action even when it does not
reconstruct the full state. I am reaching out because your work around
[PDE / determining modes / data assimilation / numerical NSE] seems close to
the exact sanity check we need.

The specific ask is narrow: could you tell us whether the framing below is
mathematically honest, too strong, or simply standard observer/data-assimilation
language that should be renamed?

Candidate framing:

> In a 2D Kolmogorov-flow Galerkin model (`k_f = 2`), a low-Fourier
> observation `Phi_K` is **state-insufficient** on the sampled support
> (signature-near pairs can have well-separated high modes) yet appears
> **control-sufficient** for a registered low-band-energy proxy action
> (local action-mixing vanishes as signature tolerance goes to zero, up to
> a small decision surface). The pattern holds at two Grashof values
> (`G = 200` and `G = 300`) under a held-out-quantile objective.

What we most need checked:

1. Is "state-insufficient on the sampled support" fair for the twin-state
   certificate, or should it be weakened?
2. Is "control-sufficient on `Phi_K`-fibers up to a measure-zero decision
   surface" fair for the kNN/disintegration read, or overstated?
3. Is the held-out look-ahead-max quantile a defensible way to define a
   regime-portable proxy action, given that the older burn-in trigger went
   vacuous at `G = 300`?
4. Would you call this a real finite-Galerkin separation between
   state-reconstruction and action sufficiency, or ordinary
   data-assimilation / observer behavior that should not be made load-bearing?

Scope: this is early-stage and numerical. It is not a Navier-Stokes regularity
claim, not an existence claim, not a theorem about the infinite-dimensional
attractor, and not a request for endorsement. We are specifically trying to
prevent overclaiming before paper-facing use.

Packet and suggested read order:

1. `PDE_C1_REGIME_GENERALITY_v1.md` - current result, scope, and gates.
2. `PDE_C1_KNN_CONVERGENCE_CHECK.md` - control-sufficiency adjudicator.
3. `PDE_C1_TWIN_STATE_CERTIFICATE.md` - sampled-support state-insufficiency
   bridge.
4. `SUNDOG_V_NAVIERSTOKES.md` - project ledger and claim boundary.

If you have time in the next [1-2 weeks], a paragraph or a few bullet points
would be ideal. "This seems conservative," "weaken X," "cite Y," or "do not
use this language because Z" are all useful outcomes. Email comments are
perfect; I would also be happy to do a short call if that is easier.

Thanks,
Jeffery Hughes Jr.

## Technical Short Version

**Subject:** Bounded sanity check — finite-Galerkin signature/control-sufficiency framing

Hi [Name],

Could I ask for a bounded framing check on a small finite-Galerkin
Navier-Stokes note? It is **not** a regularity or existence claim and
**not** a theorem about the infinite-dimensional attractor. The question
is whether we are stating a numerical signature/control-sufficiency
candidate result honestly.

In determining-modes terms, the candidate framing is a *separation*:

> In a 2D Kolmogorov-flow Galerkin model (`k_f = 2`), a low-Fourier
> observation `Phi_K` is **state-insufficient** (a twin-state search
> finds signature-near pairs with well-separated high modes on the
> sampled SRB-like support) yet appears **sufficient for a registered
> low-band-energy control action** — the local action-mixing vanishes as
> the signature tolerance → 0, consistent with control-sufficiency up to
> a measure-zero decision surface. The pattern holds at two Grashof
> values (`G = 200` and `G = 300`) under a held-out-quantile objective
> that stays discriminative across both, and the control half is also
> stable to halving the signature dimension and to swapping the
> objective construction.

It stays finite-Galerkin, sampled-support, numerical, and proxy-based,
and it is a **two-point Grashof-axis result only** — `k_f` is fixed and
the objective family is single, so we are not claiming generality across
forcing or objective.

What I need is not endorsement but a quick read on whether that framing is
earned:

1. Given the twin-state certificate is sampled-SRB / finite-Galerkin
   (not an exact attractor theorem), is the **"state-insufficient on the
   support"** language reasonable, or should it be weakened?
2. Is the kNN/disintegration reading of **"control-sufficient on
   `Phi_K`-fibers up to a measure-zero decision surface"** mathematically
   fair, or overstated?
3. Does a **held-out look-ahead-max quantile** look like a legitimate way
   to define a regime-portable objective (it replaced a burn-in-percentile
   trigger that went vacuous at `G = 300`)?
4. Would you read this as a genuine *"a sub-determining mode set can be
   control-sufficient without being state-reconstructive"* separation, or
   as ordinary data-assimilation / observer language that should not be
   made load-bearing?

Minimal packet:

- `PDE_C1_REGIME_GENERALITY_v1.md` — current result, scope, and gates.
- `PDE_C1_KNN_CONVERGENCE_CHECK.md` — the control-sufficiency adjudicator.
- `PDE_C1_TWIN_STATE_CERTIFICATE.md` — the sampled-support
  state-insufficiency bridge.
- `SUNDOG_V_NAVIERSTOKES.md` — ledger and claim boundary.

A one-paragraph reply is plenty: "framing seems conservative," "weaken
X," "this is standard, cite Y," or "don't call this support-level /
fiber-sufficient because Z." A negative answer is the most valuable
outcome — the goal is to prevent overclaiming.

Thanks,
Jeffery Hughes Jr.

## Slightly Warmer Version

**Subject:** Small 2D NSE / control-sufficiency sanity check

Hi [Name],

A bounded PDE sanity-check request if you have the bandwidth. We have
been testing a coarse-graining reading against a finite-Galerkin 2D
Kolmogorov-flow model. It is not a Clay-problem claim and not a new PDE
theorem; the point is narrow and I mostly want to know if we are
overstating it.

The observed pattern, in determining-modes language:

- a low-Fourier observation `Phi_K` does **not** reconstruct the full
  sampled state — a twin-state certificate finds signature-near pairs
  with separated high modes on the sampled SRB-like support;
- the same `Phi_K` nevertheless looks **sufficient for a registered
  low-band-energy control action** — kNN/disintegration checks show the
  local action-mixing decaying to zero as the signature radius → 0;
- this state-insufficient / control-sufficient pattern holds at both
  `G = 200` and `G = 300` (for `k_f = 2`), after we replaced a
  non-portable burn-in-percentile trigger with a held-out-quantile
  objective. The control half is also stable across a halving of the
  signature dimension and across the two objective constructions.

The internal conclusion is deliberately scoped:

> two-regime (Grashof-axis only), finite-Galerkin, sampled-support,
> numerical evidence for an observation that is not state-reconstructive
> but is control-sufficient for the registered proxy action. Proxy
> faithfulness and PDE review are open; no infinite-dimensional NSE claim
> is made.

Could you sanity-check whether that is a mathematically honest framing —
in particular whether the kNN-fiber/disintegration language and the
sampled-support twin-state language are acceptable, or should be
downgraded before any paper-facing use? A short reply is enough, and a
negative one is useful.

Packet: [link or attachment]

Thanks,
Jeffery Hughes Jr.

## Follow-Up If They Say Yes

Thank you. The shortest path:

1. `PDE_C1_REGIME_GENERALITY_v1.md` — §1 (the fix), §3 (objective), §6
   (portability gate), §8 (interpretation), §12 (result).
2. Skim the branch/verdict summaries in
   `PDE_C1_KNN_CONVERGENCE_CHECK.md` and
   `PDE_C1_TWIN_STATE_CERTIFICATE.md`.
3. Reply with any of: "framing seems conservative" / "weaken or rename
   this" / "support-level language too strong" / "fiber-sufficiency
   language too strong" / "this is standard data-assimilation, cite X" /
   "I only checked the kNN / twin-state / objective piece."

Not asking for a referee report — a paragraph is ideal.

## Follow-Up If They Decline

No worries at all, and thanks for considering it. If someone else is a
better fit for a quick 2D NSE / determining-modes / data-assimilation
framing check, a pointer would be appreciated.

## Reviewer Fit

Best fit:

- PDE analyst familiar with 2D NSE, determining modes, or data
  assimilation / observer theory;
- numerical analyst with Galerkin-NSE and invariant-measure-sampling
  experience.

Also useful:

- control / data-assimilation researcher comfortable with conditional
  observability and finite-dimensional observer diagnostics;
- turbulence / shell-model person for the empirical/objective side;
- mathematical-statistics person for the kNN/disintegration language.

## Notes For The Sender (not part of the email)

- The strongest honest selling points, if asked "why should I trust the
  numbers": the control verdict is robust to signature dimension
  (`d_K = 18` vs `32`), to sample budget (50k vs 200k), and to the
  objective construction (burn-in-percentile vs held-out-quantile both
  POSITIVE at `G = 200`); and the adjudicator overturned its **own** first
  mechanical reading on a pre-registered convergence check, so the
  pipeline is not tuned to a desired answer.
- **Adversarial robustness (2026-05-31), pre-empting the likely first
  objection ("isn't this just near-determination?"):** we pre-registered and
  ran a probe designed to *demote* the result — testing whether
  control-sufficiency is merely `Φ_K` near-determining the whole state, so the
  objective is "controlled" only because it is predictable. It did **not**
  reduce to that. Across a 6-objective band/dissipation slate, `Φ_K` predicts
  every objective well (`R² 0.76–1.00`, including the high-band and dissipation
  range) yet control-sufficiency does **not** track predictability (Spearman
  `−0.75 / −1.0`), so the separation is not a closed-subspace artifact. One
  open puzzle is flagged honestly — a predictable-but-not-control-sufficient
  dissipation objective (`palinstrophy`). Detail:
  `PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md`.
- Do **not** soften the scope clauses to make the result sound bigger.
  The two-point / `k_f`-fixed / single-objective-family / finite-Galerkin
  caveats are load-bearing and a reviewer flagging them as missing would
  be a worse outcome than a reviewer confirming a conservative framing.
- If the reviewer asks "is `pi_hat` the optimal selector?": the honest
  answer is no — it is a registered proxy (the burn-in/held-out overshoot
  trigger), not a derived `J`-optimal selector; that is exactly the open
  proxy-faithfulness item.
