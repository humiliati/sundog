# PDE C1 External Review Email Draft

## Short Version

**Subject:** Quick PDE sanity check: finite-Galerkin signature/control witness

Hi [Name],

Could I ask for a bounded sanity check on a small Navier-Stokes /
data-assimilation-adjacent note?

This is **not** a Navier-Stokes regularity claim, not an existence claim,
and not a theorem about the infinite-dimensional attractor. It is a
finite-Galerkin numerical/proof-track question about whether we are
framing a signature/control-sufficiency result honestly.

The claim I want checked is:

> In a 2D Kolmogorov-flow Galerkin model at `k_f = 2`, a low Fourier
> signature `Phi_K` is state-insufficient on sampled SRB-like support
> but still sufficient for a registered safety/proxy control action,
> up to a measure-zero decision surface. This complete regime-2 witness
> now appears at two Grashof values, `G = 200` and `G = 300`, under a
> portable held-out-quantile objective. It remains finite-Galerkin,
> sampled-support, numerical, and proxy-based.

What I most need is not an endorsement, but a quick answer to:

1. Is the support-level language reasonable, given the twin-state
   certificate is sampled-SRB / finite-Galerkin rather than an exact
   attractor theorem?
2. Is the kNN/disintegration reading of "control-sufficient on
   `Phi_K`-fibers up to a decision surface" mathematically fair, or
   should it be weakened?
3. Does the portable held-out-quantile objective look like a legitimate
   way to avoid the G=300 vacuity found under the original burn-in
   percentile rule?
4. Would you treat this as a real Postulate-1 / determining-modes-style
   witness, or as ordinary data-assimilation/numerical-observer language
   that should not be made load-bearing?

Minimal packet:

- `docs/proof/PDE_C1_REGIME_GENERALITY_v1.md` - current result and scope.
- `docs/proof/PDE_C1_KNN_CONVERGENCE_CHECK.md` - control-sufficiency
  adjudicator.
- `docs/proof/PDE_C1_TWIN_STATE_CERTIFICATE.md` - sampled-support
  state-insufficiency bridge.
- `docs/SUNDOG_V_NAVIERSTOKES.md` - ledger / claim boundary.

The most useful response would be one paragraph: "framing seems
conservative," "weaken X," "this is standard / cite Y," or "do not call
this support-level / fiber-sufficient because Z."

Thanks,
[Your name]

## Slightly Warmer Version

**Subject:** Small Navier-Stokes/control-sufficiency sanity check

Hi [Name],

I have a bounded PDE sanity-check request if you have the bandwidth.
We have been testing a Sundog/Postulate-1 reading against a finite
Galerkin 2D Kolmogorov-flow model. The result is not a Clay-problem
claim and not a new PDE theorem; the point is much narrower.

The observed pattern is:

- low Fourier signature `Phi_K` cannot reconstruct the full sampled
  state: a twin-state certificate finds signature-near pairs with
  separated high modes on the sampled SRB-like support;
- the same `Phi_K` nevertheless appears sufficient for a registered
  low-band-energy proxy action: kNN/disintegration checks show local
  action mixing decays to zero as the signature radius goes to zero;
- this complete state-insufficient / control-sufficient pattern holds
  at `G = 200` and `G = 300` for `k_f = 2`, after replacing a
  non-portable burn-in-percentile trigger with a held-out quantile
  objective that stays discriminative across regimes.

The internal conclusion is deliberately scoped:

> two-regime, finite-Galerkin, sampled-support, numerical evidence for a
> signature that is not state-reconstructive but is control-sufficient
> for the registered proxy action. Proxy faithfulness and PDE review are
> still open; no infinite-dimensional NSE claim is made.

Could you sanity-check whether that is a mathematically honest framing?
In particular, I would value your judgment on whether the kNN
fiber/disintegration language and sampled-support twin-state language
are acceptable, or whether they should be downgraded before any public
or paper-facing use.

Packet: [link or attachment]

A short reply is enough. A negative answer is useful here; the goal is
to prevent overclaiming.

Thanks,
[Your name]

## Follow-Up If They Say Yes

Thank you. The shortest review path is:

1. Read `PDE_C1_REGIME_GENERALITY_v1.md` sections 1, 3, 6, 8, and 12.
2. Skim the branch summaries in `PDE_C1_KNN_CONVERGENCE_CHECK.md` and
   `PDE_C1_TWIN_STATE_CERTIFICATE.md`.
3. Reply with any of:
   - "framing seems conservative";
   - "weaken / rename this";
   - "support-level language is too strong";
   - "fiber-sufficiency language is too strong";
   - "this is standard data-assimilation language; cite X";
   - "I only checked the kNN / twin-state / objective piece."

I am not asking for a referee report.

## Follow-Up If They Decline

No worries at all, and thank you for considering it. If someone else is
a better fit for a quick 2D NSE / determining-modes / data-assimilation
sanity check, I would be grateful for a pointer.

## Reviewer Fit

Best fit:

- PDE analyst familiar with 2D NSE, determining modes, or data
  assimilation / observer theory;
- numerical analyst with experience in Galerkin Navier-Stokes and
  invariant-measure sampling;
- control/data-assimilation researcher comfortable with conditional
  observability and finite-dimensional observer diagnostics.

Less ideal but still useful:

- turbulence / shell-model person for the empirical/objective side;
- mathematical statistics person for the kNN/disintegration language.
