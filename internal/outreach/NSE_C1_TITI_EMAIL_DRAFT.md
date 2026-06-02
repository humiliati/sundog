# NSE C1 Email Draft - Edriss Titi

Date: 2026-06-02

Purpose: a slightly warmer, Titi-specific version of the PDE C1 framing-check
email. Keep it one-to-one. Do not attach IP strategy docs or mechanism-level
patent material.

## Subject

```text
tiny NS/data-assimilation fidelity ask from hobbyist
```

## Body

```text
Dear Professor Titi,

I hope you do not mind a small cold email. I am Jeffery Hughes Jr., working on
Sundog, an independent research project around coarse observations and control.
I am reaching out because your work on determining modes, data assimilation, and
dissipative systems is close to the exact sanity check we need, and because you
have a reputation for being generous with young or outside researchers trying to
find the honest boundary of a problem.

The ask is deliberately narrow. We have a finite-Galerkin 2D Navier-Stokes
numerical result, and I would value even a one-paragraph read on whether our
language is mathematically fair, too strong, or simply standard
data-assimilation/observer language that should be renamed.

The candidate framing is:

In a 2D Kolmogorov-flow Galerkin model at two Grashof values, a low-Fourier
observation Phi_K appears state-insufficient on the sampled support: there are
signature-near pairs with separated high modes. The same Phi_K appears
control-sufficient for a registered low-band-energy proxy action: local
action-mixing goes to zero in signature neighborhoods, up to a small decision
surface. The objective is a held-out look-ahead-max quantile, not an
instantaneous function of Phi_K.

This is a two-point Grashof result only: the forcing wavenumber k_f is fixed and
the objective family is single, so we are not claiming generality across forcing
or objective.

We are not claiming Navier-Stokes regularity, not claiming an
infinite-dimensional theorem, and not asking for endorsement. The goal is to
avoid overclaiming before this becomes paper-facing.

The three questions I would most value your view on:

1. Is "state-insufficient on the sampled support" a fair phrase for the
   twin-state certificate, or should it be weakened?
2. Is "control-sufficient on Phi_K fibers up to a small decision surface" a fair
   reading of the kNN/disintegration evidence, or is that overstated?
3. Would you read the whole thing as a real finite-Galerkin separation between
   state reconstruction and action sufficiency, or as ordinary data-assimilation /
   observer behavior that we should not make load-bearing?

I have attached a short self-contained packet: the separation statement, the
two-regime result, and the two adjudicator checks with their run receipts.

If you have time for a short reply, even "this is standard, cite X" or "do not
use this language because Y" would be genuinely helpful. A negative answer is
probably the most valuable outcome for us.

Thank you for considering it,

Jeffery Hughes Jr.
Sundog
```

## Sender Note

get proof read