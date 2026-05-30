# PDE C1 — Mechanism / Novelty Recon (2026-05-29)

> Lane-3/B de-risking pass before investing in a mechanism proof: **is the
> "low-band energy is decision-sufficient while the state is not"
> phenomenon already known, and in what language?** Recon depth =
> title/abstract level across four literatures (not deep-read). Purpose:
> avoid reproving folklore and locate the genuinely novel axis.

## 1. The four established literatures and how C1 relates

| Literature | Core claim | Relation to C1 |
| --- | --- | --- |
| **Functional observability** (Montanari–Duan–Aguirre–Motter, PNAS 2022, arXiv 2201.07256; nonlinear: arXiv 2301.04108, 2501.00167) | A *functional* of the state can be estimated from outputs without full-state observability. | **The control-theory home of the separation.** But developed for linear / finite network/ODE systems; not instantiated on an NSE attractor, and for *estimation* of a functional, not a *decision/control* objective. |
| **Determining functionals / modes / forms** (Cockburn–Jones–Titi, Math. Comp. 1997; Foias–Jolly–Kravchenko–Titi determining form, arXiv 1208.5134) | Finitely many functionals **determine** the long-time state / dynamics. | The *reconstruction pole*. C1's `Φ_K` is deliberately **below** the determining threshold (certified non-injective) — the complement of this theory, by design. |
| **Approximate inertial manifolds** (Foias–Manley–Temam; Titi 1990; postprocessed Galerkin, García-Archilla–Novo–Titi) | High modes are **slaved** to low modes, `q ≈ Φ(p)`, given enough low modes. | **Resolves a trap:** if slaving held at `K=3`, `Φ_K` would be state-*sufficient* (no twin states). Our certified non-injectivity says `K=3` is **below** the slaving/AIM threshold — so the mechanism is **not** slaving. |
| **Mori–Zwanzig / optimal prediction closure** (Chorin–Hald–Kupferman; Parish–Duraisamy MZ-LES, Phys. Rev. Fluids 2017, arXiv 1611.03311; arXiv 1611.06277) | A coarse observable's evolution splits into resolved (Markovian) + memory + orthogonal-dynamics noise; the unresolved contribution to the **energy/dissipation budget** is structured and approximable. | **The mechanism's true language.** `dE_K/dt = [low-closed] + R(t)`, with `R` the MZ memory/orthogonal-dynamics coupling, is exactly the MZ decomposition. MZ-LES already relies on `R` being small/modelable for the energy budget. |

## 2. Novelty verdict

**The mechanism is largely known.** "The resolved (low-band) energy budget
is approximately closed / the unresolved coupling into it is small" is the
working premise of Mori–Zwanzig-based LES closure and AIM postprocessing.
We must **not** present that as a discovery, and the energy-budget
diagnostic must be framed as *measuring the known MZ coupling for our
specific objective*, not as finding a new closure.

**The novel axis is the framing, not the mechanism.** What the recon did
**not** find is the specific combination:

1. A **separation between state-reconstruction sufficiency and
   decision/control sufficiency** for a registered objective, on an NSE
   attractor;
2. backed by a **measured, pre-registered non-injectivity certificate**
   (twin-state) for state-insufficiency **and** a **paired fiber-constancy**
   for control-sufficiency on the same pairs;
3. stated in **functional-observability** language (control), at a
   coarseness **deliberately below** the determining/AIM threshold — so it
   is *not* a slaving/closure-accuracy claim but a "the decision is coarser
   than the state" claim.

The non-injectivity certificate is what separates this from LES folklore:
LES/AIM would say "coarse predicts coarse because coarse slaves fine"
(state-sufficient); C1 says "the coarse *decision* is robust to the
genuinely **unresolved** fine state" (state-insufficient, certified).

## 3. Consequence for the lane

- **Reframe B (mechanism).** Do **not** chase a new mechanism theorem. The
  energy-budget diagnostic is still the right next experiment, but as
  *instantiation + measurement + boundary-layer explanation*, citing MZ:
  decompose the τ-lookahead `E_K` tendency into low-closed vs. MZ coupling
  `R`, and show `∫R` is sub-boundary-layer except on ~`D_witness` mass —
  explaining the measured 3.7% residual as the MZ-coupling boundary layer.
  Certified-empirical, not a closure discovery.
- **Elevate D (language).** The novelty lives here. The reviewer-facing
  claim should be stated as **functional observability of a decision event
  on the NSE attractor, below the determining-modes threshold** — with the
  twin-state certificate as the state-unobservability witness. Four precise
  hooks now available: functional observability, determining functionals
  (complement), below-AIM coarseness, MZ closure (mechanism).
- **Honest reviewer framing.** Pre-empt the "this is just LES/closure"
  reaction by leading with the certified non-injectivity (the decision is
  robust to *unreconstructable* state), which closure/AIM does not address.

## 4. Sources

- Functional observability: [PNAS 2022 / arXiv 2201.07256](https://arxiv.org/pdf/2201.07256); [nonlinear, arXiv 2301.04108](https://arxiv.org/pdf/2301.04108); [functional observers, arXiv 2501.00167](https://arxiv.org/pdf/2501.00167)
- Determining functionals / form: [Cockburn–Jones–Titi, Math. Comp. 1997](https://www.sciencedirect.com/science/article/pii/S0022247X11000175); [determining form, arXiv 1208.5134](https://arxiv.org/pdf/1208.5134)
- Approximate inertial manifolds: [Titi 1990, 2D NSE AIM](https://www.sciencedirect.com/science/article/pii/0022247X9290048I)
- Mori–Zwanzig closure: [Parish–Duraisamy MZ-LES, arXiv 1611.03311](https://arxiv.org/abs/1611.03311); [a priori memory estimation, arXiv 1611.06277](https://arxiv.org/pdf/1611.06277)

**Deep-read update (2026-05-29).** Confirmed the functional-observability
gap directly: Montanari–Motter (PNAS 2022) defines functional observability
for **linear, finite-dimensional, networked** systems and for **estimation**
(no PDE / fluid / attractor application; no control-decision selection); the
nonlinear extension (arXiv 2301.04108) generalizes to reconstructing a
nonlinear functional but its examples are finite chaotic/time-series systems
and it explicitly leaves **"whether functional observability conditions
scale to spatially-extended systems, or whether turbulent observables (e.g.
energy, enstrophy) satisfy the reconstructability framework"** as an open
question. So C1's setting — energy decision-observability on an NSE
attractor below the determining threshold, with a measured non-injectivity
certificate — is in genuinely open territory. The MZ-LES energy-budget
deep-read (whether the unresolved coupling is *small* on short horizons vs
merely *approximable* via memory) remains the registered follow-up, tied to
the energy-budget diagnostic (mechanism lane B), not the framing.
