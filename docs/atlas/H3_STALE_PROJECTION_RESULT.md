# H3-SP Result — GRACEFUL-COLLAPSE (K1): the stale reporter shrinks to the prior; no determinable-subspace lock (clean null = SUCCESS)

> **Run 2026-06-11 against the FROZEN pre-registration** (`H3_STALE_PROJECTION_PREREG.md`, frozen
> 2026-06-11 post-review B1–B6 + disclosed pilot). HS2 of slate 2026-06-10 — completes the
> pooled-shadow triptych (HS4 → (c) bounded-partial; HS3/H10 → clean kill; HS2 → this). Exact
> unchanged commands: `python scripts/shadow_stale_projection.py` ·
> `python scripts/test_shadow_stale_projection.py` (7/7). Published as a verification receipt (owner
> decision 2026-06-11; un-promoted; not peer-reviewed). **Language rule:** the
> "report" is a frozen regression head in a ground-truth synthetic substrate — a stale demodulator's
> output structure, not introspection/confabulation. Attribution: H3 v2; Nisbett & Wilson 1977;
> Turpin et al. 2023; HS4 probe conventions.

## Headline

**Pre-registered outcome K1 fired, 10/10 cells, both heads: the stale strategy degrades gracefully
toward mean-reporting.** Reporters trained to estimate `c` where it is accessible (λ ~ U[0, 0.5]; both
heads train-fit 0.983–0.985, λ=0 control 0.993–0.999, full-spread output Var(report)/Var(c) =
0.92–1.08), then frozen and deployed at λ = 2.0 (washed), **collapse their output spread ~9× — to
0.094–0.129 of Var(c)** (floor 0.25), rendering "I don't know" as numbers near the prior mean.

- **No determinable-subspace lock: d-partial ≤ 0.0034 in every cell.** The genuinely open question
  (the in-distribution discrete survivor, which the c-objective never incentivized) answered:
  nothing. The gain family is also tiny (≤ 0.064), and the g-counterfactual decomposition attributes
  essentially all of it to generic scale-sensitivity (share ≈ the g-partial, cell by cell) —
  confirming the review's foreordained-channel analysis even at this near-zero magnitude.
- **Residual weak coupling 0.152–0.223** (the pre-registered weakly-coupled band; no cell crossed the
  0.3 STALE-GENERALIZES bar): the accessible-regime demodulator transfers a little — consistent with
  the reviewer's intermodulation-component mechanism — but mostly dies with the fringe.
- **Architecture-independent:** both heads (linear and MLP) read CO in 5/5 seeds each; verdict D
  (head-dependence) did not fire.

## The audit-relevant reading (the headline bound, stated within the language rule)

In this substrate, a proxy-correlation audit **cannot** flag stale reports — there is no projection
signature to find (K1 ∧ d ≈ 0). The detectable staleness signature is **variance collapse itself**:
the same frozen reporter that spreads over the full latent range where access exists (0.92–1.08)
contracts ~9× where access is gone. An auditor watching output *spread* against a calibrated baseline
catches what an auditor watching output *correlations* cannot. Both halves are now banked: HS4 bounds
what probes recover from the body; HS2 bounds what report-structure audits recover from the output.

## Scorecard

| Gate / readout | Result | Value |
|---|---|---|
| C0 wash (modified generator, deploy draw) | PASS | −0.0100 (≤ 0.05) |
| MI gate g / mag (ridge & strong probe) | PASS | all ≤ −0.0009 (≤ 0.01); d bal-acc 0.499 (≤ 0.52) |
| Train-fit gate (10 cells) | PASS | 0.983–0.985 (≥ 0.7) |
| λ=0 control (10 cells) | PASS | 0.993–0.999 (≥ 0.9) |
| Coupling band | all weakly-coupled | 0.152–0.223 (none > 0.3 ⇒ no SG) |
| Structure floor | **all CO** | Var-ratio 0.094–0.129 (< 0.25) vs λ=0 baseline 0.92–1.08 |
| d-partial / gain-family | ≈ 0 | max 0.0034 / max 0.064 (g-cf share ≈ all of it) |
| Verdict | **K1** | modal CO 5/5 in BOTH heads |

Reproducibility: the first 7 cells of an earlier run (host killed externally mid-run) reproduce
**byte-identically** in the completed rerun — determinism verified across host restarts.

## Honest boundaries

- Synthetic substrate; a regression head, not a language-model self-report; the declared proxy set
  {d, g, mag} is what an auditor would have. The probe-vs-introspection gap is the standing limitation.
- K1 is the clean-null SUCCESS branch as pre-registered: it **bounds** report-audits (nothing to flag
  in correlations) rather than demonstrating a flag. The variance-collapse observation is banked as
  the constructive complement, with its obvious caveat: a reporter *trained with incentives to stay
  confident* might not collapse — that is the natural follow-on (stale + confidence-regularized), not
  claimed here.
- The weak residual coupling (0.15–0.22) means the deploy-shift produced *mostly*-stale, not perfectly
  stale, reporters at this λ; the K1 verdict is robust to this (collapse fired regardless), and the
  band was pre-registered.
- HS3/H10's lesson binds: existence of structure ≠ trainability of detection; no trained-auditor
  claims are made.

## Files
- `scripts/shadow_stale_projection.py` + `scripts/test_shadow_stale_projection.py` (7/7).
- `results/atlas/h3/stale_projection_result.json` (per-cell table, all partials, baselines,
  g-counterfactual shares, gates).
- `docs/atlas/H3_STALE_PROJECTION_PREREG.md` (FROZEN 2026-06-11).
- Slate: internal hypothesis slate 2026-06-10, HS2 (gitignored internal document).
