# Non-Sovereignty Task Family

Status: **release smoke green / Apache-2.0 licensed**. This directory is the public
entry point for the non-sovereignty benchmark used by the paper spine
[`docs/NON_SOVEREIGNTY_PAPER_SPINE.md`](../../docs/NON_SOVEREIGNTY_PAPER_SPINE.md).

The task family tests a narrow claim:

> In this controlled continuous-control environment, the safety gains attributed
> to role separation are explained by a structural authority bound around a
> frozen presider channel. A capped no-role controller matches the role-separated
> controller on the registered axes.

The benchmark includes:

- a forked field/reward navigation task;
- a costly shutdown channel and `Corr_k` metric;
- an arbiter-authority cap and full-episode `Sov_opt` audit;
- an endogenous review-band shutdown signal for safe-interruptibility;
- matched uncapped, capped no-role, and capped role-separated controls.

Start with:

```powershell
npm install
npm run mesa:ns:release-smoke
```

The smoke verifies manifest integrity, JS/Python parity, fixed-control admission,
cap validity, and NS-2 table regeneration from recorded artifacts. Long PPO
bindings are staged in [REPRODUCE.md](REPRODUCE.md); they are not CI tasks.
First green local smoke: **PASS** in 3.129 s. Fresh runtimes are written to
`results/mesa/non-sovereignty/release_smoke/summary.json`.

Scope boundary: this is a Small/Medium in-vitro task family, not a foundation
model result and not a deployed-agent safety claim.

## License & citation

The files enumerated in [`MANIFEST.json`](MANIFEST.json) are released under the
**Apache License 2.0** ([`LICENSE`](LICENSE), [`LICENSE.md`](LICENSE.md));
copyright 2026 Stellar Aqua LLC. The rest of the repository remains `UNLICENSED`.
If you use this benchmark, please cite it via [`CITATION.cff`](CITATION.cff).
