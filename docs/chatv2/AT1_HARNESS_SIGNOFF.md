# AT-1 Harness Sign-Off

> 2026-07-02. Resolution of owner decision #5 in
> `NSE_ATTRACTOR_TAIL_HYPOTHESES.md`. Status:
> **`AT1_HARNESS_SIGNED_OFF`**. This is a harness-scope decision only: no
> verdict-bearing AT-1 run has been launched, and no AT-1 gates are frozen here.

## Decision

Owner sign-off is granted to touch `scripts/pde_c1_kolmogorov_cell.py` for the
AT-1 palinstrophy boundary-layer probe, but only as an additive export path.

The frozen C1 results and receipts are not reopened. Existing presets must keep
their old semantics when the new AT-1 export path is not requested.

## Authorized Harness Surface

Allowed changes:

- Add AT-1-only CLI flags and/or a new AT-1 export mode for per-sample emission
  from `lock_disc_g200` and, if the frozen spec requests it, `lock_disc_g300`.
- Emit a schema-versioned side artifact such as `at1-samples.npz` or
  `at1-samples.csv` containing sample index, step/time, `Phi_K`, objective
  value(s), registered threshold(s), margin-to-threshold, action label, and the
  objective name.
- Add post-processing code for the margin-band excision curve. This should read
  the emitted sample artifact rather than changing the simulator's old scoring
  path.
- Add a sibling-objective export only after the AT-1 frozen spec names it
  exactly: fixed-horizon palinstrophy value or lookahead-mean, not a post-read
  rescue objective.
- Add a small smoke/schema test under the repo's 10-minute rule.

Forbidden changes:

- No change to the existing integrator, forcing, timestep, random seed,
  burn-in/sampling schedule, K, G, or old discriminator objective definitions.
- No change to the existing `lock_disc_g200` / `lock_disc_g300` verdict logic
  in the no-export path.
- No post-read edits to the band grid, sibling objective, thresholds, or power
  gate.
- No reinterpretation of `PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md` or the
  banked C1 receipts.

## Regression Gate

Before any owner-run AT-1 lock run, the implementation must show:

- `python scripts/pde_c1_kolmogorov_cell.py --self-test` passes.
- A capped smoke with and without the AT-1 export path agrees on the existing
  manifest `config` and pre-existing result fields, ignoring timestamp,
  environment, elapsed-time, and file-list differences introduced solely by the
  new side artifact.
- The emitted artifact schema is written into the AT-1 frozen spec before the
  first full read.

If an old preset's no-export behavior drifts, file `AT1_HARNESS_VOID` and do
not interpret the AT-1 run.

## Next Artifact

The next document should be the frozen run spec, tentatively
`AT1_PALINSTROPHY_BOUNDARY_LAYER_SPEC.md`, with:

- exact commands for the capped smoke and owner-run lock;
- exact output paths;
- the registered margin-band grid;
- the sibling objective, if used;
- the branch table for `AT1_BOUNDARY_LAYER_ARTIFACT`,
  `AT1_TWO_POLE_CONFIRMED`, `AT1_UNDERPOWERED`, and `AT1_NEG_B`.

This sign-off authorizes the harness work needed to write that spec. It does
not promote AT-1, edit public surfaces, or claim anything about infinite-
dimensional NSE.
