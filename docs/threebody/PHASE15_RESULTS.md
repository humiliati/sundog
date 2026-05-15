# Three-Body Phase 15 - Forward-Oracle / Precision Lock Results

Status: pre-registered and lock-reviewed on 2026-05-15. No Phase 15 harness code
has been written, no smoke has run, and no full-lock result exists yet.

Phase 15 is intentionally stopped at the spec lock. The next implementation pass
must add the forward-oracle strict mode and precision receipts behind the
Phase 15 flags, rerun the exact Phase 13 and Phase 14 regression gates
bit-for-bit, then run only `npm run threebody:phase15:smoke`.

Before the full lock is interpreted, record:

- the exact smoke command and transcript
- `D_smoke`, the maximum passive/off `finalRelEnergyDrift` observed in the smoke
- the pre-registered full-lock drift bound `2 * D_smoke`
- whether the smoke supports starting the staged multi-hour full lock unchanged

The full lock remains pending until that smoke readback is recorded here.
