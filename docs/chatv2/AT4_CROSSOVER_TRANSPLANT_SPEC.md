# AT-4 — Crossover Transplant Spec (frozen; ledger vs order-blind surface, one stream)

> 2026-07-04. Lift of slate entry AT-4 into its frozen pre-registration. **The F3
> unification, instantiated: one observation stream; the surface = its order-blind window
> statistics; the carrier = AT-3's maintained ledger; the slice = the decision-ambiguous
> band.** All prior mandates wired: AT-6 (SNR-aware surface: quantized readouts; label
> frequency inside the window grid), AT-1 (margin-band idiom; thresholds inherited),
> AT-3 (carrier = the confirmed split cell; scrambled ledger = the order control),
> V3-0b/A1 (slice + liveness + order-shuffle discipline). Relative margins only (F1).
> New script, read-only imports; no harness change. Non-promotional; licensed grammar:
> "the maintained ledger reads the decision better than the registered surface statistic
> allows, on the decision-ambiguous slice of this cell" — nothing more.

## 1. Objects (frozen)

- **Cell & carrier:** G=200 (G=300 labels are degenerate per AT-3 — excluded, recorded);
  truth u (seed 0, burn-in 100k, calibration-first ordering, 500k-step window — AT-3 v1.1
  numerics verbatim). **Carrier = the AT-3 ledger at (K_obs=1, μ=10)** — the confirmed
  split cell (sub-sync, decision-carrying). **Order control = the AT-3 scrambled ledger**
  at the same cell (seed 2). **Ceiling = Φ_K3(u)** (truth signature read).
- **The stream (what the surface sees = what the ledger sees):** the observed mode's
  complex coefficient per step (select(1) = the forced mode; Re/Im recorded full-rate).
- **Label (primary):** the frozen J_q(τ=500) action of the truth (lookahead-max E_low_K3
  > q=0.70 calibration threshold) at eval instants (every 50 steps, post-transient 25%,
  contiguous 70/30 split, 2,500-step gap, seed 0 — AT-3's readout protocol verbatim).
- **Slice (decision-ambiguous, the AT-1 band idiom):** eval instants with |m − e_max| ≤
  the 30th percentile of |margin| over post-transient eval instants (mass 0.30 by
  construction; conditions the *comparison*, symmetrically for all readers).
  **N_min = 800** slice-test... slice total ≥ 800; bulk (no-slice) numbers reported for
  contrast, not gated.
- **Surface family (frozen; AT-6-mandated):** trailing windows [s−W, s] of the stream,
  **W ∈ {250, 500, 1000}** (label horizon 500 sits inside the grid). Probes per W:
  (a) **moments arm** (declared no-noise-model): mean/std/min/max/abs-mean of Re & Im
  (10 features); (b) **quantile arm**: deciles of Re & Im (20); (c) **SNR-aware gram
  arm**: symbols = 8 calibration-quantile bins of Re (Im separately reported),
  w-gram counts, w ∈ {1, 2, 4, 8}, hashed to ≤ 4,096 features. Each probe = logistic,
  same split. **surface_max = max accuracy over ALL probes × ALL W** (the surface gets
  its strongest shot).
- **Liveness axis (bag-determined by construction):** window-mean(Re) > calibration
  median at matched W=500 — the surface's moment arm must read it ≥ 0.95 **on-slice**,
  else `AT4_DEAD_APPARATUS` (void, fix, re-run).

## 2. Branch table (frozen; δ = 0.10, all on the slice)

| branch | fires iff |
| --- | --- |
| `AT4_CROSSOVER_CONFIRMED` | acc_ledger ≥ surface_max + 0.10 ∧ acc_ledger ≥ acc_scrambled + 0.10 ∧ liveness ≥ 0.95 ∧ slice N ≥ 800 |
| `AT4_SURFACE_SUFFICIENT` | surface_max ≥ acc_ledger − 0.05 — the label is window-statistic-determined on-slice; recorded, feeds the AT-6 table, not a failure |
| `AT4_SLICE_THIN` | slice N < 800 after construction — the V3-0b outcome, pre-planned: report mass + skew as the finding |
| `AT4_DEAD_APPARATUS` | liveness fails on-slice — void |
| `AT4_NEG_B` | more than one same-day re-registration of slice or surface family — voids (no verdict-shopping) |

Reported (non-gate): bulk-vs-slice contrast for every reader; the ceiling Φ_K3(u);
surface order-sanity (probes on permuted windows ≈ unchanged — order-blind by
construction); per-probe table so the AT-6 taxonomy gains rows either way.

## 3. Command & deliverables

`python scripts/at4_crossover_transplant.py --out results/proof/at4-g200`
(≈ 20–30 min: truth series + 2 ledger configs + probes; agent-run, background).
Receipt: `AT4_CROSSOVER_TRANSPLANT_RECEIPT.md` — the slice table (ledger / scrambled /
ceiling / per-probe surface × W), bulk contrast, branch verdict, and the AT-6 taxonomy
feedback rows.

## 4. Does not claim (inherited)

Nothing about the ∞-dim attractor; no "world model"; the relay-form mechanism note from
AT-3 carries over (the carrier's advantage may be relay + temporal pairing — the
crossover gate does not distinguish relay from emergent computation and the receipt will
not either); slice conditioning is label-adjacent and symmetric across readers (declared);
G=300 exclusion is a power fact, not a result; the slate do-not-say list binds in full.
