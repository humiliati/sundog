#!/usr/bin/env python
"""Frozen test for BoxSEL Phase-3 ordinary restart sampler.

Locks the observed layer I_sample for the Helly-seed query: zero-loss ordinary restarts sit inside
I* and I_box, but their sampled lower endpoint stays well above the exact box lower endpoint.
Run: python scripts/test_boxsel_phase3_restart_sampler.py
"""

import sys

sys.path.insert(0, "scripts")
import boxsel_phase3_restart_sampler as sampler
import boxsel_phase4k_dimension_compression as closed

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("(1) a single ordinary restart is zero-loss feasible:")
single = sampler.ordinary_restart_report(dim=2, restarts=1, seed=101)
trace = single.traces[0]
check("one accepted trace is returned", single.accepted == 1 and trace.attempts >= 1)
check("atom volumes are all 1/2 up to float tolerance",
      all(abs(v - sampler.ATOM_TARGET) < 1e-12 for v in trace.atom_volumes),
      f"{trace.atom_volumes}")
check("pair overlaps satisfy the seed lower bounds",
      all(v >= sampler.PAIR_TARGET - 1e-12 for v in trace.pair_overlaps),
      f"{trace.pair_overlaps}")
check("loss is below tolerance and q is a probability",
      trace.loss <= sampler.DEFAULT_TOLERANCE and 0.0 <= trace.q <= 1.0,
      f"loss={trace.loss}, q={trace.q}")

print("(2) deterministic I_sample report quantifies a lower search gap:")
report = sampler.ordinary_restart_report(dim=2, restarts=128, seed=314159)
lo, hi = report.sample_interval
box_lo, box_hi = report.exact_box_interval
check("all requested restarts are accepted", report.accepted == report.restarts == 128)
check("sample interval is nested inside exact I*=[0,1]",
      report.exact_interval == (0.0, 1.0) and 0.0 <= lo <= hi <= 1.0,
      f"I_sample={report.sample_interval}")
check("sample interval is nested inside exact I_box^2",
      box_lo <= lo <= hi <= box_hi,
      f"I_sample={report.sample_interval}, I_box={report.exact_box_interval}")
check("ordinary restarts miss the exact lower endpoint by a visible margin",
      report.lower_search_gap > 0.08,
      f"gap={report.lower_search_gap:.6f}, sample_lo={lo:.6f}, exact_box_lo={box_lo:.6f}")
check("all sampled losses are zero-ish and constraints have nonnegative slack",
      report.max_loss <= sampler.DEFAULT_TOLERANCE and report.min_slack >= -1e-12,
      f"max_loss={report.max_loss}, min_slack={report.min_slack}")

print("(3) cumulative endpoint trace has the right monotonicity:")
movement = sampler.cumulative_endpoint_trace(report)
check("endpoint trace has one row per restart", len(movement) == report.restarts)
check("lower endpoint only moves downward as restarts accumulate",
      all(movement[i][1] <= movement[i - 1][1] + 1e-15 for i in range(1, len(movement))))
check("upper endpoint only moves upward as restarts accumulate",
      all(movement[i][2] >= movement[i - 1][2] - 1e-15 for i in range(1, len(movement))))
check("final cumulative endpoint equals report interval",
      abs(movement[-1][1] - lo) < 1e-15 and abs(movement[-1][2] - hi) < 1e-15)

print("(4) seed variance is measured and still misses the exact lower endpoint:")
variance = sampler.seed_variance_report(dim=2, restarts=64, seeds=(11, 23, 37, 53))
lows = [r.sample_interval[0] for r in variance]
check("variance report returns one report per seed",
      len(variance) == 4 and all(r.accepted == 64 for r in variance))
check("every seed's sampled lower endpoint stays above exact I_box lower",
      all(low > float(closed.exact_global_infimum()) for low in lows),
      f"lows={[round(x, 6) for x in lows]}")
check("different seeds produce non-identical sampled lower endpoints",
      len({round(low, 12) for low in lows}) > 1,
      f"lows={[round(x, 6) for x in lows]}")

print("(5) dimension and exact-anchor sanity:")
report3 = sampler.ordinary_restart_report(dim=3, restarts=96, seed=271828)
check("dim=3 sampler is also zero-loss and misses the exact lower endpoint",
      report3.max_loss <= sampler.DEFAULT_TOLERANCE
      and report3.lower_search_gap > 0.08,
      f"dim3 gap={report3.lower_search_gap:.6f}, interval={report3.sample_interval}")
anchor = sampler.rational_witness_summary()
check("sampler is anchored to the exact Phase-4 endpoint, not the old rational witness",
      anchor["exact_box_lower"] == float(closed.exact_global_infimum())
      and anchor["rational_witness_q"] > anchor["exact_box_lower"])
check("summary exposes the needed Phase-6 trace fields",
      set(sampler.sampler_summary(report)).issuperset(
          {"sample_interval", "exact_box_interval", "lower_search_gap", "max_loss", "min_slack"}
      ))

print(f"\n{'ALL PASS -- Phase-3 restart sampler: zero-loss I_sample observed; sampled lower endpoint visibly above exact I_box lower' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
