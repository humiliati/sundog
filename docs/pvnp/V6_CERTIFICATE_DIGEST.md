# P-vs-NP Verifier — v6 Certificate Digest

A one-page digest of the Sundog P-vs-NP verifier arc, for readers who want the
shape before the receipts. Every number here is copied from a filed receipt
(cited inline). This is a communication note, not a result, and it makes no new
claim.

> **What this is not.** Bounded alignment-verification that *borrows* the
> finding-vs-checking asymmetry as vocabulary. **Not** the Millennium P-vs-NP
> problem, not a complexity-theoretic theorem, not a claim that wall-time
> verification is cheap. "Cheap to check" means **operation-count**, in a toy /
> mesa-local envelope, against a registered comparator.

## The certificate in one line

> Recompute a few **reward-blind** response summaries from raw rollout logs and
> accept iff the signature response clears a frozen line. Checking costs fewer
> operations than finding did.

## The number that earned "bounded positive" (v6)

From [`receipts/2026-05-31_phase1_toy_verifier_v6.md`](receipts/2026-05-31_phase1_toy_verifier_v6.md):

| Quantity | Value |
| --- | ---: |
| finding — `C_rollout_ops` | 555,876 |
| checking — `C_total_signature_ops` | 527,297 |
| **check / find ratio** | **0.948587**  (gate: ≤ 1.0 → pass) |
| ratio, calibration-inflation removed | ~0.879 |
| false accepts | 0 / 2304 |
| field-spoof / source-spoof | 0 / 453 · 0 / 453 |
| integrity probes | 5 / 5 |
| out-of-promise accepts | 0 / 768 |

The one honest asterisk that makes it real: this is **op-count**. The same lane's
wall-time "cheap" claim swung 3.5× across runs and was **withdrawn**. The
asymmetry holds only in the metric that reproduced.

## The arc (why a cheap certificate is a target)

| Stage | What happened | Verdict |
| --- | --- | --- |
| **v6** | Cheap-to-check certificate, safety-complete in the toy envelope. | **bounded positive** |
| **v0** | Same certificate, mesa bridge. A registered unsafe controller (capacity-breach, λ=0.7) replayed on 4 seed blocks; one block's mean drifted to signature **0.23545148 ≥ 0.23** and the verifier accepted it — a spoof at **6.25%** of the attacker budget. | **falsified** |
| **v1** | Repair: accept only on **3-of-4** block consensus. The single-block spoof (1 of 4) no longer promotes — the v0 break is closed. But the *same drift* hides in the disclosure flag of a **protected** cell (observation straddles 0.5, flag fires 2 of 4 < 3), so the run still can't promote. | **named quarantine** (consensus-only repair) |

The twist is the lesson: **a single block is a coin a drifting mean can flip.**
Consensus fixes that for the accept decision, and the same fragility reappears one
layer over in the disclosure flag. v0 source:
[`receipts/2026-05-31_phase3_capacity_one_wayness_v0.md`](receipts/2026-05-31_phase3_capacity_one_wayness_v0.md);
v1 source:
[`receipts/2026-06-01_phase3_capacity_one_wayness_v1.md`](receipts/2026-06-01_phase3_capacity_one_wayness_v1.md).

## See it move

```bash
node scripts/pvnp-verifier-spoofer-demo.mjs
```

A terminal replay of the three acts: the cheap check, the seed-block that drifts
over the accept line, and the consensus rule that holds the unsafe line while the
disclosure flag flickers. Deterministic, no new computation — it replays the
filed numbers so the asymmetry is tactile.

## Boundary, restated

Op-count bounded, safety-complete in the toy envelope, wall-time diagnostic-only,
mesa-local. The body-resistance boundary holds (mesa is marginal:
`FVE(net.7 | 5D) ~ 0.97–0.99`), so this is certificate-discipline transfer to a
control substrate, not signature verification where a body genuinely resists its
shadow — and not progress on P vs NP.
