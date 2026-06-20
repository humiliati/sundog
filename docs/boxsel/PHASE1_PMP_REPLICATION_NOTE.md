# BoxSEL Phase-1 — PMP Replication Gate (result note)

**Date:** 2026-06-20  
**Gates:** §7 Phase 1 of [`../SUNDOG_V_BOXSEL.md`](../SUNDOG_V_BOXSEL.md). Closes the replication gate
that must clear before any published BoxSEL interval is treated as ground truth.  
**Artifacts:** `scripts/boxsel_pmp.py` (calculator) + `scripts/test_boxsel_pmp.py` (frozen test,
**21/21 PASS**, `python scripts/test_boxsel_pmp.py` → exit 0). Prior-art + confirmations in
[`BOXSEL_LITPASS_MEMO.md`](BOXSEL_LITPASS_MEMO.md) Track C.

> **Not an attack on the paper** (falsifier `BOXSEL-PMP-TYPO-AS-ATTACK`). This note locks the body
> formula and *characterizes* two real Appendix Algorithm 2 artifacts; their effect on the paper's
> **reported metrics** is unresolved here and needs the authors' evaluation code.

## What was checked

For point premises `(A|Q1)[q1]`, `(Q2 | A and Q1)[q2]`, the body Proposition 2 ("Probabilistic
Modus Ponens") bound on `(Q2|Q1)`:

```
lower = q1*q2,   upper = min(1, q1*q2 + 1 - q1)
```

### 1. Body formula — CONFIRMED

All four locked hand cases reproduce, plus the paper's toy *exact* interval:

| case | result |
| --- | --- |
| `q1=1, q2=r`      | `[r, r]`        (slack collapses; the q1=1 sanity check) |
| `q2=1, q1=r`      | `[r, 1]` |
| `q1=0, q2=r`      | `[0, 1]` |
| `q1=0.2, q2=0.8`  | `[0.16, 0.96]`  (= the paper's Example exact interval) |

The toy example's *sampled* `[0.43, 0.83]` is the min/max of two embedding estimates
(`17/40`, `50/60`); its *exact* `[0.16, 0.96]` is exactly this PMP bound at `q1=0.2, q2=0.8`, so the
calculator and the paper's worked example validate each other.

### 2. Algorithm 2 — slack typo (CONFIRMED, characterized)

Appendix Algorithm 2 line 16 prints `min(1, q1*q2 + 1 - q2)` (verified 2026-06-20 vs arXiv HTML
`2407.11821v2`). It diverges from the body: at `q1=0.2, q2=0.8` it returns `0.36` vs the correct
`0.96` (gap 0.60), and it **fails the `q1=1` sanity check** — it returns a vacuous `1.0` where the
answer must be `q2`. The `1 - q1` (escape-mass) slack is correct; `1 - q2` is the typo.

### 3. Algorithm 2 — premise-shape drift (CONFIRMED real; load-bearing)

Algorithm 2 line 13 declares the second premise `(Q2 | A)[q2]`, dropping the `and Q1` that
Proposition 2 requires (`(E | C and D)` = `(Q2 | A and Q1)`). A finite-model counterexample
(`premise_shape_counterexample()`) shows this is **not harmless in general**:

```
Q1 = {1..5},  A = {1,2,3,4, 6..11} (A not subset Q1),  Q2 = {1,2,3,4}
P(A|Q1) = 4/5,  P(Q2 | A and Q1) = 1,  P(Q2 | A) = 2/5,  true P(Q2|Q1) = 4/5
```

Plugging the marginal `P(Q2|A)=2/5` (Algorithm 2) into the *correct* slack form gives an upper
bound of `0.52` — **below** the true `0.80`, a soundness break. Restoring the correct premise
`P(Q2 | A and Q1)=1` gives `1.0 ≥ 0.80` (sound). The drift is harmless only when `A ⊆ Q1` (then
`A and Q1 = A`) or `Q2 ⊥ Q1 | A`. **Whether the paper's construction forces `A ⊆ Q1`** cannot be
settled here — the lit-pass found no public evaluation repository.

**Bug interaction (matters for the blast radius).** The two artifacts can *compensate*. On this
same counterexample, the as-printed Algorithm 2 — both bugs together — computes
`min(1, q1*q2 + 1 - q2) = 0.8·0.4 + 1 - 0.4 = 0.92 ≥ 0.80` (sound, if loose): the slack typo
*inflates* the bound and accidentally masks the premise-drift break. So the premise drift is a
soundness risk *in isolation* (correct slack), but proving the *shipped* Algorithm 2 is unsound on
some instance needs a search where **both** bugs survive together — a concrete Phase-2 task, not a
settled claim. This is exactly why the blast-radius codes stay held.

**UPDATE 2026-06-20 (Phase 2):** that search succeeded. `find_alg2_shipped_counterexample`
(`scripts/boxsel_exact_oracle.py`) exhibits `U={1,2,3}, Q1={1,2}, A={1,3}, Q2={1,2}`, where the
as-printed Algorithm 2 returns upper `q1·q2 + 1 − q2 = 3/4` while the true `P(Q2|Q1) = 1` — so the
shipped formula **is unsound on a finite model** (both bugs survive together, do not compensate).
This settles the *formula* question; the effect on the paper's **reported metrics** is still gated
on their evaluation code. See [`PHASE2_EXACT_MICRO_SEL_ORACLE_START.md`](PHASE2_EXACT_MICRO_SEL_ORACLE_START.md).

### 4. Sharpness footnote (no error)

The paper's general upper `min(1, u1*u2 + 1 - l1)` is **sound but not tight**; the sharp max over
models is `min(1, l1*u2 + 1 - l1)` (e.g. `l1=0.5,u1=1,l2=0,u2=0.5` → sharp `0.75` vs paper `1.0`).
They coincide for point premises, so the Phase-1 gate is unaffected; an interval-premise oracle
(Phase 2+) should prefer the sharp form or knowingly match the paper.

## Verdict

```
PMP_BODY_CONFIRMED              ← body Prop 2 form locked + tested (21/21); reproduces the toy exact interval.
PMP_PREMISE_SHAPE_UNRESOLVED    ← drift is REAL and load-bearing (counterexample banked); harmlessness
                                   depends on the authors' construction (A subset Q1?) — needs their code.
(blast radius)                  ← HELD. Neither PMP_ALG2_TYPO_NO_METRIC_BLAST_RADIUS nor
                                   PMP_ALG2_TYPO_AFFECTS_METRICS can fire: no public eval repo surfaced,
                                   so we cannot tell whether reported metrics use the printed Alg 2 or Prop 2.
```

## Disposition

- **Phase 1 clears its own gate.** The body formula is the trusted oracle for hand-derived PMP
  cases (Phase 2 will check the exact micro-SEL solver against it). The two Algorithm 2 artifacts
  are characterized, not weaponized.
- **Unblocks Phase 2** (exact micro-SEL oracle) — it must agree with these hand cases + the toy.
- **External item (owner-gated):** request the authors' evaluation code / correspond with the
  authors to settle the blast radius and whether the construction forces `A ⊆ Q1`. Until then the
  blast-radius codes stay HELD and no public statement about the paper's metrics is made.

---

*Sundog Research Lab — BoxSEL Phase-1 PMP replication note. Internal; gate result, not a metric claim.*
