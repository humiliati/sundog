# NSE-H2 — N-Refinement Receipt (R1–R4, owner-run 2026-07-06)

> Lock runs of `NSE_H2_V7_N48_SPEC.md` (frozen pre-run). Artifacts
> `results/proof/c1-h2-g200-n48-{knn-sweep,twin}/`; the two G=300 runs produced
> **no verdict artifact** (solver blow-up before adjudication — see §3).
> **Non-promotional.** No gate was widened; no constant retuned after any read.

## 1. Run table

| run | verdict | wall-clock |
| --- | --- | --- |
| R1 `lock_v7_g200_n48` kNN-sweep | **`STRICTNESS_WITNESS_POSITIVE`** | 3,242 s (54 min; 1,575 steps/s) |
| R2 `lock_v7_g200_n48` twin-state | **`TWIN_STATE_CERTIFIED`** + `PAIRED_FIBER_CONSTANCY_POSITIVE` | 3,131 s (52 min) |
| R3 `lock_v7_g300_n48` kNN-sweep | **no verdict** — overflow/NaN during integration; adjudicator rejected NaN signatures | 3,351 s wasted |
| R4 `lock_v7_g300_n48` twin-state | **no verdict** — same blow-up | 4,172 s wasted |

## 2. G=200 comparability report (banked grid-32 vs measured n48)

| quantity | grid 32 (banked) | grid 48 (measured) |
| --- | --- | --- |
| portability damp (gate [0.20,0.40]) | 0.3003 | **0.30004** (calib 0.300) |
| E_max (portable, held-out) | 0.7344 | **0.734395** |
| ε_K (re-derived by rule) | 0.060598 | **0.060597** |
| kNN a_mm (gate ≤0.005) | −0.00079 | **+0.000168** |
| kNN slope | 0.736 | 0.707 |
| twin unique witness pairs | 693,795 | **689,250** |
| fiber disagree fraction (unique) | 0.0367 | 0.0376 |
| δ_H (re-derived by rule) | 0.011667 | **0.011667** |
| high-mode complement dimension | ~422 | **1,070** |

Same verdict labels under identical gates — and quantitatively near-invariant:
E_max and ε_K reproduce to 4–5 decimals, δ_H to 6, twin pair counts within 0.7%,
while the state complement the signature must *not* determine grew 2.5×. The
scope's pair-count-drop caveat (the `lock_v5_n48` 25,979 precedent) did not
recur; the v7 protocol reproduces grid-32 pair counts nearly exactly.

## 3. G=300: solver blow-up, typed

Overflow warnings appeared **before the first progress print** (within the first
~510k steps — the burn-in / early-calibration transient), then NaN propagated
through the full 5.1M-step integration and both adjudicators rejected non-finite
signatures. Gate 1 (solver diagnostics) fails; **no verdict field was ever
produced or read.**

**Diagnosis — explicit-advection CFL boundary, probe-confirmed (non-verdict
apparatus check, `scripts/nse_h2_g300n48_dt_probe.py`):** at G=300/grid-48,
`dt = 0.01` goes non-finite at **step 300** (the initial transient); `dt = 0.005`
is **stable through the full 600,000-step soak** (3,000 time units; E_low bounded
by the initial transient peak 4.686, turbulent envelope 0.84–1.34). A diagnosis
aid, not a certificate for the 10.2M-step lock — the lock itself is the soak.
Consistent with the geometry: dx shrinks 1.5× at grid
48 while G=300 carries faster velocities — G=300/32 and G=200/48 sit inside the
stability region, G=300/48 sits outside. This is a *numerics-of-the-probe* fact,
not a flow fact.

## 4. Verdict (per the frozen table)

- **`NSE-H2-V7-N48-STABLE`** — the G=200 v7 portable witness survives the
  registered N-refinement (R1+R2, all gates).
- **`NSE-H2-NUMERIC-WALL(g300)`** — diagnostics prevented a fair G=300 read
  (the slate's pre-registered expected wall; not a rescue prompt).

Licensed claim exactly as the slate words it, restricted to G=200: the witness is
now **two-objective, two-resolution at G=200** (v5+v7 objectives × grid 32+48),
still finite, numerical, and proxy-relative (H1's `NSE-H1-PROXY-ONLY` untouched).
The two-regime refinement lift remains open at the wall.

## 5. Owner fork (staged, not spent)

1. **Commission a bounded v1.2 solver-stability amendment** (recommended;
   probe-backed): new preset variant `lock_v7_g300_n48_dt5` with **one physical
   change — dt 0.01 → 0.005** — and the four step-count constants rescaled to
   hold every *physical* quantity fixed (burn-in 200k steps, sample interval 100,
   lookahead 1,000 steps = the same τ = 5.0 time units, calibration gap 10k;
   sample counts unchanged). Cost ≈ 10.2M steps ≈ 2–2.3 h per run × 2 runs,
   owner-run. Pre-committed stop: if it also fails diagnostics, `NUMERIC-WALL`
   stands final. Needs the same 3-site harness sign-off as §7 of the scope.
2. **Close H2 here** at `V7-N48-STABLE` + `NUMERIC-WALL(g300)` — a complete,
   typed outcome; the slate explicitly licenses this close.

Also noted (optional, separate sign-off, no urgency): the harness integrated
5.1M steps of NaNs twice (~2 h wasted) before failing at adjudication — an
additive early-abort diagnostic (non-finite E_low check during integration)
would have stopped both runs within seconds. Not applied; owner-gated.

Cross-refs: `NSE_H2_V7_N48_SPEC.md`, `NSE_H2_V7_N48_SCOPE.md`,
`PDE_C1_REGIME_GENERALITY_v1.md` §12 (targets),
`results/proof/c1-h2-g200-n48-{knn-sweep,twin}/manifest.json`,
`NSE_H1_FIBER_RECEIPT.md` (proxy-relative typing carried),
`NSE_STATIONARITY_GATE_CHECKLIST.md`.
