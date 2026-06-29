#!/usr/bin/env python3
r"""C4 - the order-relative law on a 4th KIND of axis: find/check (verify-mode vs predict-mode).

The slate's order-relative law held on three axes that each assign ONE order to a target -- a bounded
process with budget k resolves iff target-order <= k: determination (sigma-order), search reach (C1),
pressure reach/repertoire (C2 / Phase 7g). This tests a 4th KIND, the find/check ledger (suffstat H3),
where the SAME target carries TWO orders at once:

  verify-order  : the witness-check budget -- ops to CONFIRM the answer given a structural witness.
                  parity lambda(n): Omega(n) given the factorization -> FINITE (<= log2 n).
  predict-order : the history budget that DETERMINES the next value.
                  parity from its lambda-history: no finite order -> sigma_predict = infinity.

Pre-registered (break-first; both outcomes live):
  EXTENDS_mode_relative : parity has verify-order FINITE and predict-order INFINITE -- the two modes
                          give DIFFERENT orders for ONE target => no single scalar budget governs; the
                          law holds PER-MODE only. A finite-sigma control (LFSR) has BOTH orders finite
                          (modes agree) and the predict-meter detects it (the liveness guard).
  SINGLE_BUDGET         : parity verify-order == predict-order (one scalar even here).  [predicted false]
  VOID                  : the predict-meter cannot detect the finite control (dead apparatus).

The 4th-kind finding: on find/check the "order" is a MODE-VECTOR, not a scalar -- the find/check analog
of the sigma-schema's ">=6 filtrations, not one comparable scalar." And it EXPLAINS C2: a pressure
detector's "stable under the pressures I applied" is a VERIFY-mode signal (finite budget); soundness
("genuinely closed") is a PREDICT-mode property (order = predict-order = infinity for sigma=inf
closures). C2's break IS that mode-confusion -- substituting the finite verify-budget for the infinite
predict-order. Reuses suffstat_h3_verify_vs_predict + order_meter. Exact / deterministic.
"""
import numpy as np

import suffstat_h3_verify_vs_predict as h3   # verify_lambda, liouville, predict_order_from_history
import order_meter as om                     # lfsr (finite-sigma control)

N = 200_000
PREDICT_K = 16          # history-order ladder for the exact predict-meter
LFSR_TAP = 5            # finite-sigma control: b[n] = b[n-1] ^ b[n-LFSR_TAP]
INF = float("inf")


def max_omega(limit: int) -> int:
    """max Omega(n) over 2..limit = worst-case VERIFY witness-check cost (prime factors w/ multiplicity)."""
    omega = np.zeros(limit + 1, dtype=np.int16)
    comp = np.zeros(limit + 1, dtype=bool)
    for p in range(2, limit + 1):
        if not comp[p]:
            comp[p * p::p] = True
            pe = p
            while pe <= limit:
                omega[pe::pe] += 1
                pe *= p
    return int(omega[2:].max())


def mean_omega(limit: int) -> float:
    omega = np.zeros(limit + 1, dtype=np.int16)
    comp = np.zeros(limit + 1, dtype=bool)
    for p in range(2, limit + 1):
        if not comp[p]:
            comp[p * p::p] = True
            pe = p
            while pe <= limit:
                omega[pe::pe] += 1
                pe *= p
    return float(omega[2:].mean())


def predict_order(seq) -> float:
    """least history order k <= PREDICT_K determining the next symbol; INF if none (sigma_predict)."""
    k = h3.predict_order_from_history(np.asarray(seq, dtype=np.int8), PREDICT_K)
    return INF if k is None else float(k)


def find_check_report() -> dict:
    # --- RESIST target: parity / Liouville lambda ---
    lam = h3.liouville(N)
    parity_verify = float(max_omega(N))          # finite, ~ log2 N
    parity_predict = predict_order(lam)          # INF (sigma = infinity)

    # --- finite-sigma control: an LFSR (modes should AGREE, both finite) ---
    lf = om.lfsr(LFSR_TAP, N)
    lfsr_verify = float(LFSR_TAP)                 # state-window witness; one XOR to check
    lfsr_predict = predict_order(lf)              # finite (the predict-meter detects it)

    parity_diverges = parity_verify < INF and parity_predict == INF      # finite vs infinite
    control_agrees = lfsr_verify < INF and lfsr_predict < INF            # both finite
    meter_live = lfsr_predict < INF                                      # predict-meter detects a finite order

    # per-mode law: a budget >= the mode's order resolves; below it does not.
    verify_law = parity_verify < INF             # a finite verify-budget confirms parity
    predict_law = parity_predict == INF          # no finite predict-budget (<= K) resolves parity (resist)

    extends = parity_diverges and control_agrees and meter_live and verify_law and predict_law
    if not meter_live:
        verdict = "VOID_dead_meter"
    elif not parity_diverges:
        verdict = "SINGLE_BUDGET"
    else:
        verdict = "EXTENDS_mode_relative"

    # the C2 mode-confusion gap: soundness lives at predict-order, a bounded detector has verify-order
    mode_confusion_gap = parity_predict == INF and parity_verify < INF

    return {
        "N": N,
        "predictK": PREDICT_K,
        "parity": {"verifyOrder": parity_verify, "predictOrder": parity_predict,
                   "meanVerify": round(mean_omega(N), 3)},
        "control_lfsr": {"tap": LFSR_TAP, "verifyOrder": lfsr_verify, "predictOrder": lfsr_predict},
        "parityDiverges": parity_diverges,
        "controlAgrees": control_agrees,
        "meterLive": meter_live,
        "verifyLawHolds": verify_law,
        "predictLawHolds": predict_law,
        "modeConfusionGap": mode_confusion_gap,
        "extends": extends,
        "verdict": verdict,
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    r = find_check_report()
    fmt = lambda x: "inf" if x == INF else (f"{x:.0f}" if float(x).is_integer() else f"{x}")
    print("C4 - find/check axis: is the order-relative law mode-relative? (N =", r["N"], ")")
    print()
    print("  target            verify-order (witness)   predict-order (history)")
    p, c = r["parity"], r["control_lfsr"]
    print(f"  parity lambda     {fmt(p['verifyOrder']):>10}  (max Omega; mean {p['meanVerify']})   "
          f"{fmt(p['predictOrder']):>6}  (sigma_predict)")
    print(f"  LFSR({c['tap']}) control  {fmt(c['verifyOrder']):>10}  (state window)            "
          f"{fmt(c['predictOrder']):>6}")
    print()
    print(f"  parity modes DIVERGE (verify finite, predict inf): {r['parityDiverges']}")
    print(f"  control modes AGREE (both finite): {r['controlAgrees']}   meter live: {r['meterLive']}")
    print(f"  VERDICT: {r['verdict']}  (extends = {r['extends']})")
    if r["modeConfusionGap"]:
        print()
        print("  => 4th KIND: on find/check the 'order' is a MODE-VECTOR, not a scalar. The law holds")
        print("     per-mode (finite verify-budget confirms parity; no finite predict-budget does), but")
        print("     the two orders DIVERGE for one target -> no single budget. This EXPLAINS C2: a")
        print("     pressure detector's 'stable under applied pressure' is a VERIFY-mode signal (finite),")
        print("     while soundness is a PREDICT-mode property (predict-order = inf). C2's break = that")
        print("     mode-confusion.")
