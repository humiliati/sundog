#!/usr/bin/env python3
"""Sufficient-statistic-order slate, H4 (internal): the order-meter as a reusable
APPARATUS-LIVENESS / faithfulness instrument.

Generalizes the ad-hoc P-2 control rig into one helper. Any "the model can't see X"
null is only meaningful if the probe CAN see a known-order positive control. The
order-meter classifies a claimed null into exactly one of:

  INVISIBLE     — controls detected, target not  -> real null, apparatus live
  DEAD APPARATUS— controls NOT detected          -> probe broken; the null is uninformative
  NOT A NULL    — target itself detected          -> investigate / known-bias check

Hardening claim (falsifiable): the meter discriminates all three on KNOWN ground
truth. Self-test below builds one scenario per verdict and checks the meter calls it.
Run: python scripts/order_meter.py
"""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


def own_r2(X, y):
    """Faithful copy of epsilon_machine_shadow.own_r2: 4-fold CV max(0, linear, MLP)."""
    X = np.asarray(X, float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    kf = KFold(4, shuffle=True, random_state=0)
    lin = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean()
    mlp = cross_val_score(MLPRegressor(hidden_layer_sizes=(24,), max_iter=300, random_state=0),
                          X, y, cv=kf, scoring="r2").mean()
    return max(0.0, lin, mlp)


# ---- the reusable instrument ------------------------------------------------
def order_meter(probe, target, controls, thr=0.10):
    """probe: (X,y)->detection score in [0,1].  target: (X,y).  controls: list of
    (name, (X,y)) known-detectable signals.  Returns (verdict, scores)."""
    ctrl_scores = {name: probe(X, y) for name, (X, y) in controls}
    tgt = probe(*target)
    all_controls_live = all(s > thr for s in ctrl_scores.values())
    if not all_controls_live:
        verdict = "DEAD APPARATUS"
    elif tgt > thr:
        verdict = "NOT A NULL (target detected)"
    else:
        verdict = "INVISIBLE (real null, apparatus live)"
    return verdict, {"target": tgt, **ctrl_scores}


def order_meter_from_scores(target_score, control_scores, thr=0.5):
    """When the probe has already run elsewhere: classify a null from precomputed
    detection scores (control_scores: name->score). Same three-way verdict. Lets the
    meter audit a paused lane's *recorded* numbers without re-running its substrate."""
    if not all(s > thr for s in control_scores.values()):
        return "DEAD APPARATUS"
    return "NOT A NULL (target detected)" if target_score > thr else "INVISIBLE (real null, apparatus live)"


# ---- signal sources ---------------------------------------------------------
def lfsr(d, N, seed=0xACE):
    b = np.zeros(N + 1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    b[1:d + 1] = rng.integers(0, 2, size=d)
    for n in range(d + 1, N + 1):
        b[n] = b[n - 1] ^ b[n - d]
    return (1 - 2 * b).astype(float)


def residue(q, N):
    n = np.arange(N + 1)
    return np.where(n % q == 0, 1.0, -1.0)


def liouville(N):
    omega = np.zeros(N + 1, dtype=np.int8)
    comp = np.zeros(N + 1, dtype=bool)
    for p in range(2, N + 1):
        if not comp[p]:
            comp[p * p::p] = True
            pe = p
            while pe <= N:
                omega[pe::pe] += 1
                pe *= p
    return (1 - 2 * (omega & 1)).astype(float)


def history_xy(seq, K, M, rng):
    pos = rng.integers(K + 1, len(seq), size=M)
    X = np.column_stack([seq[pos - j] for j in range(1, K + 1)])
    return X, seq[pos]


# ---- self-test: one scenario per verdict, KNOWN ground truth ----------------
def main():
    K, M = 8, 3000
    rng = lambda s: np.random.default_rng(s)
    lam = liouville(200_000)
    lf5 = lfsr(5, 200_000)
    res3 = residue(3, 200_000)
    lf3 = lfsr(3, 200_000)

    controls = [("lfsr5", history_xy(lf5, K, M, rng(1))),
                ("residue3", history_xy(res3, K, M, rng(2)))]

    print("ORDER_METER self-test (one scenario per verdict; ground truth in brackets)\n")

    # A. real invisible: target = Liouville (genuinely finite-order-unpredictable)
    vA, sA = order_meter(own_r2, history_xy(lam, K, M, rng(3)), controls)
    print(f"  A [truth=INVISIBLE]  target=Liouville      -> {vA}")
    print(f"       scores: {{ {', '.join(f'{k}={v:.3f}' for k,v in sA.items())} }}")

    # B. dead apparatus: same data, a BROKEN probe (always 0)
    broken = lambda X, y: 0.0
    vB, sB = order_meter(broken, history_xy(lam, K, M, rng(3)), controls)
    print(f"  B [truth=DEAD]       probe=broken(returns 0)-> {vB}")
    print(f"       scores: {{ {', '.join(f'{k}={v:.3f}' for k,v in sB.items())} }}")

    # C. not a null: target is actually an order-5 signal mislabeled "invisible"
    controlsC = [("residue3", history_xy(res3, K, M, rng(2))),
                 ("lfsr3", history_xy(lf3, K, M, rng(4)))]
    vC, sC = order_meter(own_r2, history_xy(lf5, K, M, rng(5)), controlsC)
    print(f"  C [truth=NOT A NULL] target=lfsr5(order 5)  -> {vC}")
    print(f"       scores: {{ {', '.join(f'{k}={v:.3f}' for k,v in sC.items())} }}")

    ok = (vA.startswith("INVISIBLE") and vB.startswith("DEAD") and vC.startswith("NOT A NULL"))
    print(f"\n  DISCRIMINATION: {'PASS — all three verdicts correct' if ok else 'FAIL'}")

    # scores-API: audit a paused lane's RECORDED null without re-running its substrate.
    print("\n  scores-API (synthetic): classify from precomputed detection scores")
    print(f"    control live (0.90), target 0.00 -> {order_meter_from_scores(0.00, {'ctrl': 0.90})}")
    print(f"    control dying(0.28), target 0.00 -> {order_meter_from_scores(0.00, {'ctrl': 0.28})}")
    print("  (Apply to any 'the model can't see X' null by supplying its probe + a known-order")
    print("   positive control; classify the null as INVISIBLE / DEAD APPARATUS / NOT A NULL.)")


if __name__ == "__main__":
    main()
