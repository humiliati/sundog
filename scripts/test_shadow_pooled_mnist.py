"""Frozen test for H3 Substrate B — MNIST CNN pooled-shadow (pre-reg §B).

Asserts the QUALITATIVE pre-registered outcome with tolerances against the
result JSON produced by scripts/shadow_pooled_mnist.py:

  P-B1 (resist)   : theta-R^2 DROPS post-GAP vs pre-GAP (the continuous nuisance
                    is washed by the spatial average) — by a meaningful margin.
  P-B2 (determine): y-acc stays HIGH post-GAP (the class survives; partly trivial
                    as it is the training target — flagged honestly in the receipt).
  P-B3 (sweep)    : ensemble-spread lambda washes theta-R^2 further toward chance.
  Controls        : label-permutation drives both recoveries to chance (no leakage).

  KILL-B1 check   : the test FAILS loudly if theta is fully recoverable post-GAP
                    (theta-R^2 post ~= pre), i.e. the resist did not happen. (That
                    would itself be a valid informative null, but it must NOT be
                    silently recorded as a pass of P-B1.)

Run:  python -m pytest scripts/test_shadow_pooled_mnist.py -q
  or  python scripts/test_shadow_pooled_mnist.py
"""

from __future__ import annotations

import json
import os

RESULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "atlas", "h3", "mnist_result.json"
)


def _load():
    with open(os.path.normpath(RESULT_PATH)) as f:
        return json.load(f)


def test_result_exists_and_wellformed():
    r = _load()
    for k in (
        "theta_r2_pre", "theta_r2_post", "y_acc_pre", "y_acc_post",
        "chance_y", "sweep", "control_permuted_y_acc_post",
        "control_permuted_theta_r2_post",
    ):
        assert k in r, f"missing key {k}"
    assert len(r["sweep"]) >= 3, "sweep must have >=3 lambda points"


def test_p_b1_theta_resist_post_lt_pre():
    """P-B1: theta-R^2 drops post-GAP vs pre-GAP by a meaningful margin."""
    r = _load()
    pre, post = r["theta_r2_pre"], r["theta_r2_post"]
    assert post < pre - 0.10, (
        f"P-B1 FAIL: theta-R2 did not drop enough post-GAP (pre={pre:.3f} "
        f"post={post:.3f}). If post ~= pre this is KILL-B1 (theta fully "
        f"recoverable post-GAP) — a valid null, but not a P-B1 pass."
    )


def test_kill_b1_not_triggered():
    """KILL-B1 guard: theta must NOT be fully recoverable post-GAP."""
    r = _load()
    pre, post = r["theta_r2_pre"], r["theta_r2_post"]
    assert post < pre - 0.05, (
        f"KILL-B1 TRIGGERED: theta-R2 post ({post:.3f}) ~= pre ({pre:.3f}); "
        f"the CNN does NOT resist the continuous nuisance via GAP."
    )


def test_p_b1_direction_robust_to_probe():
    """P-B1 honesty: the pre>post direction must survive a fair NON-PCA probe.

    The static gap is probe-sensitive (a naive ridge on the raw 1568-dim
    pre-pool map can flip the sign). We require the direction to hold under a
    strong-ridge (alpha=200) probe that de-overfits the high-dim map — i.e.
    P-B1 is a representation property, not merely a PCA artifact.
    """
    r = _load()
    rob = r["probe_robustness_noPCA"]["alpha_200"]
    assert rob["pre_minus_post"] > 0.05, (
        f"P-B1 direction NOT robust: even under fair strong-ridge probing the "
        f"pre/post gap is {rob['pre_minus_post']:+.3f} "
        f"(pre={rob['theta_r2_pre_noPCA']:.3f} post={rob['theta_r2_post_noPCA']:.3f}). "
        f"P-B1 would then be a PCA artifact, not a resist property."
    )


def test_p_b2_y_determine_post_high():
    """P-B2: y-acc stays high post-GAP (well over chance)."""
    r = _load()
    post_y = r["y_acc_post"]
    chance = r["chance_y"]
    assert post_y >= max(0.80, 4 * chance), (
        f"P-B2 FAIL: post-GAP y-acc {post_y:.3f} not high vs chance {chance:.3f}"
    )


def test_p_b2_y_acc_robust_to_pooling():
    """y-determine survives pooling (post not much worse than pre)."""
    r = _load()
    assert r["y_acc_post"] >= r["y_acc_pre"] - 0.15, (
        f"y-acc collapsed under pooling: pre={r['y_acc_pre']:.3f} "
        f"post={r['y_acc_post']:.3f} (would be KILL-A2-like)"
    )


def test_p_b3_sweep_washes_theta():
    """P-B3: theta-R^2 of ensemble-averaged GAP washes toward chance with lambda."""
    r = _load()
    sweep = sorted(r["sweep"], key=lambda d: d["lambda"])
    theta0 = sweep[0]["theta_r2"]
    thetaN = sweep[-1]["theta_r2"]
    assert thetaN < theta0 - 0.03, (
        f"P-B3 FAIL: theta-R2 did not wash with lambda "
        f"(lambda0={theta0:.3f} -> lambdaN={thetaN:.3f})"
    )


def test_p_b3_sweep_y_holds():
    """The sweep washes theta while y-acc holds (determine survives lossiness)."""
    r = _load()
    sweep = sorted(r["sweep"], key=lambda d: d["lambda"])
    chance = r["chance_y"]
    yN = sweep[-1]["y_acc"]
    assert yN >= max(0.70, 3 * chance), (
        f"y-acc collapsed across the sweep (lambdaN y-acc={yN:.3f} vs "
        f"chance {chance:.3f})"
    )


def test_controls_permutation_chance():
    """Label-permutation control: recoveries collapse to chance (no leakage)."""
    r = _load()
    chance = r["chance_y"]
    perm_y = r["control_permuted_y_acc_post"]
    perm_theta = r["control_permuted_theta_r2_post"]
    assert perm_y < chance + 0.10, f"permuted y-acc {perm_y:.3f} not at chance"
    assert perm_theta < 0.10, f"permuted theta-R2 {perm_theta:.3f} not at chance"


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL  {name}: {e}")
    if failures:
        print(f"\n{len(failures)} test(s) FAILED")
        raise SystemExit(1)
    print("\nALL TESTS PASSED")
