#!/usr/bin/env python
"""Frozen test for H3 Substrate A — pooled-shadow synthetic.

Asserts the QUALITATIVE outcome that was ACTUALLY observed against the pre-reg
(docs/atlas/H3_POOLED_SHADOW_PREREG.md, §A), with tolerances (bands not points):

  DETERMINE half HOLDS (as predicted):
   * P-A1: lambda=0 post-pool c-R2 HIGH (>=0.6) for all bodies (lossiness-essential).
   * P-A3: post-pool d-acc stays HIGH (>=0.85) across the whole lambda grid, all bodies.
   * controls clean: majority d-acc = 0.5; label-permutation c-R2 ~ 0 and d-acc ~ chance.

  RESIST half FAILS -> KILL-A1 (the pre-registered, important null):
   * post-pool c-R2 for clf_d STAYS HIGH at lambda=2 (> 0.5) -> the mean-pool does NOT
     wash the continuous; it acts as a noise-averaging estimator of the shared mean c.
   * the pre/post gap is NOT large-positive (pooling does not destroy c vs a single unit);
     in fact a single un-pooled unit recovers c WORSE than the pool at high lambda.

  c-resist IS lambda-dependent (post-pool c-R2 declines as lambda grows) -> KILL-A3 NOT fired
  (the small decline is a real lossiness effect, just far too weak to wash c).

Run:  python scripts/test_shadow_pooled_synthetic.py     (or pytest)
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "results" / "atlas" / "h3" / "synthetic_result.json"
SCRIPT = ROOT / "scripts" / "shadow_pooled_synthetic.py"


def _load():
    if not RESULT.exists():
        subprocess.run([sys.executable, str(SCRIPT)], check=True, cwd=str(ROOT))
    return json.loads(RESULT.read_text())


def _post_c(r, obj, lam): return r["results"][obj]["post"][lam]["c_r2"]
def _post_d(r, obj, lam): return r["results"][obj]["post"][lam]["d_acc"]
def _pre_c(r, obj, lam):  return r["results"][obj]["pre"][lam]["c_r2"]


def test_controls_clean():
    r = _load()
    c = r["controls"]
    assert abs(c["majority_d_balanced_acc"] - 0.5) < 1e-9
    # label permutation -> no leakage: c-R2 near 0, d-acc near chance
    assert c["label_perm_post_c_r2"] < 0.1, c["label_perm_post_c_r2"]
    assert abs(c["label_perm_post_d_acc"] - 0.5) < 0.1, c["label_perm_post_d_acc"]


def test_P_A1_lossiness_essential():
    """lambda=0 post-pool c-R2 HIGH for all bodies (a lossless ensemble loses nothing)."""
    r = _load()
    for obj in ["clf_d", "reg_c", "recon"]:
        assert _post_c(r, obj, "0.0") >= 0.6, (obj, _post_c(r, obj, "0.0"))


def test_P_A3_determine_half_holds():
    """post-pool d-acc stays HIGH (>=0.85) across the whole grid for all bodies."""
    r = _load()
    grid = [str(l) for l in r["meta"]["lambda_grid"]]
    for obj in ["clf_d", "reg_c", "recon"]:
        for lam in grid:
            assert _post_d(r, obj, lam) >= 0.85, (obj, lam, _post_d(r, obj, lam))


def test_KILL_A1_resist_fails_clf_d():
    """The pre-registered important NULL: clf_d post-pool c-R2 STAYS HIGH (>0.5) at lambda=2.

    The mean-pool does NOT wash the continuous; it noise-averages back to the shared mean c.
    """
    r = _load()
    assert _post_c(r, "clf_d", "2.0") > 0.5, _post_c(r, "clf_d", "2.0")
    assert "KILL-A1" in r["kills_triggered"]


def test_pooling_does_not_open_a_resist_gap():
    """Pre/post gap is NOT a large positive (pooling does not destroy c vs a single unit).

    Observed: a single un-pooled unit recovers c WORSE than the pool at high lambda, so the
    pre/post 'pooling kills c' gap is absent (in fact negative). Assert it is well below the
    pre-reg's >=0.4 'pooling washes c' threshold.
    """
    r = _load()
    gap = _pre_c(r, "clf_d", "2.0") - _post_c(r, "clf_d", "2.0")
    assert gap < 0.4, gap


def test_KILL_A3_not_fired_resist_is_lambda_dependent():
    """post-pool c-R2 for clf_d DOES decline with lambda -> the (weak) resist is a lossiness
    effect, so KILL-A3 (lambda-independent / flat) must NOT fire."""
    r = _load()
    grid = [str(l) for l in r["meta"]["lambda_grid"]]
    curve = [_post_c(r, "clf_d", lam) for lam in grid]
    assert (max(curve) - min(curve)) >= 0.05, (max(curve) - min(curve))
    assert curve[0] > curve[-1]  # monotone-ish decline from lambda=0 to lambda=2
    assert "KILL-A3" not in r["kills_triggered"]


def test_objective_ordering_qualitative():
    """reg_c (incentivized to keep c) keeps AT LEAST as much c post-pool as clf_d at lambda=2.

    P-A5's >=0.2 margin is NOT met (both keep lots of c), but the ordering direction predicted
    by objective-dependence still holds qualitatively: reg_c >= clf_d.
    """
    r = _load()
    assert _post_c(r, "reg_c", "2.0") >= _post_c(r, "clf_d", "2.0") - 0.02


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
