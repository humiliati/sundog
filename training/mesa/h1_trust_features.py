"""Shared H1 local/trust feature helpers for Python trainers.

Mirrors scripts/h1-trust-features.mjs. Reads only the local observation stream,
frozen head proposals, and previous committed actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

BASE_H1_FEATURES = [
    "obs0", "obs1", "obs2", "obs3", "obs4", "obs5",
    "fa_x", "fa_y", "ra_x", "ra_y", "fa_norm", "ra_norm",
    "disagree_l2", "cos_agree", "fd_grad_norm", "hist_act_norm_prev", "hist_sLocal_prev",
]

TRUST_K = 8

TRUST_FEATURES = [
    "sample_dispersion",
    "sLocal_var_K",
    "grad_norm_var_K",
    "grad_dir_stability_K",
    "disagree_mean_K",
    "act_dir_consistency_K",
]


def h1_features_for_mode(feature_mode: str = "base") -> list[str]:
    return BASE_H1_FEATURES + TRUST_FEATURES if feature_mode == "trust" else list(BASE_H1_FEATURES)


def norm2(v: np.ndarray | list[float]) -> float:
    return float(math.hypot(float(v[0]), float(v[1])))


def cos2(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    na = norm2(a)
    nb = norm2(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    return float(np.dot(av, bv) / (na * nb))


def _variance_zero_padded(values: list[float], k: int = TRUST_K) -> float:
    padded = list(values[-k:])
    while len(padded) < k:
        padded.insert(0, 0.0)
    arr = np.asarray(padded, dtype=np.float32)
    return float(np.var(arr))


def _mean_zero_padded(values: list[float], k: int = TRUST_K) -> float:
    padded = list(values[-k:])
    while len(padded) < k:
        padded.insert(0, 0.0)
    return float(np.mean(np.asarray(padded, dtype=np.float32)))


def _mean_consecutive_cos(vectors: list[np.ndarray]) -> float:
    xs = [v for v in vectors[-TRUST_K:] if norm2(v) >= 1e-9]
    if len(xs) < 2:
        return 0.0
    vals = [cos2(xs[i - 1], xs[i]) for i in range(1, len(xs))]
    return float(np.mean(vals)) if vals else 0.0


@dataclass
class H1FeatureState:
    s_local: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    grad_dir: list[np.ndarray] = field(default_factory=list)
    disagree: list[float] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    prev_act_norm: float = 0.0
    prev_s_local: float = 0.0

    def reset(self, obs: list[float], info: dict[str, Any] | None = None) -> None:
        self.s_local.clear()
        self.grad_norm.clear()
        self.grad_dir.clear()
        self.disagree.clear()
        self.actions.clear()
        self.prev_act_norm = 0.0
        self.prev_s_local = float((info or {}).get("s_local", sum(obs[2:6]) / 4))

    def note_action(self, action: np.ndarray, info: dict[str, Any] | None = None, obs: list[float] | None = None) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        while len(self.actions) > TRUST_K:
            self.actions.pop(0)
        self.prev_act_norm = norm2(action)
        if info is not None:
            self.prev_s_local = float(info.get("s_local", self.prev_s_local))
        elif obs is not None:
            self.prev_s_local = float(sum(obs[2:6]) / 4)


def build_h1_local_features(
    obs: list[float],
    fa: np.ndarray,
    ra: np.ndarray,
    *,
    eps: float,
    state: H1FeatureState,
    feature_mode: str = "base",
) -> dict[str, float]:
    samples = [float(obs[2]), float(obs[3]), float(obs[4]), float(obs[5])]
    fd = np.asarray([(samples[0] - samples[1]) / (2 * eps), (samples[2] - samples[3]) / (2 * eps)], dtype=np.float32)
    fd_norm = norm2(fd)
    disagree = norm2(fa - ra)
    s_local = float(sum(samples) / 4)

    state.s_local.append(s_local)
    state.grad_norm.append(fd_norm)
    state.grad_dir.append(fd)
    state.disagree.append(disagree)
    while len(state.s_local) > TRUST_K:
        state.s_local.pop(0)
    while len(state.grad_norm) > TRUST_K:
        state.grad_norm.pop(0)
    while len(state.grad_dir) > TRUST_K:
        state.grad_dir.pop(0)
    while len(state.disagree) > TRUST_K:
        state.disagree.pop(0)

    fmap = {
        "obs0": float(obs[0]),
        "obs1": float(obs[1]),
        "obs2": float(obs[2]),
        "obs3": float(obs[3]),
        "obs4": float(obs[4]),
        "obs5": float(obs[5]),
        "fa_x": float(fa[0]),
        "fa_y": float(fa[1]),
        "ra_x": float(ra[0]),
        "ra_y": float(ra[1]),
        "fa_norm": norm2(fa),
        "ra_norm": norm2(ra),
        "disagree_l2": disagree,
        "cos_agree": cos2(fa, ra),
        "fd_grad_norm": fd_norm,
        "hist_act_norm_prev": state.prev_act_norm,
        "hist_sLocal_prev": state.prev_s_local,
    }

    if feature_mode == "trust":
        arr = np.asarray(samples, dtype=np.float32)
        fmap.update({
            "sample_dispersion": float(np.std(arr)),
            "sLocal_var_K": _variance_zero_padded(state.s_local),
            "grad_norm_var_K": _variance_zero_padded(state.grad_norm),
            "grad_dir_stability_K": _mean_consecutive_cos(state.grad_dir),
            "disagree_mean_K": _mean_zero_padded(state.disagree),
            "act_dir_consistency_K": _mean_consecutive_cos(state.actions),
        })

    return fmap


def trust_feature_audit(feature_mode: str, inference_features: list[str]) -> dict[str, Any]:
    forbidden = [
        f for f in inference_features
        if any(s in f.lower() for s in ("x_false", "x_goal", "terminal", "true_", "basin", "cell", "seed", "label", "metric"))
    ]
    return {
        "feature_mode": feature_mode,
        "window_k": TRUST_K,
        "base_feature_count": len(BASE_H1_FEATURES),
        "trust_feature_count": len([f for f in TRUST_FEATURES if f in inference_features]),
        "inference_feature_count": len(inference_features),
        "trust_features": TRUST_FEATURES,
        "missing_base_features": [f for f in BASE_H1_FEATURES if f not in inference_features],
        "missing_trust_features": [f for f in TRUST_FEATURES if f not in inference_features] if feature_mode == "trust" else [],
        "forbidden_feature_scan": forbidden,
        "no_privileged_feature_names": not forbidden,
    }
