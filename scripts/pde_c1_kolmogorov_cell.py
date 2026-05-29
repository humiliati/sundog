"""Execute the PDE C1 Kolmogorov-flow v0 cell.

Implements the runnable side of:

  docs/proof/PDE_C1_CELLSET_KOLMOGOROV.md
  docs/proof/PDE_C1_FIBER_PROTOCOL.md

The default preset is a smoke. Use `--preset lock` for the registered v0
sample count. Smoke runs are machinery checks only and never file a C1 verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "proof" / "c1-kolmogorov-v0-smoke"

VERDICT_BEARING_PRESETS = {
    "lock",
    "fallback",
    "lock_v1",
    "fallback_v1",
    "lock_v2",
    "fallback_v2",
    "lock_v3",
    "fallback_v3",
    "lock_v4",
    "fallback_v4",
    "lock_v5",
    "fallback_v5",
}


@dataclass(frozen=True)
class RunConfig:
    preset: str
    grid_size: int
    n_modes: int
    k_signature: int
    forcing_wavenumber: int
    grashof: float
    forcing_amplitude: float
    viscosity: float
    dt: float
    burnin_steps: int
    sample_count: int
    sample_interval_steps: int
    lookahead_steps: int
    n_min: int
    delta_action: float
    s_pos: float
    delta_proxy_min: float
    e_max_burnin_fraction: float
    random_seed: int
    integrator: str
    signature_dimension: int
    action_tiebreak: str
    adjudicator: str
    k_neighbors: int
    delta_incompat: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=[
            "smoke",
            "lock",
            "fallback",
            "lock_v1",
            "fallback_v1",
            "lock_v2",
            "fallback_v2",
            "lock_v3",
            "fallback_v3",
            "lock_v4",
            "fallback_v4",
            "lock_v5",
            "fallback_v5",
        ],
        default="smoke",
    )
    parser.add_argument(
        "--adjudicator",
        choices=["bin", "knn"],
        default="bin",
        help="Fiber-locality adjudicator: hard binning (default) or kNN/disintegration.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write-samples", action="store_true", help="Write per-sample rows even for large runs.")
    parser.add_argument("--self-test", action="store_true", help="Run a tiny deterministic self-test and exit.")
    parser.add_argument(
        "--allow-unregistered-overrides",
        action="store_true",
        help="Allow manual smoke-only overrides. Such runs are never verdict-bearing.",
    )
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--burnin-steps", type=int)
    parser.add_argument("--sample-interval-steps", type=int)
    parser.add_argument("--lookahead-steps", type=int)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    # Cell-set v0 / v1 / v2 / v3 / v4 / v5 pinned values. v0..v4 use K = 4
    # (signature dim 32). v5 uses K = 3 (signature dim 18) to address the
    # curse-of-dimensionality coverage failure observed at K = 4 with the
    # G = 200, k_f = 2 attractor.
    grid_size = 32
    n_modes = 16
    forcing_amplitude = 1.0
    dt = 0.01
    sample_interval_steps = 50
    lookahead_steps = int(round(5.0 / dt))
    n_min = 30
    delta_action = 0.10
    s_pos = 0.50
    delta_proxy_min = 0.01
    # E_max windowing: cells v0..v3 use full burn-in (1.0). v4+ may pin smaller
    # fractions to exclude transient contamination of the 95th percentile.
    e_max_burnin_fraction = 1.0
    # kNN adjudication (adopted 2026-05-28, Fork A): k neighbours including self;
    # delta_incompat is the positive-mass threshold for filing PDE-C1-NEG-A.
    k_neighbors = 30
    delta_incompat = 0.01
    random_seed = 20260528

    if args.preset == "lock":
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 4
        grashof = 100.0
        k_signature = 4
    elif args.preset == "fallback":
        burnin_steps = 100_000
        sample_count = 200_000
        kf = 4
        grashof = 100.0
        k_signature = 4
    elif args.preset == "lock_v1":
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 100.0
        k_signature = 4
    elif args.preset == "fallback_v1":
        burnin_steps = 100_000
        sample_count = 200_000
        kf = 2
        grashof = 100.0
        k_signature = 4
    elif args.preset == "lock_v2":
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 300.0
        k_signature = 4
    elif args.preset == "fallback_v2":
        burnin_steps = 100_000
        sample_count = 200_000
        kf = 2
        grashof = 300.0
        k_signature = 4
    elif args.preset == "lock_v3":
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 200.0
        k_signature = 4
    elif args.preset == "fallback_v3":
        burnin_steps = 100_000
        sample_count = 200_000
        kf = 2
        grashof = 200.0
        k_signature = 4
    elif args.preset == "lock_v4":
        # v4: same regime as v3 (k_f=2, G=200) but with the E_max amendment
        # — E_max from the last 25% of burn-in to exclude transient
        # contamination.
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 200.0
        e_max_burnin_fraction = 0.25
        k_signature = 4
    elif args.preset == "fallback_v4":
        burnin_steps = 100_000
        sample_count = 200_000
        kf = 2
        grashof = 200.0
        e_max_burnin_fraction = 0.25
        k_signature = 4
    elif args.preset == "lock_v5":
        # v5: same regime + E_max amendment as v4, but with K = 3 (signature
        # dim 18 instead of 32) to address the curse-of-dimensionality
        # coverage failure observed in v2/v4 at K = 4.
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 200.0
        e_max_burnin_fraction = 0.25
        k_signature = 3
    elif args.preset == "fallback_v5":
        burnin_steps = 100_000
        sample_count = 200_000
        kf = 2
        grashof = 200.0
        e_max_burnin_fraction = 0.25
        k_signature = 3
    else:
        # Smoke is intentionally not the registered cell. It exists to validate
        # the integrator, binning, and receipt plumbing under the repo's
        # ~10-minute inline rule.
        burnin_steps = 2_000
        sample_count = 200
        sample_interval_steps = 10
        lookahead_steps = 100
        kf = 4
        grashof = 100.0
        k_signature = 4

    # Dimensionless normalization: G = forcing_amplitude / nu^2.
    viscosity = math.sqrt(forcing_amplitude / grashof)

    overrides = {
        "sample_count": args.sample_count,
        "burnin_steps": args.burnin_steps,
        "sample_interval_steps": args.sample_interval_steps,
        "lookahead_steps": args.lookahead_steps,
    }
    if any(value is not None for value in overrides.values()) and not args.allow_unregistered_overrides:
        raise SystemExit("Manual overrides require --allow-unregistered-overrides; override runs are smoke-only.")
    if args.allow_unregistered_overrides:
        if args.sample_count is not None:
            sample_count = args.sample_count
        if args.burnin_steps is not None:
            burnin_steps = args.burnin_steps
        if args.sample_interval_steps is not None:
            sample_interval_steps = args.sample_interval_steps
        if args.lookahead_steps is not None:
            lookahead_steps = args.lookahead_steps

    return RunConfig(
        preset=args.preset,
        grid_size=grid_size,
        n_modes=n_modes,
        k_signature=k_signature,
        forcing_wavenumber=kf,
        grashof=grashof,
        forcing_amplitude=forcing_amplitude,
        viscosity=viscosity,
        dt=dt,
        burnin_steps=burnin_steps,
        sample_count=sample_count,
        sample_interval_steps=sample_interval_steps,
        lookahead_steps=lookahead_steps,
        n_min=n_min,
        delta_action=delta_action,
        s_pos=s_pos,
        delta_proxy_min=delta_proxy_min,
        e_max_burnin_fraction=e_max_burnin_fraction,
        random_seed=random_seed,
        integrator="pseudo_spectral_vorticity_semi_implicit_euler",
        signature_dimension=2 * k_signature * k_signature,
        action_tiebreak="no_op",
        adjudicator=args.adjudicator,
        k_neighbors=k_neighbors,
        delta_incompat=delta_incompat,
    )


class KolmogorovStepper:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        m = cfg.grid_size
        wave = np.fft.fftfreq(m, d=1.0 / m)
        self.kx, self.ky = np.meshgrid(wave, wave, indexing="ij")
        self.k2 = self.kx * self.kx + self.ky * self.ky
        self.k2_nozero = self.k2.copy()
        self.k2_nozero[0, 0] = 1.0
        self.dealias_mask = (np.abs(self.kx) <= (m / 3.0)) & (np.abs(self.ky) <= (m / 3.0))
        self.zero_mean = (self.kx == 0) & (self.ky == 0)

        y = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False)
        _, yy = np.meshgrid(y, y, indexing="ij")
        curl_forcing = -cfg.forcing_amplitude * cfg.forcing_wavenumber * np.cos(cfg.forcing_wavenumber * yy)
        self.forcing_hat = np.fft.fft2(curl_forcing)
        self.forcing_hat[~self.dealias_mask] = 0.0
        self.forcing_hat[0, 0] = 0.0
        self.low_indices = select_low_modes(wave, cfg.k_signature, cfg.forcing_wavenumber)

    def initial_state(self) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.random_seed)
        m = self.cfg.grid_size
        y = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False)
        _, yy = np.meshgrid(y, y, indexing="ij")
        omega = -(self.cfg.forcing_amplitude / (self.cfg.viscosity * self.cfg.forcing_wavenumber)) * np.cos(
            self.cfg.forcing_wavenumber * yy
        )
        omega += 0.05 * rng.standard_normal((m, m))
        omega_hat = np.fft.fft2(omega)
        omega_hat[~self.dealias_mask] = 0.0
        omega_hat[0, 0] = 0.0
        return omega_hat

    def velocity_hat(self, omega_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi_hat = omega_hat / self.k2_nozero
        psi_hat[0, 0] = 0.0
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        return u_hat, v_hat

    def nonlinear_hat(self, omega_hat: np.ndarray) -> np.ndarray:
        u_hat, v_hat = self.velocity_hat(omega_hat)
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        omega_x = np.fft.ifft2(1j * self.kx * omega_hat).real
        omega_y = np.fft.ifft2(1j * self.ky * omega_hat).real
        adv_hat = np.fft.fft2(u * omega_x + v * omega_y)
        adv_hat[~self.dealias_mask] = 0.0
        adv_hat[0, 0] = 0.0
        return adv_hat

    def step(self, omega_hat: np.ndarray) -> np.ndarray:
        dt = self.cfg.dt
        explicit = omega_hat + dt * (-self.nonlinear_hat(omega_hat) + self.forcing_hat)
        next_hat = explicit / (1.0 + dt * self.cfg.viscosity * self.k2)
        next_hat[~self.dealias_mask] = 0.0
        next_hat[0, 0] = 0.0
        return next_hat

    def signature(self, omega_hat: np.ndarray) -> np.ndarray:
        scale = float(self.cfg.grid_size * self.cfg.grid_size)
        omega_norm = omega_hat / scale
        values: list[float] = []
        for ix, iy in self.low_indices:
            k_norm = math.sqrt(float(self.k2[ix, iy]))
            amp = omega_norm[ix, iy] / k_norm
            values.append(float(amp.real))
            values.append(float(amp.imag))
        return np.asarray(values, dtype=np.float64)

    def low_energy(self, omega_hat: np.ndarray) -> float:
        sig = self.signature(omega_hat)
        return float(np.dot(sig, sig))


def select_low_modes(wave: np.ndarray, k_signature: int, forced_k: int) -> list[tuple[int, int]]:
    modes: list[tuple[float, float, int, int]] = []
    for ix, kx in enumerate(wave):
        for iy, ky in enumerate(wave):
            if kx == 0 and ky == 0:
                continue
            if abs(kx) > k_signature or abs(ky) > k_signature:
                continue
            # Half-plane representatives for a real field.
            if ky > 0 or (ky == 0 and kx > 0):
                modes.append((max(abs(kx), abs(ky)), kx * kx + ky * ky, ix, iy))
    modes.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    selected = [(ix, iy) for _, _, ix, iy in modes[: k_signature * k_signature]]
    forced = None
    for ix, kx in enumerate(wave):
        for iy, ky in enumerate(wave):
            if kx == 0 and ky == forced_k:
                forced = (ix, iy)
                break
        if forced is not None:
            break
    if forced is not None and forced not in selected:
        selected[-1] = forced
        selected = sorted(set(selected), key=lambda pair: (max(abs(wave[pair[0]]), abs(wave[pair[1]])), pair[0], pair[1]))
    if len(selected) != k_signature * k_signature:
        raise ValueError(f"expected {k_signature * k_signature} low modes, got {len(selected)}")
    return selected


def run_cell(cfg: RunConfig) -> dict:
    stepper = KolmogorovStepper(cfg)
    omega_hat = stepper.initial_state()
    total_steps = cfg.burnin_steps + (cfg.sample_count - 1) * cfg.sample_interval_steps + cfg.lookahead_steps
    sample_steps = {cfg.burnin_steps + i * cfg.sample_interval_steps: i for i in range(cfg.sample_count)}

    burnin_energies: list[float] = []
    energy_by_step = np.empty(total_steps + 1, dtype=np.float64)
    sample_signatures = np.empty((cfg.sample_count, cfg.signature_dimension), dtype=np.float64)
    sample_energy = np.empty(cfg.sample_count, dtype=np.float64)
    sig_min = np.full(cfg.signature_dimension, np.inf)
    sig_max = np.full(cfg.signature_dimension, -np.inf)

    started = time.perf_counter()
    progress_stride = max(1, total_steps // 10)
    for step in range(total_steps + 1):
        energy = stepper.low_energy(omega_hat)
        energy_by_step[step] = energy
        if step <= cfg.burnin_steps:
            sig = stepper.signature(omega_hat)
            sig_min = np.minimum(sig_min, sig)
            sig_max = np.maximum(sig_max, sig)
            burnin_energies.append(energy)
        if step in sample_steps:
            sample_index = sample_steps[step]
            sig = stepper.signature(omega_hat)
            sample_signatures[sample_index, :] = sig
            sample_energy[sample_index] = energy
        if step < total_steps:
            omega_hat = stepper.step(omega_hat)
        if step > 0 and step % progress_stride == 0:
            elapsed = time.perf_counter() - started
            print(f"[pde-c1] step {step}/{total_steps} ({100 * step / total_steps:.0f}%), elapsed {elapsed:.1f}s", flush=True)

    # E_max windowing: use the last fraction of burn-in to exclude transients
    # that would bias the 95th percentile above the steady-state attractor.
    burnin_array = np.asarray(burnin_energies)
    window_len = max(1, int(cfg.e_max_burnin_fraction * len(burnin_array)))
    e_max = float(np.percentile(burnin_array[-window_len:], 95))
    epsilon_k = 0.05 * math.sqrt(max(0.0, 2.0 * e_max))
    h_k = epsilon_k / math.sqrt(cfg.signature_dimension) if epsilon_k > 0 else 1.0
    actions = label_samples(energy_by_step, cfg, e_max)
    if cfg.adjudicator == "knn":
        result, bin_rows, sample_rows, knn_histogram = aggregate_knn(
            sample_signatures, actions, epsilon_k, cfg
        )
    else:
        bin_rows, sample_rows = aggregate_bins(sample_signatures, sample_energy, actions, sig_min, h_k, cfg)
        result = summarize(bin_rows, cfg)
        knn_histogram = []
    result.update(
        {
            "adjudicator": cfg.adjudicator,
            "knn_histogram": knn_histogram,
            "e_max": e_max,
            "epsilon_k": epsilon_k,
            "h_k": h_k,
            "burnin_energy_min": float(np.min(burnin_energies)),
            "burnin_energy_max": float(np.max(burnin_energies)),
            "burnin_energy_mean": float(np.mean(burnin_energies)),
            "sample_energy_min": float(np.min(sample_energy)),
            "sample_energy_max": float(np.max(sample_energy)),
            "sample_energy_mean": float(np.mean(sample_energy)),
            "sample_rows": sample_rows,
            "bin_rows": bin_rows,
            "low_modes": [
                {"kx": int(stepper.kx[ix, iy]), "ky": int(stepper.ky[ix, iy])}
                for ix, iy in stepper.low_indices
            ],
            "elapsed_seconds": time.perf_counter() - started,
            "total_steps": total_steps,
            "steps_per_second": total_steps / max(1e-9, time.perf_counter() - started),
        }
    )
    return result


def label_samples(energy_by_step: np.ndarray, cfg: RunConfig, e_max: float) -> np.ndarray:
    actions = np.zeros(cfg.sample_count, dtype=np.int8)
    for i in range(cfg.sample_count):
        start = cfg.burnin_steps + i * cfg.sample_interval_steps
        end = start + cfg.lookahead_steps
        if float(np.max(energy_by_step[start : end + 1])) > e_max:
            actions[i] = 1
    return actions


def aggregate_bins(
    signatures: np.ndarray,
    sample_energy: np.ndarray,
    actions: np.ndarray,
    origin: np.ndarray,
    h_k: float,
    cfg: RunConfig,
) -> tuple[list[dict], list[dict]]:
    bins: dict[tuple[int, ...], dict[str, int]] = defaultdict(lambda: {"n": 0, "no_op": 0, "damp_low_band": 0})
    sample_rows: list[dict] = []
    for i, sig in enumerate(signatures):
        idx = tuple(np.floor((sig - origin) / h_k).astype(np.int64).tolist())
        rec = bins[idx]
        rec["n"] += 1
        if actions[i] == 1:
            rec["damp_low_band"] += 1
            action = "damp_low_band"
        else:
            rec["no_op"] += 1
            action = "no_op"
        if cfg.sample_count <= 5_000:
            sample_rows.append(
                {
                    "sample_index": i,
                    "time": (cfg.burnin_steps + i * cfg.sample_interval_steps) * cfg.dt,
                    "energy": float(sample_energy[i]),
                    "action": action,
                    "bin_key": bin_key(idx),
                    "signature_norm": float(np.linalg.norm(sig)),
                }
            )

    rows: list[dict] = []
    for idx, counts in bins.items():
        no_op = counts["no_op"]
        damp = counts["damp_low_band"]
        n = counts["n"]
        if damp > no_op:
            majority = "damp_low_band"
            majority_count = damp
        else:
            # Locked action tie-break: prefer no_op.
            majority = "no_op"
            majority_count = no_op
        minority_fraction = 1.0 - (majority_count / n)
        evaluated = n >= cfg.n_min
        rows.append(
            {
                "bin_key": bin_key(idx),
                "n": n,
                "no_op_count": no_op,
                "damp_low_band_count": damp,
                "majority_action": majority,
                "minority_fraction": minority_fraction,
                "evaluated": evaluated,
                "fiber_incompatible": bool(evaluated and minority_fraction > cfg.delta_action),
            }
        )
    rows.sort(key=lambda row: (-int(row["n"]), row["bin_key"]))
    return rows, sample_rows


def summarize(bin_rows: list[dict], cfg: RunConfig) -> dict:
    occupied = len(bin_rows)
    evaluated = sum(1 for row in bin_rows if row["evaluated"])
    incompatible = [row for row in bin_rows if row["fiber_incompatible"]]
    s_eval = evaluated / occupied if occupied else 0.0
    no_op = sum(int(row["no_op_count"]) for row in bin_rows)
    damp = sum(int(row["damp_low_band_count"]) for row in bin_rows)

    verdict_bearing_presets = VERDICT_BEARING_PRESETS
    damp_fraction = damp / max(1, no_op + damp)
    proxy_constant = (
        damp_fraction < cfg.delta_proxy_min
        or damp_fraction > 1.0 - cfg.delta_proxy_min
    )
    proxy_structural_constant = damp == 0 or no_op == 0
    if cfg.preset not in verdict_bearing_presets:
        verdict = "SMOKE_ONLY"
        verdict_label = ""
        interpretable = False
    elif proxy_structural_constant:
        # Structural-vacuity precedence: when damp_fraction is exactly 0 or 1,
        # increasing N_sample cannot resolve the proxy. Vacuity takes precedence
        # over the coverage gate per the 2026-05-28 protocol amendment.
        verdict = "DEFERRED_VACUITY"
        verdict_label = "proxy_selector_structurally_constant"
        interpretable = False
    elif s_eval < cfg.s_pos:
        verdict = "DEFERRED_COVERAGE"
        verdict_label = ""
        interpretable = False
    elif proxy_constant:
        verdict = "DEFERRED_VACUITY"
        verdict_label = "proxy_selector_near_constant_on_sampled_support"
        interpretable = False
    elif incompatible:
        verdict = "PDE-C1-NEG-A"
        verdict_label = "fiber_incompatibility"
        interpretable = True
    else:
        verdict = "STRICTNESS_WITNESS_POSITIVE"
        verdict_label = "proxy_fiber_constancy_on_evaluated_bins"
        interpretable = True

    return {
        "verdict": verdict,
        "verdict_label": verdict_label,
        "interpretable": interpretable,
        "occupied_bin_count": occupied,
        "evaluated_bin_count": evaluated,
        "incompatible_bin_count": len(incompatible),
        "s_eval": s_eval,
        "s_pos": cfg.s_pos,
        "no_op_count": no_op,
        "damp_low_band_count": damp,
        "damp_fraction": damp / max(1, no_op + damp),
        "max_minority_fraction": max((float(row["minority_fraction"]) for row in bin_rows), default=0.0),
        "max_evaluated_minority_fraction": max(
            (float(row["minority_fraction"]) for row in bin_rows if row["evaluated"]),
            default=0.0,
        ),
    }


def aggregate_knn(
    signatures: np.ndarray,
    actions: np.ndarray,
    epsilon_k: float,
    cfg: RunConfig,
) -> tuple[dict, list, list, list]:
    """kNN / disintegration adjudication (Fork A, adopted 2026-05-28).

    For each sample, examine its k nearest signature-space neighbours
    (including self). The neighbourhood radius r_k is a per-sample fidelity
    measure; the local minority fraction estimates conditional proxy
    non-constancy (a nonparametric estimate of the disintegration of mu
    over Phi_K). Spec: docs/proof/PDE_C1_KNN_ADJUDICATION_DESIGN.md.
    """
    from sklearn.neighbors import BallTree

    n = int(signatures.shape[0])
    k = min(cfg.k_neighbors, n)
    tree = BallTree(signatures, metric="euclidean")
    dist, idx = tree.query(signatures, k=k)  # self included at distance 0
    r_k = dist[:, -1]
    neighbour_actions = actions[idx]  # (n, k) of 0/1
    damp_count = neighbour_actions.sum(axis=1)
    no_op_count = k - damp_count
    majority_is_damp = damp_count > no_op_count  # strict; tie -> no_op
    majority_count = np.where(majority_is_damp, damp_count, no_op_count)
    minority = 1.0 - majority_count / float(k)

    fidelity_pass = r_k <= epsilon_k
    f_count = int(fidelity_pass.sum())
    fidelity_coverage = f_count / n if n else 0.0
    incompat_mask = fidelity_pass & (minority > cfg.delta_action)
    incompat_count = int(incompat_mask.sum())
    incompat_fraction = incompat_count / f_count if f_count else 0.0

    damp = int(actions.sum())
    no_op = int(n - damp)

    r_max = float(r_k.max()) if n else 0.0
    nbins = 20
    edges = np.linspace(0.0, max(r_max, epsilon_k * 2.0, 1e-12), nbins + 1)
    counts, _ = np.histogram(r_k, bins=edges)
    histogram = [
        {
            "r_lo": float(edges[j]),
            "r_hi": float(edges[j + 1]),
            "count": int(counts[j]),
            "within_epsilon": bool(edges[j + 1] <= epsilon_k),
        }
        for j in range(nbins)
    ]

    result = summarize_knn(
        n=n,
        k=k,
        f_count=f_count,
        fidelity_coverage=fidelity_coverage,
        incompat_count=incompat_count,
        incompat_fraction=incompat_fraction,
        no_op=no_op,
        damp=damp,
        r_k=r_k,
        minority=minority,
        fidelity_pass=fidelity_pass,
        epsilon_k=epsilon_k,
        cfg=cfg,
    )
    return result, [], [], histogram


def summarize_knn(
    *,
    n: int,
    k: int,
    f_count: int,
    fidelity_coverage: float,
    incompat_count: int,
    incompat_fraction: float,
    no_op: int,
    damp: int,
    r_k: np.ndarray,
    minority: np.ndarray,
    fidelity_pass: np.ndarray,
    epsilon_k: float,
    cfg: RunConfig,
) -> dict:
    damp_fraction = damp / max(1, n)
    proxy_structural_constant = damp == 0 or no_op == 0
    proxy_constant = (
        damp_fraction < cfg.delta_proxy_min or damp_fraction > 1.0 - cfg.delta_proxy_min
    )

    if cfg.preset not in VERDICT_BEARING_PRESETS:
        verdict, verdict_label, interpretable = "SMOKE_ONLY", "", False
    elif proxy_structural_constant:
        verdict, verdict_label, interpretable = (
            "DEFERRED_VACUITY",
            "proxy_selector_structurally_constant",
            False,
        )
    elif proxy_constant:
        verdict, verdict_label, interpretable = (
            "DEFERRED_VACUITY",
            "proxy_selector_near_constant_on_sampled_support",
            False,
        )
    elif fidelity_coverage < cfg.s_pos:
        verdict, verdict_label, interpretable = (
            "DEFERRED_FIDELITY_COVERAGE",
            "insufficient_fidelity_passing_fraction",
            False,
        )
    elif incompat_fraction > cfg.delta_incompat:
        verdict, verdict_label, interpretable = (
            "PDE-C1-NEG-A",
            "fiber_incompatibility_knn",
            True,
        )
    else:
        verdict, verdict_label, interpretable = (
            "STRICTNESS_WITNESS_POSITIVE",
            "proxy_fiber_constancy_knn",
            True,
        )

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q)) if a.size else 0.0

    r_pass = r_k[fidelity_pass]
    m_pass = minority[fidelity_pass]
    return {
        "verdict": verdict,
        "verdict_label": verdict_label,
        "interpretable": interpretable,
        "n_samples": n,
        "k_neighbors_effective": k,
        "fidelity_passing_count": f_count,
        "fidelity_coverage": fidelity_coverage,
        "s_pos": cfg.s_pos,
        "incompat_count": incompat_count,
        "incompat_fraction": incompat_fraction,
        "delta_incompat": cfg.delta_incompat,
        "delta_action": cfg.delta_action,
        "no_op_count": no_op,
        "damp_low_band_count": damp,
        "damp_fraction": damp_fraction,
        "epsilon_k_radius_threshold": epsilon_k,
        "r_k_median": pct(r_k, 50),
        "r_k_p95": pct(r_k, 95),
        "r_k_min": float(r_k.min()) if n else 0.0,
        "r_k_max": float(r_k.max()) if n else 0.0,
        "r_k_median_passing": pct(r_pass, 50),
        "max_minority_fraction": float(minority.max()) if n else 0.0,
        "max_passing_minority_fraction": float(m_pass.max()) if m_pass.size else 0.0,
    }


def bin_key(idx: Iterable[int]) -> str:
    return ";".join(str(int(x)) for x in idx)


def write_outputs(out_dir: Path, cfg: RunConfig, result: dict, write_samples: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/pde_c1_kolmogorov_cell.py",
        "spec": "docs/proof/PDE_C1_CELLSET_KOLMOGOROV.md",
        "protocol": "docs/proof/PDE_C1_FIBER_PROTOCOL.md",
        "config": asdict(cfg),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": sys.platform,
            "cwd": str(ROOT),
        },
        "result": {
            k: v
            for k, v in result.items()
            if k not in ("bin_rows", "sample_rows", "knn_histogram")
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if cfg.adjudicator == "knn":
        write_csv(out_dir / "knn-radius-histogram.csv", result.get("knn_histogram", []))
    else:
        write_csv(out_dir / "bin-summary.csv", result["bin_rows"])
        if write_samples or result["sample_rows"]:
            write_csv(out_dir / "sample-actions.csv", result["sample_rows"])
    write_receipt(out_dir / "PDE_C1_KOLMOGOROV_RESULTS.md", manifest)


def write_receipt(path: Path, manifest: dict) -> None:
    r = manifest["result"]
    c = manifest["config"]
    if c.get("adjudicator") == "knn":
        write_receipt_knn(path, manifest)
        return
    lines = [
        "# PDE C1 Kolmogorov v0 Receipt",
        "",
        f"**Status:** {r['verdict']}",
        f"**Preset:** `{c['preset']}`",
        f"**Adjudicator:** `bin`",
        f"**Interpretable verdict:** `{r['interpretable']}`",
        "",
        "## Readout",
        "",
        f"- `S_eval`: `{r['s_eval']:.6g}` vs `S_pos = {r['s_pos']}`",
        f"- occupied bins: `{r['occupied_bin_count']}`",
        f"- evaluated bins: `{r['evaluated_bin_count']}`",
        f"- incompatible bins: `{r['incompatible_bin_count']}`",
        f"- damp fraction: `{r['damp_fraction']:.6g}`",
        f"- max evaluated minority fraction: `{r['max_evaluated_minority_fraction']:.6g}`",
        f"- elapsed seconds: `{r['elapsed_seconds']:.3f}`",
        f"- steps/sec: `{r['steps_per_second']:.3f}`",
        "",
        "## Branch",
        "",
    ]
    if r["verdict"] == "SMOKE_ONLY":
        lines.append("Smoke-only run; no C1 negative or positive may be filed from this receipt.")
    elif r["verdict"] == "DEFERRED_COVERAGE":
        lines.append("Coverage gate deferred; the pre-registered fallback is one `N_sample = 200000` rerun.")
    elif r["verdict"] == "DEFERRED_VACUITY":
        lines.append(
            "Proxy selector is essentially constant on the sampled support "
            "(damp_fraction outside `[delta_proxy_min, 1 - delta_proxy_min]`); the "
            "strictness predicate has no discriminative content. No verdict filed. "
            "No fall-back is admissible on this cell; re-pinning to a discriminative "
            "regime requires a new cell-set instance (e.g. v1)."
        )
    elif r["verdict"] == "PDE-C1-NEG-A":
        lines.append("Fiber-incompatible evaluated bin found; file `PDE-C1-NEG-A` for this cell.")
    else:
        lines.append("No evaluated bin crossed `delta_action` and the proxy exhibited non-trivial action discrimination; file strictness-witness positive under the proxy.")
    lines.extend(["", "## Files", "", "- `manifest.json`", "- `bin-summary.csv`"])
    if c["sample_count"] <= 5_000:
        lines.append("- `sample-actions.csv`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_receipt_knn(path: Path, manifest: dict) -> None:
    r = manifest["result"]
    c = manifest["config"]
    lines = [
        "# PDE C1 Kolmogorov kNN Receipt",
        "",
        f"**Status:** {r['verdict']}",
        f"**Preset:** `{c['preset']}`",
        f"**Adjudicator:** `knn`",
        f"**Interpretable verdict:** `{r['interpretable']}`",
        "",
        "## Readout",
        "",
        f"- samples: `{r['n_samples']}`, k (effective): `{r['k_neighbors_effective']}`",
        f"- `epsilon_K` (radius threshold): `{r['epsilon_k_radius_threshold']:.6g}`",
        f"- fidelity coverage: `{r['fidelity_coverage']:.6g}` vs `S_pos = {r['s_pos']}` "
        f"(`{r['fidelity_passing_count']}` of `{r['n_samples']}` within `epsilon_K`)",
        f"- `r_k` median / p95 / max: `{r['r_k_median']:.6g}` / `{r['r_k_p95']:.6g}` / `{r['r_k_max']:.6g}`",
        f"- `r_k` median among fidelity-passing: `{r['r_k_median_passing']:.6g}`",
        f"- damp fraction (global): `{r['damp_fraction']:.6g}`",
        f"- incompatible fraction (of passing): `{r['incompat_fraction']:.6g}` "
        f"vs `delta_incompat = {r['delta_incompat']}` (`{r['incompat_count']}` samples)",
        f"- max passing minority fraction: `{r['max_passing_minority_fraction']:.6g}` "
        f"vs `delta_action = {r['delta_action']}`",
        f"- elapsed seconds: `{r['elapsed_seconds']:.3f}`",
        "",
        "## Branch",
        "",
    ]
    if r["verdict"] == "SMOKE_ONLY":
        lines.append("Smoke-only run; no C1 negative or positive may be filed from this receipt.")
    elif r["verdict"] == "DEFERRED_VACUITY":
        lines.append(
            "Proxy selector is essentially constant on the sampled support; the "
            "strictness predicate has no discriminative content. No verdict filed."
        )
    elif r["verdict"] == "DEFERRED_FIDELITY_COVERAGE":
        lines.append(
            "Fewer than `S_pos` of samples have `k` neighbours within `epsilon_K`; the "
            "attractor is too sparse at the tolerance scale for a faithful local fiber "
            "test at this `k` and `N`. The `r_k` histogram (knn-radius-histogram.csv) "
            "maps how far from fidelity the attractor sits. Admissible responses: larger "
            "`N_sample`, a pre-registered larger `epsilon_K` (with the same fidelity "
            "caveat as binning), or reconsidering the continuous-fiber object for this "
            "attractor. Not a fall-back-by-default."
        )
    elif r["verdict"] == "PDE-C1-NEG-A":
        lines.append(
            "A positive-mass fraction of fidelity-passing samples have local minority "
            "fraction above `delta_action`; file `PDE-C1-NEG-A` (fiber incompatibility "
            "under the kNN/disintegration adjudicator)."
        )
    else:
        lines.append(
            "Across the fidelity-passing coverage, the proxy is locally constant on "
            "fibers (incompat fraction below `delta_incompat`) and non-trivially "
            "discriminative (vacuity gate passed); file strictness-witness positive "
            "under the kNN/disintegration adjudicator."
        )
    lines.extend(["", "## Files", "", "- `manifest.json`", "- `knn-radius-histogram.csv`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    args = argparse.Namespace(
        preset="smoke",
        adjudicator="bin",
        out=DEFAULT_OUT,
        write_samples=False,
        self_test=True,
        allow_unregistered_overrides=True,
        sample_count=8,
        burnin_steps=20,
        sample_interval_steps=2,
        lookahead_steps=4,
    )
    cfg = build_config(args)
    result = run_cell(cfg)
    assert result["occupied_bin_count"] > 0
    assert result["no_op_count"] + result["damp_low_band_count"] == cfg.sample_count
    assert math.isfinite(result["e_max"])
    assert result["verdict"] == "SMOKE_ONLY"
    print("[pde-c1] self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    cfg = build_config(args)
    if args.allow_unregistered_overrides and cfg.preset in ("lock", "fallback"):
        print("[pde-c1] overrides present; this receipt is not verdict-bearing despite lock/fallback preset", flush=True)
    result = run_cell(cfg)
    if args.allow_unregistered_overrides:
        result["verdict"] = "SMOKE_ONLY"
        result["verdict_label"] = "manual_overrides_non_verdict"
        result["interpretable"] = False
    write_outputs(args.out.resolve(), cfg, result, args.write_samples)
    print(f"[pde-c1] verdict: {result['verdict']}")
    print(f"[pde-c1] wrote: {args.out.resolve()}")


if __name__ == "__main__":
    main()
