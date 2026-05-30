"""Execute the PDE C1 Kolmogorov-flow v0 cell.

Implements the runnable side of:

  docs/proof/PDE_C1_CELLSET_KOLMOGOROV.md
  docs/proof/PDE_C1_FIBER_PROTOCOL.md

The default preset is a smoke. Use `--preset lock` for the registered v0
sample count. Smoke runs are machinery checks only and never file a C1 verdict.
The `twin-state` adjudicator is a support-level non-injectivity certificate,
not a proxy-control verdict.
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
    "lock_v5_n48",
    "lock_v5_k2",
    "lock_v5_k4",
    "lock_v5_enstrophy",
    "lock_v6",
    "lock_v7_g200",
    "lock_v7_g300",
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
    twin_k_neighbors: int
    twin_delta_high_fraction: float
    twin_high_norm_floor: float
    twin_min_witness_fraction: float
    twin_min_unique_pairs: int
    objective: str
    objective_quantile: float
    calibration_sample_count: int
    calibration_gap_steps: int
    objective_observable: str = "energy"


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
            "lock_v5_n48",
            "lock_v5_k2",
            "lock_v5_k4",
            "lock_v5_enstrophy",
            "lock_v6",
            "lock_v7_g200",
            "lock_v7_g300",
        ],
        default="smoke",
    )
    parser.add_argument(
        "--adjudicator",
        choices=["bin", "knn", "knn-sweep", "twin-state", "mz-budget"],
        default="bin",
        help="Fiber-locality adjudicator: hard binning (default), kNN/disintegration, "
        "knn-sweep (convergence check over k), or twin-state support certificate.",
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
    parser.add_argument(
        "--objective",
        choices=["overshoot-burnin", "portable-quantile"],
        default=None,
        help="Override the preset's objective. Manual use is treated as an "
        "unregistered override (requires --allow-unregistered-overrides; forces SMOKE_ONLY).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    # Cell-set v0 / v1 / v2 / v3 / v4 / v5 / v6 pinned values. v0..v4 use K = 4
    # (signature dim 32). v5 uses K = 3 (signature dim 18) to address the
    # curse-of-dimensionality coverage failure observed at K = 4 with the
    # G = 200, k_f = 2 attractor. v6 keeps K = 3 and changes the regime to
    # G = 300 for the pre-registered regime-generality test.
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
    # Twin-state support certificate defaults (pre-registered in
    # PDE_C1_TWIN_STATE_CERTIFICATE.md): k neighbours including self; delta_H is
    # a fixed fraction of the sample median high-mode norm.
    twin_k_neighbors = 50
    twin_delta_high_fraction = 0.05
    twin_high_norm_floor = 1e-6
    twin_min_witness_fraction = 0.01
    twin_min_unique_pairs = 100
    # Objective (regime-generality v1, PDE_C1_REGIME_GENERALITY_v1.md): v0..v6
    # use the burn-in overshoot proxy; lock_v7_* use a held-out look-ahead-max
    # quantile that pins damp_fraction = 1 - q by construction at every regime.
    objective = "overshoot-burnin"
    objective_quantile = 0.70
    calibration_sample_count = 0
    calibration_gap_steps = 0
    # Trigger observable for the objective: low-band energy (default) or low-band
    # enstrophy (robustness wave, objective axis). The signature Phi_K is
    # unchanged; only the quantity the safety trigger watches changes.
    objective_observable = "energy"
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
    elif args.preset == "lock_v5_n48":
        # Robustness wave, N-refinement: identical to lock_v5 (k_f=2, G=200,
        # K=3, overshoot-burnin objective + E_max last-25% amendment, same nu,
        # dt, sampling) EXCEPT the Galerkin resolution is refined grid 32 -> 48
        # (dealias cutoff ~10 -> ~16 modes; n_modes 16 -> 24). nu is unchanged
        # (G fixed by grashof), the K=3 signature is the same 9 modes. Tests
        # refinement-invariance of the regime-2 separation. Spec:
        # docs/proof/PDE_C1_ROBUSTNESS_WAVE.md.
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 200.0
        e_max_burnin_fraction = 0.25
        k_signature = 3
        grid_size = 48
        n_modes = 24
    elif args.preset in ("lock_v5_k2", "lock_v5_k4"):
        # Robustness wave, K-window: identical to lock_v5 except the signature
        # band K (observation-choice axis). k2 is the coarser lower-bracket
        # test; k4 is the finer upper probe (expected coverage-limited per the
        # v2/v4 curse-of-dimensionality finding that motivated K=3). Spec:
        # docs/proof/PDE_C1_ROBUSTNESS_WAVE.md.
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 200.0
        e_max_burnin_fraction = 0.25
        k_signature = 2 if args.preset == "lock_v5_k2" else 4
    elif args.preset == "lock_v5_enstrophy":
        # Robustness wave, objective axis: identical to lock_v5 except the
        # safety trigger watches low-band ENSTROPHY (Sum_low |omega|^2) instead
        # of energy. The signature Phi_K is unchanged. Tests clause (ii)
        # objective-robustness beyond the energy proxy.
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 200.0
        e_max_burnin_fraction = 0.25
        k_signature = 3
        objective_observable = "enstrophy"
    elif args.preset == "fallback_v5":
        burnin_steps = 100_000
        sample_count = 200_000
        kf = 2
        grashof = 200.0
        e_max_burnin_fraction = 0.25
        k_signature = 3
    elif args.preset == "lock_v6":
        # v6 / regime-generality v0: same objective, k_f, K, sampling, and
        # E_max amendment as v5, but at higher Grashof G = 300.
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 300.0
        e_max_burnin_fraction = 0.25
        k_signature = 3
    elif args.preset in ("lock_v7_g200", "lock_v7_g300"):
        # Regime-generality v1 (PDE_C1_REGIME_GENERALITY_v1.md): portable
        # objective = held-out look-ahead-max quantile (q=0.70), calibrated on a
        # disjoint post-burn-in window. g200 is the mandatory positive control;
        # g300 is the generality test. K=3, k_f=2 inherited from v5/v6.
        burnin_steps = 100_000
        sample_count = 50_000
        kf = 2
        grashof = 200.0 if args.preset == "lock_v7_g200" else 300.0
        k_signature = 3
        objective = "portable-quantile"
        objective_quantile = 0.70
        calibration_sample_count = 50_000
        calibration_gap_steps = 5_000
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
        "objective": args.objective,
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
        if args.objective is not None:
            objective = args.objective

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
        objective=objective,
        objective_quantile=objective_quantile,
        calibration_sample_count=calibration_sample_count,
        calibration_gap_steps=calibration_gap_steps,
        objective_observable=objective_observable,
        twin_k_neighbors=twin_k_neighbors,
        twin_delta_high_fraction=twin_delta_high_fraction,
        twin_high_norm_floor=twin_high_norm_floor,
        twin_min_witness_fraction=twin_min_witness_fraction,
        twin_min_unique_pairs=twin_min_unique_pairs,
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
        self.high_indices = select_high_modes(wave, cfg.n_modes, self.dealias_mask, self.low_indices)
        # Low-pass mask for the MZ energy-budget decomposition: True only at the
        # signature low modes and their conjugates (so the low-passed field is a
        # real-valued band projection). Spec: PDE_C1_MZ_ENERGY_BUDGET.md.
        self.low_pass_mask = np.zeros((m, m), dtype=bool)
        for ix, iy in self.low_indices:
            self.low_pass_mask[ix, iy] = True
            self.low_pass_mask[(m - ix) % m, (m - iy) % m] = True

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

    def mode_vector(self, omega_hat: np.ndarray, indices: list[tuple[int, int]]) -> np.ndarray:
        scale = float(self.cfg.grid_size * self.cfg.grid_size)
        omega_norm = omega_hat / scale
        values: list[float] = []
        for ix, iy in indices:
            k_norm = math.sqrt(float(self.k2[ix, iy]))
            amp = omega_norm[ix, iy] / k_norm
            values.append(float(amp.real))
            values.append(float(amp.imag))
        return np.asarray(values, dtype=np.float64)

    def signature(self, omega_hat: np.ndarray) -> np.ndarray:
        return self.mode_vector(omega_hat, self.low_indices)

    def high_mode_vector(self, omega_hat: np.ndarray) -> np.ndarray:
        return self.mode_vector(omega_hat, self.high_indices)

    def low_energy(self, omega_hat: np.ndarray) -> float:
        sig = self.signature(omega_hat)
        return float(np.dot(sig, sig))

    def low_enstrophy(self, omega_hat: np.ndarray) -> float:
        """Low-band enstrophy Sum_low |omega_hat/scale|^2 (energy without the
        1/|k|^2 weighting). Robustness-wave objective-axis trigger; the
        signature Phi_K is unchanged."""
        scale = float(self.cfg.grid_size * self.cfg.grid_size)
        total = 0.0
        for ix, iy in self.low_indices:
            amp = omega_hat[ix, iy] / scale
            total += float((amp * amp.conjugate()).real)
        return total

    def energy_budget(self, omega_hat: np.ndarray) -> dict:
        """Mori-Zwanzig decomposition of the low-band energy tendency.

        `dE_low/dt = g(Phi_K) + R`, where `g = D_low + F_low + T_LLL` (low
        dissipation + forcing + band-closed nonlinear transfer) depends only
        on the low modes, and `R` is the transfer involving >=1 high mode
        (the orthogonal-dynamics / MZ coupling), obtained by re-evaluating the
        nonlinear term on the low-passed field. Exact; no closure model.
        Sums over the same low modes as `low_energy`. Spec:
        docs/proof/PDE_C1_MZ_ENERGY_BUDGET.md.
        """
        scale2 = float(self.cfg.grid_size) ** 4
        nu = self.cfg.viscosity
        nl_full = self.nonlinear_hat(omega_hat)
        nl_low = self.nonlinear_hat(omega_hat * self.low_pass_mask)
        f = self.forcing_hat
        t_low = 0.0
        t_lll = 0.0
        d_low = 0.0
        f_low = 0.0
        for ix, iy in self.low_indices:
            k2 = float(self.k2[ix, iy])
            w = omega_hat[ix, iy]
            wc = w.conjugate()
            t_low += -(2.0 / (scale2 * k2)) * float((wc * nl_full[ix, iy]).real)
            t_lll += -(2.0 / (scale2 * k2)) * float((wc * nl_low[ix, iy]).real)
            d_low += -(2.0 * nu / scale2) * float(abs(w) ** 2)
            f_low += (2.0 / (scale2 * k2)) * float((wc * f[ix, iy]).real)
        r = t_low - t_lll
        g = d_low + f_low + t_lll
        return {
            "dEdt": g + r,
            "g": g,
            "R": r,
            "D_low": d_low,
            "F_low": f_low,
            "T_low": t_low,
            "T_lll": t_lll,
        }


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


def select_high_modes(
    wave: np.ndarray,
    n_modes: int,
    dealias_mask: np.ndarray,
    low_indices: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    low = set(low_indices)
    modes: list[tuple[float, float, int, int]] = []
    for ix, kx in enumerate(wave):
        for iy, ky in enumerate(wave):
            if kx == 0 and ky == 0:
                continue
            if not bool(dealias_mask[ix, iy]):
                continue
            if abs(kx) > n_modes or abs(ky) > n_modes:
                continue
            # Half-plane representatives for a real field.
            if not (ky > 0 or (ky == 0 and kx > 0)):
                continue
            if (ix, iy) in low:
                continue
            modes.append((max(abs(kx), abs(ky)), kx * kx + ky * ky, ix, iy))
    modes.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return [(ix, iy) for _, _, ix, iy in modes]


def run_cell(cfg: RunConfig) -> dict:
    stepper = KolmogorovStepper(cfg)
    omega_hat = stepper.initial_state()
    interval = cfg.sample_interval_steps
    look = cfg.lookahead_steps
    burn = cfg.burnin_steps
    portable = cfg.objective == "portable-quantile"

    # Sampling schedule. For the portable-quantile objective a held-out
    # calibration block precedes the adjudication block, separated by a
    # decorrelation gap; E_max is the q-quantile of the calibration block's
    # look-ahead-max excursions. For overshoot-burnin only the adjudication
    # block exists and E_max comes from the burn-in window. The overshoot path
    # is byte-identical to the prior schedule (adj_start0 == burn).
    if portable:
        calib_starts = [burn + i * interval for i in range(cfg.calibration_sample_count)]
        calib_block_end = (calib_starts[-1] + look) if calib_starts else burn
        adj_start0 = calib_block_end + cfg.calibration_gap_steps
    else:
        calib_starts = []
        adj_start0 = burn
    adj_starts = [adj_start0 + i * interval for i in range(cfg.sample_count)]
    total_steps = adj_starts[-1] + look
    adj_step_to_idx = {s: i for i, s in enumerate(adj_starts)}

    burnin_energies: list[float] = []
    energy_by_step = np.empty(total_steps + 1, dtype=np.float64)
    sample_signatures = np.empty((cfg.sample_count, cfg.signature_dimension), dtype=np.float64)
    sample_energy = np.empty(cfg.sample_count, dtype=np.float64)
    sample_high_modes = (
        np.empty((cfg.sample_count, 2 * len(stepper.high_indices)), dtype=np.float64)
        if cfg.adjudicator == "twin-state"
        else None
    )
    sample_mz = (
        np.empty((cfg.sample_count, 7), dtype=np.float64)
        if cfg.adjudicator == "mz-budget"
        else None
    )
    sig_min = np.full(cfg.signature_dimension, np.inf)
    sig_max = np.full(cfg.signature_dimension, -np.inf)

    started = time.perf_counter()
    progress_stride = max(1, total_steps // 10)
    observable = stepper.low_enstrophy if cfg.objective_observable == "enstrophy" else stepper.low_energy
    for step in range(total_steps + 1):
        energy = observable(omega_hat)
        energy_by_step[step] = energy
        if step <= burn:
            sig = stepper.signature(omega_hat)
            sig_min = np.minimum(sig_min, sig)
            sig_max = np.maximum(sig_max, sig)
            burnin_energies.append(energy)
        idx = adj_step_to_idx.get(step)
        if idx is not None:
            sig = stepper.signature(omega_hat)
            sample_signatures[idx, :] = sig
            sample_energy[idx] = energy
            if sample_high_modes is not None:
                sample_high_modes[idx, :] = stepper.high_mode_vector(omega_hat)
            if sample_mz is not None:
                b = stepper.energy_budget(omega_hat)
                sample_mz[idx, :] = (
                    b["dEdt"], b["g"], b["R"], b["D_low"], b["F_low"], b["T_low"], b["T_lll"]
                )
        if step < total_steps:
            omega_hat = stepper.step(omega_hat)
        if step > 0 and step % progress_stride == 0:
            elapsed = time.perf_counter() - started
            print(f"[pde-c1] step {step}/{total_steps} ({100 * step / total_steps:.0f}%), elapsed {elapsed:.1f}s", flush=True)

    def _lookahead_max(starts: list[int]) -> np.ndarray:
        return np.array(
            [float(np.max(energy_by_step[s : s + look + 1])) for s in starts],
            dtype=np.float64,
        )

    burnin_array = np.asarray(burnin_energies)
    if portable:
        # Held-out look-ahead-max quantile: E_max = q-quantile of the calibration
        # block's excursions. Pins damp_fraction = 1 - q by construction.
        calib_lookahead_max = _lookahead_max(calib_starts)
        e_max = float(np.quantile(calib_lookahead_max, cfg.objective_quantile))
        calibration_damp_fraction = float(np.mean(calib_lookahead_max > e_max))
    else:
        # E_max windowing: last fraction of burn-in to exclude transients.
        calib_lookahead_max = np.asarray([], dtype=np.float64)
        calibration_damp_fraction = float("nan")
        window_len = max(1, int(cfg.e_max_burnin_fraction * len(burnin_array)))
        e_max = float(np.percentile(burnin_array[-window_len:], 95))
    epsilon_k = 0.05 * math.sqrt(max(0.0, 2.0 * e_max))
    h_k = epsilon_k / math.sqrt(cfg.signature_dimension) if epsilon_k > 0 else 1.0
    adj_lookahead_max = _lookahead_max(adj_starts)
    actions = (adj_lookahead_max > e_max).astype(np.int8)
    knn_histogram: list = []
    knn_sweep_rows: list = []
    twin_witness_rows: list = []
    if cfg.adjudicator == "knn":
        result, bin_rows, sample_rows, knn_histogram = aggregate_knn(
            sample_signatures, actions, epsilon_k, cfg
        )
    elif cfg.adjudicator == "knn-sweep":
        result, bin_rows, sample_rows, knn_sweep_rows = aggregate_knn_sweep(
            sample_signatures, actions, epsilon_k, cfg
        )
    elif cfg.adjudicator == "twin-state":
        result, bin_rows, sample_rows, twin_witness_rows = aggregate_twin_state(
            sample_signatures, sample_high_modes, actions, epsilon_k, cfg
        )
    elif cfg.adjudicator == "mz-budget":
        result, bin_rows, sample_rows, _ = aggregate_mz_budget(
            sample_signatures, sample_mz, actions, epsilon_k, cfg
        )
    else:
        bin_rows, sample_rows = aggregate_bins(sample_signatures, sample_energy, actions, sig_min, h_k, cfg)
        result = summarize(bin_rows, cfg)
    result.update(
        {
            "adjudicator": cfg.adjudicator,
            "objective": cfg.objective,
            "objective_quantile": cfg.objective_quantile if portable else None,
            "calibration_sample_count": cfg.calibration_sample_count if portable else 0,
            "calibration_damp_fraction": calibration_damp_fraction,
            "knn_histogram": knn_histogram,
            "knn_sweep_rows": knn_sweep_rows,
            "twin_witness_rows": twin_witness_rows,
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
            "high_mode_count": len(stepper.high_indices),
            "high_mode_dimension": 2 * len(stepper.high_indices),
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


def aggregate_knn_sweep(
    signatures: np.ndarray,
    actions: np.ndarray,
    epsilon_k: float,
    cfg: RunConfig,
) -> tuple[dict, list, list, list]:
    """kNN convergence check (pre-registered: PDE_C1_KNN_CONVERGENCE_CHECK.md).

    Query the BallTree once at k = max(sweep), sub-slice for each k, and report
    incompat_fraction vs neighbourhood radius. Distinguishes genuine
    fiber-incompatibility (incompat plateaus as r_k -> 0) from a finite-radius
    boundary-straddling artifact (incompat decays to 0).
    """
    from sklearn.neighbors import BallTree

    n = int(signatures.shape[0])
    # Amended pre-registration (PDE_C1_KNN_CONVERGENCE_CHECK.md §6):
    # denser low-k sweep, all expected full-coverage; no coverage-failing point.
    k_sweep = [k for k in (10, 15, 20, 25, 30, 40, 50) if k <= n]
    k_query = min(max(k_sweep), n) if k_sweep else min(1, n)
    tree = BallTree(signatures, metric="euclidean")
    dist, idx = tree.query(signatures, k=k_query)
    damp = int(actions.sum())
    no_op = int(n - damp)

    rows: list[dict] = []
    for k in k_sweep:
        r_k = dist[:, k - 1]
        nbr = actions[idx[:, :k]]
        damp_count = nbr.sum(axis=1)
        no_op_count = k - damp_count
        majority_count = np.maximum(damp_count, no_op_count)
        minority = 1.0 - majority_count / float(k)
        fidelity_pass = r_k <= epsilon_k
        f_count = int(fidelity_pass.sum())
        fcov = f_count / n if n else 0.0
        if f_count:
            incompat = float(((minority > cfg.delta_action) & fidelity_pass).sum()) / f_count
            mean_minority = float(np.mean(minority[fidelity_pass]))
        else:
            incompat = 0.0
            mean_minority = 0.0
        rows.append(
            {
                "k": k,
                "r_k_median": float(np.median(r_k)),
                "r_k_median_passing": float(np.median(r_k[fidelity_pass])) if f_count else 0.0,
                "fidelity_coverage": fcov,
                "incompat_fraction": incompat,
                "mean_minority": mean_minority,
            }
        )

    # Fit only over coverage-passing points (amended pre-registration §6 fix 1).
    fit_rows = [row for row in rows if row["fidelity_coverage"] >= cfg.s_pos]
    if len(fit_rows) >= 2:
        xs = np.array([row["r_k_median"] for row in fit_rows], dtype=np.float64)
        slope_mm, intercept_mm = (float(v) for v in np.polyfit(xs, np.array([r["mean_minority"] for r in fit_rows]), 1))
        slope_inc, intercept_inc = (float(v) for v in np.polyfit(xs, np.array([r["incompat_fraction"] for r in fit_rows]), 1))
    else:
        slope_mm = intercept_mm = slope_inc = intercept_inc = 0.0

    result = summarize_knn_sweep(
        rows=rows,
        fit_rows=fit_rows,
        intercept_mm=intercept_mm,
        slope_mm=slope_mm,
        intercept_inc=intercept_inc,
        slope_inc=slope_inc,
        damp=damp,
        no_op=no_op,
        n=n,
        epsilon_k=epsilon_k,
        cfg=cfg,
    )
    return result, [], [], rows


def summarize_knn_sweep(
    *,
    rows: list,
    fit_rows: list,
    intercept_mm: float,
    slope_mm: float,
    intercept_inc: float,
    slope_inc: float,
    damp: int,
    no_op: int,
    n: int,
    epsilon_k: float,
    cfg: RunConfig,
) -> dict:
    # Amended classification (PDE_C1_KNN_CONVERGENCE_CHECK.md §6): primary
    # statistic is the threshold-free mean local minority fraction; classify
    # on its r_k -> 0 intercept a_mm over coverage-passing points only.
    A_MM_LO = 0.005
    A_MM_HI = 0.015
    damp_fraction = damp / max(1, n)
    proxy_structural_constant = damp == 0 or no_op == 0
    proxy_constant = (
        damp_fraction < cfg.delta_proxy_min or damp_fraction > 1.0 - cfg.delta_proxy_min
    )

    if cfg.preset not in VERDICT_BEARING_PRESETS:
        verdict, verdict_label, interpretable = "SMOKE_ONLY", "", False
    elif proxy_structural_constant or proxy_constant:
        verdict, verdict_label, interpretable = (
            "DEFERRED_VACUITY",
            "proxy_selector_constant_on_sampled_support",
            False,
        )
    elif len(fit_rows) < 2:
        verdict, verdict_label, interpretable = (
            "INCONCLUSIVE_CONVERGENCE",
            "insufficient_coverage_passing_sweep_points",
            False,
        )
    elif intercept_mm <= A_MM_LO:
        verdict, verdict_label, interpretable = (
            "STRICTNESS_WITNESS_POSITIVE",
            "mean_minority_decays_to_zero_boundary_artifact",
            True,
        )
    elif intercept_mm >= A_MM_HI:
        verdict, verdict_label, interpretable = (
            "PDE-C1-NEG-A",
            "mean_minority_plateau_nonzero_genuine",
            True,
        )
    else:
        verdict, verdict_label, interpretable = (
            "INCONCLUSIVE_CONVERGENCE",
            "mean_minority_intercept_in_ambiguous_band",
            False,
        )

    return {
        "verdict": verdict,
        "verdict_label": verdict_label,
        "interpretable": interpretable,
        "n_samples": n,
        "mean_minority_intercept": intercept_mm,
        "mean_minority_slope": slope_mm,
        "incompat_intercept": intercept_inc,
        "incompat_slope": slope_inc,
        "fit_point_count": len(fit_rows),
        "a_mm_lo": A_MM_LO,
        "a_mm_hi": A_MM_HI,
        "delta_incompat": cfg.delta_incompat,
        "delta_action": cfg.delta_action,
        "no_op_count": no_op,
        "damp_low_band_count": damp,
        "damp_fraction": damp_fraction,
        "epsilon_k_radius_threshold": epsilon_k,
        "sweep_k_values": [row["k"] for row in rows],
        "sweep_mean_minority": [row["mean_minority"] for row in rows],
        "sweep_incompat_fractions": [row["incompat_fraction"] for row in rows],
        "sweep_r_k_medians": [row["r_k_median"] for row in rows],
        "sweep_fidelity_coverages": [row["fidelity_coverage"] for row in rows],
    }


def aggregate_twin_state(
    signatures: np.ndarray,
    high_vectors: np.ndarray | None,
    actions: np.ndarray,
    epsilon_k: float,
    cfg: RunConfig,
) -> tuple[dict, list, list, list]:
    """Support-level non-injectivity certificate.

    Search the sampled SRB-like support for signature-near pairs whose
    complementary high-mode coordinates are separated by a pre-registered
    threshold delta_H. This is a state-insufficiency certificate, not a
    proxy-control adjudication.
    """
    if high_vectors is None:
        raise ValueError("twin-state adjudicator requires captured high-mode vectors")

    from sklearn.neighbors import BallTree

    n = int(signatures.shape[0])
    high_norms = np.linalg.norm(high_vectors, axis=1) if high_vectors.size else np.zeros(n)
    high_norm_median = float(np.median(high_norms)) if n else 0.0
    delta_h_raw = cfg.twin_delta_high_fraction * high_norm_median
    delta_h = max(cfg.twin_high_norm_floor, delta_h_raw)
    damp = int(actions.sum())
    no_op = int(n - damp)

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q)) if a.size else 0.0

    if n < 2:
        result = summarize_twin_state(
            n=n,
            k=0,
            epsilon_k=epsilon_k,
            delta_h=delta_h,
            high_norms=high_norms,
            candidate_sample_count=0,
            candidate_pair_count_directed=0,
            candidate_pair_count_unique=0,
            witness_sample_count=0,
            witness_pair_count_directed=0,
            witness_pair_count_unique=0,
            candidate_action_disagree_directed=0,
            candidate_action_disagree_unique=0,
            witness_action_disagree_directed=0,
            witness_action_disagree_unique=0,
            max_candidate_high_distance=0.0,
            witness_high_distances=np.asarray([], dtype=np.float64),
            witness_signature_distances=np.asarray([], dtype=np.float64),
            damp=damp,
            no_op=no_op,
            cfg=cfg,
        )
        return result, [], [], []

    k = min(max(2, cfg.twin_k_neighbors), n)
    tree = BallTree(signatures, metric="euclidean")
    dist, idx = tree.query(signatures, k=k)
    nbr_dist = dist[:, 1:]
    nbr_idx = idx[:, 1:]

    candidate_sample = np.zeros(n, dtype=bool)
    witness_sample = np.zeros(n, dtype=bool)
    candidate_pair_count_directed = 0
    witness_pair_count_directed = 0
    # Paired fiber-constancy accumulators (PDE_C1_PAIRED_FIBER_CONSTANCY.md):
    # for each signature-near pair, does the proxy action agree? This turns the
    # matched-radius composition (twin-state non-injectivity + kNN control read)
    # into a paired test on the SAME pairs.
    candidate_action_disagree_directed = 0
    witness_action_disagree_directed = 0
    candidate_disagree_code_chunks: list[np.ndarray] = []
    witness_disagree_code_chunks: list[np.ndarray] = []
    max_candidate_high_distance = 0.0
    candidate_code_chunks: list[np.ndarray] = []
    witness_code_chunks: list[np.ndarray] = []
    witness_high_chunks: list[np.ndarray] = []
    witness_sig_chunks: list[np.ndarray] = []
    witness_rows: list[dict] = []
    row_limit = 200

    chunk_size = 512
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        row_ids = np.arange(start, end, dtype=np.int64)
        dist_chunk = nbr_dist[start:end]
        idx_chunk = nbr_idx[start:end]
        candidate_mask = dist_chunk <= epsilon_k
        if not bool(candidate_mask.any()):
            continue

        candidate_sample[start:end] = candidate_mask.any(axis=1)
        candidate_pair_count_directed += int(candidate_mask.sum())
        diff = high_vectors[idx_chunk] - high_vectors[row_ids[:, None]]
        high_dist = np.linalg.norm(diff, axis=2)
        candidate_high = high_dist[candidate_mask]
        if candidate_high.size:
            max_candidate_high_distance = max(max_candidate_high_distance, float(candidate_high.max()))

        lo = np.minimum(row_ids[:, None], idx_chunk)
        hi = np.maximum(row_ids[:, None], idx_chunk)
        codes = lo * n + hi
        candidate_code_chunks.append(codes[candidate_mask])

        # Paired fiber-constancy read: do the two members of a signature-near
        # pair require the same proxy action? Action label is symmetric in the
        # pair, so directed disagreement de-dupes to the same unique fraction.
        action_diff = actions[idx_chunk] != actions[row_ids[:, None]]
        candidate_disagree = candidate_mask & action_diff
        candidate_action_disagree_directed += int(candidate_disagree.sum())
        candidate_disagree_code_chunks.append(codes[candidate_disagree])

        witness_mask = candidate_mask & (high_dist >= delta_h)
        if not bool(witness_mask.any()):
            continue

        witness_sample[start:end] = witness_mask.any(axis=1)
        witness_pair_count_directed += int(witness_mask.sum())
        witness_code_chunks.append(codes[witness_mask])
        witness_high_chunks.append(high_dist[witness_mask])
        witness_sig_chunks.append(dist_chunk[witness_mask])

        witness_disagree = witness_mask & action_diff
        witness_action_disagree_directed += int(witness_disagree.sum())
        witness_disagree_code_chunks.append(codes[witness_disagree])

        if len(witness_rows) < row_limit:
            positions = np.argwhere(witness_mask)
            for local_i, local_j in positions:
                i = int(row_ids[local_i])
                j = int(idx_chunk[local_i, local_j])
                witness_rows.append(
                    {
                        "sample_i": i,
                        "sample_j": j,
                        "signature_distance": float(dist_chunk[local_i, local_j]),
                        "high_mode_distance": float(high_dist[local_i, local_j]),
                        "high_distance_over_delta_h": float(high_dist[local_i, local_j] / delta_h)
                        if delta_h > 0
                        else 0.0,
                        "sample_i_high_norm": float(high_norms[i]),
                        "sample_j_high_norm": float(high_norms[j]),
                        "action_i": int(actions[i]),
                        "action_j": int(actions[j]),
                        "action_agree": bool(actions[i] == actions[j]),
                    }
                )
                if len(witness_rows) >= row_limit:
                    break

    candidate_pair_count_unique = unique_pair_count(candidate_code_chunks)
    witness_pair_count_unique = unique_pair_count(witness_code_chunks)
    candidate_action_disagree_unique = unique_pair_count(candidate_disagree_code_chunks)
    witness_action_disagree_unique = unique_pair_count(witness_disagree_code_chunks)
    witness_high_distances = (
        np.concatenate(witness_high_chunks) if witness_high_chunks else np.asarray([], dtype=np.float64)
    )
    witness_signature_distances = (
        np.concatenate(witness_sig_chunks) if witness_sig_chunks else np.asarray([], dtype=np.float64)
    )

    result = summarize_twin_state(
        n=n,
        k=k,
        epsilon_k=epsilon_k,
        delta_h=delta_h,
        high_norms=high_norms,
        candidate_sample_count=int(candidate_sample.sum()),
        candidate_pair_count_directed=candidate_pair_count_directed,
        candidate_pair_count_unique=candidate_pair_count_unique,
        witness_sample_count=int(witness_sample.sum()),
        witness_pair_count_directed=witness_pair_count_directed,
        witness_pair_count_unique=witness_pair_count_unique,
        candidate_action_disagree_directed=candidate_action_disagree_directed,
        candidate_action_disagree_unique=candidate_action_disagree_unique,
        witness_action_disagree_directed=witness_action_disagree_directed,
        witness_action_disagree_unique=witness_action_disagree_unique,
        max_candidate_high_distance=max_candidate_high_distance,
        witness_high_distances=witness_high_distances,
        witness_signature_distances=witness_signature_distances,
        damp=damp,
        no_op=no_op,
        cfg=cfg,
    )
    result.update(
        {
            "high_norm_p05": pct(high_norms, 5),
            "high_norm_p50": pct(high_norms, 50),
            "high_norm_p95": pct(high_norms, 95),
        }
    )
    return result, [], [], witness_rows


def unique_pair_count(code_chunks: list[np.ndarray]) -> int:
    nonempty = [chunk for chunk in code_chunks if chunk.size]
    if not nonempty:
        return 0
    return int(np.unique(np.concatenate(nonempty)).size)


def summarize_twin_state(
    *,
    n: int,
    k: int,
    epsilon_k: float,
    delta_h: float,
    high_norms: np.ndarray,
    candidate_sample_count: int,
    candidate_pair_count_directed: int,
    candidate_pair_count_unique: int,
    witness_sample_count: int,
    witness_pair_count_directed: int,
    witness_pair_count_unique: int,
    candidate_action_disagree_directed: int = 0,
    candidate_action_disagree_unique: int = 0,
    witness_action_disagree_directed: int = 0,
    witness_action_disagree_unique: int = 0,
    max_candidate_high_distance: float,
    witness_high_distances: np.ndarray,
    witness_signature_distances: np.ndarray,
    damp: int,
    no_op: int,
    cfg: RunConfig,
) -> dict:
    high_norm_median = float(np.median(high_norms)) if high_norms.size else 0.0
    candidate_sample_fraction = candidate_sample_count / n if n else 0.0
    witness_sample_fraction = witness_sample_count / n if n else 0.0
    witness_fraction_of_candidates = (
        witness_sample_count / candidate_sample_count if candidate_sample_count else 0.0
    )
    damp_fraction = damp / max(1, n)

    if cfg.preset not in VERDICT_BEARING_PRESETS:
        verdict, verdict_label, interpretable = "SMOKE_ONLY", "", False
    elif high_norm_median <= cfg.twin_high_norm_floor:
        verdict, verdict_label, interpretable = (
            "TWIN_STATE_DEFERRED_HIGH_MODE_FLOOR",
            "sampled_support_high_modes_numerically_flat",
            False,
        )
    elif candidate_sample_fraction < cfg.s_pos:
        verdict, verdict_label, interpretable = (
            "TWIN_STATE_DEFERRED_COVERAGE",
            "insufficient_signature_near_pair_coverage",
            False,
        )
    elif (
        witness_sample_fraction >= cfg.twin_min_witness_fraction
        and witness_pair_count_unique >= cfg.twin_min_unique_pairs
    ):
        verdict, verdict_label, interpretable = (
            "TWIN_STATE_CERTIFIED",
            "signature_near_high_mode_separated_twins",
            True,
        )
    else:
        verdict, verdict_label, interpretable = (
            "TWIN_STATE_NO_CERTIFICATE",
            "no_pre_registered_positive_mass_twin_witness",
            False,
        )

    # Paired fiber-constancy: among the state-separated (witness) pairs the
    # certificate already found, what fraction require DIFFERENT proxy actions?
    # This is the fiber criterion Phi_K(x0)=Phi_K(x1) => pi*(x0)=pi*(x1) tested
    # directly on the certified non-injective pairs. Spec:
    # docs/proof/PDE_C1_PAIRED_FIBER_CONSTANCY.md.
    witness_action_disagree_fraction_unique = (
        witness_action_disagree_unique / witness_pair_count_unique
        if witness_pair_count_unique
        else 0.0
    )
    witness_action_disagree_fraction_directed = (
        witness_action_disagree_directed / witness_pair_count_directed
        if witness_pair_count_directed
        else 0.0
    )
    candidate_action_disagree_fraction_unique = (
        candidate_action_disagree_unique / candidate_pair_count_unique
        if candidate_pair_count_unique
        else 0.0
    )
    # Secondary verdict; does NOT override the primary twin-state verdict. Only
    # interpretable when the certificate stands and the proxy is non-constant.
    proxy_structural_constant = damp == 0 or no_op == 0
    if cfg.preset not in VERDICT_BEARING_PRESETS:
        paired_fiber_verdict = "SMOKE_ONLY"
    elif verdict != "TWIN_STATE_CERTIFIED" or witness_pair_count_unique == 0:
        paired_fiber_verdict = "PAIRED_FIBER_UNDEFINED"
    elif proxy_structural_constant:
        paired_fiber_verdict = "PAIRED_FIBER_DEFERRED_VACUITY"
    elif witness_action_disagree_fraction_unique <= cfg.delta_action:
        paired_fiber_verdict = "PAIRED_FIBER_CONSTANCY_POSITIVE"
    else:
        paired_fiber_verdict = "PDE-C1-PAIRED-NEG"

    def pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q)) if a.size else 0.0

    return {
        "verdict": verdict,
        "verdict_label": verdict_label,
        "interpretable": interpretable,
        "paired_fiber_verdict": paired_fiber_verdict,
        "witness_action_disagree_unique": witness_action_disagree_unique,
        "witness_action_disagree_directed": witness_action_disagree_directed,
        "witness_action_disagree_fraction_unique": witness_action_disagree_fraction_unique,
        "witness_action_disagree_fraction_directed": witness_action_disagree_fraction_directed,
        "candidate_action_disagree_unique": candidate_action_disagree_unique,
        "candidate_action_disagree_fraction_unique": candidate_action_disagree_fraction_unique,
        "paired_fiber_delta_action": cfg.delta_action,
        "n_samples": n,
        "k_neighbors_effective": k,
        "epsilon_k_radius_threshold": epsilon_k,
        "delta_h": delta_h,
        "twin_delta_high_fraction": cfg.twin_delta_high_fraction,
        "twin_high_norm_floor": cfg.twin_high_norm_floor,
        "twin_min_witness_fraction": cfg.twin_min_witness_fraction,
        "twin_min_unique_pairs": cfg.twin_min_unique_pairs,
        "candidate_sample_count": candidate_sample_count,
        "candidate_sample_fraction": candidate_sample_fraction,
        "candidate_pair_count_directed": candidate_pair_count_directed,
        "candidate_pair_count_unique": candidate_pair_count_unique,
        "witness_sample_count": witness_sample_count,
        "witness_sample_fraction": witness_sample_fraction,
        "witness_fraction_of_candidates": witness_fraction_of_candidates,
        "witness_pair_count_directed": witness_pair_count_directed,
        "witness_pair_count_unique": witness_pair_count_unique,
        "max_candidate_high_distance": max_candidate_high_distance,
        "witness_high_distance_p50": pct(witness_high_distances, 50),
        "witness_high_distance_p95": pct(witness_high_distances, 95),
        "witness_signature_distance_p50": pct(witness_signature_distances, 50),
        "witness_signature_distance_p95": pct(witness_signature_distances, 95),
        "high_norm_median": high_norm_median,
        "high_norm_min": float(high_norms.min()) if high_norms.size else 0.0,
        "high_norm_max": float(high_norms.max()) if high_norms.size else 0.0,
        "no_op_count": no_op,
        "damp_low_band_count": damp,
        "damp_fraction": damp_fraction,
        "s_pos": cfg.s_pos,
    }


def aggregate_mz_budget(
    sample_signatures: np.ndarray,
    sample_mz: np.ndarray,
    actions: np.ndarray,
    epsilon_k: float,
    cfg: RunConfig,
) -> tuple[dict, list, list, list]:
    """Mori-Zwanzig coupling-disintegration diagnostic (Level 1 v2).

    EXPLANATORY, not promotion-bearing. The energy budget is dE_low/dt =
    g(Phi_K) + R, with g = D_low + F_low (low-determined; the band-closed
    transfer T_LLL is identically 0 by energy conservation) and R = T_low the
    full inter-scale transfer. The mechanism test is whether R is *predictable
    from Phi_K*: disintegrate R over the signature with kNN and compare its
    conditional variance to a perfect-function floor (g) and a no-dependence
    ceiling (permuted R). Spec: docs/proof/PDE_C1_MZ_ENERGY_BUDGET.md.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    dEdt = sample_mz[:, 0]
    g = sample_mz[:, 1]
    R = sample_mz[:, 2]
    D_low = sample_mz[:, 3]
    F_low = sample_mz[:, 4]
    T_low = sample_mz[:, 5]
    n = int(sample_mz.shape[0])

    def rms(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean(a * a))) if a.size else 0.0

    # Predictability of a target from the signature, as held-out R^2 of a
    # flexible regressor. Block split (first 70% train / last 30% test) avoids
    # temporal leakage. Unbiased and steepness-agnostic (unlike kNN conditional
    # variance, which the 18-dim neighbourhood width + g's quadratic steepness
    # confound). R^2 -> 1 means the target is a function of Phi_K.
    ntr = max(2, int(0.7 * n))
    x_tr, x_te = sample_signatures[:ntr], sample_signatures[ntr:]

    def r2_from_signature(target: np.ndarray) -> float:
        y_tr, y_te = target[:ntr], target[ntr:]
        if y_te.size < 2 or float(np.var(y_te)) <= 0.0:
            return float("nan")
        model = HistGradientBoostingRegressor(max_iter=200, random_state=0)
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        ss_res = float(np.sum((y_te - pred) ** 2))
        ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
        return float(1.0 - ss_res / (ss_tot + 1e-300))

    rng = np.random.default_rng(cfg.random_seed + 1)
    r2_R = r2_from_signature(R)
    r2_g = r2_from_signature(g)  # positive control: g is an exact f(Phi_K) -> ~1
    r2_perm = r2_from_signature(R[rng.permutation(n)])  # negative control -> ~0
    slaving_index = r2_R  # fraction of Var(R) explained by the signature (held-out)

    damp = int(actions.sum())
    no_op = int(n - damp)
    d_low_max = float(np.max(D_low)) if n else 0.0
    # Calibration gates: regressor must recover the exact function g (>0.9) and
    # must not spuriously fit permuted R (<0.1); dissipation must be <= 0.
    estimator_ok = (r2_g > 0.90) and (r2_perm < 0.10) and (d_low_max <= 1e-12)

    interpretable = cfg.preset in VERDICT_BEARING_PRESETS
    if not interpretable:
        verdict, verdict_label = "SMOKE_ONLY", ""
    elif not estimator_ok:
        verdict, verdict_label = "MZ_COUPLING_ESTIMATOR_INVALID", "control_or_structural_gate_failed"
    elif slaving_index >= 0.70:
        verdict, verdict_label = "COUPLING_SIGNATURE_SLAVED", "R_predictable_from_signature"
    elif slaving_index >= 0.30:
        verdict, verdict_label = "COUPLING_PARTIALLY_SLAVED", "R_partly_predictable_from_signature"
    else:
        verdict, verdict_label = "COUPLING_NOT_SLAVED", "R_not_signature_pinned_see_level2"

    result = {
        "verdict": verdict,
        "verdict_label": verdict_label,
        "interpretable": interpretable and estimator_ok,
        "adjudicator": "mz-budget",
        "n_samples": n,
        # --- coupling predictability (the mechanism test): held-out R^2 ---
        "r2_R_from_signature": r2_R,
        "r2_g_control": r2_g,
        "r2_perm_control": r2_perm,
        "slaving_index": slaving_index,
        "train_count": ntr,
        "test_count": n - ntr,
        # --- energy-conservation finding (v1 record) ---
        "rms_R_over_rms_dEdt": rms(R) / (rms(dEdt) + 1e-300),
        "rms_R_over_rms_g": rms(R) / (rms(g) + 1e-300),
        "rms_R_over_rms_T_low": rms(R) / (rms(T_low) + 1e-300),
        "corr_g_R": float(np.corrcoef(g, R)[0, 1]) if n > 1 else 0.0,
        "rms_R": rms(R),
        "rms_g": rms(g),
        "rms_dEdt": rms(dEdt),
        "D_low_mean": float(np.mean(D_low)) if n else 0.0,
        "D_low_max": d_low_max,
        "F_low_mean": float(np.mean(F_low)) if n else 0.0,
        "no_op_count": no_op,
        "damp_low_band_count": damp,
        "damp_fraction": damp / max(1, n),
    }
    sample_rows = [
        {"i": i, "dEdt": float(dEdt[i]), "g": float(g[i]), "R": float(R[i])}
        for i in range(min(n, 200))
    ]
    return result, [], sample_rows, []


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
            if k
            not in ("bin_rows", "sample_rows", "knn_histogram", "knn_sweep_rows", "twin_witness_rows")
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if cfg.adjudicator == "knn-sweep":
        write_csv(out_dir / "knn-sweep.csv", result.get("knn_sweep_rows", []))
    elif cfg.adjudicator == "knn":
        write_csv(out_dir / "knn-radius-histogram.csv", result.get("knn_histogram", []))
    elif cfg.adjudicator == "twin-state":
        write_csv(out_dir / "twin-state-witnesses.csv", result.get("twin_witness_rows", []))
    elif cfg.adjudicator == "mz-budget":
        write_csv(out_dir / "mz-budget-samples.csv", result.get("sample_rows", []))
    else:
        write_csv(out_dir / "bin-summary.csv", result["bin_rows"])
        if write_samples or result["sample_rows"]:
            write_csv(out_dir / "sample-actions.csv", result["sample_rows"])
    write_receipt(out_dir / "PDE_C1_KOLMOGOROV_RESULTS.md", manifest)


def write_receipt(path: Path, manifest: dict) -> None:
    r = manifest["result"]
    c = manifest["config"]
    if c.get("adjudicator") == "knn-sweep":
        write_receipt_knn_sweep(path, manifest)
        return
    if c.get("adjudicator") == "knn":
        write_receipt_knn(path, manifest)
        return
    if c.get("adjudicator") == "twin-state":
        write_receipt_twin_state(path, manifest)
        return
    if c.get("adjudicator") == "mz-budget":
        write_receipt_mz_budget(path, manifest)
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


def write_receipt_knn_sweep(path: Path, manifest: dict) -> None:
    r = manifest["result"]
    c = manifest["config"]
    lines = [
        "# PDE C1 Kolmogorov kNN Convergence-Check Receipt",
        "",
        f"**Status:** {r['verdict']}",
        f"**Preset:** `{c['preset']}`",
        f"**Adjudicator:** `knn-sweep`",
        f"**Interpretable verdict:** `{r['interpretable']}`",
        "",
        "## Sweep (vs neighbourhood radius; primary statistic = mean_minority)",
        "",
        "| k | r_k median | fidelity coverage | mean_minority | incompat fraction |",
        "|---:|---:|---:|---:|---:|",
    ]
    ks = r.get("sweep_k_values", [])
    rks = r.get("sweep_r_k_medians", [])
    fcs = r.get("sweep_fidelity_coverages", [])
    mms = r.get("sweep_mean_minority", [])
    incs = r.get("sweep_incompat_fractions", [])
    for j in range(len(ks)):
        lines.append(
            f"| {ks[j]} | {rks[j]:.6g} | {fcs[j]:.6g} | {mms[j]:.6g} | {incs[j]:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over "
            f"`{r['fit_point_count']}` coverage-passing points: intercept "
            f"`a_mm = {r['mean_minority_intercept']:.6g}`, slope `{r['mean_minority_slope']:.6g}`",
            f"- secondary (diagnostic) `incompat_fraction` fit intercept: "
            f"`{r['incompat_intercept']:.6g}` (grain-confounded; not gated)",
            f"- damp fraction (global): `{r['damp_fraction']:.6g}`",
            f"- classification (pre-registered §6): `a_mm <= {r['a_mm_lo']}` => POSITIVE; "
            f"`a_mm >= {r['a_mm_hi']}` => NEG-A; else INCONCLUSIVE",
            f"- elapsed seconds: `{r['elapsed_seconds']:.3f}`",
            "",
            "## Branch",
            "",
        ]
    )
    if r["verdict"] == "SMOKE_ONLY":
        lines.append("Smoke-only run; no C1 verdict may be filed from this receipt.")
    elif r["verdict"] == "DEFERRED_VACUITY":
        lines.append("Proxy selector essentially constant; no verdict filed.")
    elif r["verdict"] == "PDE-C1-NEG-A":
        lines.append(
            "`mean_minority` extrapolates to a clearly positive value as `r_k -> 0` "
            f"(`a_mm >= {r['a_mm_hi']}`): the conditional proxy non-constancy survives "
            "the zero-radius limit and is **genuine** fiber-incompatibility, not a "
            "finite-radius boundary artifact. The provisional v4 `PDE-C1-NEG-A` is "
            "confirmed."
        )
    elif r["verdict"] == "STRICTNESS_WITNESS_POSITIVE":
        lines.append(
            "`mean_minority` extrapolates to ~zero as `r_k -> 0` "
            f"(`a_mm <= {r['a_mm_lo']}`): the observed mixing is a finite-radius "
            "boundary-straddling artifact around a clean decision surface. The proxy "
            "is control-sufficient on fibers at this cell (Reading-2 regime 2); the "
            "provisional v4 `PDE-C1-NEG-A` is **overturned**."
        )
    else:
        lines.append(
            f"`a_mm` in the ambiguous band `({r['a_mm_lo']}, {r['a_mm_hi']})`, or too few "
            "coverage-passing sweep points to fit. `INCONCLUSIVE_CONVERGENCE` — a "
            "larger `N` or wider clean-`k` range is needed. No verdict filed."
        )
    lines.extend(["", "## Files", "", "- `manifest.json`", "- `knn-sweep.csv`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_receipt_twin_state(path: Path, manifest: dict) -> None:
    r = manifest["result"]
    c = manifest["config"]
    lines = [
        "# PDE C1 Twin-State Certificate Receipt",
        "",
        f"**Status:** {r['verdict']}",
        f"**Preset:** `{c['preset']}`",
        f"**Adjudicator:** `twin-state`",
        f"**Interpretable certificate:** `{r['interpretable']}`",
        "",
        "## Readout",
        "",
        f"- samples: `{r['n_samples']}`, k (effective): `{r['k_neighbors_effective']}`",
        f"- `epsilon_K` (signature radius): `{r['epsilon_k_radius_threshold']:.6g}`",
        f"- `delta_H`: `{r['delta_h']:.6g}` "
        f"(`{r['twin_delta_high_fraction']}` x median high-mode norm, floor `{r['twin_high_norm_floor']}`)",
        f"- high-mode norm median / min / max: `{r['high_norm_median']:.6g}` / "
        f"`{r['high_norm_min']:.6g}` / `{r['high_norm_max']:.6g}`",
        f"- signature-near sample coverage: `{r['candidate_sample_fraction']:.6g}` "
        f"vs `S_pos = {r['s_pos']}` (`{r['candidate_sample_count']}` of `{r['n_samples']}`)",
        f"- candidate pairs unique / directed: `{r['candidate_pair_count_unique']}` / "
        f"`{r['candidate_pair_count_directed']}`",
        f"- witness sample fraction: `{r['witness_sample_fraction']:.6g}` "
        f"vs `{r['twin_min_witness_fraction']}` (`{r['witness_sample_count']}` samples)",
        f"- witness pairs unique / directed: `{r['witness_pair_count_unique']}` / "
        f"`{r['witness_pair_count_directed']}` vs min unique `{r['twin_min_unique_pairs']}`",
        f"- witness high-distance p50 / p95: `{r['witness_high_distance_p50']:.6g}` / "
        f"`{r['witness_high_distance_p95']:.6g}`",
        f"- elapsed seconds: `{r['elapsed_seconds']:.3f}`",
        "",
        "## Paired fiber-constancy",
        "",
        f"**Paired verdict:** `{r.get('paired_fiber_verdict', 'n/a')}`",
        "",
        f"- witness-pair action disagreement (unique): "
        f"`{r.get('witness_action_disagree_fraction_unique', 0.0):.6g}` "
        f"(`{r.get('witness_action_disagree_unique', 0)}` of `{r['witness_pair_count_unique']}`) "
        f"vs `delta_action = {r.get('paired_fiber_delta_action', 0.1)}`",
        f"- witness-pair action disagreement (directed): "
        f"`{r.get('witness_action_disagree_fraction_directed', 0.0):.6g}`",
        f"- candidate-pair action disagreement (unique, comparator): "
        f"`{r.get('candidate_action_disagree_fraction_unique', 0.0):.6g}` "
        f"(`{r.get('candidate_action_disagree_unique', 0)}` of `{r['candidate_pair_count_unique']}`)",
        "",
        "## Branch",
        "",
    ]
    pf = r.get("paired_fiber_verdict", "")
    if pf == "PAIRED_FIBER_CONSTANCY_POSITIVE":
        lines.append(
            "**Paired fiber-constancy POSITIVE.** The state-separated (witness) pairs "
            "the certificate found almost all require the SAME proxy action: action "
            "disagreement on certified non-injective pairs is at or below `delta_action`. "
            "This composes the non-injectivity and control-sufficiency reads on the SAME "
            "pairs, not just at a matched radius."
        )
    elif pf == "PDE-C1-PAIRED-NEG":
        lines.append(
            "**Paired fiber-constancy NEGATIVE (regime 3 on the witnessed pairs).** A "
            "non-trivial fraction of state-separated signature-near pairs require "
            "incompatible actions: the certified non-injective fibers are control-"
            "insufficient for this objective. File as the named negative, do not rescue."
        )
    elif pf == "PAIRED_FIBER_DEFERRED_VACUITY":
        lines.append(
            "Proxy action is structurally constant on this run; the paired test is "
            "vacuous and no fiber-constancy claim is filed."
        )
    elif pf == "PAIRED_FIBER_UNDEFINED":
        lines.append(
            "No certified witness pairs (or no twin-state certificate), so the paired "
            "fiber-constancy test is undefined for this run."
        )
    if r["verdict"] == "SMOKE_ONLY":
        lines.append("Smoke-only run; no support-level state-insufficiency certificate may be filed.")
    elif r["verdict"] == "TWIN_STATE_DEFERRED_HIGH_MODE_FLOOR":
        lines.append(
            "The sampled support is numerically flat in the complementary high modes at "
            "the pre-registered scale. No support-level non-injectivity certificate is filed."
        )
    elif r["verdict"] == "TWIN_STATE_DEFERRED_COVERAGE":
        lines.append(
            "Too few samples have a signature-near neighbour within `epsilon_K`; this "
            "run cannot adjudicate support-level non-injectivity."
        )
    elif r["verdict"] == "TWIN_STATE_CERTIFIED":
        lines.append(
            "A positive-mass fraction of sampled states has a signature-near twin with "
            "high-mode separation above `delta_H`. This certifies `Phi_K` non-injective "
            "on the sampled SRB-like support for this cell."
        )
    else:
        lines.append(
            "Signature-near coverage was adequate, but the pre-registered positive-mass "
            "witness threshold did not clear. This is a no-certificate receipt, not a "
            "proof of injectivity."
        )
    lines.extend(["", "## Files", "", "- `manifest.json`", "- `twin-state-witnesses.csv`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_receipt_mz_budget(path: Path, manifest: dict) -> None:
    r = manifest["result"]
    c = manifest["config"]
    lines = [
        "# PDE C1 Mori-Zwanzig Coupling-Disintegration Receipt (Level 1 v2)",
        "",
        f"**Status:** {r['verdict']}  (explanatory, NOT promotion-bearing)",
        f"**Preset:** `{c['preset']}`",
        f"**Adjudicator:** `mz-budget`",
        "",
        "## Decomposition `dE_low/dt = g(Phi_K) + R`",
        "",
        "`g = D_low + F_low` (dissipation + forcing; the band-closed transfer "
        "`T_LLL` is identically 0 by energy conservation). `R = T_low` is the "
        "full inter-scale (out-of-band-mediated) transfer.",
        "",
        "## Coupling predictability (the mechanism test): held-out R^2",
        "",
        f"- **slaving_index** = held-out R^2 of R predicted from Phi_K "
        f"(1 = fully signature-determined, 0 = unpredictable): "
        f"`{r.get('slaving_index', 0.0):.4g}`",
        f"- `R^2(R | Phi_K)`: `{r.get('r2_R_from_signature', 0.0):.4g}`",
        f"- `R^2(g)` positive control (g is an exact f(Phi_K); want > 0.90): "
        f"`{r.get('r2_g_control', 0.0):.4g}`",
        f"- `R^2(permuted R)` negative control (want < 0.10): "
        f"`{r.get('r2_perm_control', 0.0):.4g}`",
        f"- train / test (block split, no temporal leakage): "
        f"`{r.get('train_count', 0)}` / `{r.get('test_count', 0)}`",
        "",
        "## Energy-conservation finding (v1 record)",
        "",
        f"- `rms(R)/rms(dE_low/dt)`: `{r.get('rms_R_over_rms_dEdt', 0.0):.4g}`  "
        f"`rms(R)/rms(g)`: `{r.get('rms_R_over_rms_g', 0.0):.4g}`  "
        f"`rms(R)/rms(T_low)`: `{r.get('rms_R_over_rms_T_low', 0.0):.4g}` (=1 => T_LLL=0)",
        f"- `corr(g,R)`: `{r.get('corr_g_R', 0.0):.4g}` (negative => quasi-equilibrium cancellation)",
        f"- `D_low` mean/max: `{r.get('D_low_mean', 0.0):.4g}` / `{r.get('D_low_max', 0.0):.4g}` "
        f"(max<=0)  `F_low` mean: `{r.get('F_low_mean', 0.0):.4g}` (>0)",
        f"- `damp_fraction`: `{r.get('damp_fraction', 0.0):.4g}`  samples: `{r.get('n_samples', 0)}`",
        "",
        "## Reading",
        "",
    ]
    si = r.get("slaving_index", 0.0)
    if r["verdict"] == "SMOKE_ONLY":
        lines.append("Smoke-only preset; no diagnostic reading filed.")
    elif r["verdict"] == "MZ_COUPLING_ESTIMATOR_INVALID":
        lines.append(
            "Calibration controls failed (need eta_g < 0.10 floor, eta_perm > 0.70 "
            "ceiling, D_low_max <= 0). The disintegration estimator is not "
            "trustworthy on this run; do not interpret."
        )
    elif r["verdict"] == "COUPLING_SIGNATURE_SLAVED":
        lines.append(
            f"**Coupling signature-slaved** (slaving_index `{si:.3g}`): the net "
            "high-mode energy-transfer `R` into the band is pinned by `Phi_K` even "
            "though the high modes themselves roam (twin states). This is the "
            "mechanism for control-sufficiency: the *relevant functional* is "
            "slaved to the signature, not the state. Explanatory; C1 unchanged."
        )
    elif r["verdict"] == "COUPLING_PARTIALLY_SLAVED":
        lines.append(
            f"**Partially slaved** (slaving_index `{si:.3g}`): `R` is partly "
            "signature-predictable; the remainder must be carried by the "
            "lookahead-integrated coupling (Level 2). Explanatory; C1 unchanged."
        )
    else:
        lines.append(
            f"**Coupling not signature-pinned** (slaving_index `{si:.3g}`): "
            "instantaneous `R` is not Phi_K-determined; control-sufficiency must "
            "come from the integrated/averaged coupling over the lookahead "
            "(Level 2). Explanatory; C1 unchanged."
        )
    lines.extend(["", "## Files", "", "- `manifest.json`", "- `mz-budget-samples.csv`"])
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
        objective=None,
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
