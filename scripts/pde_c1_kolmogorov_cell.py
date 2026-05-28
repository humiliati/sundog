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
    random_seed: int
    integrator: str
    signature_dimension: int
    action_tiebreak: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "lock", "fallback"], default="smoke")
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
    # Cell-set v0 pinned values.
    grid_size = 32
    n_modes = 16
    k_signature = 4
    kf = 4
    forcing_amplitude = 1.0
    grashof = 100.0
    # Dimensionless normalization: G = forcing_amplitude / nu^2.
    viscosity = math.sqrt(forcing_amplitude / grashof)
    dt = 0.01
    sample_interval_steps = 50
    lookahead_steps = int(round(5.0 / dt))
    n_min = 30
    delta_action = 0.10
    s_pos = 0.50
    random_seed = 20260528

    if args.preset == "lock":
        burnin_steps = 100_000
        sample_count = 50_000
    elif args.preset == "fallback":
        burnin_steps = 100_000
        sample_count = 200_000
    else:
        # Smoke is intentionally not the registered cell. It exists to validate
        # the integrator, binning, and receipt plumbing under the repo's
        # ~10-minute inline rule.
        burnin_steps = 2_000
        sample_count = 200
        sample_interval_steps = 10
        lookahead_steps = 100

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
        random_seed=random_seed,
        integrator="pseudo_spectral_vorticity_semi_implicit_euler",
        signature_dimension=2 * k_signature * k_signature,
        action_tiebreak="no_op",
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

    e_max = float(np.percentile(np.asarray(burnin_energies), 95))
    epsilon_k = 0.05 * math.sqrt(max(0.0, 2.0 * e_max))
    h_k = epsilon_k / math.sqrt(cfg.signature_dimension) if epsilon_k > 0 else 1.0
    actions = label_samples(energy_by_step, cfg, e_max)
    bin_rows, sample_rows = aggregate_bins(sample_signatures, sample_energy, actions, sig_min, h_k, cfg)
    result = summarize(bin_rows, cfg)
    result.update(
        {
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

    if cfg.preset != "lock" and cfg.preset != "fallback":
        verdict = "SMOKE_ONLY"
        verdict_label = ""
        interpretable = False
    elif s_eval < cfg.s_pos:
        verdict = "DEFERRED_COVERAGE"
        verdict_label = ""
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
        "result": {k: v for k, v in result.items() if k not in ("bin_rows", "sample_rows")},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "bin-summary.csv", result["bin_rows"])
    if write_samples or result["sample_rows"]:
        write_csv(out_dir / "sample-actions.csv", result["sample_rows"])
    write_receipt(out_dir / "PDE_C1_KOLMOGOROV_RESULTS.md", manifest)


def write_receipt(path: Path, manifest: dict) -> None:
    r = manifest["result"]
    c = manifest["config"]
    lines = [
        "# PDE C1 Kolmogorov v0 Receipt",
        "",
        f"**Status:** {r['verdict']}",
        f"**Preset:** `{c['preset']}`",
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
    elif r["verdict"] == "PDE-C1-NEG-A":
        lines.append("Fiber-incompatible evaluated bin found; file `PDE-C1-NEG-A` for this cell.")
    else:
        lines.append("No evaluated bin crossed `delta_action`; file strictness-witness positive under the proxy.")
    lines.extend(["", "## Files", "", "- `manifest.json`", "- `bin-summary.csv`"])
    if c["sample_count"] <= 5_000:
        lines.append("- `sample-actions.csv`")
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
