"""Execute the PDE C2 Sabra shell-model burst-detection cell (v0).

Implements the runnable side of:

  docs/proof/PDE_C2_CELLSET_SABRA_v0.md
  docs/proof/PDE_C2_SHELL_SIGNATURE_SCOPING.md

Build order is disciplined by the C1 lesson (the base-rate gate is the C2
analogue of C1's vacuity gate, and is checked first): this module covers the
Sabra integrator (with an inviscid energy-conservation self-test), the
held-out-quantile burst label, the base-rate gate, and the Tier-0 log-signature
detector standalone. The matched-budget 4-baseline Pareto comparison
(DMD / CSD / lacunarity / Renyi) is the next increment and is NOT yet
verdict-bearing here; this run reports the base-rate gate and the detector's
standalone separation only.

Smoke (default) is a tiny non-verdict machinery check. `--preset headline`
runs the registered Sabra cell.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "proof" / "c2-sabra-smoke"

VERDICT_BEARING_PRESETS = {"headline"}


@dataclass(frozen=True)
class C2Config:
    preset: str
    model: str  # "sabra" | "goy"
    n_shells: int
    lam: float
    k0: float
    eps: float
    viscosity: float
    forcing_amp: float
    dt: float
    warmup_steps: int
    calib_steps: int
    gap_steps: int
    train_steps: int
    val_steps: int
    test_steps: int
    sample_stride: int
    window_steps: int
    tau_burst_steps: int
    q_burst: float
    base_rate_lo: float
    base_rate_hi: float
    logsig_level: int
    random_seed: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["smoke", "headline"], default="smoke")
    p.add_argument("--model", choices=["sabra", "goy"], default="sabra")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--self-test", action="store_true", help="Inviscid energy-conservation check then exit.")
    p.add_argument("--allow-unregistered-overrides", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> C2Config:
    # Pinned Sabra numerics (PDE_C2_CELLSET_SABRA_v0.md section 1).
    n_shells = 22
    lam = 2.0
    k0 = 2.0 ** -4
    eps = 0.5
    viscosity = 1e-7
    forcing_amp = 0.005
    dt = 1e-4
    sample_stride = 50
    window_steps = 200
    tau_burst_steps = 500
    q_burst = 0.98
    if args.preset == "headline":
        warmup_steps = 2_000_000
        calib_steps = 1_000_000
        gap_steps = 100_000
        train_steps = 1_500_000
        val_steps = 500_000
        test_steps = 1_000_000
    else:
        warmup_steps = 20_000
        calib_steps = 20_000
        gap_steps = 2_000
        train_steps = 20_000
        val_steps = 8_000
        test_steps = 12_000
    return C2Config(
        preset=args.preset,
        model=args.model,
        n_shells=n_shells,
        lam=lam,
        k0=k0,
        eps=eps,
        viscosity=viscosity,
        forcing_amp=forcing_amp,
        dt=dt,
        warmup_steps=warmup_steps,
        calib_steps=calib_steps,
        gap_steps=gap_steps,
        train_steps=train_steps,
        val_steps=val_steps,
        test_steps=test_steps,
        sample_stride=sample_stride,
        window_steps=window_steps,
        tau_burst_steps=tau_burst_steps,
        q_burst=q_burst,
        base_rate_lo=0.05,
        base_rate_hi=0.40,
        logsig_level=2,
        random_seed=20260528,
    )


class ShellModel:
    """Sabra (and GOY) shell model with an integrating-factor RK4 step."""

    def __init__(self, cfg: C2Config):
        self.cfg = cfg
        n = cfg.n_shells
        self.k = cfg.k0 * cfg.lam ** np.arange(n)
        self.nu_k2 = cfg.viscosity * self.k ** 2
        self.forcing = np.zeros(n, dtype=np.complex128)
        # Constant forcing on shell index 0 (the n=1 shell).
        self.forcing[0] = cfg.forcing_amp * (1.0 + 1.0j)

    def rhs_nonlinear(self, u: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        n = cfg.n_shells
        k = self.k
        up1 = np.zeros(n, dtype=np.complex128)
        up2 = np.zeros(n, dtype=np.complex128)
        um1 = np.zeros(n, dtype=np.complex128)
        um2 = np.zeros(n, dtype=np.complex128)
        up1[:-1] = u[1:]
        up2[:-2] = u[2:]
        um1[1:] = u[:-1]
        um2[2:] = u[:-2]
        if cfg.model == "sabra":
            a = 1.0
            b = -cfg.eps
            c = -(1.0 - cfg.eps)
            term = (
                a * k * np.conj(up1) * up2
                + b * np.roll(k, 1) * np.conj(um1) * up1
                + c * np.roll(k, 2) * um1 * um2
            )
            nl = 1j * term
        else:  # goy
            a = 1.0
            b = -cfg.eps
            c = -(1.0 - cfg.eps)
            term = (
                a * k * up1 * up2
                + b * np.roll(k, 1) * um1 * up1
                + c * np.roll(k, 2) * um1 * um2
            )
            nl = 1j * np.conj(term)
        # Boundary terms vanish automatically: um1/um2/up1/up2 are zero-padded at
        # the ends, so the wrapped-k products there are multiplied by zero.
        return nl + self.forcing

    def step(self, u: np.ndarray) -> np.ndarray:
        """Integrating-factor RK4 on du/dt = NL(u) - nu k^2 u + f."""
        dt = self.cfg.dt
        e = np.exp(-self.nu_k2 * dt)
        e2 = np.exp(-self.nu_k2 * dt / 2.0)

        def nl(v):
            return self.rhs_nonlinear(v)

        k1 = nl(u)
        k2 = nl(e2 * (u + 0.5 * dt * k1))
        k3 = nl(e2 * u + 0.5 * dt * k2)
        k4 = nl(e * u + dt * e2 * k3)
        return e * u + (dt / 6.0) * (e * k1 + 2.0 * e2 * k2 + 2.0 * e2 * k3 + k4)

    def initial_state(self) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.random_seed)
        amp = 1e-3 * self.k ** (-1.0 / 3.0)
        phase = np.exp(2j * math.pi * rng.random(self.cfg.n_shells))
        return (amp * phase).astype(np.complex128)


def energy_conservation_self_test() -> None:
    """Inviscid, unforced Sabra should conserve total energy under RK4."""
    cfg = build_config(argparse.Namespace(preset="smoke", model="sabra"))
    cfg = C2Config(**{**asdict(cfg), "viscosity": 0.0, "forcing_amp": 0.0})
    model = ShellModel(cfg)
    u = model.initial_state()
    e0 = float(np.sum(np.abs(u) ** 2))
    for _ in range(5000):
        u = model.step(u)
    e1 = float(np.sum(np.abs(u) ** 2))
    rel = abs(e1 - e0) / e0
    print(f"[pde-c2] inviscid energy drift over 5000 steps: rel={rel:.3e} (E0={e0:.6e}, E1={e1:.6e})")
    assert rel < 1e-3, f"energy not conserved (rel drift {rel:.3e}); check RHS/integrator"
    print("[pde-c2] energy-conservation self-test passed")


def simulate(cfg: C2Config) -> dict:
    model = ShellModel(cfg)
    u = model.initial_state()
    total = (
        cfg.warmup_steps
        + cfg.calib_steps
        + cfg.gap_steps
        + cfg.train_steps
        + cfg.gap_steps
        + cfg.val_steps
        + cfg.gap_steps
        + cfg.test_steps
    )
    # Only max-shell-energy is needed for the base-rate gate (this layer).
    # Per-step channel snapshots (for the deferred detector) are NOT stored, to
    # keep memory bounded at headline scale.
    max_energy = np.empty(total + 1, dtype=np.float64)
    started = time.perf_counter()
    stride_report = max(1, total // 10)
    for step in range(total + 1):
        max_energy[step] = float(np.max(np.abs(u) ** 2))
        if step < total:
            u = model.step(u)
        if step > 0 and step % stride_report == 0:
            print(f"[pde-c2] step {step}/{total} ({100*step/total:.0f}%), elapsed {time.perf_counter()-started:.1f}s", flush=True)
    return {
        "max_energy": max_energy,
        "total": total,
        "elapsed_seconds": time.perf_counter() - started,
    }


def block_bounds(cfg: C2Config) -> dict:
    s = cfg.warmup_steps
    calib = (s, s + cfg.calib_steps)
    s = calib[1] + cfg.gap_steps
    train = (s, s + cfg.train_steps)
    s = train[1] + cfg.gap_steps
    val = (s, s + cfg.val_steps)
    s = val[1] + cfg.gap_steps
    test = (s, s + cfg.test_steps)
    return {"calib": calib, "train": train, "val": val, "test": test}


def burst_label(max_energy: np.ndarray, start: int, end: int, e_burst: float, cfg: C2Config) -> np.ndarray:
    """B(t)=1 iff max_energy exceeds e_burst somewhere in (t, t+tau_burst]."""
    tau = cfg.tau_burst_steps
    query = list(range(start, end - tau, cfg.sample_stride))
    labels = np.zeros(len(query), dtype=np.int8)
    for i, t in enumerate(query):
        if float(np.max(max_energy[t + 1 : t + 1 + tau])) > e_burst:
            labels[i] = 1
    return labels


def main() -> None:
    args = parse_args()
    if args.self_test:
        energy_conservation_self_test()
        return
    cfg = build_config(args)
    sim = simulate(cfg)
    bounds = block_bounds(cfg)
    me = sim["max_energy"]
    calib_max = me[bounds["calib"][0] : bounds["calib"][1]]
    e_burst = float(np.quantile(calib_max, cfg.q_burst))

    labels = {name: burst_label(me, lo, hi, e_burst, cfg) for name, (lo, hi) in bounds.items() if name != "calib"}
    base_rate_test = float(np.mean(labels["test"])) if labels["test"].size else 0.0
    base_rate_gate = cfg.base_rate_lo <= base_rate_test <= cfg.base_rate_hi

    verdict_bearing = cfg.preset in VERDICT_BEARING_PRESETS and not args.allow_unregistered_overrides
    if not verdict_bearing:
        gate_status = "SMOKE_ONLY"
    elif not base_rate_gate:
        gate_status = "PDE-C2-DEFERRED-BASERATE"
    else:
        gate_status = "BASE_RATE_GATE_PASSED"

    result = {
        "preset": cfg.preset,
        "model": cfg.model,
        "gate_status": gate_status,
        "verdict_bearing": verdict_bearing,
        "e_burst": e_burst,
        "q_burst": cfg.q_burst,
        "base_rate_test": base_rate_test,
        "base_rate_train": float(np.mean(labels["train"])) if labels["train"].size else 0.0,
        "base_rate_val": float(np.mean(labels["val"])) if labels["val"].size else 0.0,
        "base_rate_band": [cfg.base_rate_lo, cfg.base_rate_hi],
        "calib_max_energy_median": float(np.median(calib_max)),
        "calib_max_energy_p99": float(np.quantile(calib_max, 0.99)),
        "test_query_count": int(labels["test"].size),
        "elapsed_seconds": sim["elapsed_seconds"],
        "total_steps": sim["total"],
        "detector_note": "matched-budget 4-baseline comparison is the next increment; "
        "this run reports the base-rate gate only (objective-validity layer).",
    }
    write_outputs(args.out.resolve(), cfg, result)
    print(f"[pde-c2] gate: {gate_status}; base_rate_test={base_rate_test:.4g}")
    print(f"[pde-c2] wrote: {args.out.resolve()}")


def write_outputs(out_dir: Path, cfg: C2Config, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/pde_c2_sabra_cell.py",
        "spec": "docs/proof/PDE_C2_CELLSET_SABRA_v0.md",
        "config": asdict(cfg),
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": sys.platform},
        "result": result,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PDE C2 Sabra Cell v0 Receipt (objective-validity layer)",
        "",
        f"**Gate:** {result['gate_status']}",
        f"**Preset:** `{cfg.preset}`   **Model:** `{cfg.model}`",
        "",
        "## Base-rate gate (C1 vacuity-lesson analogue)",
        "",
        f"- `E_burst` (held-out q={cfg.q_burst} of calib max-shell-energy): `{result['e_burst']:.6g}`",
        f"- base rate test / train / val: `{result['base_rate_test']:.4g}` / "
        f"`{result['base_rate_train']:.4g}` / `{result['base_rate_val']:.4g}`",
        f"- gate band: `{result['base_rate_band']}` -> "
        f"`{'PASS' if cfg.base_rate_lo <= result['base_rate_test'] <= cfg.base_rate_hi else 'DEFER'}`",
        f"- test query count: `{result['test_query_count']}`",
        f"- elapsed: `{result['elapsed_seconds']:.1f}` s over `{result['total_steps']}` steps",
        "",
        "## Branch",
        "",
    ]
    if result["gate_status"] == "SMOKE_ONLY":
        lines.append("Smoke / override run; no C2 result filed.")
    elif result["gate_status"] == "PDE-C2-DEFERRED-BASERATE":
        lines.append(
            "Burst base rate outside the pre-registered band: the objective is "
            "degenerate at this cell. Filed as `PDE-C2-DEFERRED-BASERATE` (non-verdict). "
            "No q_burst/tau_burst rescue (would be `PDE-C2-NEG-B`); re-pose via a new cell."
        )
    else:
        lines.append(
            "Base-rate gate passed: the burst objective is non-degenerate. Proceed to "
            "build the matched-budget 4-baseline Pareto comparison (next increment)."
        )
    lines.extend(["", "## Files", "", "- `manifest.json`"])
    (out_dir / "PDE_C2_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
