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
    forcing_scheme: str  # "additive" (v0) | "fixed-amplitude" (v1)
    n_shells: int
    lam: float
    k0: float
    eps: float
    viscosity: float
    forcing_amp: float  # additive: |f_1|/sqrt2; fixed-amplitude: target |u_1|
    diagnostic_steps: int
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
    target_base_rate: float
    base_rate_pairwise_tol: float
    base_rate_lo: float
    base_rate_hi: float
    logsig_level: int
    random_seed: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["smoke", "headline"], default="smoke")
    p.add_argument("--model", choices=["sabra", "goy"], default="sabra")
    p.add_argument(
        "--forcing",
        choices=["additive", "fixed-amplitude"],
        default="fixed-amplitude",
        help="v1 default fixed-amplitude (hold |u_1|) for a steady cascade; additive reproduces v0.",
    )
    p.add_argument("--diagnostic", action="store_true", help="Run the stationarity diagnostic (T_eq / T_burst) then exit; files no verdict.")
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
    dt = 1e-4
    sample_stride = 50
    window_steps = 200
    tau_burst_steps = 500
    q_burst = 0.98  # retained for reference; v1 label uses target_base_rate
    # v1 (PDE_C2_CELLSET_SABRA_v1.md section 11): the C1-proven construction —
    # E_burst is the (1 - target_base_rate)-quantile of the held-out calib
    # block's look-ahead-max eps, pinning the calib base rate by construction.
    target_base_rate = 0.15
    base_rate_pairwise_tol = 0.10
    forcing_scheme = getattr(args, "forcing", "fixed-amplitude")
    # v1: fixed-amplitude forcing holds |u_1| at a fixed target (1.0) for a
    # statistically steady cascade; additive reproduces v0 (|f_1| = 0.005*sqrt2).
    forcing_amp = 1.0 if forcing_scheme == "fixed-amplitude" else 0.005
    if args.preset == "headline":
        warmup_steps = 2_000_000
        calib_steps = 1_000_000
        gap_steps = 100_000
        train_steps = 1_500_000
        val_steps = 500_000
        test_steps = 1_000_000
        diagnostic_steps = 3_000_000
    else:
        warmup_steps = 20_000
        calib_steps = 20_000
        gap_steps = 2_000
        train_steps = 20_000
        val_steps = 8_000
        test_steps = 12_000
        diagnostic_steps = 200_000
    return C2Config(
        preset=args.preset,
        model=args.model,
        forcing_scheme=forcing_scheme,
        diagnostic_steps=diagnostic_steps,
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
        if cfg.forcing_scheme == "additive":
            # Constant additive forcing on shell index 0 (the n=1 shell).
            self.forcing[0] = cfg.forcing_amp * (1.0 + 1.0j)
        # fixed-amplitude forcing carries no additive term; it is applied in
        # step() by renormalising |u_0| to cfg.forcing_amp each step.

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
        u_new = e * u + (dt / 6.0) * (e * k1 + 2.0 * e2 * k2 + 2.0 * e2 * k3 + k4)
        if self.cfg.forcing_scheme == "fixed-amplitude":
            # Hold |u_1| at the target modulus (phase free) for a steady cascade.
            a0 = abs(u_new[0])
            if a0 > 1e-30:
                u_new[0] = u_new[0] * (self.cfg.forcing_amp / a0)
        return u_new

    def dissipation(self, u: np.ndarray) -> float:
        """Energy dissipation rate eps(t) = nu * sum_n k_n^2 |u_n|^2.

        The v1 burst observable (PDE_C2_CELLSET_SABRA_v1.md): the canonical
        shell-model intermittency measure, concentrated in the dissipation
        range (k_n^2 weighting) rather than the pinned forcing shell.
        """
        return float(np.sum(self.nu_k2 * np.abs(u) ** 2))

    def initial_state(self) -> np.ndarray:
        rng = np.random.default_rng(self.cfg.random_seed)
        amp = 1e-3 * self.k ** (-1.0 / 3.0)
        phase = np.exp(2j * math.pi * rng.random(self.cfg.n_shells))
        return (amp * phase).astype(np.complex128)


def energy_conservation_self_test() -> None:
    """Inviscid, unforced Sabra should conserve total energy under RK4."""
    cfg = build_config(argparse.Namespace(preset="smoke", model="sabra", forcing="additive"))
    cfg = C2Config(**{**asdict(cfg), "viscosity": 0.0, "forcing_amp": 0.0, "forcing_scheme": "additive"})
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


def run_diagnostic(cfg: C2Config) -> dict:
    """Stationarity diagnostic: report total/max energy series, T_eq, T_burst.

    Files no verdict. Distinguishes drift (energy never flattens) from
    rare-cluster intermittency (flattens, long T_burst) and sets the v1 cell's
    warmup/block lengths via the pre-registered rule (warmup>=3*T_eq,
    block>=50*T_burst).
    """
    model = ShellModel(cfg)
    u = model.initial_state()
    n_steps = cfg.diagnostic_steps
    downsample = 200
    te: list[float] = []  # total energy (for the stationarity/plateau check)
    diss: list[float] = []  # dissipation rate eps (the v1 burst observable)
    started = time.perf_counter()
    stride_report = max(1, n_steps // 10)
    for step in range(n_steps + 1):
        if step % downsample == 0:
            e = np.abs(u) ** 2
            te.append(float(e.sum()))
            diss.append(float(np.sum(model.nu_k2 * e)))
        if step < n_steps:
            u = model.step(u)
        if step > 0 and step % stride_report == 0:
            print(f"[pde-c2-diag] step {step}/{n_steps} ({100*step/n_steps:.0f}%), elapsed {time.perf_counter()-started:.1f}s", flush=True)
    te_arr = np.asarray(te)
    me_arr = np.asarray(diss)
    n = len(te_arr)
    rec_dt = downsample * cfg.dt
    # T_eq via 10-segment plateau detection on total energy.
    seg = max(1, n // 10)
    ref = float(np.mean(te_arr[-2 * seg:]))
    t_eq_idx = n
    for s in range(10):
        lo = s * seg
        hi = (s + 1) * seg if s < 9 else n
        if abs(float(np.mean(te_arr[lo:hi])) - ref) <= 0.10 * ref:
            ok = all(
                abs(float(np.mean(te_arr[ss * seg : (ss + 1) * seg if ss < 9 else n])) - ref) <= 0.15 * ref
                for ss in range(s, 10)
            )
            if ok:
                t_eq_idx = lo
                break
    plateaued = t_eq_idx < n
    t_eq = float(t_eq_idx * rec_dt)
    # T_burst: mean inter-burst interval of max-energy above its post-eq q98.
    post = me_arr[t_eq_idx:] if plateaued else me_arr[n // 2 :]
    thr = float(np.quantile(post, 0.98)) if post.size else float("nan")
    t_burst = float("nan")
    n_bursts = 0
    if post.size:
        exceed = np.where(post > thr)[0]
        if exceed.size >= 2:
            gaps = np.diff(exceed)
            inter = gaps[gaps > 1]
            n_bursts = int(inter.size) + 1
            t_burst = float(np.mean(inter) * rec_dt) if inter.size else float(np.mean(gaps) * rec_dt)
    half = n // 2
    result = {
        "mode": "diagnostic",
        "forcing_scheme": cfg.forcing_scheme,
        "model": cfg.model,
        "total_energy_first_half_mean": float(np.mean(te_arr[:half])),
        "total_energy_second_half_mean": float(np.mean(te_arr[half:])),
        "total_energy_min": float(np.min(te_arr)),
        "total_energy_max": float(np.max(te_arr)),
        "eps_median": float(np.median(me_arr)),
        "eps_p98_posteq": thr,
        "plateaued": plateaued,
        "T_eq_time": t_eq,
        "T_burst_time": t_burst,
        "n_bursts_posteq": n_bursts,
        "rec_dt": rec_dt,
        "suggested_warmup_steps": int(math.ceil(3 * t_eq / cfg.dt)) if plateaued else None,
        "suggested_block_steps": int(math.ceil(50 * t_burst / cfg.dt)) if (t_burst == t_burst) else None,
        "elapsed_seconds": time.perf_counter() - started,
        "n_steps": n_steps,
    }
    return result


def write_diagnostic_outputs(out_dir: Path, cfg: C2Config, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/pde_c2_sabra_cell.py",
        "spec": "docs/proof/PDE_C2_CELLSET_SABRA_v1.md",
        "mode": "diagnostic",
        "config": asdict(cfg),
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": sys.platform},
        "result": result,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    fh = result["total_energy_first_half_mean"]
    sh = result["total_energy_second_half_mean"]
    drift = abs(sh - fh) / fh if fh else float("nan")
    lines = [
        "# PDE C2 Sabra v1 Stationarity Diagnostic",
        "",
        f"**Forcing:** `{cfg.forcing_scheme}`   **Model:** `{cfg.model}`",
        "",
        "## Stationarity",
        "",
        f"- total energy first-half / second-half mean: `{fh:.6g}` / `{sh:.6g}` "
        f"(drift `{drift:.3g}`)",
        f"- total energy min / max: `{result['total_energy_min']:.6g}` / `{result['total_energy_max']:.6g}`",
        f"- plateaued: `{result['plateaued']}`   T_eq: `{result['T_eq_time']:.4g}` time units",
        "",
        "## Burst timescale",
        "",
        f"- dissipation ε median / post-eq q98: `{result['eps_median']:.6g}` / `{result['eps_p98_posteq']:.6g}`",
        f"- post-eq burst count: `{result['n_bursts_posteq']}`   T_burst: `{result['T_burst_time']:.4g}` time units",
        "",
        "## Suggested v1 cell lengths (pre-registered rule)",
        "",
        f"- warmup >= 3*T_eq -> `{result['suggested_warmup_steps']}` steps",
        f"- each block >= 50*T_burst -> `{result['suggested_block_steps']}` steps",
        "",
        "## Read",
        "",
    ]
    if not result["plateaued"]:
        lines.append(
            "Total energy did NOT plateau within the diagnostic window: this "
            "forcing/regime still drifts. Lengthen warmup or reconsider the "
            "forcing before the verdict-bearing cell."
        )
    elif drift < 0.10:
        lines.append(
            "Total energy plateaus (first/second-half drift < 10%): the cascade "
            "is statistically steady under this forcing. Pin the v1 cell lengths "
            "from the suggestions above and proceed to the verdict-bearing run."
        )
    else:
        lines.append(
            "Total energy plateau is marginal (first/second-half drift >= 10%): "
            "lengthen warmup before committing the cell."
        )
    lines.extend(["", "## Files", "", "- `manifest.json`"])
    (out_dir / "PDE_C2_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    # Only the burst observable (v1: dissipation rate eps) is needed for the
    # base-rate gate (this layer). Per-step channel snapshots (for the deferred
    # detector) are NOT stored, to keep memory bounded at headline scale.
    burst_obs = np.empty(total + 1, dtype=np.float64)
    started = time.perf_counter()
    stride_report = max(1, total // 10)
    for step in range(total + 1):
        burst_obs[step] = model.dissipation(u)
        if step < total:
            u = model.step(u)
        if step > 0 and step % stride_report == 0:
            print(f"[pde-c2] step {step}/{total} ({100*step/total:.0f}%), elapsed {time.perf_counter()-started:.1f}s", flush=True)
    return {
        "burst_obs": burst_obs,
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


def burst_label(obs_series: np.ndarray, start: int, end: int, e_burst: float, cfg: C2Config) -> np.ndarray:
    """B(t)=1 iff the burst observable (eps) exceeds e_burst in (t, t+tau_burst]."""
    tau = cfg.tau_burst_steps
    query = list(range(start, end - tau, cfg.sample_stride))
    labels = np.zeros(len(query), dtype=np.int8)
    for i, t in enumerate(query):
        if float(np.max(obs_series[t + 1 : t + 1 + tau])) > e_burst:
            labels[i] = 1
    return labels


def main() -> None:
    args = parse_args()
    if args.self_test:
        energy_conservation_self_test()
        return
    cfg = build_config(args)
    if args.diagnostic:
        result = run_diagnostic(cfg)
        write_diagnostic_outputs(args.out.resolve(), cfg, result)
        print(
            f"[pde-c2-diag] plateaued={result['plateaued']} "
            f"T_eq={result['T_eq_time']:.4g} T_burst={result['T_burst_time']:.4g}"
        )
        print(f"[pde-c2-diag] wrote: {args.out.resolve()}")
        return
    sim = simulate(cfg)
    bounds = block_bounds(cfg)
    obs = sim["burst_obs"]
    calib_obs = obs[bounds["calib"][0] : bounds["calib"][1]]
    e_burst = float(np.quantile(calib_obs, cfg.q_burst))

    labels = {name: burst_label(obs, lo, hi, e_burst, cfg) for name, (lo, hi) in bounds.items() if name != "calib"}
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
        "burst_observable": "dissipation_rate_eps",
        "calib_eps_median": float(np.median(calib_obs)),
        "calib_eps_p99": float(np.quantile(calib_obs, 0.99)),
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
        f"- `E_burst` (held-out q={cfg.q_burst} of calib dissipation ε): `{result['e_burst']:.6g}`",
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
