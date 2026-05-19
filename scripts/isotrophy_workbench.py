"""Executable smoke workbench for docs/sundog_v_isotrophy.md.

This is deliberately a small harness, not the full K_facet experiment. It
parses the Li-Liao compact supplementary ansatz, integrates selected rows with
DOP853, and evaluates concrete spacetime-isotropy generators through the same
residual gate specified in docs/isotrophy/files.math:

    generator action -> explicit spatial matrix -> SO(3) Procrustes +
    phase minimization.

The smoke target is parser/integrator/gate sanity, especially:
    - sigma3 and residual generators use the same detector path;
    - F_beta is structural for the ansatz;
    - F_delta is tested as emergent, not quotiented away.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


SUPPLEMENT_URLS = {
    "A": "https://numericaltank.sjtu.edu.cn/three-body/three-body/gif/supplementary-A.txt",
    "B": "https://numericaltank.sjtu.edu.cn/three-body/three-body/gif/supplementary-B.txt",
}
DEFAULT_RTOL = 1e-12
DEFAULT_ATOL = 1e-12
DEFAULT_MAX_STEP_FRACTION = 0.02
DEFAULT_SIGMA_TOLERANCE = 1e-5
DEFAULT_EXPECTED_SIGMA_COUNT = None
DEFAULT_CLOSURE_FLOOR = 1e-15

FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ROW_RE = re.compile(
    rf"O_\{{(?P<index>\d+)\}}\((?P<m3>{FLOAT})\)\s+"
    rf"(?P<z0>{FLOAT})\s+(?P<vx>{FLOAT})\s+(?P<vy>{FLOAT})\s+"
    rf"(?P<vz>{FLOAT})\s+(?P<period>{FLOAT})\s+(?P<stability>[SU])"
)


I3 = np.diag([1.0, 1.0, 1.0])
Z_MIRROR = np.diag([1.0, 1.0, -1.0])
R_PI = np.diag([-1.0, -1.0, 1.0])
POINT_INVERSION = R_PI @ Z_MIRROR

PERMUTATIONS = {
    "identity": (0, 1, 2),
    "swap12": (1, 0, 2),
    # Group-action convention: (P x)_i = x_{p^{-1}(i)}.
    "cycle123": (2, 0, 1),
    "cycle132": (1, 2, 0),
}


@dataclass(frozen=True)
class Generator:
    name: str
    klass: str
    perm: tuple[int, int, int]
    shift_fraction: float
    treverse: bool
    spatial: np.ndarray


GENERATORS = {
    "sigma3": Generator("sigma3", "choreography", PERMUTATIONS["cycle123"], 1.0 / 3.0, False, I3),
    # True inverse of (cycle123, +T/3): inverse permutation and -T/3 == +2T/3.
    "sigma3_inverse": Generator(
        "sigma3_inverse", "choreography", PERMUTATIONS["cycle132"], 2.0 / 3.0, False, I3
    ),
    "sigma3_opposite_orientation": Generator(
        "sigma3_opposite_orientation", "choreography", PERMUTATIONS["cycle132"], 1.0 / 3.0, False, I3
    ),
    # Inverse of the opposite-orientation element (cycle132, +T/3):
    # its square is (cycle123, +2T/3); the two generate the full opposite Z3.
    "sigma3_opposite_inverse": Generator(
        "sigma3_opposite_inverse", "choreography", PERMUTATIONS["cycle123"], 2.0 / 3.0, False, I3
    ),
    "alpha_I": Generator("alpha_I", "alpha", PERMUTATIONS["swap12"], 1.0 / 2.0, False, I3),
    "beta_I": Generator("beta_I", "beta", PERMUTATIONS["swap12"], 0.0, True, I3),
    "gamma_Z": Generator("gamma_Z", "gamma", PERMUTATIONS["swap12"], 1.0 / 2.0, False, Z_MIRROR),
    "delta_Z": Generator("delta_Z", "delta", PERMUTATIONS["swap12"], 0.0, True, Z_MIRROR),
    "F_beta": Generator("F_beta", "beta", PERMUTATIONS["swap12"], 0.0, True, R_PI),
    "F_delta": Generator("F_delta", "delta", PERMUTATIONS["swap12"], 0.0, True, POINT_INVERSION),
}


@dataclass(frozen=True)
class OrbitRow:
    source: str
    line_no: int
    index: int
    m3: float
    z0: float
    vx: float
    vy: float
    vz: float
    period: float
    stability: str

    @property
    def masses(self) -> np.ndarray:
        return np.array([1.0, 1.0, self.m3], dtype=float)

    @property
    def label(self) -> str:
        return f"O_{{{self.index}}}({self.m3:g})"


@dataclass(frozen=True)
class IntegratedOrbit:
    row: OrbitRow
    solution: object
    y0: np.ndarray
    elapsed_seconds: float
    closure_position_inf: float
    closure_velocity_inf: float
    inertia_degenerate: bool


def read_text(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    return Path(path_or_url).read_text(encoding="utf-8")


def parse_rows(text: str, source: str) -> list[OrbitRow]:
    rows: list[OrbitRow] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = ROW_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        rows.append(
            OrbitRow(
                source=source,
                line_no=line_no,
                index=int(groups["index"]),
                m3=float(groups["m3"]),
                z0=float(groups["z0"]),
                vx=float(groups["vx"]),
                vy=float(groups["vy"]),
                vz=float(groups["vz"]),
                period=float(groups["period"]),
                stability=groups["stability"],
            )
        )
    return rows


def expand_initial_state(row: OrbitRow, center_com: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses = row.masses
    x = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, row.z0],
        ],
        dtype=float,
    )
    v = np.array(
        [
            [row.vx, row.vy, row.vz],
            [row.vx, row.vy, -row.vz],
            [-2.0 * row.vx / row.m3, -2.0 * row.vy / row.m3, 0.0],
        ],
        dtype=float,
    )
    if center_com:
        x = x - np.average(x, axis=0, weights=masses)
    return masses, x, v


def pack_state(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([x.reshape(9), v.reshape(9)])


def rhs_factory(masses: np.ndarray):
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        x = y[:9].reshape(3, 3)
        v = y[9:].reshape(3, 3)
        a = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                delta = x[j] - x[i]
                r2 = float(np.dot(delta, delta))
                r = math.sqrt(r2)
                a[i] += masses[j] * delta / (r2 * r)
        return pack_state(v, a)

    return rhs


def sample_positions(integrated: IntegratedOrbit, times: np.ndarray) -> np.ndarray:
    period = integrated.row.period
    wrapped = np.mod(times, period)
    y = integrated.solution.sol(wrapped)
    return y[:9, :].T.reshape(len(wrapped), 3, 3)


def integrate_orbit(row: OrbitRow, rtol: float, atol: float, max_step_fraction: float) -> IntegratedOrbit:
    masses, x0, v0 = expand_initial_state(row, center_com=True)
    y0 = pack_state(x0, v0)
    max_step = row.period * max_step_fraction if max_step_fraction > 0 else np.inf
    started = time.perf_counter()
    solution = solve_ivp(
        rhs_factory(masses),
        (0.0, row.period),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )
    elapsed = time.perf_counter() - started
    if not solution.success:
        raise RuntimeError(f"integration failed for {row.label}: {solution.message}")
    y_final = solution.y[:, -1]
    closure_position = np.linalg.norm((y_final[:9] - y0[:9]).reshape(3, 3), axis=1).max() / 2.0
    closure_velocity = np.linalg.norm((y_final[9:] - y0[9:]).reshape(3, 3), axis=1).max()
    return IntegratedOrbit(
        row=row,
        solution=solution,
        y0=y0,
        elapsed_seconds=elapsed,
        closure_position_inf=float(closure_position),
        closure_velocity_inf=float(closure_velocity),
        inertia_degenerate=abs(row.m3 * row.z0 * row.z0 - 2.0) < 1e-3,
    )


def best_so3_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    h = source.T @ target
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return r


def transform_positions(integrated: IntegratedOrbit, generator: Generator, times: np.ndarray) -> np.ndarray:
    period = integrated.row.period
    shift = generator.shift_fraction * period
    eps = -1.0 if generator.treverse else 1.0
    source_times = eps * (times - shift)
    positions = sample_positions(integrated, source_times)
    positions = positions[:, generator.perm, :]
    return positions @ generator.spatial.T


def residual_for_generator(
    integrated: IntegratedOrbit,
    generator: Generator,
    n_samples: int,
    phase_grid: int,
) -> dict[str, float | str | int | bool]:
    period = integrated.row.period
    times = np.linspace(0.0, period, n_samples, endpoint=False)
    generated = transform_positions(integrated, generator, times)
    generated_flat = generated.reshape(-1, 3)

    def objective(phi: float) -> tuple[float, float, np.ndarray]:
        target = sample_positions(integrated, times + phi).reshape(-1, 3)
        rotation = best_so3_rotation(generated_flat, target)
        aligned = generated_flat @ rotation.T
        diffs = target - aligned
        norms = np.linalg.norm(diffs, axis=1)
        return float(norms.max() / 2.0), float(math.sqrt(np.mean(norms * norms)) / 2.0), rotation

    grid = np.linspace(0.0, period, phase_grid, endpoint=False)
    grid_values = [objective(phi)[0] for phi in grid]
    best_index = int(np.argmin(grid_values))
    grid_step = period / phase_grid
    center = float(grid[best_index])

    def scalar(phi: float) -> float:
        return objective(phi)[0]

    local = minimize_scalar(
        scalar,
        bounds=(center - grid_step, center + grid_step),
        method="bounded",
        options={"xatol": max(period * 1e-10, 1e-12), "maxiter": 80},
    )
    phase = float(local.x % period)
    residual_inf, residual_rms, rotation = objective(phase)
    return {
        "label": integrated.row.label,
        "index": integrated.row.index,
        "m3": integrated.row.m3,
        "period": integrated.row.period,
        "generator": generator.name,
        "class": generator.klass,
        "residual_inf": residual_inf,
        "residual_rms": residual_rms,
        "phase": phase,
        "det_rotation": float(np.linalg.det(rotation)),
        "closure_position_inf": integrated.closure_position_inf,
        "closure_velocity_inf": integrated.closure_velocity_inf,
        "residual_to_closure": residual_inf / max(integrated.closure_position_inf, DEFAULT_CLOSURE_FLOOR),
        "inertia_degenerate": integrated.inertia_degenerate,
        "integration_seconds": integrated.elapsed_seconds,
        "n_samples": n_samples,
        "phase_grid": phase_grid,
    }


def select_rows(
    rows: Iterable[OrbitRow],
    m3: float | None,
    limit: int,
    sort_period: bool,
    indices: set[int] | None = None,
) -> list[OrbitRow]:
    selected = [
        row
        for row in rows
        if (m3 is None or abs(row.m3 - m3) < 5e-13) and (indices is None or row.index in indices)
    ]
    if sort_period:
        selected.sort(key=lambda row: (row.period, row.index))
    if limit > 0:
        selected = selected[:limit]
    return selected


def parse_indices(indices: str | None) -> set[int] | None:
    if not indices:
        return None
    parsed: set[int] = set()
    for part in indices.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            parsed.add(int(part))
        except ValueError as exc:
            raise SystemExit(f"invalid orbit index {part!r} in --indices") from exc
    return parsed


def write_outputs(out_dir: Path, summary: dict[str, object], records: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not records:
        return
    fieldnames = list(records[0].keys())
    with (out_dir / "residuals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def scan_record_from_residuals(
    integrated: IntegratedOrbit,
    sigma3: dict[str, object],
    sigma3_inverse: dict[str, object],
    f_beta: dict[str, object],
    sigma_tolerance: float,
    closure_floor: float,
) -> dict[str, object]:
    sigma3_residual = float(sigma3["residual_inf"])
    sigma3_inverse_residual = float(sigma3_inverse["residual_inf"])
    if sigma3_residual <= sigma3_inverse_residual:
        sigma_best_generator = "sigma3"
        sigma_best_residual = sigma3_residual
        sigma_best_phase = float(sigma3["phase"])
    else:
        sigma_best_generator = "sigma3_inverse"
        sigma_best_residual = sigma3_inverse_residual
        sigma_best_phase = float(sigma3_inverse["phase"])
    sigma_group_residual = max(sigma3_residual, sigma3_inverse_residual)

    closure_scale = max(integrated.closure_position_inf, closure_floor)
    f_beta_residual = float(f_beta["residual_inf"])
    return {
        "label": integrated.row.label,
        "index": integrated.row.index,
        "line_no": integrated.row.line_no,
        "m3": integrated.row.m3,
        "z0": integrated.row.z0,
        "period": integrated.row.period,
        "stability": integrated.row.stability,
        "inertia_degenerate": integrated.inertia_degenerate,
        "closure_position_inf": integrated.closure_position_inf,
        "closure_velocity_inf": integrated.closure_velocity_inf,
        "integration_seconds": integrated.elapsed_seconds,
        "sigma3_residual_inf": sigma3_residual,
        "sigma3_inverse_residual_inf": sigma3_inverse_residual,
        "sigma_best_generator": sigma_best_generator,
        "sigma_best_residual_inf": sigma_best_residual,
        "sigma_best_phase": sigma_best_phase,
        "sigma_best_to_closure": sigma_best_residual / closure_scale,
        "sigma_group_residual_inf": sigma_group_residual,
        "sigma_group_to_closure": sigma_group_residual / closure_scale,
        "sigma_candidate": sigma_group_residual <= sigma_tolerance,
        "F_beta_residual_inf": f_beta_residual,
        "F_beta_to_closure": f_beta_residual / closure_scale,
    }


def command_parse(args: argparse.Namespace) -> int:
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    by_m3: dict[float, int] = {}
    for row in rows:
        by_m3[row.m3] = by_m3.get(row.m3, 0) + 1
    summary = {
        "source": source,
        "rows": len(rows),
        "m3_counts": {f"{key:g}": by_m3[key] for key in sorted(by_m3)},
    }
    print(json.dumps(summary, indent=2))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    selected = select_rows(rows, args.m3, args.limit, args.sort_period, parse_indices(args.indices))
    if not selected:
        raise SystemExit("no rows selected")

    generator_names = [name.strip() for name in args.generators.split(",") if name.strip()]
    missing = [name for name in generator_names if name not in GENERATORS]
    if missing:
        raise SystemExit(f"unknown generators: {', '.join(missing)}")

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for row in selected:
        integrated = integrate_orbit(row, args.rtol, args.atol, args.max_step_fraction)
        for name in generator_names:
            records.append(residual_for_generator(integrated, GENERATORS[name], args.n_samples, args.phase_grid))

    summary: dict[str, object] = {
        "mode": "smoke",
        "source": source,
        "rows_total": len(rows),
        "rows_selected": len(selected),
        "selected_labels": [row.label for row in selected],
        "generators": generator_names,
        "n_samples": args.n_samples,
        "phase_grid": args.phase_grid,
        "rtol": args.rtol,
        "atol": args.atol,
        "elapsed_seconds": time.perf_counter() - started,
        "note": "Smoke only: residual magnitudes are not binding K_facet evidence.",
    }

    print(json.dumps(summary, indent=2))
    if records:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    if args.out:
        write_outputs(Path(args.out), summary, records)
    return 0


def command_sigma3_scan(args: argparse.Namespace) -> int:
    source = args.source.upper()
    text = read_text(args.path or SUPPLEMENT_URLS[source])
    rows = parse_rows(text, source)
    selected = select_rows(rows, args.m3, args.limit, args.sort_period, parse_indices(args.indices))
    if not selected:
        raise SystemExit("no rows selected")

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for row_index, row in enumerate(selected, start=1):
        integrated = integrate_orbit(row, args.rtol, args.atol, args.max_step_fraction)
        sigma3 = residual_for_generator(integrated, GENERATORS["sigma3"], args.n_samples, args.phase_grid)
        sigma3_inverse = residual_for_generator(
            integrated,
            GENERATORS["sigma3_inverse"],
            args.n_samples,
            args.phase_grid,
        )
        f_beta = residual_for_generator(integrated, GENERATORS["F_beta"], args.n_samples, args.phase_grid)
        records.append(
            scan_record_from_residuals(
                integrated,
                sigma3,
                sigma3_inverse,
                f_beta,
                args.sigma_tolerance,
                args.closure_floor,
            )
        )
        if args.report_every and row_index % args.report_every == 0:
            candidates = sum(1 for record in records if record["sigma_candidate"])
            print(
                f"[isotrophy] scanned {row_index}/{len(selected)} rows; "
                f"sigma_candidates={candidates}",
                file=sys.stderr,
                flush=True,
            )

    sigma_candidates = [record for record in records if record["sigma_candidate"]]
    expectation_met = (
        None if args.expected_sigma_count is None else len(sigma_candidates) == args.expected_sigma_count
    )
    summary: dict[str, object] = {
        "mode": "sigma3_scan",
        "source": source,
        "rows_total": len(rows),
        "rows_selected": len(selected),
        "m3": args.m3,
        "rtol": args.rtol,
        "atol": args.atol,
        "n_samples": args.n_samples,
        "phase_grid": args.phase_grid,
        "sigma_tolerance": args.sigma_tolerance,
        "expected_sigma_count": args.expected_sigma_count,
        "sigma_candidate_count": len(sigma_candidates),
        "sigma_candidate_labels": [record["label"] for record in sigma_candidates],
        "expectation_met": expectation_met,
        "elapsed_seconds": time.perf_counter() - started,
        "note": "G.2 scan: not a K_facet result or daughter-family count.",
    }

    print(json.dumps(summary, indent=2))
    if args.print_records and records:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    if args.out:
        write_outputs(Path(args.out), summary, records)
    if args.strict_expected and expectation_met is not True:
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sundog isotrophy workbench smoke harness")
    sub = parser.add_subparsers(dest="command", required=True)

    parse_cmd = sub.add_parser("parse", help="fetch/parse a supplementary file and report row counts")
    parse_cmd.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    parse_cmd.add_argument("--path", help="local supplementary file instead of the live URL")
    parse_cmd.set_defaults(func=command_parse)

    smoke = sub.add_parser("smoke", help="run a capped integration/residual smoke")
    smoke.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    smoke.add_argument("--path", help="local supplementary file instead of the live URL")
    smoke.add_argument("--m3", type=float, default=1.0)
    smoke.add_argument("--limit", type=int, default=1)
    smoke.add_argument("--indices", help="comma-separated orbit indices to include before limit is applied")
    smoke.add_argument("--sort-period", action="store_true", default=True)
    smoke.add_argument("--no-sort-period", dest="sort_period", action="store_false")
    smoke.add_argument("--generators", default="sigma3,sigma3_inverse,F_beta,F_delta")
    smoke.add_argument("--n-samples", type=int, default=181)
    smoke.add_argument("--phase-grid", type=int, default=25)
    smoke.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    smoke.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    smoke.add_argument("--max-step-fraction", type=float, default=DEFAULT_MAX_STEP_FRACTION)
    smoke.add_argument("--out", help="optional output directory for manifest.json and residuals.csv")
    smoke.set_defaults(func=command_smoke)

    scan = sub.add_parser("sigma3-scan", help="scan selected m3 rows for sigma3 precondition candidates")
    scan.add_argument("--source", choices=sorted(SUPPLEMENT_URLS), default="A")
    scan.add_argument("--path", help="local supplementary file instead of the live URL")
    scan.add_argument("--m3", type=float, default=1.0)
    scan.add_argument("--limit", type=int, default=0)
    scan.add_argument("--indices", help="comma-separated orbit indices to include before limit is applied")
    scan.add_argument("--sort-period", action="store_true", default=True)
    scan.add_argument("--no-sort-period", dest="sort_period", action="store_false")
    scan.add_argument("--n-samples", type=int, default=1009)
    scan.add_argument("--phase-grid", type=int, default=73)
    scan.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    scan.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    scan.add_argument("--max-step-fraction", type=float, default=DEFAULT_MAX_STEP_FRACTION)
    scan.add_argument("--sigma-tolerance", type=float, default=DEFAULT_SIGMA_TOLERANCE)
    scan.add_argument("--expected-sigma-count", type=int, default=DEFAULT_EXPECTED_SIGMA_COUNT)
    scan.add_argument("--closure-floor", type=float, default=DEFAULT_CLOSURE_FLOOR)
    scan.add_argument("--strict-expected", action="store_true")
    scan.add_argument("--print-records", action="store_true")
    scan.add_argument("--report-every", type=int, default=25)
    scan.add_argument("--out", help="optional output directory for manifest.json and residuals.csv")
    scan.set_defaults(func=command_sigma3_scan)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
