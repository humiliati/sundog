"""Bead-maze render of the detector-recovered strict three-body choreographies.

Honest-provenance anniversary visual (sundog_v_isotrophy.md, signed-off
"honest open-bet + visual" framing):

  * The strict-21 index set is read **from the detector receipt**
    results/isotrophy/m3eq1-sigma3-precondition-fixed-inverse-orientation-25/
    residuals.csv (columns sigma_strict_single_curve_candidate /
    sigma_opposite_strict_single_curve_candidate). Numbers are read, never
    re-derived from prose (spec self-consistency).
  * Orbits are integrated with the workbench's own physics
    (scripts/isotrophy_workbench.py) — no reimplementation.
  * On a choreography all three bodies trace ONE shared closed curve, so each
    panel is one "wire" (body-0 path over a full period) with three "beads"
    (the three bodies at t=0) — a doctor's-office bead-maze of the catalog.
  * The baked-in caption states the artifact and the OPEN bet (piano-trio
    descent count K_facet vs K_emp). It does NOT claim a theorem result;
    the theorem-killing outcome stays live, exactly as the roadmap says.

SVG is hand-emitted (no matplotlib; matches the generate-sundog-logo-toolkit
precedent) so the output is a crisp, dependency-free, shareable vector asset.

Run:  python scripts/isotrophy_bead_maze.py
Out:  public/media/isotrophy-bead-maze.svg
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import isotrophy_workbench as wb  # noqa: E402  (import-safe: __main__ guard)

MIRROR = ROOT / "docs" / "isotrophy" / "supplementary-A_periodic-3d_mirror.txt"
RECEIPT = (
    ROOT
    / "results"
    / "isotrophy"
    / "m3eq1-sigma3-precondition-fixed-inverse-orientation-25"
    / "residuals.csv"
)
OUT = ROOT / "public" / "media" / "isotrophy-bead-maze.svg"

# Integration accuracy: this is a *picture*, not the detector gate. 1e-11 gives
# visually-closed loops fast; we do NOT rerun the ~8.18h G.2 scan.
RTOL = ATOL = 1e-11
MAX_STEP_FRACTION = 0.01
N_SAMPLES = 540

# Fixed isometric projection, identical for every panel so the 21 are visually
# comparable (azimuth about world-z, then tilt about screen-x).
_AZ = math.radians(27.0)
_EL = math.radians(22.0)
_CA, _SA = math.cos(_AZ), math.sin(_AZ)
_CE, _SE = math.cos(_EL), math.sin(_EL)

PALETTE = {
    "ink": "#0c1c2b",
    "board": "#102a3f",
    "panel": "#0e2336",
    "frame": "#7da2b8",
    "wire": "#f4c430",
    "wire_glow": "#b97812",
    "bead": ("#ffd76b", "#ff7a45", "#79bde8"),
    "text": "#dfe9ef",
    "muted": "#9fb3c0",
    "accent": "#f4c430",
}


def project(p):
    """3D world point -> 2D screen (x right, y down)."""
    x, y, z = p
    xr = _CA * x - _SA * y
    yr = _SA * x + _CA * y
    sx = xr
    sy = _CE * yr - _SE * z
    return sx, -sy


def read_strict_indices(receipt_path: Path):
    """Read the strict-21 straight from the detector receipt CSV.

    canonical-strict := sigma_strict_single_curve_candidate is True
    opposite-strict  := sigma_opposite_strict_single_curve_candidate is True
    (the two sets are disjoint per the roadmap; we assert that here).
    """
    canonical, opposite = [], []
    with receipt_path.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            idx = int(rec["index"])
            if rec["sigma_strict_single_curve_candidate"].strip().lower() == "true":
                canonical.append(idx)
            elif rec["sigma_opposite_strict_single_curve_candidate"].strip().lower() == "true":
                opposite.append(idx)
    overlap = set(canonical) & set(opposite)
    if overlap:
        raise SystemExit(f"strict canonical/opposite overlap (unexpected): {sorted(overlap)}")
    return sorted(canonical), sorted(opposite)


def panel_svg(orbit, x0, y0, w, h, label, tag):
    pad = 16.0
    inner = (x0 + pad, y0 + pad + 14, w - 2 * pad, h - 2 * pad - 22)
    times = [orbit.row.period * i / N_SAMPLES for i in range(N_SAMPLES + 1)]
    pos = wb.sample_positions(orbit, __import__("numpy").asarray(times))  # (N,3,3)

    wire = [project(pos[i, 0]) for i in range(len(times))]  # body-0 = the shared loop
    beads = [project(pos[0, b]) for b in range(3)]          # 3 bodies at t=0

    xs = [p[0] for p in wire] + [p[0] for p in beads]
    ys = [p[1] for p in wire] + [p[1] for p in beads]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny, 1e-9)
    scale = 0.86 * min(inner[2], inner[3]) / span
    cx_data, cy_data = (minx + maxx) / 2, (miny + maxy) / 2
    cx_box, cy_box = inner[0] + inner[2] / 2, inner[1] + inner[3] / 2

    def tx(p):
        return (
            cx_box + (p[0] - cx_data) * scale,
            cy_box + (p[1] - cy_data) * scale,
        )

    wpts = [tx(p) for p in wire]
    d = "M " + " L ".join(f"{px:.2f} {py:.2f}" for px, py in wpts) + " Z"

    out = []
    out.append(
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="14" '
        f'fill="{PALETTE["panel"]}" stroke="{PALETTE["frame"]}" '
        f'stroke-opacity="0.34" stroke-width="1.2"/>'
    )
    # the wire: a soft underglow then the bright brass loop
    out.append(f'<path d="{d}" fill="none" stroke="{PALETTE["wire_glow"]}" '
               f'stroke-width="5.5" stroke-opacity="0.45" '
               f'stroke-linejoin="round" stroke-linecap="round"/>')
    out.append(f'<path d="{d}" fill="none" stroke="{PALETTE["wire"]}" '
               f'stroke-width="2.1" stroke-linejoin="round" stroke-linecap="round"/>')
    # three beads at t=0
    for b in range(3):
        bx, by = tx(beads[b])
        out.append(f'<circle cx="{bx:.2f}" cy="{by:.2f}" r="6.4" '
                   f'fill="{PALETTE["bead"][b]}" stroke="{PALETTE["ink"]}" '
                   f'stroke-width="1.1"/>')
    out.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 + h - 9:.1f}" text-anchor="middle" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace" font-size="12" '
        f'font-weight="700" fill="{PALETTE["text"]}">{label}'
        f'<tspan fill="{PALETTE["muted"]}" font-weight="400"> · {tag}</tspan></text>'
    )
    return "".join(out)


def main() -> int:
    if not MIRROR.exists():
        raise SystemExit(f"catalog mirror missing: {MIRROR}")
    if not RECEIPT.exists():
        raise SystemExit(f"detector receipt missing: {RECEIPT}")

    canonical, opposite = read_strict_indices(RECEIPT)
    print(f"strict canonical={len(canonical)} opposite={len(opposite)} "
          f"total={len(canonical) + len(opposite)}")
    if (len(canonical), len(opposite)) != (13, 8):
        raise SystemExit(
            f"strict split {len(canonical)}+{len(opposite)} != expected 13+8 "
            f"— receipt changed; re-confirm before rendering."
        )

    order = [(i, "canon") for i in canonical] + [(i, "opp") for i in opposite]
    want = {i for i, _ in order}
    rows = wb.select_rows(
        wb.parse_rows(wb.read_text(str(MIRROR)), "A"),
        m3=1.0, limit=0, sort_period=False, indices=want,
    )
    by_index = {r.index: r for r in rows}
    missing = sorted(want - set(by_index))
    if missing:
        raise SystemExit(f"indices not found in mirror m3=1 slice: {missing}")

    # AGENTS.md ~10-min rule: measure the first integration, extrapolate, abort
    # to a staged command if the full 21 would blow the budget.
    integrated = {}
    t_first = None
    for n, (idx, _tag) in enumerate(order):
        t0 = time.perf_counter()
        integrated[idx] = wb.integrate_orbit(by_index[idx], RTOL, ATOL, MAX_STEP_FRACTION)
        dt = time.perf_counter() - t0
        if n == 0:
            t_first = dt
            projected = dt * len(order)
            print(f"first integration {dt:.1f}s -> projected ~{projected:.0f}s for 21")
            if projected > 540:
                raise SystemExit(
                    f"projected {projected:.0f}s exceeds ~9min inline budget; "
                    f"stage as an operator command instead (AGENTS.md ~10-min rule)."
                )

    cols, gap = 7, 22
    pw, ph = 296.0, 248.0
    mx, top, caph = 40.0, 132.0, 150.0
    rows_n = math.ceil(len(order) / cols)
    W = mx * 2 + cols * pw + (cols - 1) * gap
    H = top + rows_n * ph + (rows_n - 1) * gap + caph

    s = []
    s.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
        f'role="img" aria-label="Bead-maze of 21 detector-recovered strict '
        f'three-body choreographies">'
    )
    s.append(
        "<metadata>Provenance: indices read from results/isotrophy/"
        "m3eq1-sigma3-precondition-fixed-inverse-orientation-25/residuals.csv "
        "(sigma_strict / sigma_opposite_strict single-curve flags); orbits "
        "integrated via scripts/isotrophy_workbench.py from "
        "docs/isotrophy/supplementary-A_periodic-3d_mirror.txt. Artifact "
        "(21 strict choreographies recovered exactly) — NOT a theorem result; "
        "the K_facet vs K_emp piano-trio descent count is an open bet.</metadata>"
    )
    s.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="{PALETTE["ink"]}"/>')
    s.append(f'<rect x="14" y="14" width="{W - 28:.0f}" height="{H - 28:.0f}" '
             f'rx="20" fill="{PALETTE["board"]}" stroke="{PALETTE["frame"]}" '
             f'stroke-opacity="0.28"/>')
    s.append(
        f'<text x="{W / 2:.0f}" y="62" text-anchor="middle" '
        f'font-family="Georgia,\'Times New Roman\',serif" font-size="34" '
        f'font-weight="700" fill="{PALETTE["text"]}">'
        f'Twenty-one three-body choreographies, strung on their wires</text>'
    )
    s.append(
        f'<text x="{W / 2:.0f}" y="94" text-anchor="middle" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace" font-size="14" '
        f'fill="{PALETTE["accent"]}">recovered exactly by the σ₃ detector from '
        f'the Li–Liao 10,059-orbit catalog · 13 canonical + 8 opposite, '
        f'single closed curve each</text>'
    )

    for n, (idx, tag) in enumerate(order):
        r, c = divmod(n, cols)
        x0 = mx + c * (pw + gap)
        y0 = top + r * (ph + gap)
        s.append(panel_svg(integrated[idx], x0, y0, pw, ph, f"O_{{{idx}}}", tag))

    cy = H - caph + 6
    s.append(f'<line x1="{mx:.0f}" y1="{cy:.0f}" x2="{W - mx:.0f}" y2="{cy:.0f}" '
             f'stroke="{PALETTE["frame"]}" stroke-opacity="0.25"/>')
    line1 = ("What this is: a detector receipt. The σ₃ isotropy detector pulled "
             "these 21 strict equal-mass choreographies out of 10,059 orbits — "
             "the count matches the catalog exactly.")
    line2 = ("The open bet (anniversary): does the sundog refinement predict the "
             "piano-trio descent count K_facet from each choreography's residual "
             "spacetime isotropy? K_emp is not yet measured.")
    line3 = ("Not a theorem result. The outcome that kills the refinement — "
             "K_facet and K_emp unrelated after facet conditioning — is still "
             "live. Workbench: sundog_v_isotrophy.md.")
    for i, ln in enumerate((line1, line2, line3)):
        s.append(
            f'<text x="{mx:.0f}" y="{cy + 30 + i * 26:.0f}" '
            f'font-family="Georgia,\'Times New Roman\',serif" font-size="16" '
            f'fill="{PALETTE["text"] if i != 2 else PALETTE["muted"]}">{ln}</text>'
        )
    s.append("</svg>\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(s), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
