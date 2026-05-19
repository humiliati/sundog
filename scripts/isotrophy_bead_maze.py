"""Bead-maze render of the detector-recovered three-body choreographies.

Honest-provenance anniversary visual. Framing follows
`internal/anniversary/isotrophy_handoff_note.md` (2026-05-19), which
supersedes the earlier "open bet" framing:

  * G.2 is a real, durable DETECTOR + LITERATURE-COUNT win, not theorem
    evidence: the hardened sigma_3 detector recovers the 21 strict
    single-inertial-curve equal-mass choreographies and cleanly splits off
    4 relative/rotating ones (a 2*pi/3 global rotation), explaining why a
    gauge-invariant gate sees 25 where the literature says 21.
  * K1 is a CLEAN NEGATIVE: the v0.2 daughter-count operator reduced to the
    equivariance-only null (K_facet = 0; generically Z3 ∩ Z2 = {e}). It was
    retired at the cheap precheck, not patched. A success of the process.

Do-Not-Say (baked into the artifact text): isotrophy does NOT prove the
theorem; K_facet = 0 does NOT mean "no piano-trios exist"; the 4 relatives
are OUTSIDE the strict single-closed-trajectory convention, NOT literature
mistakes.

Provenance discipline:
  * The three classes are read from the detector receipt
    results/isotrophy/m3eq1-sigma3-precondition-fixed-inverse-orientation-25/
    residuals.csv. Numbers are read and asserted, never re-derived from prose.
  * Orbits are integrated with the workbench's own physics
    (scripts/isotrophy_workbench.py) — no reimplementation.
  * Each panel is one shared closed "wire" (body-0 path over a period) with
    three "beads" (the bodies at t=0): a doctor's-office bead-maze.
  * SVG is hand-emitted (no matplotlib) — crisp, dependency-free, shareable.

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

_AZ = math.radians(27.0)
_EL = math.radians(22.0)
_CA, _SA = math.cos(_AZ), math.sin(_AZ)
_CE, _SE = math.cos(_EL), math.sin(_EL)

PALETTE = {
    "ink": "#0c1c2b",
    "board": "#102a3f",
    "panel": "#0e2336",
    "frame_strict": "#f4c430",   # brass = strict single-curve choreographies
    "frame_rel": "#a98cff",      # violet = relative / rotating (2*pi/3)
    "wire": "#f4c430",
    "wire_glow": "#b97812",
    "bead": ("#ffd76b", "#ff7a45", "#79bde8"),
    "text": "#dfe9ef",
    "muted": "#9fb3c0",
    "accent": "#f4c430",
}


def project(p):
    x, y, z = p
    xr = _CA * x - _SA * y
    yr = _SA * x + _CA * y
    return xr, -(_CE * yr - _SE * z)


def read_classes(receipt_path: Path):
    """Read the three detector classes straight from the receipt CSV.

    canonical-strict := sigma_strict_single_curve_candidate
    opposite-strict  := sigma_opposite_strict_single_curve_candidate
    relative/rotating := admitted under some orientation but NOT strict
                         single-curve (sigma_any_orientation_candidate and
                         not sigma_any_strict_single_curve_candidate)

    The three sets must be disjoint and total the receipt's row count.
    """
    canonical, opposite, relative = [], [], []
    total = 0
    with receipt_path.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            total += 1
            idx = int(rec["index"])

            def t(col):
                return rec[col].strip().lower() == "true"

            if t("sigma_strict_single_curve_candidate"):
                canonical.append(idx)
            elif t("sigma_opposite_strict_single_curve_candidate"):
                opposite.append(idx)
            elif t("sigma_any_orientation_candidate") and not t(
                "sigma_any_strict_single_curve_candidate"
            ):
                relative.append(idx)
            else:
                raise SystemExit(
                    f"row O_{idx} fits no detector class — receipt schema "
                    f"changed; re-confirm against the handoff before rendering."
                )
    sets = [set(canonical), set(opposite), set(relative)]
    if set.intersection(*sets):
        raise SystemExit("detector classes overlap (unexpected).")
    if (len(canonical), len(opposite), len(relative)) != (13, 8, 4):
        raise SystemExit(
            f"class split {len(canonical)}+{len(opposite)}+{len(relative)} != "
            f"expected 13+8+4 (handoff G.2). Receipt changed — stop."
        )
    if total != 25:
        raise SystemExit(f"receipt has {total} rows, expected 25.")
    return sorted(canonical), sorted(opposite), sorted(relative)


def panel_svg(orbit, x0, y0, w, h, label, tag, frame, rotating):
    pad = 16.0
    inner = (x0 + pad, y0 + pad + 14, w - 2 * pad, h - 2 * pad - 22)
    times = [orbit.row.period * i / N_SAMPLES for i in range(N_SAMPLES + 1)]
    pos = wb.sample_positions(orbit, __import__("numpy").asarray(times))

    wire = [project(pos[i, 0]) for i in range(len(times))]
    beads = [project(pos[0, b]) for b in range(3)]

    xs = [p[0] for p in wire] + [p[0] for p in beads]
    ys = [p[1] for p in wire] + [p[1] for p in beads]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny, 1e-9)
    scale = 0.84 * min(inner[2], inner[3]) / span
    cx_d, cy_d = (minx + maxx) / 2, (miny + maxy) / 2
    cx_b, cy_b = inner[0] + inner[2] / 2, inner[1] + inner[3] / 2

    def tx(p):
        return cx_b + (p[0] - cx_d) * scale, cy_b + (p[1] - cy_d) * scale

    d = "M " + " L ".join(f"{px:.2f} {py:.2f}" for px, py in (tx(p) for p in wire)) + " Z"
    out = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="14" fill="{PALETTE["panel"]}" stroke="{frame}" '
        f'stroke-opacity="{0.7 if rotating else 0.42}" '
        f'stroke-width="{1.8 if rotating else 1.2}"/>',
        f'<path d="{d}" fill="none" stroke="{PALETTE["wire_glow"]}" '
        f'stroke-width="5.5" stroke-opacity="0.45" stroke-linejoin="round" '
        f'stroke-linecap="round"/>',
        f'<path d="{d}" fill="none" stroke="{PALETTE["wire"]}" '
        f'stroke-width="2.1" stroke-linejoin="round" stroke-linecap="round"/>',
    ]
    for b in range(3):
        bx, by = tx(beads[b])
        out.append(
            f'<circle cx="{bx:.2f}" cy="{by:.2f}" r="6.4" '
            f'fill="{PALETTE["bead"][b]}" stroke="{PALETTE["ink"]}" '
            f'stroke-width="1.1"/>'
        )
    if rotating:
        out.append(
            f'<text x="{x0 + w - 14:.1f}" y="{y0 + 26:.1f}" text-anchor="end" '
            f'font-family="ui-monospace,Menlo,Consolas,monospace" '
            f'font-size="12" font-weight="700" fill="{PALETTE["frame_rel"]}">'
            f'↻ 2π/3</text>'
        )
    out.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 + h - 9:.1f}" text-anchor="middle" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace" font-size="12" '
        f'font-weight="700" fill="{PALETTE["text"]}">{label}'
        f'<tspan fill="{PALETTE["muted"]}" font-weight="400"> · {tag}</tspan>'
        f'</text>'
    )
    return "".join(out)


def main() -> int:
    if not MIRROR.exists():
        raise SystemExit(f"catalog mirror missing: {MIRROR}")
    if not RECEIPT.exists():
        raise SystemExit(f"detector receipt missing: {RECEIPT}")

    canonical, opposite, relative = read_classes(RECEIPT)
    print(
        f"strict canonical={len(canonical)} opposite={len(opposite)} "
        f"relative={len(relative)} total={len(canonical)+len(opposite)+len(relative)}"
    )

    order = (
        [(i, "single-curve · canon", False) for i in canonical]
        + [(i, "single-curve · opp", False) for i in opposite]
        + [(i, "relative", True) for i in relative]
    )
    want = {i for i, _, _ in order}
    rows = wb.select_rows(
        wb.parse_rows(wb.read_text(str(MIRROR)), "A"),
        m3=1.0, limit=0, sort_period=False, indices=want,
    )
    by_index = {r.index: r for r in rows}
    missing = sorted(want - set(by_index))
    if missing:
        raise SystemExit(f"indices not found in mirror m3=1 slice: {missing}")

    integrated = {}
    for n, (idx, _tag, _rot) in enumerate(order):
        t0 = time.perf_counter()
        integrated[idx] = wb.integrate_orbit(by_index[idx], RTOL, ATOL, MAX_STEP_FRACTION)
        if n == 0:
            projected = (time.perf_counter() - t0) * len(order)
            print(f"first integration -> projected ~{projected:.0f}s for {len(order)}")
            if projected > 540:
                raise SystemExit(
                    f"projected {projected:.0f}s exceeds ~9min inline budget; "
                    f"stage as an operator command (AGENTS.md ~10-min rule)."
                )

    cols, gap = 7, 22
    pw, ph = 296.0, 248.0
    mx, top, caph = 40.0, 138.0, 184.0
    rows_n = math.ceil(len(order) / cols)
    W = mx * 2 + cols * pw + (cols - 1) * gap
    H = top + rows_n * ph + (rows_n - 1) * gap + caph

    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
        f'role="img" aria-label="Bead-maze of 25 detector-recovered three-body '
        f'choreographies: 21 strict single-curve and 4 relative (2pi/3 '
        f'rotation). A detector and literature-count reconciliation, not '
        f'theorem evidence.">',
        "<metadata>Classes read from results/isotrophy/"
        "m3eq1-sigma3-precondition-fixed-inverse-orientation-25/residuals.csv; "
        "orbits integrated via scripts/isotrophy_workbench.py from the "
        "supplementary-A mirror. G.2 = detector + literature-count "
        "reconciliation (21 strict single-curve + 4 relative = 25 "
        "gauge-invariant, vs 21 in the literature). NOT theorem evidence. The "
        "v0.2 daughter-count test was retired at K1 as the equivariance-only "
        "null (K_facet=0). K_facet=0 does NOT mean no piano-trios; the 4 "
        "relatives are outside the strict single-curve convention, not "
        "literature mistakes. See "
        "internal/anniversary/isotrophy_handoff_note.md.</metadata>",
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="{PALETTE["ink"]}"/>',
        f'<rect x="14" y="14" width="{W - 28:.0f}" height="{H - 28:.0f}" '
        f'rx="20" fill="{PALETTE["board"]}" stroke="{PALETTE["frame_strict"]}" '
        f'stroke-opacity="0.24"/>',
        f'<text x="{W / 2:.0f}" y="60" text-anchor="middle" '
        f'font-family="Georgia,\'Times New Roman\',serif" font-size="34" '
        f'font-weight="700" fill="{PALETTE["text"]}">Three-body '
        f'choreographies, sorted by the wire they ride</text>',
        f'<text x="{W / 2:.0f}" y="92" text-anchor="middle" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace" font-size="14" '
        f'fill="{PALETTE["accent"]}">σ₃ detector vs the Li–Liao '
        f'10,059-orbit catalog: 21 strict single-curve + 4 relative '
        f'(↻ 2π/3) = 25 gauge-invariant, reconciling the '
        f'literature’s 21 — a detector win, not theorem evidence</text>',
        f'<text x="{W / 2:.0f}" y="116" text-anchor="middle" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace" font-size="12" '
        f'fill="{PALETTE["muted"]}">'
        f'<tspan fill="{PALETTE["frame_strict"]}">■</tspan> strict '
        f'single-curve (21)    '
        f'<tspan fill="{PALETTE["frame_rel"]}">■</tspan> relative, '
        f'2π/3 global rotation (4) — outside the strict convention, '
        f'not literature errors</text>',
    ]

    for n, (idx, tag, rot) in enumerate(order):
        r, c = divmod(n, cols)
        x0 = mx + c * (pw + gap)
        y0 = top + r * (ph + gap)
        frame = PALETTE["frame_rel"] if rot else PALETTE["frame_strict"]
        s.append(panel_svg(integrated[idx], x0, y0, pw, ph,
                            f"O_{{{idx}}}", tag, frame, rot))

    cy = H - caph + 4
    s.append(
        f'<line x1="{mx:.0f}" y1="{cy:.0f}" x2="{W - mx:.0f}" y2="{cy:.0f}" '
        f'stroke="{PALETTE["frame_strict"]}" stroke-opacity="0.22"/>'
    )
    line1 = ("What this is — a detector and literature-count win. The "
             "σ₃ workbench recovered the 21 strict equal-mass "
             "choreographies and cleanly split off 4 relative ones (a 2π/3 "
             "global rotation), explaining why a gauge-invariant gate sees 25 "
             "where the literature says 21. It is not theorem evidence.")
    line2 = ("The proposed daughter-count theorem test was retired at its "
             "cheap K1 precheck: under v0.2 it reduced to the "
             "equivariance-only null (K_facet = 0; generically "
             "Z₃ ∩ Z₂ = {e}). The cheap falsifier ran before "
             "the multi-hour sweep.")
    line3 = ("K_facet = 0 does not mean “no piano-trios,” and the 4 "
             "relatives are outside the strict single-curve convention, not "
             "literature errors. This is a receipt for the discipline: publish "
             "the boundary, do not patch the operator. "
             "sundog_v_isotrophy.md · isotrophy_handoff_note.md")
    for i, ln in enumerate((line1, line2, line3)):
        s.append(
            f'<text x="{mx:.0f}" y="{cy + 32 + i * 30:.0f}" '
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
