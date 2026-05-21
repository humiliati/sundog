"""Post-process the rendered bead-maze SVG to wrap each panel in a <g>.

The threebody.html "catalog sidecar" knob (rotation-angle threshold) needs
to address each panel individually from JS. The Python renderer
(isotrophy_bead_maze.py) emits panels as flat element runs without group
wrappers, so this script reads the rendered SVG + the detector receipt and
emits an annotated copy with one <g class="iso-cell" data-...> per panel.

Inputs:
  public/media/isotrophy-bead-maze.svg           (rendered, do not modify)
  results/isotrophy/m3eq1-sigma3-precondition-fixed-inverse-orientation-25/residuals.csv

Output:
  public/media/isotrophy-bead-maze-annotated.svg

Per panel <g> attributes:
  class="iso-cell"
  data-label   = e.g. "O_62"
  data-index   = catalog index
  data-strict  = "true" | "false"
  data-orient  = "canonical" | "opposite" | "relative"
  data-rotation-rad = SO(3)-min rotation angle in radians (catalog-accepted orientation)
  data-period  = orbit period

No re-integration is performed; this is a pure text/structure rewrite.
Run:  python scripts/isotrophy_bead_maze_annotate.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "public" / "media" / "isotrophy-bead-maze.svg"
RECEIPT = (
    ROOT
    / "results"
    / "isotrophy"
    / "m3eq1-sigma3-precondition-fixed-inverse-orientation-25"
    / "residuals.csv"
)
OUT = ROOT / "public" / "media" / "isotrophy-bead-maze-annotated.svg"

# A panel background rect is the one with rx="14" + fill="#0e2336" (panel color).
# The outer board has rx="20" so it does not collide. The label <text> closes
# the panel; it always contains 'O_{<index>}<tspan'.
RECT_RE = re.compile(
    r'<rect [^>]*rx="14"[^>]*fill="#0e2336"[^>]*/>',
)
LABEL_RE = re.compile(r'<text [^>]*>O_\{(\d+)\}<tspan[^>]*>([^<]*)</tspan></text>')


def read_receipt(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            idx = int(rec["index"])
            strict = rec["sigma_any_strict_single_curve_candidate"].strip().lower() == "true"
            canon_strict = (
                rec["sigma_strict_single_curve_candidate"].strip().lower() == "true"
            )
            opp_strict = (
                rec["sigma_opposite_strict_single_curve_candidate"].strip().lower() == "true"
            )
            canon_cand = rec["sigma_candidate"].strip().lower() == "true"
            # Pick the rotation angle from whichever orientation actually
            # admitted the row. Canonical first if it admitted, otherwise
            # opposite. Relative rows admit only via SO(3) gauge in one
            # orientation; we pick the accepted one.
            if canon_cand or canon_strict:
                orient = "canonical"
                angle = float(rec["sigma_group_rotation_angle_rad"])
            else:
                orient = "opposite"
                angle = float(rec["sigma_opposite_group_rotation_angle_rad"])
            if not strict:
                orient = "relative"
            out[idx] = {
                "strict": strict,
                "orient": orient,
                "rotation_rad": angle,
                "period": float(rec["period"]),
            }
    return out


def main() -> int:
    if not SRC.exists():
        print(f"missing source SVG: {SRC}", file=sys.stderr)
        return 1
    if not RECEIPT.exists():
        print(f"missing detector receipt: {RECEIPT}", file=sys.stderr)
        return 1

    svg = SRC.read_text(encoding="utf-8")
    by_index = read_receipt(RECEIPT)

    # Find all panel rects + all label texts in document order. Each panel is
    # the contiguous slice [rect_start, label_end) where label_end is the
    # closing </text> of the *next* label after this rect.
    rect_spans = [m.span() for m in RECT_RE.finditer(svg)]
    label_matches = list(LABEL_RE.finditer(svg))
    if len(rect_spans) != 25 or len(label_matches) != 25:
        print(
            f"expected 25 panels; got {len(rect_spans)} rects and "
            f"{len(label_matches)} labels — schema drift, aborting.",
            file=sys.stderr,
        )
        return 2

    parts = []
    cursor = 0
    annotated = 0
    for rect_span, label_match in zip(rect_spans, label_matches):
        rect_start, _ = rect_span
        label_end = label_match.end()
        idx = int(label_match.group(1))
        meta = by_index.get(idx)
        if meta is None:
            print(f"panel O_{{{idx}}} not in receipt — schema drift", file=sys.stderr)
            return 3

        # Emit prefix (everything since the previous slice ended) untouched,
        # then the wrapped panel.
        parts.append(svg[cursor:rect_start])
        parts.append(
            f'<g class="iso-cell" '
            f'data-label="O_{idx}" '
            f'data-index="{idx}" '
            f'data-strict="{"true" if meta["strict"] else "false"}" '
            f'data-orient="{meta["orient"]}" '
            f'data-rotation-rad="{meta["rotation_rad"]:.6e}" '
            f'data-period="{meta["period"]:.6f}">'
        )
        parts.append(svg[rect_start:label_end])
        parts.append("</g>")
        cursor = label_end
        annotated += 1

    parts.append(svg[cursor:])
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} with {annotated} annotated panels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
