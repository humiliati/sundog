#!/usr/bin/env python3
"""Generate Sundog OG cards (1200x630) from a config-driven template.

Each card: eyebrow + headline (1-3 lines, serif) + lede (1-3 lines, sans) +
URL footer + a per-page viz snippet on the right. Sundog house style: warm
paper gradient, brass accent, ink type. ImageMagick rasterizes the SVG.

Usage: python3 _generate.py   (rebuilds every entry in CARDS)
       python3 _generate.py index sundog   (rebuilds only those names)
"""

from __future__ import annotations
import sys, subprocess, pathlib, textwrap

OUT_DIR = pathlib.Path(__file__).parent


def text_lines(x: int, y: int, lines: list[str], *, family: str, size: int, weight: int, fill: str, line_height: float | None = None, letter_spacing: float = 0) -> str:
    """Render a stack of text lines at (x, y) (y = first baseline)."""
    if line_height is None:
        line_height = size * 1.08
    out = []
    for i, line in enumerate(lines):
        yy = y + i * line_height
        # XML-escape & ' " < >
        esc = (line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                   .replace('"', '&quot;'))
        ls_attr = f' letter-spacing="{letter_spacing}"' if letter_spacing else ''
        out.append(
            f'<text x="{x}" y="{int(yy)}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}"'
            f'{ls_attr}>{esc}</text>'
        )
    return '\n  '.join(out)


def build_svg(eyebrow: str, headline: list[str], lede: list[str], url: str, viz: str) -> str:
    SERIF = "Georgia, 'Times New Roman', serif"
    SANS = "Verdana, Geneva, sans-serif"
    MONO = "Courier New, Courier, monospace"

    eyebrow_block = text_lines(60, 100, [eyebrow], family=SANS, size=20, weight=700,
                               fill="#2A5570", letter_spacing=3)

    # Headline: 1-3 lines. Size depends on line count. Sized so the longest
    # plausible line (~22 chars) fits left of the right-side viz column.
    h_count = len(headline)
    h_size = {1: 68, 2: 56, 3: 48}.get(h_count, 48)
    h_y_start = {1: 220, 2: 200, 3: 180}.get(h_count, 180)
    headline_block = text_lines(60, h_y_start, headline, family=SERIF, size=h_size,
                                weight=700, fill="#1A3A52", line_height=h_size * 1.06)

    # Lede: 1-3 lines.
    l_y_start = h_y_start + h_size * 1.06 * h_count + 48
    lede_block = text_lines(60, int(l_y_start), lede, family=SANS, size=22, weight=400,
                            fill="#40505C", line_height=32)

    url_block = text_lines(60, 560, [url], family=MONO, size=22, weight=700, fill="#1A3A52")

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630" width="1200" height="630">
  <defs>
    <linearGradient id="paper" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.72"/>
      <stop offset="100%" stop-color="#FFFAF0" stop-opacity="0.94"/>
    </linearGradient>
    <pattern id="ruled" x="0" y="0" width="1200" height="28" patternUnits="userSpaceOnUse">
      <line x1="0" y1="27" x2="1200" y2="27" stroke="#63A3DC" stroke-opacity="0.075" stroke-width="1"/>
    </pattern>
    <radialGradient id="halo" cx="0.18" cy="0.12" r="0.6">
      <stop offset="0%" stop-color="#F4C430" stop-opacity="0.10"/>
      <stop offset="100%" stop-color="#F4C430" stop-opacity="0"/>
    </radialGradient>
    <filter id="cardShadow" x="-10%" y="-10%" width="120%" height="125%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="6"/>
      <feOffset dx="0" dy="4"/>
      <feComponentTransfer><feFuncA type="linear" slope="0.15"/></feComponentTransfer>
      <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <!-- Paper-ruled base: warm cream + horizontal blue rules + red legal-pad margin -->
  <rect width="1200" height="630" fill="#FEFDFB"/>
  <rect width="1200" height="630" fill="url(#paper)"/>
  <rect width="1200" height="630" fill="url(#ruled)"/>
  <rect x="90" y="0" width="2" height="630" fill="#FF6363" fill-opacity="0.12"/>
  <rect width="1200" height="630" fill="url(#halo)"/>

  <!-- Top accent rule -->
  <rect x="60" y="60" width="80" height="3" fill="#C99517"/>

  <!-- Eyebrow -->
  {eyebrow_block}

  <!-- Headline -->
  {headline_block}

  <!-- Lede -->
  {lede_block}

  <!-- URL -->
  {url_block}

  <!-- Per-page visualization (right side, ~620 to ~1140) -->
  {viz}

  <!-- Bottom-right mark -->
  <text x="1140" y="600" font-family="Verdana, Geneva, sans-serif" font-size="14" font-weight="700" letter-spacing="2" fill="#9AA4AC" text-anchor="end">sundog.cc</text>
</svg>
"""


# ---------- Per-page viz snippets (positioned within the full 1200x630 viewBox) ----------

def viz_halo_atlas() -> str:
    """Concentric halo rings + parhelia + parhelic line. For index.html and sundog.html."""
    cx, cy = 880, 305
    return f"""
  <g transform="translate({cx},{cy})">
    <!-- 22 degree halo -->
    <circle cx="0" cy="0" r="150" fill="none" stroke="#1A3A52" stroke-width="2.4" opacity="0.85"/>
    <!-- 46 degree halo (faint, partial) -->
    <circle cx="0" cy="0" r="220" fill="none" stroke="#1A3A52" stroke-width="1.2" opacity="0.32" stroke-dasharray="6 6"/>
    <!-- parhelic circle (horizontal) -->
    <line x1="-260" y1="0" x2="260" y2="0" stroke="#2A5570" stroke-width="1.5" opacity="0.55"/>
    <!-- sun column (vertical light pillar) -->
    <line x1="0" y1="-260" x2="0" y2="260" stroke="#C99517" stroke-width="1.5" opacity="0.4" stroke-dasharray="4 4"/>
    <!-- Sun (compass rose / diamond) -->
    <polygon points="0,-22 22,0 0,22 -22,0" fill="#F4C430" stroke="#C99517" stroke-width="2"/>
    <!-- Parhelia (left and right of sun, outward-facing daggers) -->
    <polygon points="-150,-10 -130,0 -150,10 -170,0" fill="#F4C430" stroke="#C99517" stroke-width="1.5"/>
    <polygon points="150,-10 170,0 150,10 130,0" fill="#F4C430" stroke="#C99517" stroke-width="1.5"/>
    <!-- Circumzenithal arc (smile, top) -->
    <path d="M -130,-200 Q 0,-260 130,-200" fill="none" stroke="#C99517" stroke-width="2.4" opacity="0.85"/>
  </g>
  <text x="880" y="555" font-family="Verdana, Geneva, sans-serif" font-size="13" fill="#6A7680" text-anchor="middle">22° halo · parhelic circle · parhelia · CZA</text>
"""


def viz_about_ladder() -> str:
    """Evidence-tier ladder. Boxes stacked, each labeled."""
    x, y = 820, 140  # shifted right to clear the 64pt headline
    rows = [
        ("Operating envelope", "#F4C430", "#C99517"),
        ("Research result",    "#2A5570", "#1A3A52"),
        ("Instrumented prototype", "#FFFFFF", "#9AA4AC"),
        ("Plausible",          "#FFFFFF", "#9AA4AC"),
        ("Aspirational",       "#FFFFFF", "#CFD8DC"),
    ]
    width = 310  # narrower so the ladder fits in 820..1140
    parts = [f'<g transform="translate({x},{y})">']
    parts.append(f'<text x="{width//2}" y="-12" font-family="Verdana, Geneva, sans-serif" font-size="13" font-weight="700" letter-spacing="2" fill="#6A7680" text-anchor="middle">EVIDENCE TIER LADDER</text>')
    for i, (label, fill, stroke) in enumerate(rows):
        text_color = "#1A3A52" if fill in ("#FFFFFF",) else "#FFFFFF"
        if fill == "#F4C430":
            text_color = "#1A3A52"
        parts.append(f'<rect x="0" y="{i*54}" width="{width}" height="44" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        parts.append(f'<text x="18" y="{i*54+28}" font-family="Verdana, Geneva, sans-serif" font-size="16" font-weight="700" fill="{text_color}">{label}</text>')
    parts.append('</g>')
    return '\n  '.join(parts)


def viz_alignment_split() -> str:
    """Bayes vs Sundog split: posterior heatmap on the left, signal trace on the right.
    Total viz width ~300, sits at translate(840,160) to clear the 64pt headline."""
    return """
  <g transform="translate(840,160)">
    <text x="150" y="-15" font-family="Verdana, Geneva, sans-serif" font-size="13" font-weight="700" letter-spacing="2" fill="#6A7680" text-anchor="middle">BAYES · SUNDOG</text>
    <!-- Left panel: posterior heatmap -->
    <rect x="0" y="0" width="140" height="170" rx="6" fill="#FFFFFF" stroke="#CFD8DC" stroke-width="1.5"/>
    <text x="70" y="190" font-family="Verdana, Geneva, sans-serif" font-size="12" fill="#1A3A52" text-anchor="middle">posterior</text>
    <!-- 4x4 heatmap cells: 26x26 each, 4 gap, 14 padding -->
    <rect x="14"  y="14"  width="26" height="26" fill="#1A3A52" opacity="0.20"/>
    <rect x="44"  y="14"  width="26" height="26" fill="#1A3A52" opacity="0.42"/>
    <rect x="74"  y="14"  width="26" height="26" fill="#1A3A52" opacity="0.62"/>
    <rect x="104" y="14"  width="26" height="26" fill="#1A3A52" opacity="0.45"/>
    <rect x="14"  y="44"  width="26" height="26" fill="#1A3A52" opacity="0.30"/>
    <rect x="44"  y="44"  width="26" height="26" fill="#1A3A52" opacity="0.55"/>
    <rect x="74"  y="44"  width="26" height="26" fill="#1A3A52" opacity="0.85"/>
    <rect x="104" y="44"  width="26" height="26" fill="#1A3A52" opacity="0.55"/>
    <rect x="14"  y="74"  width="26" height="26" fill="#1A3A52" opacity="0.18"/>
    <rect x="44"  y="74"  width="26" height="26" fill="#1A3A52" opacity="0.40"/>
    <rect x="74"  y="74"  width="26" height="26" fill="#1A3A52" opacity="0.58"/>
    <rect x="104" y="74"  width="26" height="26" fill="#1A3A52" opacity="0.34"/>
    <rect x="14"  y="104" width="26" height="26" fill="#1A3A52" opacity="0.10"/>
    <rect x="44"  y="104" width="26" height="26" fill="#1A3A52" opacity="0.22"/>
    <rect x="74"  y="104" width="26" height="26" fill="#1A3A52" opacity="0.32"/>
    <rect x="104" y="104" width="26" height="26" fill="#1A3A52" opacity="0.18"/>

    <!-- Right panel: signal trace -->
    <rect x="160" y="0" width="140" height="170" rx="6" fill="#FFFFFF" stroke="#CFD8DC" stroke-width="1.5"/>
    <text x="230" y="190" font-family="Verdana, Geneva, sans-serif" font-size="12" fill="#1A3A52" text-anchor="middle">signal trace</text>
    <line x1="160" y1="50"  x2="300" y2="50"  stroke="#E2E6E9" stroke-width="1"/>
    <line x1="160" y1="90"  x2="300" y2="90"  stroke="#E2E6E9" stroke-width="1"/>
    <line x1="160" y1="130" x2="300" y2="130" stroke="#E2E6E9" stroke-width="1"/>
    <path d="M 160,145 L 175,135 L 190,128 L 205,110 L 220,90 L 235,70 L 250,55 L 265,44 L 280,38 L 300,34"
          fill="none" stroke="#C99517" stroke-width="2.6"/>
    <circle cx="175" cy="135" r="2.5" fill="#C99517"/>
    <circle cx="220" cy="90"  r="2.5" fill="#C99517"/>
    <circle cx="265" cy="44"  r="2.5" fill="#C99517"/>
  </g>
"""


def viz_balance_cartpole() -> str:
    """Cart-pole with shadow on a baseline."""
    return """
  <g transform="translate(700,180)">
    <text x="220" y="-25" font-family="Verdana, Geneva, sans-serif" font-size="13" font-weight="700" letter-spacing="2" fill="#6A7680" text-anchor="middle">CART · SHADOW · POLE</text>
    <!-- floor -->
    <line x1="0" y1="200" x2="440" y2="200" stroke="#1A3A52" stroke-width="2"/>
    <!-- wall (shadow surface) -->
    <line x1="440" y1="0" x2="440" y2="200" stroke="#1A3A52" stroke-width="2"/>
    <!-- cart -->
    <rect x="160" y="160" width="100" height="40" rx="4" fill="#2A5570" stroke="#1A3A52" stroke-width="1.5"/>
    <circle cx="180" cy="200" r="9" fill="#1A3A52"/>
    <circle cx="240" cy="200" r="9" fill="#1A3A52"/>
    <!-- pole (rotated slightly) -->
    <line x1="210" y1="160" x2="265" y2="40" stroke="#C99517" stroke-width="6" stroke-linecap="round"/>
    <circle cx="265" cy="40" r="11" fill="#F4C430" stroke="#C99517" stroke-width="2"/>
    <!-- shadow of pole on the wall -->
    <line x1="440" y1="60" x2="440" y2="160" stroke="#1A3A52" stroke-width="6" opacity="0.30" stroke-linecap="round"/>
    <!-- shadow projection rays (subtle) -->
    <line x1="265" y1="40" x2="440" y2="60" stroke="#9AA4AC" stroke-width="0.8" stroke-dasharray="3 3" opacity="0.5"/>
    <line x1="210" y1="160" x2="440" y2="160" stroke="#9AA4AC" stroke-width="0.8" stroke-dasharray="3 3" opacity="0.5"/>
    <text x="450" y="115" font-family="Verdana, Geneva, sans-serif" font-size="11" fill="#1A3A52" font-style="italic">shadow</text>
  </g>
"""


def viz_threebody_orbits() -> str:
    """Three orbital traces around a central neighborhood."""
    return """
  <g transform="translate(880,305)">
    <text x="0" y="-225" font-family="Verdana, Geneva, sans-serif" font-size="13" font-weight="700" letter-spacing="2" fill="#6A7680" text-anchor="middle">LOCAL POCKET</text>
    <!-- Orbit 1: tight inner loop -->
    <ellipse cx="-30" cy="-10" rx="85" ry="55" fill="none" stroke="#1A3A52" stroke-width="2" opacity="0.85" transform="rotate(-20)"/>
    <!-- Orbit 2: bigger looping -->
    <path d="M -120,40 C -80,-100 80,-100 120,40 C 80,160 -80,160 -120,40 Z"
          fill="none" stroke="#C99517" stroke-width="2.4" opacity="0.85"/>
    <!-- Orbit 3: escaping spiral -->
    <path d="M 0,0 C 50,-50 130,-30 160,40 C 180,90 170,150 200,200"
          fill="none" stroke="#D45A4A" stroke-width="2" opacity="0.7" stroke-dasharray="4 4"/>
    <!-- Bodies -->
    <circle cx="-100" cy="-30" r="10" fill="#1A3A52"/>
    <circle cx="85"  cy="20"   r="10" fill="#C99517" stroke="#9C7A12" stroke-width="1"/>
    <circle cx="170" cy="100"  r="8"  fill="#D45A4A" stroke="#9C2D21" stroke-width="1"/>
    <!-- escape arrow -->
    <line x1="180" y1="120" x2="220" y2="200" stroke="#D45A4A" stroke-width="1.5" marker-end="url(#arrowred)"/>
  </g>
  <defs>
    <marker id="arrowred" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M 0,0 L 10,5 L 0,10 z" fill="#D45A4A"/>
    </marker>
  </defs>
"""


def viz_mines_grid() -> str:
    """A small minesweeper-style grid with pressure shading and one revealed mine."""
    return """
  <g transform="translate(700,160)">
    <text x="180" y="-15" font-family="Verdana, Geneva, sans-serif" font-size="13" font-weight="700" letter-spacing="2" fill="#6A7680" text-anchor="middle">PRESSURE FIELD · SAFE POCKET</text>
    <!-- 6x6 grid, cell 50x50, gap 4 -->
    <!-- Pressure shading: increases toward bottom-right -->
""" + "".join([
        f'    <rect x="{c*54}" y="{r*54}" width="50" height="50" rx="3" fill="#F4C430" fill-opacity="{0.05 + (r+c) * 0.05:.2f}" stroke="#C99517" stroke-opacity="{0.20 + (r+c)*0.06:.2f}" stroke-width="1.4"/>'
        for r in range(6) for c in range(6)
    ]) + """
    <!-- revealed mine marker bottom-right -->
    <circle cx="297" cy="297" r="14" fill="#D45A4A" stroke="#9C2D21" stroke-width="2"/>
    <line x1="288" y1="288" x2="306" y2="306" stroke="#FFFFFF" stroke-width="2"/>
    <line x1="306" y1="288" x2="288" y2="306" stroke="#FFFFFF" stroke-width="2"/>
    <!-- safe pocket annotation top-left -->
    <circle cx="25"  cy="25"  r="5" fill="#2E7D5A"/>
    <text x="38" y="29" font-family="Verdana, Geneva, sans-serif" font-size="12" fill="#2E7D5A">safe</text>
  </g>
"""


# ---------- Card configs ----------

def viz_navierstokes_witness() -> str:
    """NSE C1 Reading-2 witness: two Grashof regime boxes with witness-pair beads."""
    return '\n'.join([
        '<g transform="translate(660,160)" filter="url(#cardShadow)">',
        '  <rect x="0" y="0" width="200" height="130" rx="10" fill="#FFFFFF" stroke="#B8831E" stroke-width="2"/>',
        '  <text x="100" y="34" font-family="Verdana, Geneva, sans-serif" font-size="14" font-weight="800" letter-spacing="2" fill="#684811" text-anchor="middle">G = 200</text>',
        '  <text x="100" y="62" font-family="Verdana, Geneva, sans-serif" font-size="11" fill="#684811" text-anchor="middle">Reading-2 witness</text>',
        '  <text x="100" y="80" font-family="Verdana, Geneva, sans-serif" font-size="11" fill="#684811" text-anchor="middle">d = 32</text>',
        '  <line x1="20" y1="92" x2="180" y2="92" stroke="#CFD8DC" stroke-width="1"/>',
        '  <text x="100" y="112" font-family="Courier New, monospace" font-size="12" font-weight="700" fill="#1A3A52" text-anchor="middle">a_mm = -0.00079</text>',
        '</g>',
        '<g transform="translate(900,160)" filter="url(#cardShadow)">',
        '  <rect x="0" y="0" width="200" height="130" rx="10" fill="#FFFFFF" stroke="#B8831E" stroke-width="3"/>',
        '  <text x="100" y="34" font-family="Verdana, Geneva, sans-serif" font-size="14" font-weight="800" letter-spacing="2" fill="#684811" text-anchor="middle">G = 300</text>',
        '  <text x="100" y="62" font-family="Verdana, Geneva, sans-serif" font-size="11" fill="#684811" text-anchor="middle">generality test</text>',
        '  <text x="100" y="80" font-family="Verdana, Geneva, sans-serif" font-size="11" fill="#684811" text-anchor="middle">d = 18</text>',
        '  <line x1="20" y1="92" x2="180" y2="92" stroke="#CFD8DC" stroke-width="1"/>',
        '  <text x="100" y="112" font-family="Courier New, monospace" font-size="12" font-weight="700" fill="#1A3A52" text-anchor="middle">942,834 pairs</text>',
        '</g>',
        '<g>',
        '  <line x1="860" y1="225" x2="900" y2="225" stroke="#B8831E" stroke-width="2" stroke-dasharray="4 4"/>',
        '  <text x="880" y="218" font-family="Verdana, Geneva, sans-serif" font-size="10" font-weight="700" fill="#684811" text-anchor="middle">eps_K</text>',
        '  <text x="880" y="245" font-family="Courier New, monospace" font-size="11" font-weight="800" fill="#1A3A52" text-anchor="middle">0.0664</text>',
        '</g>',
        # 12 witness-pair beads
        '<g>',
        '  <circle cx="660" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="700" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="740" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="780" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="820" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="860" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="900" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="940" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="980" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="1020" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="1060" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '  <circle cx="1100" cy="360" r="5.5" fill="#F4C430" stroke="#C99517" stroke-width="1"/>',
        '</g>',
        '<text x="880" y="390" font-family="Verdana, Geneva, sans-serif" font-size="11" font-weight="700" fill="#6A7680" text-anchor="middle">witness pairs at eps_K</text>',
        # Review-gated badge
        '<g transform="translate(720,430)" filter="url(#cardShadow)">',
        '  <rect x="0" y="0" width="320" height="46" rx="23" fill="#FFFFFF" stroke="#B8831E" stroke-width="2.2"/>',
        '  <text x="160" y="29" font-family="Verdana, Geneva, sans-serif" font-size="13" font-weight="800" letter-spacing="2" fill="#684811" text-anchor="middle">REVIEW-GATED  C1 UNPROMOTED</text>',
        '</g>',
    ])


def viz_generality_matrix() -> str:
    """Six-lane mini-matrix for the generality umbrella."""
    lanes = [
        (660, 140, 230, 130, "Navier-Stokes C1",  "942,834 WITNESS PAIRS",  "#FFF4D6", "#B8831E", "#684811", True),
        (910, 140, 230, 130, "Riemann",           "3 LANES, 3 BOUNDED NULLS", "#FFFFFF", "#B8831E", "#684811", False),
        (660, 290, 230, 130, "Yang-Mills",        "4 PROBES, 1 BOUNDED NULL", "#FFFFFF", "#B8831E", "#684811", False),
        (910, 290, 230, 130, "P-vs-NP",           "SAFETY GREEN, COST HOLD",  "#FFFFFF", "#B53A2C", "#7A1F14", False),
        (660, 440, 230, 95,  "ARC-AGI",           "PHASE 3E DESIGN HOLD",     "#EDF2F7", "#64748B", "#37465A", False),
        (910, 440, 230, 95,  "Three-Body 15C",    "COMPUTE-BLOCKED",          "#EDF2F7", "#64748B", "#37465A", False),
    ]
    parts = []
    for (x, y, w, h, label, stat, sfill, sstroke, stext, prominent) in lanes:
        sw = 3 if prominent else 1.5
        parts.append(f'<g filter="url(#cardShadow)"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#FFFFFF" stroke="{sstroke}" stroke-width="{sw}"/></g>')
        label_size = 18 if prominent else 16
        parts.append(f'<text x="{x + 14}" y="{y + 30}" font-family="Georgia, serif" font-size="{label_size}" font-weight="700" fill="#1A3A52">{label}</text>')
        bx = x + 14
        by = y + (50 if h > 100 else 46)
        bw = w - 28
        parts.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="22" rx="11" fill="{sfill}" stroke="{sstroke}" stroke-width="1"/>')
        parts.append(f'<text x="{bx + bw/2}" y="{by + 15}" font-family="Verdana, Geneva, sans-serif" font-size="10" font-weight="800" fill="{stext}" text-anchor="middle">{stat}</text>')
        if h > 100:
            note = {
                "Navier-Stokes C1": ["Reading-2 witness G=200+300", "deep dive at /navierstokes"],
                "Riemann":          ["no structural-zero edge", "external review filed"],
                "Yang-Mills":       ["YM-P2-NEG-A across slate", "no mass-gap claim"],
                "P-vs-NP":          ["v0-v5, 0 false accepts", "2 required reruns"],
            }.get(label, ["", ""])
            parts.append(f'<text x="{x + 14}" y="{y + 102}" font-family="Verdana, Geneva, sans-serif" font-size="11" fill="#40505C">{note[0]}</text>')
            parts.append(f'<text x="{x + 14}" y="{y + 118}" font-family="Courier New, monospace" font-size="10" fill="#1A3A52">{note[1]}</text>')
    parts.append('<text x="900" y="565" font-family="Verdana, Geneva, sans-serif" font-size="12" fill="#6A7680" text-anchor="middle">six transfer-test lanes</text>')
    return "\n  ".join(parts)


CARDS = {
    "index": {
        "filename": "home.png",
        "eyebrow": "SUNDOG · RESEARCH LAB",
        "headline": ["Alignment", "without sight."],
        "lede": [
            "Halo geometry from the math —",
            "and traceability discipline carried into",
            "control, agents, and workbenches.",
        ],
        "url": "sundog.cc",
        "viz": viz_halo_atlas,
    },
    "about": {
        "filename": "about.png",
        "eyebrow": "SUNDOG · ABOUT",
        "headline": ["A traceability harness", "for indirect inference."],
        "lede": [
            "Hidden-state control, route fidelity,",
            "and explicit failure boundaries.",
        ],
        "url": "sundog.cc/about",
        "viz": viz_about_ladder,
    },
    "alignment": {
        "filename": "alignment.png",
        "eyebrow": "SUNDOG · ALIGNMENT",
        "headline": ["Route fidelity under", "partial observation."],
        "lede": [
            "Bayes turns evidence into belief.",
            "Sundog turns response into control.",
        ],
        "url": "sundog.cc/alignment",
        "viz": viz_alignment_split,
    },
    "balance": {
        "filename": "balance.png",
        "eyebrow": "SUNDOG · BALANCE WORKBENCH",
        "headline": ["Cart-pole, with", "the body angle denied."],
        "lede": [
            "Shadow-derived balance under",
            "partial observation. Operating",
            "envelope mapped, boundary named.",
        ],
        "url": "sundog.cc/balance",
        "viz": viz_balance_cartpole,
    },
    "threebody": {
        "filename": "threebody.png",
        "eyebrow": "SUNDOG · THREE-BODY",
        "headline": ["Local signals survive", "the near-escape pocket."],
        "lede": [
            "Indirect signatures in chaotic",
            "dynamics — not the whole regime,",
            "but a named operating envelope.",
        ],
        "url": "sundog.cc/threebody",
        "viz": viz_threebody_orbits,
    },
    "mines": {
        "filename": "mines.png",
        "eyebrow": "SUNDOG · PRESSURE MINES",
        "headline": ["Pressure maps", "become decisions."],
        "lede": [
            "A noisy pressure field buys safe",
            "progress inside a narrow mapped",
            "pocket — and fails cleanly past it.",
        ],
        "url": "sundog.cc/mines",
        "viz": viz_mines_grid,
    },
    "sundog": {
        "filename": "atlas.png",
        "eyebrow": "SUNDOG · ATLAS",
        "headline": ["The geometry of", "a sun dog."],
        "lede": [
            "Every named arc in a halo",
            "display, calibrated to photographs,",
            "with the math visible at every step.",
        ],
        "url": "sundog.cc/sundog",
        "viz": viz_halo_atlas,
    },
    "navierstokes": {
        "filename": "navierstokes.png",
        "eyebrow": "SUNDOG · NAVIER-STOKES C1",
        "headline": ["A witness at", "two Grashof", "regimes."],
        "lede": [
            "State-insufficient AND",
            "control-sufficient. Not a",
            "Clay-problem claim.",
        ],
        "url": "sundog.cc/navierstokes",
        "viz": viz_navierstokes_witness,
    },
    "generality": {
        "filename": "generality.png",
        "eyebrow": "SUNDOG · HIGH-STAKES GENERALITY",
        "headline": ["Transfer tests,", "walls, and", "review gates."],
        "lede": [
            "Six lanes asking whether",
            "signatures survive substrate",
            "changes. Receipts, not claims.",
        ],
        "url": "sundog.cc/generality",
        "viz": viz_generality_matrix,
    },
}


def render(name: str) -> pathlib.Path:
    cfg = CARDS[name]
    svg = build_svg(
        eyebrow=cfg["eyebrow"],
        headline=cfg["headline"],
        lede=cfg["lede"],
        url=cfg["url"],
        viz=cfg["viz"](),
    )
    svg_path = OUT_DIR / cfg["filename"].replace(".png", ".svg")
    png_path = OUT_DIR / cfg["filename"]
    svg_path.write_text(svg)
    # CairoSVG rasterize (ImageMagick was compiled --without-rsvg, so its
    # internal MSVG parser fails on stop-opacity / fill-opacity, producing
    # solid-colour backgrounds. CairoSVG is a real SVG renderer.)
    import cairosvg
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(png_path),
        output_width=1200,
        output_height=630,
    )
    return png_path


def main(argv: list[str]) -> int:
    names = argv[1:] if len(argv) > 1 else list(CARDS.keys())
    for n in names:
        if n not in CARDS:
            print(f"skip: unknown card '{n}'", file=sys.stderr)
            continue
        path = render(n)
        size = path.stat().st_size
        print(f"  rendered {n:<10} -> {path.name} ({size//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
