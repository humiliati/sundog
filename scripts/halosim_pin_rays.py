#!/usr/bin/env python
"""Phase 14E helper: byte-safe ray-count pin for a HaloSim .sim.

Unlike `halosim_gen_frames.py` (which sweeps sun altitude and, for
Camera-View templates, syncs camera-aim to the sun), this edits **only**
the ray count on the T-3 line and leaves every other byte — sun altitude,
camera aim/azimuth, crystal blocks, view, filters — exactly as the library
authored it. That is the faithful configuration for a 14E reproduction
receipt: "the shipped recipe, as-is, just rendered at a reliable ray count."

It is needed because several bundled sims are Camera-View with the camera
deliberately decoupled from the sun (e.g. `Subhorizon arcs.sim`: sun 18.5,
camera -11; `Parhelic circ and more.sim`: sun 22, camera-aim 25) — there,
gen_frames' camera-sync would corrupt the framing.

.sim format (HaloSim 3.6.1; `Type9` constant line is the structural
anchor T, per the gen_frames decode):

    T-4   sun altitude (deg)                         <-- reported, untouched
    T-3   "<plotStyle> <rays> <levels> <maxFaces>"   <-- ONLY <rays> changed
    T+0   "Type9"                                     <-- anchor

CRLF / latin-1, trailing blank line; bytes preserved outside the one token.

Usage:
    python scripts/halosim_pin_rays.py --src "C:/Users/hughe/Lowitz arcs.sim" \
        --out docs/calibration/halosim_outputs/phase14e_frames/lowitz.sim \
        --rays 1000000
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

TOKEN = re.compile(r"\S+")


def split_sim(raw: bytes) -> list[str]:
    return raw.decode("latin-1").split("\r\n")


def join_sim(lines: list[str]) -> bytes:
    return ("\r\n".join(lines)).encode("latin-1")


def find_anchor(lines: list[str]) -> int:
    for i, l in enumerate(lines):
        if l.strip() == "Type9":
            return i
    for i, l in enumerate(lines):
        if l.strip().startswith("Type"):
            return i
    raise SystemExit("ERROR: no 'Type9' anchor line found — not a HaloSim .sim?")


def replace_token(line: str, index: int, value: str) -> str:
    spans = [m.span() for m in TOKEN.finditer(line)]
    if index >= len(spans):
        raise SystemExit(f"ERROR: token #{index} not present in line {line!r}")
    s, e = spans[index]
    return line[:s] + value + line[e:]


def get_token(line: str, index: int) -> str:
    toks = TOKEN.findall(line)
    return toks[index] if index < len(toks) else ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 14E rays-only .sim pin")
    ap.add_argument("--src", required=True, help="source library .sim")
    ap.add_argument("--out", required=True, help="output .sim path")
    ap.add_argument("--rays", type=int, required=True, help="ray count to pin")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_file():
        raise SystemExit(f"ERROR: source not found: {src}")
    raw = src.read_bytes()
    lines = split_sim(raw)
    T = find_anchor(lines)
    sun_i, rays_i = T - 4, T - 3
    if sun_i < 0:
        raise SystemExit("ERROR: anchor too close to file edge — unexpected .sim layout")

    rl = TOKEN.findall(lines[rays_i])
    if len(rl) < 3 or not rl[1].isdigit() or not rl[2].isdigit():
        raise SystemExit(
            f"ERROR: line T-3 ({rays_i+1}) {lines[rays_i]!r} is not '<style> <rays> <levels> ...'"
        )
    sun_alt = get_token(lines[sun_i], 0)
    cam_alt = get_token(lines[T + 14], 0) if T + 14 < len(lines) else ""
    view = "camera-view" if cam_alt not in ("", "0") else "fisheye/sky-fixed"
    tpl_rays = rl[1]

    out = Path(a.out)
    if out.resolve().parent == Path("C:/Users/hughe").resolve():
        raise SystemExit("ERROR: refusing to write into the HaloSim home directory")
    out.parent.mkdir(parents=True, exist_ok=True)

    L = list(lines)
    L[rays_i] = replace_token(L[rays_i], 1, str(a.rays))
    data = join_sim(L)

    chk = split_sim(data)
    diffs = [
        i
        for i in range(max(len(chk), len(lines)))
        if (chk[i] if i < len(chk) else None) != (lines[i] if i < len(lines) else None)
    ]
    if diffs != [rays_i]:
        raise SystemExit(f"ERROR: unexpected line changes {[d + 1 for d in diffs]} (only T-3 allowed)")
    out.write_bytes(data)

    print(
        f"{src.name}: sun_alt={sun_alt} view={view} "
        f"rays {tpl_rays} -> {a.rays}  ->  {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
