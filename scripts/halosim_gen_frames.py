#!/usr/bin/env python
"""HS-1: HaloSim .sim frame generator for the cinematic sun-altitude sweep.

Emits one `.sim` per frame from a template, varying ONLY the sun altitude
(and, for Camera-View templates, the camera-aim altitude that tracks the
sun) and pinning the ray count / levels. Everything else — view, crystal
blocks, colours, filters, FOV, sun diameter, description — is inherited
verbatim from the template.

.sim format (HaloSim 3.6.1, reverse-engineered from 5 known sims; the
`Type9` line is a constant section marker used as the structural anchor T):

    T-4   sun altitude (degrees)            <-- varied per frame
    T-3   "<plotStyle> <rays> <levels> <maxFaces>"  <-- rays/levels pinned
    T-1   "<a> <viewMode> <b> <cameraAlt>"  <-- cameraAlt synced if Camera View
    T+0   "Type9"                           <-- anchor (constant)
    T+14  "<cameraAlt> <cameraAz>"          <-- cameraAlt synced if Camera View

Fisheye templates keep the camera fields at 0 (sun position is intrinsic
to the projection), so only T-4 / T-3 change — this also sidesteps the
HS-3 Camera-View auto-zoom risk and is the recommended cinematic path.

Files are CRLF / latin-1 with a trailing blank line; this tool preserves
bytes exactly outside the targeted numeric tokens.

Usage:
    python scripts/halosim_gen_frames.py --template "C:/Users/hughe/46halo.sim" \
        --out docs/calibration/halosim_outputs/hs_frames \
        --start 0 --stop 60 --step 2 --rays 300000 [--levels 256] [--dry-run]
"""
from __future__ import annotations
import argparse, re, sys
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
    """Replace the index-th whitespace-delimited token, preserving every
    run of surrounding whitespace exactly."""
    spans = [m.span() for m in TOKEN.finditer(line)]
    if index >= len(spans):
        raise SystemExit(
            f"ERROR: token #{index} not present in line {line!r}"
        )
    s, e = spans[index]
    return line[:s] + value + line[e:]


def get_token(line: str, index: int) -> str:
    toks = TOKEN.findall(line)
    return toks[index] if index < len(toks) else ""


def main() -> int:
    ap = argparse.ArgumentParser(description="HS-1 HaloSim frame .sim generator")
    ap.add_argument("--template", required=True, help="path to a template .sim")
    ap.add_argument("--out", required=True, help="output directory for frame sims")
    ap.add_argument("--start", type=int, default=0, help="first sun altitude (deg)")
    ap.add_argument("--stop", type=int, default=60, help="last sun altitude (deg)")
    ap.add_argument("--step", type=int, default=2, help="altitude step (deg)")
    ap.add_argument("--rays", type=int, default=None, help="pin ray count (else keep template's)")
    ap.add_argument("--levels", type=int, default=None, help="pin levels (else keep template's)")
    ap.add_argument("--prefix", default="hs_frame", help="output filename prefix")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    a = ap.parse_args()

    tpl = Path(a.template)
    if not tpl.is_file():
        raise SystemExit(f"ERROR: template not found: {tpl}")
    raw = tpl.read_bytes()
    lines = split_sim(raw)
    T = find_anchor(lines)
    sun_i, rays_i, cam1_i, cam14_i = T - 4, T - 3, T - 1, T + 14
    if sun_i < 0 or cam14_i >= len(lines):
        raise SystemExit("ERROR: anchor too close to file edge — unexpected .sim layout")

    # Validate the structural assumptions against the template.
    try:
        float(get_token(lines[sun_i], 0))
    except ValueError:
        raise SystemExit(
            f"ERROR: line T-4 ({sun_i+1}) {lines[sun_i]!r} is not a sun-altitude number"
        )
    rl = TOKEN.findall(lines[rays_i])
    if len(rl) < 3 or not rl[1].isdigit() or not rl[2].isdigit():
        raise SystemExit(
            f"ERROR: line T-3 ({rays_i+1}) {lines[rays_i]!r} is not '<style> <rays> <levels> ...'"
        )

    tpl_cam_alt = get_token(lines[cam14_i], 0)
    camera_view = tpl_cam_alt not in ("", "0")
    tpl_rays, tpl_levels = rl[1], rl[2]

    out = Path(a.out)
    # Safety: never write into the HaloSim asset library / user home.
    if out.resolve() == Path("C:/Users/hughe").resolve():
        raise SystemExit("ERROR: refusing to write frames into the HaloSim home directory")
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    alts = list(range(a.start, a.stop + 1, a.step))
    print(f"template : {tpl}")
    print(f"anchor   : Type9 at line {T+1}  (sun=L{sun_i+1}, rays/levels=L{rays_i+1})")
    print(f"view     : {'Camera-View (camera alt synced to sun)' if camera_view else 'Fisheye/sky-fixed (only sun altitude changes)'}")
    print(f"rays     : {tpl_rays} -> {a.rays if a.rays is not None else tpl_rays}")
    print(f"levels   : {tpl_levels} -> {a.levels if a.levels is not None else tpl_levels}")
    print(f"frames   : {len(alts)}  ({a.start}..{a.stop} step {a.step})  -> {out}")
    print(f"mode     : {'DRY-RUN (nothing written)' if a.dry_run else 'WRITE'}")

    written = []
    for alt in alts:
        L = list(lines)
        L[sun_i] = replace_token(L[sun_i], 0, str(alt))
        if a.rays is not None:
            L[rays_i] = replace_token(L[rays_i], 1, str(a.rays))
        if a.levels is not None:
            L[rays_i] = replace_token(L[rays_i], 2, str(a.levels))
        if camera_view:
            L[cam1_i] = replace_token(L[cam1_i], 3, str(alt))
            L[cam14_i] = replace_token(L[cam14_i], 0, str(alt))
        name = f"{a.prefix}_{alt:03d}deg.sim"
        if not a.dry_run:
            data = join_sim(L)
            (out / name).write_bytes(data)
            # Integrity: only the intended lines may differ from the template.
            chk = split_sim(data)
            diffs = [i for i in range(max(len(chk), len(lines)))
                     if (chk[i] if i < len(chk) else None) != (lines[i] if i < len(lines) else None)]
            allowed = {sun_i, rays_i}
            if camera_view:
                allowed |= {cam1_i, cam14_i}
            stray = [d + 1 for d in diffs if d not in allowed]
            if stray:
                raise SystemExit(f"ERROR: {name} changed unexpected lines {stray}")
        written.append(name)

    print(f"\n{'would write' if a.dry_run else 'wrote'} {len(written)} files: "
          f"{written[0]} .. {written[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
