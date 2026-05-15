"""Phase 15 follow-up #1' -- per-ring HaloSim ray-filter frame generator.

Pyramidal odd-radius rings will not separate in a blended all-sky render
(verified: 1M = 0 rings, 6M = 1 ring). The definitive method is to render
each ring ALONE by restricting HaloSim's ray filter to that ring's
entry/exit crystal faces (Tape AH-CH10 p6 wedge table). This emits one
byte-safe `.sim` per wedge, reusing the proven HS-0 Startup.sim-swap loop
-- no GUI face-typing.

`.sim` ray-filter fields decoded by correlating the Pyramidal /
supralateral / circumscribed library sims against their Ray-Filters
panel (the `Type9` line is the structural anchor T):

    T-3   "<plotStyle> <rays> <levels> <maxFaces>"   <- <rays> pinned
    T+8   "<entrance faces>"   (quoted string)        <- set per wedge
    T+9   "<internal>"         (quoted; left as-is)
    T+10  "<exit faces>"       (quoted string)        <- set per wedge

The filter is already ON in the pyramidal library sim (T+4 = "True"),
so isolation = restrict entrance to the wedge's entry face and exit to
its exit face. Everything else (sun alt, crystals, view, split-sky) is
inherited verbatim. CRLF / latin-1; bytes preserved outside the three
edited lines (integrity-checked, aborts otherwise).

Usage:
    python scripts/halosim_filter_frames.py --src "C:/Users/hughe/Pyramidal 20-35d halos.sim" \
        --out <dir> --entrance 3 --exit 26 --rays 4000000 --tag w9
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

TOKEN = re.compile(r"\S+")
LEAD_WS = re.compile(r"^(\s*)")


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
    raise SystemExit("ERROR: no 'Type9' anchor -- not a HaloSim .sim?")


def set_rays(line: str, rays: int) -> str:
    spans = [m.span() for m in TOKEN.finditer(line)]
    if len(spans) < 3:
        raise SystemExit(f"ERROR: T-3 not '<style> <rays> <levels> ...': {line!r}")
    s, e = spans[1]
    return line[:s] + str(rays) + line[e:]


def set_quoted(line: str, value: str) -> str:
    """Filter lines are a single quoted string (possibly indented, and
    the value may contain a space e.g. "not tested" -- so a \\S+ token
    split is wrong here). Preserve leading whitespace, rewrite the quoted
    field."""
    ws = LEAD_WS.match(line).group(1)
    if '"' not in line:
        raise SystemExit(f"ERROR: expected a quoted filter field, got {line!r}")
    return f'{ws}"{value}"'


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 15 ray-filter frame generator")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--entrance", required=True, help="entrance face(s), e.g. 3")
    ap.add_argument("--exit", dest="exit_", required=True, help="exit face(s), e.g. 26")
    ap.add_argument("--rays", type=int, required=True)
    ap.add_argument("--tag", required=True, help="filename tag, e.g. w9")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_file():
        raise SystemExit(f"ERROR: source not found: {src}")
    lines = split_sim(src.read_bytes())
    T = find_anchor(lines)
    rays_i, ent_i, exit_i = T - 3, T + 8, T + 10
    if rays_i < 0 or exit_i >= len(lines):
        raise SystemExit("ERROR: anchor too close to file edge -- unexpected layout")
    if '"' not in lines[ent_i] or '"' not in lines[exit_i]:
        raise SystemExit(
            f"ERROR: T+8/T+10 not quoted filter fields "
            f"(L{ent_i+1}={lines[ent_i]!r} L{exit_i+1}={lines[exit_i]!r})"
        )

    out = Path(a.out)
    if out.resolve() == Path("C:/Users/hughe").resolve():
        raise SystemExit("ERROR: refusing to write into the HaloSim home directory")
    out.mkdir(parents=True, exist_ok=True)

    L = list(lines)
    L[rays_i] = set_rays(L[rays_i], a.rays)
    L[ent_i] = set_quoted(L[ent_i], a.entrance)
    L[exit_i] = set_quoted(L[exit_i], a.exit_)
    data = join_sim(L)

    chk = split_sim(data)
    diffs = [
        i for i in range(max(len(chk), len(lines)))
        if (chk[i] if i < len(chk) else None) != (lines[i] if i < len(lines) else None)
    ]
    if sorted(diffs) != sorted({rays_i, ent_i, exit_i}):
        raise SystemExit(
            f"ERROR: unexpected line changes {[d+1 for d in diffs]} "
            f"(only T-3,T+8,T+10 allowed)"
        )

    name = f"pyr_{a.tag}_e{a.entrance}_x{a.exit_}.sim"
    (out / name).write_bytes(data)
    print(f"{src.name}: entrance={a.entrance} exit={a.exit_} "
          f"rays->{a.rays}  ->  {out.name}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
