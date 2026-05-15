#!/usr/bin/env python
"""HS-2: standalone local HaloSim batch render runner (cinematic sweep).

Runs entirely on the local machine (NOT through the agent/MCP), so a
1000-frame / multi-hour sweep is not bound by any agent session limit.
The operator brings HaloSim to the foreground, then launches this once.

Per frame, the proven HS-0/HS-1 primitive:
  1. copy the frame's .sim over C:\\Users\\hughe\\Startup.sim
  2. click Reset  (reloads Startup.sim)         -- fixed coords
  3. click Start  (renders)                      -- fixed coords
  4. POLL autosave.bmp mtime until it advances and the size is stable
     (robust at ANY ray count: 2s @ 200k .. minutes @ 10M; no fixed wait)
  5. verify non-blank, harvest -> _staging/<run>/<frame>.png (+ .bmp)

Fixed coords are intentional: the operator keeps the HaloSim window
placed consistently between sessions (project assumption 2026-05-14).
A stall guard aborts if N consecutive frames produce no fresh BMP —
that is the cheap, vision-free way to catch a moved/closed window.

PyMacroRecord was evaluated as the input driver and rejected: it only
replays fixed delays, which desyncs on variable render time. This
controller polls instead. PyMacroRecord remains a no-dependency fallback
only for a strictly uniform ray-count sweep.

Usage:
  # 1. read true-resolution button coordinates:
  python scripts/halosim_run_sweep.py calibrate
  # 2. run the sweep (HaloSim must be foreground):
  python scripts/halosim_run_sweep.py run \\
      --frames-dir docs/calibration/halosim_outputs/hs_frames \\
      --reset 611,372 --start 767,367 [--limit 3] [--resume] [--no-bmp]
"""
from __future__ import annotations
import argparse, json, shutil, sys, time
from datetime import datetime
from pathlib import Path

AUTOSAVE = Path(r"C:\Users\hughe\autosave.bmp")
STARTUP = Path(r"C:\Users\hughe\Startup.sim")
HOME = Path(r"C:\Users\hughe")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs/calibration/halosim_outputs/_staging"
CONFIG = Path(__file__).resolve().parent / "halosim_sweep_config.json"


def grab():
    from PIL import ImageGrab
    return ImageGrab.grab()


def cmd_calibrate(_a):
    img = grab()
    out = Path(__file__).resolve().parent / "_halosim_calib.png"
    img.save(out, "PNG")
    print(f"true-resolution screenshot: {img.size}  -> {out}")
    print("Read the Reset and Start button pixel coords from that image,")
    print("then either pass --reset X,Y --start X,Y to `run`, or save them:")
    print(f'  {{"reset_xy":[X,Y],"start_xy":[X,Y]}}  ->  {CONFIG}')
    return 0


def is_blank(path: Path) -> bool:
    from PIL import Image
    ex = Image.open(path).convert("RGB").getextrema()
    return all(lo == hi for lo, hi in ex)


def stable_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="HS-2 standalone HaloSim sweep runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate", help="save a true-res screenshot to read button coords")
    r = sub.add_parser("run", help="drive the sweep")
    r.add_argument("--frames-dir", required=True, help="dir of hs_frame_*deg.sim")
    r.add_argument("--config", default=str(CONFIG))
    r.add_argument("--reset", help="Reset button 'X,Y' (overrides config)")
    r.add_argument("--start", help="Start button 'X,Y' (overrides config)")
    r.add_argument("--out", default=str(DEFAULT_OUT), help="staging root (gitignored)")
    r.add_argument("--timeout", type=float, default=900.0, help="per-frame render cap (s)")
    r.add_argument("--settle", type=float, default=1.5, help="pause between Reset and Start (s)")
    r.add_argument("--poll", type=float, default=0.75, help="autosave poll interval (s)")
    r.add_argument("--limit", type=int, default=0, help="render at most N frames (0=all)")
    r.add_argument("--start-index", type=int, default=0, help="skip the first K frames")
    r.add_argument("--resume", action="store_true", help="skip frames already harvested non-blank")
    r.add_argument("--no-bmp", action="store_true", help="keep only PNG in staging (not the 4MB BMP)")
    r.add_argument("--stall-abort", type=int, default=3, help="abort after N consecutive no-render frames")
    a = ap.parse_args()

    if a.cmd == "calibrate":
        return cmd_calibrate(a)

    import pyautogui
    pyautogui.FAILSAFE = True   # slam mouse to a screen corner to abort
    pyautogui.PAUSE = 0.3

    # Resolve button coordinates (CLI > config).
    cfg = {}
    if Path(a.config).is_file():
        cfg = json.loads(Path(a.config).read_text())
    def xy(s): x, y = s.split(","); return (int(x), int(y))
    reset_xy = xy(a.reset) if a.reset else tuple(cfg.get("reset_xy", ()))
    start_xy = xy(a.start) if a.start else tuple(cfg.get("start_xy", ()))
    if not (len(reset_xy) == 2 and len(start_xy) == 2):
        sys.exit("ERROR: need Reset/Start coords. Run `calibrate`, then pass "
                 "--reset X,Y --start X,Y or write them to the config json.")

    frames = sorted(Path(a.frames_dir).glob("hs_frame_*deg.sim"))
    if not frames:
        sys.exit(f"ERROR: no hs_frame_*deg.sim in {a.frames_dir}")
    frames = frames[a.start_index:]
    if a.limit:
        frames = frames[: a.limit]

    runtag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) / runtag
    if out.resolve() == HOME.resolve():
        sys.exit("ERROR: refusing to stage into the HaloSim home directory")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "_sweep_log.tsv"
    log.write_text("frame\talt\tdetect_s\tbytes\tstatus\n")

    backup = HOME / "Startup.sim.sweepbak"
    if STARTUP.exists() and not backup.exists():
        shutil.copyfile(STARTUP, backup)
    print(f"sweep {runtag}: {len(frames)} frames -> {out}")
    print(f"Reset={reset_xy} Start={start_xy}  (HaloSim MUST be foreground)")
    print(f"Startup.sim backed up -> {backup}")

    done = fail = 0
    consecutive_stall = 0
    try:
        for i, fr in enumerate(frames, 1):
            stem = fr.stem  # hs_frame_030deg
            alt = stem.split("_")[-1].replace("deg", "")
            png = out / f"{stem}.png"
            if a.resume and png.exists() and not is_blank(png):
                print(f"[{i}/{len(frames)}] {stem}  SKIP (already harvested)")
                done += 1
                continue

            ok = False
            for attempt in (1, 2):  # one retry, per HS-2 spec
                baseline = stable_mtime(AUTOSAVE)
                shutil.copyfile(fr, STARTUP)
                t0 = time.time()
                pyautogui.click(*reset_xy)
                time.sleep(a.settle)
                pyautogui.click(*start_xy)

                # Poll: mtime advanced AND size stable across two reads.
                last_sz = -1
                detect = None
                while time.time() - t0 < a.timeout:
                    time.sleep(a.poll)
                    if AUTOSAVE.exists() and AUTOSAVE.stat().st_mtime > baseline:
                        sz = AUTOSAVE.stat().st_size
                        if sz == last_sz and sz > 0:
                            detect = round(time.time() - t0, 1)
                            break
                        last_sz = sz
                if detect is None:
                    print(f"[{i}/{len(frames)}] {stem}  no render (attempt {attempt}, "
                          f"{a.timeout:.0f}s timeout)")
                    continue
                if is_blank(AUTOSAVE):
                    print(f"[{i}/{len(frames)}] {stem}  BLANK (attempt {attempt})")
                    continue
                # Harvest.
                from PIL import Image
                Image.open(AUTOSAVE).convert("RGB").save(png, "PNG", optimize=True)
                nbytes = png.stat().st_size
                if not a.no_bmp:
                    shutil.copyfile(AUTOSAVE, out / f"{stem}.bmp")
                with log.open("a") as fh:
                    fh.write(f"{stem}\t{alt}\t{detect}\t{nbytes}\tOK\n")
                print(f"[{i}/{len(frames)}] {stem}  h={alt}  rendered {detect}s  "
                      f"-> {png.name} ({nbytes//1024} KB)")
                ok = True
                break

            if ok:
                done += 1
                consecutive_stall = 0
            else:
                fail += 1
                consecutive_stall += 1
                with log.open("a") as fh:
                    fh.write(f"{stem}\t{alt}\t\t\tFAIL\n")
                if consecutive_stall >= a.stall_abort:
                    sys.exit(f"ABORT: {consecutive_stall} consecutive frames with no "
                             f"render. HaloSim likely not foreground / window moved / "
                             f"coords wrong. Re-run `calibrate`.")
    finally:
        if backup.exists():
            shutil.copyfile(backup, STARTUP)
            backup.unlink()
            print(f"restored Startup.sim from backup")

    print(f"\nDONE  ok={done} fail={fail}  log={log}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
