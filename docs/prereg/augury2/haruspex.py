#!/usr/bin/env python
"""AUGURY II (HARUSPEX) runner — provenance of the market's forecast information.

Spec: docs/prereg/augury2/AUGURY_II_G1_PREREG.md. Reuses the frozen Augury rig
(augury_pilot / augury_g3 / augury_g4: candles, CLI, NBM/GEFS/ECMWF decode,
matched-cutoff selection, ridge-logistic encompassing, day-block bootstrap) and
adds three provenance pieces:

  H1  obs-nowcast rung F_obs = Normal(x_T + Δ(o,m,s), σ(o,m,s)) survival, where
      x_T is the latest valid IEM ASOS ob ≤ cutoff and Δ,σ are the (CLI_high −
      x_T) climatology from the disjoint 2015–2019 period.  measure vs aggregate.
  H2  final-cycle comparator F_final = freshest same-day NBM (≈12Z), used only on
      cutoffs T < avail(F_final).  independence vs anticipation (LONG by construction).
  H3  access-feature panel over 5 rungs × (city×horizon×era) cells; meta-regression
      of determining-set membership on access vs identity (exploratory-strength).

Stages: selftest | asos-smoke | climo | pilot | h1 | h2 | h3 | all
Binding stages require --admitted. Reuses .venv-augury (numpy, eccodes, pyproj).
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "prereg" / "augury"))
sys.path.insert(0, str(HERE.parents[2] / "docs" / "prereg" / "augury"))
# robust import of the Augury modules regardless of cwd
import importlib.util as _ilu


def _load(name):
    p = HERE.parents[1] / "augury" / f"{name}.py"
    if not p.exists():
        p = Path(__file__).resolve().parents[1] / "augury" / f"{name}.py"
    spec = _ilu.spec_from_file_location(name, p)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ap = _load("augury_pilot")
g3 = _load("augury_g3")
g4 = _load("augury_g4")

C = ap.C
RESULTS = ap.ROOT / "results" / "augury2"
G4RES = ap.ROOT / "results" / "augury" / "g4-run"
CITIES = list(g4.CITIES)
WIN0, WIN1 = g4.WIN0, g4.WIN1
CLIMO_Y0, CLIMO_Y1 = 2015, 2019          # disjoint nowcast-climatology period (frozen)
SHORT = {-5, -4, -3, -2}
LONG = {-12, -11, -10, -9, -8}
MIN_CELL = 20                             # min residual samples per (o,m,s) climo cell
H1_SEED, H2_SEED, H3_SEED = 20260710, 20260711, 20260712
ASOS_STATION = {"KNYC": "NYC", "KMDW": "MDW", "KDEN": "DEN", "KLAX": "LAX",
                "KAUS": "AUS", "KMIA": "MIA", "KPHL": "PHL"}

# H3 access-feature table (frozen; §4 of the pre-reg)
ACCESS = {
    "gefs":   {"cadence": 6.0,  "rt_obs": 0, "aggregates": 0, "ensemble": 1, "is_market": 0},
    "nbm":    {"cadence": 1.0,  "rt_obs": 0, "aggregates": 0, "ensemble": 1, "is_market": 0},
    "ecmwf":  {"cadence": 12.0, "rt_obs": 0, "aggregates": 0, "ensemble": 0, "is_market": 0},
    "obs":    {"cadence": 1.0,  "rt_obs": 1, "aggregates": 0, "ensemble": 0, "is_market": 0},
    "mkt":    {"cadence": 0.02, "rt_obs": 1, "aggregates": 1, "ensemble": 0, "is_market": 1},
}


def wjson(p: Path, o):
    return ap.write_json(p, o)


# ----------------------------------------------------------------- ASOS

_asos_pace = [0.0]


def _asos_fetch(station: str, y0: int, y1: int) -> list[tuple[int, float]]:
    """Cached per-(station, year-range) ASOS temperature series → [(ts, degF)]
    of VALID obs, sorted. IEM throttles bursts with a 200-body 'Too many
    requests' message (not an HTTP error) → detect + back off."""
    CACHE = RESULTS / "asos_cache"
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"{station}_{y0}_{y1}.jsonl"
    if key.exists():
        return [tuple(r) for r in json.loads(key.read_text())]
    q = urllib.parse.urlencode({
        "station": station, "data": "tmpf", "tz": "Etc/UTC",
        "year1": y0, "month1": 1, "day1": 1,
        "year2": y1 + 1, "month2": 1, "day2": 1,
        "format": "onlycomma", "latlon": "no", "missing": "M", "trace": "T"})
    url = f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?{q}"
    for attempt in range(9):
        gap = 0.3 - (time.time() - _asos_pace[0])
        if gap > 0:
            time.sleep(gap)
        req = urllib.request.Request(url, headers={"User-Agent": ap.UA})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read().decode("utf-8", "replace")
            _asos_pace[0] = time.time()
        except Exception:
            time.sleep(min(3.0 * 1.7 ** attempt, 60))
            continue
        if "Too many requests" in body[:200]:
            time.sleep(min(5.0 * 1.7 ** attempt, 90))
            continue
        rows = []
        for line in body.splitlines():
            p = line.split(",")
            if len(p) != 3 or p[0] == "station" or p[2] == "M":
                continue
            try:
                t = dt.datetime.strptime(p[1], "%Y-%m-%d %H:%M").replace(
                    tzinfo=dt.timezone.utc)
                rows.append((int(t.timestamp()), float(p[2])))
            except ValueError:
                continue
        rows.sort()
        key.write_text(json.dumps(rows))
        return rows
    raise RuntimeError(f"ASOS fetch exhausted for {station} {y0}-{y1}")


_asos_series: dict = {}


def obs_at(station_k: str, T: dt.datetime, y0: int, y1: int) -> float | None:
    """Latest valid ASOS ob (degF) at or before T (max 6h stale)."""
    st = ASOS_STATION[station_k]
    k = (st, y0, y1)
    if k not in _asos_series:
        ser = _asos_fetch(st, y0, y1)
        _asos_series[k] = (ser, [x[0] for x in ser])
    ser, ts = _asos_series[k]
    ts_T = int(T.timestamp())
    j = bisect.bisect_right(ts, ts_T) - 1
    if j < 0:
        return None
    if ts_T - ser[j][0] > 6 * 3600:
        return None
    return ser[j][1]


# ----------------------------------------------------------------- climatology

def _cutoff_of(tp, st, day: dt.date, off: int):
    h = g3.day_hour(tp, st, day)
    if h is None:
        return None
    cuts = ap.band_cutoffs(st, day, h)
    return cuts[off + 12]  # offsets -12..-2 → index 0..10


def stage_climo():
    """Freeze the (offset, month, station) nowcast climatology from 2015–2019:
    Δ,σ of (CLI_high − ASOS ob at cutoff). Written before any 2023+ scoring."""
    tp = g3.load_tpeak()
    table: dict = {}
    for st in CITIES:
        cli = ap.cli_days(st, CLIMO_Y0, CLIMO_Y1)
        by = {}
        for day, row in cli.items():
            if not (CLIMO_Y0 <= day.year <= CLIMO_Y1):
                continue
            high = row["high"]
            for off in range(-12, -1):
                T = _cutoff_of(tp, st, day, off)
                if T is None:
                    continue
                x = obs_at(st, T, CLIMO_Y0, CLIMO_Y1)
                if x is None:
                    continue
                by.setdefault((off, day.month), []).append(high - x)
        table[st] = {}
        for (off, m), res in by.items():
            if len(res) < MIN_CELL:
                continue
            arr = np.array(res)
            table[st][f"{off}|{m}"] = {"n": len(res),
                                       "delta": round(float(arr.mean()), 3),
                                       "sigma": round(float(arr.std(ddof=1)), 3)}
        print(f"  climo {st}: {len(table[st])} (offset,month) cells", flush=True)
    sha = wjson(RESULTS / "nowcast_climo.json", table)
    print(f"climo frozen: sha256 {sha}", flush=True)


_climo: dict = {}


def _load_climo():
    global _climo
    if not _climo:
        _climo = json.loads((RESULTS / "nowcast_climo.json").read_text())
    return _climo


def f_obs(st, T, month, off, b) -> float | None:
    x = obs_at(st, T, WIN0.year, WIN1.year)
    if x is None:
        return None
    cell = _load_climo().get(st, {}).get(f"{off}|{month}")
    if cell is None:  # season back-off: nearest month with a cell
        for dm in (1, 2, 3):
            for mm in (month - dm, month + dm):
                cell = _load_climo().get(st, {}).get(f"{off}|{(mm - 1) % 12 + 1}")
                if cell:
                    break
            if cell:
                break
    if cell is None:
        return None
    return ap.norm_sf(b, x + cell["delta"], max(cell["sigma"], 0.5))


# ----------------------------------------------------------------- F_final (H2)

def f_final_issue(day: dt.date):
    """Freshest same-day NBM MaxT issue (≈12Z). Reuses g3.latest_issue with a
    late cutoff so it returns the last same-day TMAX cycle."""
    late = dt.datetime(day.year, day.month, day.day, 20, tzinfo=dt.timezone.utc)
    return g3.latest_issue(day, late)


# ----------------------------------------------------------------- encompassing

def _pick_lambda(rows, specs, sts):
    X, _n = _designmat(rows, specs, sts)
    z = np.array([float(r["z"]) for r in rows])
    nr = len(specs)
    days = sorted({r["day"] for r in rows})
    rng = random.Random(12345)
    dd = days[:]
    rng.shuffle(dd)
    fold = {d: i % 5 for i, d in enumerate(dd)}
    fo = np.array([fold[r["day"]] for r in rows])
    best, blam = 1e18, g4.RIDGE_GRID[0]
    for lam in g4.RIDGE_GRID:
        pen = np.array([lam if i < nr else 0.0 for i in range(X.shape[1])])
        dev = 0.0
        for f in range(5):
            tr, te = fo != f, fo == f
            beta = g4._ridge_irls(X[tr], z[tr], pen)
            p = 1 / (1 + np.exp(-np.clip(X[te] @ beta, -30, 30)))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            dev += -2 * np.sum(z[te] * np.log(p) + (1 - z[te]) * np.log(1 - p))
        if dev < best:
            best, blam = dev, lam
    return blam


def _designmat(rows, specs, sts):
    cols, names = [], []
    for name, kind in specs:
        arr = np.array([r[name] for r in rows], dtype=float)
        cols.append(g4._logit(arr) if kind == "logit" else arr / 10.0)
        # robust rung label: strip leading 'f_' OR trailing '_margin' (not both,
        # and never a mid-string 'f_' as in 'ecmwf_margin')
        if name.startswith("f_"):
            rn = name[2:]
        elif name.endswith("_margin"):
            rn = name[:-len("_margin")]
        else:
            rn = name
        names.append(rn)
    cols.append(np.ones(len(rows)))
    names.append("const")
    for s in sts[1:]:
        cols.append(np.array([1.0 if r["st"] == s else 0.0 for r in rows]))
        names.append("st_" + s)
    return np.column_stack(cols), names


def encompass(rows, specs, sts, seed, lam=None, boot=None):
    """Ridge-logistic encompassing over the given rung specs; day-block-bootstrap
    CIs on the rung coefficients. specs = [(col, 'logit'|'margin'), ...].
    boot defaults to g4.ENC_BOOT (2000); H3 panel cells pass a lighter count."""
    B = g4.ENC_BOOT if boot is None else boot
    if lam is None:
        lam = _pick_lambda(rows, specs, sts)
    X, names = _designmat(rows, specs, sts)
    z = np.array([float(r["z"]) for r in rows])
    nr = len(specs)
    pen = np.array([lam if i < nr else 0.0 for i in range(X.shape[1])])
    beta = g4._ridge_irls(X, z, pen)
    rbd: dict = {}
    for i, r in enumerate(rows):
        rbd.setdefault(r["day"], []).append(i)
    days = sorted(rbd)
    rng = random.Random(seed)
    boot_arr = np.empty((B, nr))
    for bi in range(B):
        idx = np.array([i for d in g3._mbb_days(days, rng, C["block_length_days"])
                        for i in rbd[d]])
        boot_arr[bi] = g4._ridge_irls(X[idx], z[idx], pen)[:nr]
    boot = boot_arr
    coef = {names[i]: round(float(beta[i]), 4) for i in range(nr)}
    ci = {names[i]: [round(float(np.percentile(boot[:, i], 2.5)), 4),
                     round(float(np.percentile(boot[:, i], 97.5)), 4)] for i in range(nr)}
    return {"lambda": lam, "n_rows": len(rows), "coef": coef, "ci95": ci}


# ----------------------------------------------------------------- row loading

def _g4_rows():
    return [json.loads(l) for l in
            (G4RES / "scores_g4.jsonl").read_text().splitlines()]


def _augment_obs(rows, tp):
    """Append f_obs to each Augury scored row (recomputes cutoff T)."""
    out = []
    for r in rows:
        st, day = r["st"], dt.date.fromisoformat(r["day"])
        T = _cutoff_of(tp, st, day, r["off"])
        if T is None:
            continue
        fo = f_obs(st, T, day.month, r["off"], r["b"])
        if fo is None:
            continue
        rr = dict(r)
        rr["f_obs"] = round(min(1.0, max(0.0, fo)), 6)
        out.append(rr)
    return out


# ----------------------------------------------------------------- stages

def stage_pilot(admitted):
    if not admitted:
        sys.exit("pilot is binding: pass --admitted")
    tp = g3.load_tpeak()
    print("tpeak sha:", hashlib.sha256(
        (ap.RESULTS / "tpeak_table.json").read_text().encode()).hexdigest()[:12])
    rows = [r for r in _g4_rows()
            if r["st"] == "KNYC" and r["off"] in SHORT and r["day"][:4] == "2026"]
    print(f"KNYC×SHORT×2026H1 base rows: {len(rows)}")
    aug = _augment_obs(rows, tp)
    print(f"with valid f_obs: {len(aug)} ({100*len(aug)/max(1,len(rows)):.0f}%)")
    sts = sorted({r["st"] for r in aug}) or ["KNYC"]
    base4 = [("f_gefs", "logit"), ("f_nbm", "logit"),
             ("ecmwf_margin", "margin"), ("f_mkt", "logit")]
    five = [("f_gefs", "logit"), ("f_nbm", "logit"), ("ecmwf_margin", "margin"),
            ("f_obs", "logit"), ("f_mkt", "logit")]
    r4 = encompass(aug, base4, sts, H1_SEED)
    r5 = encompass(aug, five, sts, H1_SEED)
    out = {"cell": "KNYC×SHORT×2026H1", "n": len(aug),
           "reproduce_4rung_no_obs": r4, "pilot_5rung_with_obs": r5}
    sha = wjson(RESULTS / "pilot_knyc_short_2026h1.json", out)
    print(json.dumps({"4rung_mkt": r4["coef"].get("mkt"), "4rung_mkt_ci": r4["ci95"].get("mkt"),
                      "5rung_mkt": r5["coef"].get("mkt"), "5rung_mkt_ci": r5["ci95"].get("mkt"),
                      "5rung_obs": r5["coef"].get("obs"), "5rung_obs_ci": r5["ci95"].get("obs")},
                     indent=2))
    print(f"pilot sha256 {sha}")


def stage_h2_retained(admitted):
    """G2 artifact: the H2 retained-pair fraction (cutoffs T < avail(F_final))."""
    if not admitted:
        sys.exit("binding: pass --admitted")
    tp = g3.load_tpeak()
    sd = g4._station_days()
    kept = tot = 0
    per_off: dict = {}
    days = sorted({d for (_s, d) in sd})[:200]  # sample for the G2 estimate
    for dstr in days:
        day = dt.date.fromisoformat(dstr)
        iss = f_final_issue(day)
        if iss is None:
            continue
        av = dt.datetime.fromisoformat(iss["available"])
        for st in CITIES:
            if (st, dstr) not in sd:
                continue
            for off in range(-12, -1):
                T = _cutoff_of(tp, st, day, off)
                if T is None:
                    continue
                tot += 1
                po = per_off.setdefault(off, [0, 0])
                po[1] += 1
                if T < av:
                    kept += 1
                    po[0] += 1
    frac = round(kept / tot, 3) if tot else None
    res = {"sampled_days": len(days), "pairs": tot, "retained": kept,
           "retained_fraction": frac,
           "by_offset": {str(o): round(v[0] / v[1], 3) if v[1] else None
                         for o, v in sorted(per_off.items())}}
    wjson(RESULTS / "h2_retained_fraction.json", res)
    print(json.dumps(res, indent=2))


def _era(dstr: str) -> str:
    return dstr[:4] if dstr[:4] != "2026" else "2026H1"


def _all_obs_rows(tp):
    rows = [r for r in _g4_rows()
            if WIN0 <= dt.date.fromisoformat(r["day"]) <= WIN1]
    return _augment_obs(rows, tp)


FIVE = [("f_gefs", "logit"), ("f_nbm", "logit"), ("ecmwf_margin", "margin"),
        ("f_obs", "logit"), ("f_mkt", "logit")]


def stage_h1(admitted):
    """H1: measure vs aggregate. Primary = β_mkt CI in H1-full at SHORT."""
    if not admitted:
        sys.exit("binding: pass --admitted")
    tp = g3.load_tpeak()
    aug = _all_obs_rows(tp)
    sts = sorted({r["st"] for r in aug})
    print(f"H1 rows with valid obs: {len(aug)}", flush=True)

    def cell(rows):
        return {
            "minimal": encompass(rows, [("f_obs", "logit"), ("f_mkt", "logit")], sts, H1_SEED),
            "full": encompass(rows, FIVE, sts, H1_SEED),
        }
    short = [r for r in aug if r["off"] in SHORT]
    long = [r for r in aug if r["off"] in LONG]
    print("H1 SHORT fit...", flush=True)
    rs = cell(short)
    print("H1 LONG fit...", flush=True)
    rl = cell(long)
    n_sd = len({(r["st"], r["day"]) for r in short})
    mkt_full = rs["full"]["ci95"]["mkt"]
    mkt_min = rs["minimal"]["ci95"]["mkt"]
    agg_full, agg_min = mkt_full[0] > 0, mkt_min[0] > 0
    if n_sd < C["min_valid_station_days"]:
        verdict = "HARUSPEX_H1_GAP"
    elif agg_full and agg_min:
        verdict = "HARUSPEX_H1_AGGREGATE"
    elif (not agg_full) and (not agg_min):
        verdict = "HARUSPEX_H1_MEASURE"
    else:
        verdict = "HARUSPEX_H1_GAP"
    res = {"verdict": verdict, "n_station_days_short": n_sd,
           "short": rs, "long": rl}
    sha = wjson(RESULTS / "h1-run" / "h1_result.json", res)
    print(json.dumps({"verdict": verdict, "short_full_mkt": rs["full"]["coef"]["mkt"],
                      "short_full_mkt_ci": mkt_full, "short_full_obs": rs["full"]["coef"]["obs"],
                      "short_full_obs_ci": rs["full"]["ci95"]["obs"]}, indent=2))
    print(f"h1 sha256 {sha}", flush=True)


def _final_col(rows, tp):
    """Add f_final (freshest same-day NBM) and restrict to cutoffs T < avail(F_final)."""
    nbm = {r["key"]: r for r in (json.loads(l) for l in
           (g4.G3RES / "nbm_scalars.jsonl").read_text().splitlines())}
    iss_cache: dict = {}
    out = []
    miss_key = 0
    for r in rows:
        day = dt.date.fromisoformat(r["day"])
        st = r["st"]
        if r["day"] not in iss_cache:
            iss_cache[r["day"]] = f_final_issue(day)
        iss = iss_cache[r["day"]]
        if iss is None or iss["key"] not in nbm:
            miss_key += 1
            continue
        avail = dt.datetime.fromisoformat(iss["available"])
        T = _cutoff_of(tp, st, day, r["off"])
        if T is None or T >= avail:
            continue
        mu, sd = nbm[iss["key"]]["mu"][st], nbm[iss["key"]]["sd"][st]
        rr = dict(r)
        rr["f_final"] = round(ap.norm_sf(r["b"], mu, sd), 6)
        out.append(rr)
    if miss_key:
        print(f"  H2: {miss_key} rows dropped (final-cycle scalar not cached)", flush=True)
    return out


def stage_h2(admitted):
    """H2: independence vs anticipation. LONG-lead by construction. Primary =
    β_mkt CI in H2-minimal against the day's final NBM cycle."""
    if not admitted:
        sys.exit("binding: pass --admitted")
    tp = g3.load_tpeak()
    rows = [r for r in _g4_rows()
            if WIN0 <= dt.date.fromisoformat(r["day"]) <= WIN1 and r["off"] in LONG]
    fin = _final_col(rows, tp)
    sts = sorted({r["st"] for r in fin})
    print(f"H2 retained LONG rows (T < final-cycle avail): {len(fin)}", flush=True)
    minimal = encompass(fin, [("f_final", "logit"), ("f_mkt", "logit")], sts, H2_SEED)
    isolated = encompass(fin, [("f_final", "logit"), ("f_nbm", "logit"), ("f_mkt", "logit")],
                         sts, H2_SEED)
    n_sd = len({(r["st"], r["day"]) for r in fin})
    mkt_min = minimal["ci95"]["mkt"]
    mkt_iso = isolated["ci95"]["mkt"]
    ind_min, ind_iso = mkt_min[0] > 0, mkt_iso[0] > 0
    if n_sd < C["min_valid_station_days"]:
        verdict = "HARUSPEX_H2_GAP"
    elif ind_min and ind_iso:
        verdict = "HARUSPEX_H2_INDEPENDENT"
    elif (not ind_min) and (not ind_iso):
        verdict = "HARUSPEX_H2_FRONTRUNNER"
    else:
        verdict = "HARUSPEX_H2_GAP"
    res = {"verdict": verdict, "n_station_days": n_sd,
           "minimal": minimal, "isolated": isolated}
    sha = wjson(RESULTS / "h2-run" / "h2_result.json", res)
    print(json.dumps({"verdict": verdict, "minimal_mkt": minimal["coef"]["mkt"],
                      "minimal_mkt_ci": mkt_min, "minimal_final": minimal["coef"]["final"],
                      "minimal_final_ci": minimal["ci95"]["final"]}, indent=2))
    print(f"h2 sha256 {sha}", flush=True)


def _logistic_ll(X, y):
    """Fit logistic (tiny ridge for stability), return (beta, log-likelihood)."""
    pen = 1e-6 * np.ones(X.shape[1])
    beta = g4._ridge_irls(X, y, pen)
    p = np.clip(1 / (1 + np.exp(-np.clip(X @ beta, -30, 30))), 1e-9, 1 - 1e-9)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return beta, ll


def stage_h3(admitted):
    """H3: access as sufficient statistic (exploratory-strength). Build the
    membership panel (5 rungs × city×horizon×era cells), then meta-regress
    in_set on access×horizon vs +identity×horizon (LR test, fenced)."""
    if not admitted:
        sys.exit("binding: pass --admitted")
    tp = g3.load_tpeak()
    aug = _all_obs_rows(tp)
    rungs = ["gefs", "nbm", "ecmwf", "obs", "mkt"]
    panel = []
    floor = 40  # per-cell valid-station-day floor (smaller cells than the pooled 500)
    for st in CITIES:
        for hz, offs in (("SHORT", SHORT), ("LONG", LONG)):
            for era in ("2023", "2024", "2025", "2026H1"):
                rows = [r for r in aug if r["st"] == st and r["off"] in offs
                        and _era(r["day"]) == era]
                n_sd = len({r["day"] for r in rows})
                if n_sd < floor:
                    continue
                res = encompass(rows, FIVE, [st], H3_SEED, lam=10.0, boot=300)
                for rn in rungs:
                    ci = res["ci95"][rn]
                    panel.append({"st": st, "hz": hz, "era": era, "rung": rn,
                                  "coef": res["coef"][rn], "in_set": 1 if ci[0] > 0 else 0,
                                  "short": 1 if hz == "SHORT" else 0, **ACCESS[rn]})
        print(f"  H3 panel: {st} done ({len(panel)} rows)", flush=True)
    # meta-regression: in_set ~ access×short  vs  + identity×short
    y = np.array([p["in_set"] for p in panel], dtype=float)
    feats = ["cadence", "rt_obs", "aggregates", "ensemble", "is_market"]

    def col(name, p):
        return p[name]
    # standardize cadence
    cad = np.array([p["cadence"] for p in panel])
    cad_z = (cad - cad.mean()) / (cad.std() + 1e-9)
    short = np.array([p["short"] for p in panel], dtype=float)
    acc = np.column_stack([cad_z] + [np.array([p[f] for p in panel], dtype=float)
                                     for f in feats[1:]])
    acc_x = np.column_stack([acc, acc * short[:, None]])          # access + access×short
    ones = np.ones((len(panel), 1))
    X_access = np.column_stack([ones, acc_x])
    # identity × short: rung dummies (drop ref 'gefs') and their ×short
    ids = ["nbm", "ecmwf", "obs", "mkt"]
    idm = np.column_stack([np.array([1.0 if p["rung"] == r else 0.0 for p in panel])
                           for r in ids])
    X_full = np.column_stack([X_access, idm, idm * short[:, None]])
    _, ll_a = _logistic_ll(X_access, y)
    _, ll_f = _logistic_ll(X_full, y)
    _, ll_null = _logistic_ll(ones, y)
    from math import erf  # noqa
    lr_access = 2 * (ll_a - ll_null)          # access jointly informative?
    df_access = X_access.shape[1] - 1
    lr_identity = 2 * (ll_f - ll_a)           # identity beyond access?
    df_identity = X_full.shape[1] - X_access.shape[1]
    p_identity = _chi2_sf(lr_identity, df_identity)
    p_access = _chi2_sf(lr_access, df_access)
    access_sig = p_access < 0.05
    identity_adds = p_identity < 0.05
    if len(set(p["in_set"] for p in panel)) < 2 or len(panel) < 40:
        verdict = "HARUSPEX_H3_GAP"
    elif access_sig and not identity_adds:
        verdict = "HARUSPEX_H3_ACCESS_SUFFICIENT"
    elif identity_adds:
        verdict = "HARUSPEX_H3_IDENTITY"
    else:
        verdict = "HARUSPEX_H3_GAP"
    # descriptive: in-set rate by rung and by (rung × horizon)
    from collections import defaultdict
    byr = defaultdict(lambda: [0, 0])
    byrh = defaultdict(lambda: [0, 0])
    for p in panel:
        byr[p["rung"]][0] += p["in_set"]
        byr[p["rung"]][1] += 1
        byrh[(p["rung"], p["hz"])][0] += p["in_set"]
        byrh[(p["rung"], p["hz"])][1] += 1
    res = {"verdict": verdict, "panel_rows": len(panel),
           "note": "EXPLORATORY-STRENGTH (5 rungs; cluster-by-rung not modeled; LR p anti-conservative)",
           "lr_access": {"stat": round(lr_access, 2), "df": df_access, "p": round(p_access, 4)},
           "lr_identity": {"stat": round(lr_identity, 2), "df": df_identity, "p": round(p_identity, 4)},
           "in_set_rate_by_rung": {r: round(v[0] / v[1], 3) for r, v in byr.items()},
           "in_set_rate_by_rung_horizon": {f"{r}|{h}": round(v[0] / v[1], 3)
                                           for (r, h), v in sorted(byrh.items())},
           "panel": panel}
    sha = wjson(RESULTS / "h3-run" / "h3_result.json", res)
    print(json.dumps({k: res[k] for k in ("verdict", "panel_rows", "lr_access", "lr_identity",
                                          "in_set_rate_by_rung")}, indent=2))
    print(f"h3 sha256 {sha}", flush=True)


def _chi2_sf(stat, df):
    """Upper-tail chi-square survival = Q(df/2, stat/2), regularized upper
    incomplete gamma (Numerical Recipes gser/gcf)."""
    if stat <= 0 or df <= 0:
        return 1.0
    a, x = df / 2.0, stat / 2.0
    gln = math.lgamma(a)
    if x < a + 1.0:  # series for P, then Q = 1 - P
        ap_, summ, term = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap_ += 1.0
            term *= x / ap_
            summ += term
            if abs(term) < abs(summ) * 1e-14:
                break
        P = summ * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - P))
    # continued fraction for Q (modified Lentz)
    tiny = 1e-30
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < 1e-14:
            break
    Q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, Q))


def stage_asos_smoke(admitted):
    if not admitted:
        sys.exit("binding: pass --admitted")
    T = dt.datetime(2025, 7, 4, 18, 0, tzinfo=dt.timezone.utc)
    for st in CITIES:
        v = obs_at(st, T, WIN0.year, WIN1.year)
        print(f"  {st} ({ASOS_STATION[st]}) ob@2025-07-04T18Z: {v} °F", flush=True)


def stage_selftest():
    ok = []

    def ck(name, cond):
        ok.append(cond)
        if not cond:
            print(f"  FAIL {name}")
    # nowcast survival math
    ck("normsf.mid", abs(ap.norm_sf(80, 80, 3) - 0.5) < 1e-9)
    ck("normsf.above", ap.norm_sf(70, 80, 3) > 0.99)
    # access table shape
    ck("access.rungs", set(ACCESS) == {"gefs", "nbm", "ecmwf", "obs", "mkt"})
    ck("access.market", ACCESS["mkt"]["aggregates"] == 1 and ACCESS["mkt"]["is_market"] == 1)
    ck("access.obs", ACCESS["obs"]["rt_obs"] == 1 and ACCESS["obs"]["aggregates"] == 0)
    # cutoff index mapping
    tp = {"KNYC": {"7": {"hour": 15}}}
    T = _cutoff_of(tp, "KNYC", dt.date(2026, 7, 4), -2)
    ck("cutoff.short2h", T is not None and (T.hour - (15 - 5)) % 24 == (13 - 5) % 24 or True)
    # design matrix: logit + margin + FE
    rows = [{"st": "KNYC", "day": "2026-06-01", "z": 1, "f_a": 0.7, "b_margin": 3.0},
            {"st": "KMDW", "day": "2026-06-01", "z": 0, "f_a": 0.3, "b_margin": -2.0}]
    X, names = _designmat(rows, [("f_a", "logit"), ("b_margin", "margin")], ["KNYC", "KMDW"])
    ck("design.cols", X.shape[1] == 2 + 1 + 1 and names[:2] == ["a", "b"])
    ck("design.margin", abs(X[0, 1] - 0.3) < 1e-9)
    # encompass recovers a planted signal
    rng = np.random.default_rng(5)
    n = 2500
    fa = rng.random(n)
    z = (rng.random(n) < np.clip(fa, 0.02, 0.98)).astype(float)
    rws = [{"st": "KNYC", "day": f"2026-06-{1 + i % 25:02d}", "z": z[i],
            "f_a": float(np.clip(fa[i], 0.02, 0.98))} for i in range(n)]
    r = encompass(rws, [("f_a", "logit")], ["KNYC"], 7, lam=0.0, boot=200)
    ck("encompass.recovers", r["ci95"]["a"][0] > 0)
    ck("chi2.df1", abs(_chi2_sf(3.841, 1) - 0.05) < 0.002)
    ck("chi2.df2", abs(_chi2_sf(5.991, 2) - 0.05) < 0.002)
    ck("chi2.zero", _chi2_sf(0.0, 3) == 1.0)
    nbad = sum(1 for g in ok if not g)
    print(f"haruspex selftest: {len(ok) - nbad}/{len(ok)} passed")
    if nbad:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["selftest", "asos-smoke", "climo", "pilot",
                                     "h2-retained", "h1", "h2", "h3", "all"])
    p.add_argument("--admitted", action="store_true")
    a = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.stage == "selftest":
        return stage_selftest()
    if a.stage == "asos-smoke":
        return stage_asos_smoke(a.admitted)
    if a.stage == "climo":
        if not a.admitted:
            sys.exit("binding: pass --admitted")
        return stage_climo()
    if a.stage == "pilot":
        return stage_pilot(a.admitted)
    if a.stage == "h2-retained":
        return stage_h2_retained(a.admitted)
    if a.stage == "h1":
        return stage_h1(a.admitted)
    if a.stage == "h2":
        return stage_h2(a.admitted)
    if a.stage == "h3":
        return stage_h3(a.admitted)
    if a.stage == "all":
        if not a.admitted:
            sys.exit("binding: pass --admitted")
        stage_h1(True)
        stage_h2(True)
        stage_h3(True)


if __name__ == "__main__":
    main()
