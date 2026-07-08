#!/usr/bin/env python
"""AUGURY G4 — the determining-shadow-set read (full pantheon).

Extends the G3 market-vs-NBM encompassing to the full ladder
{GEFS, NBM, ECMWF-ENS, MOS, market} and asks: what is the minimal subset of
forecasters that determines the outcome, and is the market in it, by horizon?
Spec: docs/prereg/augury/AUGURY_G4_PREREG.md (§1-§4). Reuses the G3 primitives
(candles, NBM scalars, CLI, scoring, day-block bootstrap, GRIB decode) at their
frozen hashes; adds GEFS + ECMWF-ENS + MOS forecaster columns.

Rungs (all -> F_k(theta) = P(high > theta) at matched availability cutoffs):
  GEFS   geavg+gespr TMAX (0.25 regular_ll); civil-day max = max of the 6-h
         blocks covering 12Z-00Z; Normal(mu,sigma) survival.
  NBM    reuse results/augury/g3-full-run/nbm_scalars.jsonl (Normal, as G3).
  ECMWF  enfo mx2t3 (0.25 regular_ll); 21 members (cf + pf 1..20); per-member
         civil-day max over the 3-h blocks; ensemble mu,sigma -> Normal.
  MOS    IEM NBS/MEX MaxT (point) -> covariate (MaxT - theta); never a dist.
  market reuse G3 candle exceedance (PAV midpoint).

Stages: selftest | plan | gefs | ecmwf | mos | score | adjudicate | all
Binding stages require --admitted. Window 2023-02-01..2026-06-30 (ECMWF start).
Requires numpy, eccodes, pyproj (.venv-augury).
"""
from __future__ import annotations

import bisect
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import augury_g3 as g3  # noqa: E402
import augury_pilot as ap  # noqa: E402

C = ap.C
RESULTS = ap.ROOT / "results" / "augury" / "g4-run"
G3RES = ap.ROOT / "results" / "augury" / "g3-full-run"
STATION_LL = g3.STATION_LL
CITIES = g3.CITIES
WIN0 = dt.date(2023, 2, 1)
WIN1 = dt.date(2026, 6, 30)
ECMWF_MEMBERS = [str(i) for i in range(1, 22)]  # 21 perturbed members (mx2t3 has no cf)
SHORT_OFFSETS = {-5, -4, -3, -2}
LONG_OFFSETS = {-12, -11, -10, -9, -8}
G4_SEED_DM = 20260708
G4_SEED_ENC = 20260709
ENC_BOOT = 2000
RIDGE_GRID = [0.0, 0.01, 0.1, 1.0, 10.0]  # lambda grid (spec §3.1); frozen

GEFS_BUCKET = "noaa-gefs-pds"
GEFS_HOST = f"https://{GEFS_BUCKET}.s3.amazonaws.com"
ECM_HOST = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"


def wjson(p: Path, o) -> str:
    return ap.write_json(p, o)


# --------------------------------------------------------------- grid decode

_geom = {}


def grid_indices(h):
    """{station: values-array index}, validated 7/7 vs find_nearest once per
    geometry. regular_ll (GEFS/ECMWF) computed directly; lambert delegates to
    the G3 boustrophedon path."""
    import eccodes
    gt = eccodes.codes_get(h, "gridType")
    if "lambert" in gt:
        return g3._grid_indices(h)
    g = lambda k: eccodes.codes_get(h, k)  # noqa: E731
    Ni, Nj = g("Ni"), g("Nj")
    lo1, la1 = g("longitudeOfFirstGridPointInDegrees"), g("latitudeOfFirstGridPointInDegrees")
    di, dj = g("iDirectionIncrementInDegrees"), g("jDirectionIncrementInDegrees")
    jpos = g("jScansPositively")
    sig = (gt, Ni, Nj, round(lo1, 4), round(la1, 4), round(di, 5), round(dj, 5), jpos)
    if sig in _geom:
        return _geom[sig]
    idx = {}
    for st, (la, lo) in STATION_LL.items():
        i = round(((lo % 360) - (lo1 % 360)) / di) % Ni
        j = round((la - la1) / dj) if jpos == 1 else round((la1 - la) / dj)
        idx[st] = int(min(max(j, 0), Nj - 1)) * Ni + int(min(max(i, 0), Ni - 1))
    vals = eccodes.codes_get_array(h, "values")
    for st, (la, lo) in STATION_LL.items():
        fn = eccodes.codes_grib_find_nearest(h, la, lo)[0].value
        if abs(float(vals[idx[st]]) - fn) > 1e-6:
            raise RuntimeError(f"grid validation FAILED {st} {sig}")
    _geom[sig] = idx
    print(f"  grid {gt} validated 7/7 (Ni={Ni} Nj={Nj})", flush=True)
    return idx


def decode_points(data: bytes) -> dict[str, float]:
    import eccodes
    h = eccodes.codes_new_from_message(data)
    try:
        idx = grid_indices(h)
        vals = eccodes.codes_get_array(h, "values")
        return {st: float(vals[i]) for st, i in idx.items()}
    finally:
        eccodes.codes_release(h)


# --------------------------------------------------------------- idx / s3

def idx_records_url(idx_url: str) -> list[dict]:
    data = ap.cached_get(idx_url, ok404=True)
    if data is None:
        return []
    recs = []
    for line in data.decode("utf-8", "replace").strip().splitlines():
        p = line.split(":")
        if len(p) < 6:
            continue
        recs.append({"start": int(p[1]), "var": p[3], "lvl": p[4],
                     "win": p[5], "extra": ":".join(p[6:]), "end": None})
    for i in range(len(recs) - 1):
        recs[i]["end"] = recs[i + 1]["start"]
    return recs


def _parse_lm(lm: str | None) -> dt.datetime | None:
    return dt.datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S %Z").replace(
        tzinfo=dt.timezone.utc) if lm else None


def s3_head_mtime(url: str) -> dt.datetime | None:
    """LastModified of an S3 object via a 1-byte ranged GET (availability)."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": ap.UA})
    req.add_header("Range", "bytes=0-0")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return _parse_lm(r.headers.get("Last-Modified"))
        except Exception:
            if attempt == 3:
                return None
            time.sleep(1.0 * 2 ** attempt)
    return None


_ecm_mt_cache: dict = {}


def s3_head_mtime_ecm(url: str) -> dt.datetime | None:
    """LastModified for an ECMWF object (503-aware, cached per url)."""
    if url in _ecm_mt_cache:
        return _ecm_mt_cache[url]
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": ap.UA})
    req.add_header("Range", "bytes=0-0")
    for attempt in range(9):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                mt = _parse_lm(r.headers.get("Last-Modified"))
                _ecm_mt_cache[url] = mt
                return mt
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < 8:
                time.sleep(min(2.0 * 2 ** attempt, 45))
                continue
            return None
        except Exception:
            if attempt < 8:
                time.sleep(min(2.0 * 2 ** attempt, 45))
                continue
            return None
    return None


# --------------------------------------------------------------- GEFS rung

def _civilday_blocks_gefs(cyc: dt.datetime, day: dt.date):
    """The two 6-h TMAX blocks tiling [D 12Z, D+1 00Z]; returns [(fhr, endZ)]."""
    d12 = dt.datetime(day.year, day.month, day.day, 12, tzinfo=dt.timezone.utc)
    ends = [d12 + dt.timedelta(hours=6), d12 + dt.timedelta(hours=12)]  # 18Z, 00Z
    out = []
    for e in ends:
        fhr = int((e - cyc).total_seconds() // 3600)
        if fhr <= 0 or fhr > 240:
            return None
        out.append((fhr, e))
    return out


def gefs_avail(cyc: dt.datetime, day: dt.date):
    """Cheap availability datetime for a GEFS cycle (idx exists + mtime), no
    field decode — used by the scan/selection so we fetch fields only once."""
    blocks = _civilday_blocks_gefs(cyc, day)
    if blocks is None:
        return None
    ymd, hh = cyc.strftime("%Y%m%d"), cyc.strftime("%H")
    base = f"{GEFS_HOST}/gefs.{ymd}/{hh}/atmos/pgrb2sp25"
    av = None
    for fhr, _e in blocks:
        f = f"{base}/geavg.t{hh}z.pgrb2s.0p25.f{fhr:03d}"
        if ap.cached_get(f + ".idx", ok404=True) is None:
            return None
        mt = s3_head_mtime(f)
        if mt is None:
            return None
        av = mt if av is None else max(av, mt)
    return av


def gefs_daily(cyc: dt.datetime, day: dt.date):
    """Civil-day-max mu,sigma per station for one GEFS cycle, or None."""
    blocks = _civilday_blocks_gefs(cyc, day)
    if blocks is None:
        return None
    ymd, hh = cyc.strftime("%Y%m%d"), cyc.strftime("%H")
    base = f"{GEFS_HOST}/gefs.{ymd}/{hh}/atmos/pgrb2sp25"
    per_block = []
    avail = None
    for fhr, _e in blocks:
        gavg = f"{base}/geavg.t{hh}z.pgrb2s.0p25.f{fhr:03d}"
        gspr = f"{base}/gespr.t{hh}z.pgrb2s.0p25.f{fhr:03d}"
        ra = idx_records_url(gavg + ".idx")
        rs = idx_records_url(gspr + ".idx")
        ma = [r for r in ra if r["var"] == "TMAX" and "2 m" in r["lvl"]]
        ms = [r for r in rs if r["var"] == "TMAX" and "2 m" in r["lvl"]]
        if not ma or not ms:
            return None
        mt = s3_head_mtime(gavg + ".idx")
        if mt is None:
            return None
        avail = mt if avail is None else max(avail, mt)
        mu = decode_points(g3._raw_get(gavg, ma[0]["start"], ma[0]["end"] or ma[0]["start"] + 20_000_000))
        sd = decode_points(g3._raw_get(gspr, ms[0]["start"], ms[0]["end"] or ms[0]["start"] + 20_000_000))
        per_block.append((mu, sd))
    out_mu, out_sd = {}, {}
    for st in STATION_LL:
        best = max(range(len(per_block)), key=lambda b: per_block[b][0][st])
        k = per_block[best][0][st]
        out_mu[st] = round((k - 273.15) * 1.8 + 32, 3)
        out_sd[st] = round(per_block[best][1][st] * 1.8, 3)
    return {"cycle": cyc.isoformat(), "available": avail.isoformat(),
            "mu": out_mu, "sd": out_sd}


# --------------------------------------------------------------- ECMWF rung

def _civilday_steps_ecmwf(cyc: dt.datetime, day: dt.date):
    """The four 3-h mx2t3 blocks tiling [D 12Z, D+1 00Z]; returns [(step,endZ)]."""
    d12 = dt.datetime(day.year, day.month, day.day, 12, tzinfo=dt.timezone.utc)
    ends = [d12 + dt.timedelta(hours=h) for h in (3, 6, 9, 12)]  # 15,18,21,00Z
    out = []
    for e in ends:
        step = int((e - cyc).total_seconds() // 3600)
        if step <= 0 or step > 240 or step % 3 != 0:
            return None
        out.append((step, e))
    return out


_ecm_pace = [0.0]


def _ecm_get(url: str, lo: int | None = None, hi: int | None = None) -> bytes | None:
    """ECMWF S3 is rate-limited (503 Slow Down). Gentle pace + long 503 backoff."""
    import urllib.error
    import urllib.request
    gap = 0.25 - (time.time() - _ecm_pace[0])
    if gap > 0:
        time.sleep(gap)
    req = urllib.request.Request(url, headers={"User-Agent": ap.UA})
    if lo is not None:
        req.add_header("Range", f"bytes={lo}-{hi - 1}")
    for attempt in range(12):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                _ecm_pace[0] = time.time()
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(min(3.0 * 1.7 ** attempt, 60))  # 503-aware; never raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(min(3.0 * 1.7 ** attempt, 60))
    _ecm_pace[0] = time.time()
    return None  # exhausted: skip this field/issue rather than crash the pull


def _ecm_cached_index(url: str):
    CACHE = ap.CACHE
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()
    f = CACHE / key
    if f.exists():
        return f.read_bytes()
    data = _ecm_get(url)
    if data is not None:
        f.write_bytes(data)
    return data


def _ecm_oper_base(cyc, step):
    ymd, hh = cyc.strftime("%Y%m%d"), cyc.strftime("%H")
    stamp = cyc.strftime("%Y%m%d%H%M%S")
    return f"{ECM_HOST}/{ymd}/{hh}z/ifs/0p25/oper/{stamp}-{step}h-oper-fc"


def ecmwf_avail(cyc: dt.datetime, day: dt.date):
    """Availability for an ECMWF-HRES (deterministic oper) cycle.

    NOTE: the ECMWF rung is HRES, not ENS. The ENS pull (21 members x 4 blocks
    = 84 fields/issue) proved infeasible from this location — the eu-central-1
    bucket 503-throttles sustained access (~500s/issue, failing). HRES is one
    deterministic mx2t3 field/block (4/issue, ~21x fewer requests) and IS
    pullable. Fenced: this is the deterministic flagship IFS as a point
    covariate, not the physics+AI ensemble distribution."""
    steps = _civilday_steps_ecmwf(cyc, day)
    if steps is None:
        return None
    av = None
    for step, _e in steps:
        base = _ecm_oper_base(cyc, step)
        if _ecm_cached_index(base + ".index") is None:
            return None
        mt = s3_head_mtime_ecm(base + ".grib2")
        if mt is None:
            return None
        av = mt if av is None else max(av, mt)
    return av


def ecmwf_daily(cyc: dt.datetime, day: dt.date):
    """ECMWF-HRES civil-day max (deg F) per station: max of the deterministic
    mx2t3 blocks covering 12Z-00Z. A point forecast (no ensemble spread)."""
    steps = _civilday_steps_ecmwf(cyc, day)
    if steps is None:
        return None
    day_max = {st: -1e9 for st in STATION_LL}
    avail = None
    for step, _e in steps:
        base = _ecm_oper_base(cyc, step)
        idx = _ecm_cached_index(base + ".index")
        if idx is None:
            return None
        recs = [json.loads(l) for l in idx.decode().strip().splitlines() if l.strip()]
        mx = [r for r in recs if r.get("param") == "mx2t3" and r.get("type") == "fc"]
        if not mx:
            return None
        mt = s3_head_mtime_ecm(base + ".grib2")
        if mt is None:
            return None
        avail = mt if avail is None else max(avail, mt)
        r = mx[0]
        data = _ecm_get(base + ".grib2", r["_offset"], r["_offset"] + r["_length"])
        if data is None:
            return None
        pts = decode_points(data)
        for st in STATION_LL:
            if pts[st] > day_max[st]:
                day_max[st] = pts[st]
    return {"cycle": cyc.isoformat(), "available": avail.isoformat(),
            "maxt": {st: round((v - 273.15) * 1.8 + 32, 3) for st, v in day_max.items()}}


# --------------------------------------------------------------- MOS rung

def mos_maxt(station: str, cyc: dt.datetime, day: dt.date) -> float | None:
    """IEM NBS MaxT (deg F) for the civil day from a given model run."""
    url = (f"https://mesonet.agron.iastate.edu/api/1/mos.json?station={station}"
           f"&model=NBS&runtime={cyc.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    d = ap.jget(url, ok404=True)
    if not d or "data" not in d:
        return None
    best = None
    for row in d["data"]:
        ft = row.get("ftime", "")
        tmp = row.get("tmp")
        if ft.startswith(day.isoformat()) and isinstance(tmp, (int, float)):
            best = tmp if best is None else max(best, tmp)
    return float(best) if best is not None else None


# --------------------------------------------------------------- issue scan

def _latest_cycle(day: dt.date, cutoff: dt.datetime, avail_fn, cyc_hours=(0, 6, 12, 18)):
    """Latest rung cycle datetime whose files are available <= cutoff (cheap
    availability check only — no field decode)."""
    c0 = cutoff.replace(minute=0, second=0, microsecond=0)
    for back in range(0, 30):  # step back hourly up to ~30h across the cycle grid
        c = c0 - dt.timedelta(hours=back)
        if c.hour not in cyc_hours:
            continue
        av = avail_fn(c, day)
        if av is not None and av <= cutoff:
            return c
    return None


def _scalar_cache(name: str) -> dict:
    p = RESULTS / name
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["k"]] = r
    return out


def _band_cutoffs_for(st, day, tp):
    h = g3.day_hour(tp, st, day)
    return None if h is None else ap.band_cutoffs(st, day, h)


def _station_days():
    """(station, day) with a G3 score row in the G4 window (reuse G3 output)."""
    rows = (G3RES / "scores.jsonl")
    sd = {}
    for line in rows.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        d = dt.date.fromisoformat(r["day"])
        if WIN0 <= d <= WIN1:
            sd.setdefault((r["st"], r["day"]), set()).add(r["off"])
    return sd


def _run_grib_rung(name: str, avail_fn, daily_fn, cyc_hours=(0, 6, 12, 18)) -> None:
    tp = g3.load_tpeak()
    have = _scalar_cache(f"{name}_scalars.jsonl")
    sd = _station_days()
    # issue union: for each (station,day) the cycles selected across band cutoffs
    need = {}
    days = sorted({d for (_st, d) in sd})
    for i, dstr in enumerate(days):
        day = dt.date.fromisoformat(dstr)
        for st in CITIES:
            if (st, dstr) not in sd:
                continue
            cuts = _band_cutoffs_for(st, day, tp)
            if cuts is None:
                continue
            for T in cuts:
                cyc = _latest_cycle(day, T, avail_fn, cyc_hours)
                if cyc:
                    k = f"{dstr}|{cyc.isoformat()}"
                    need.setdefault(k, (day, cyc))
        if (i + 1) % 100 == 0:
            print(f"  {name} scan {i+1}/{len(days)} (issues {len(need)})", flush=True)
    todo = [(k, v) for k, v in need.items() if k not in have]
    print(f"{name}: {len(need)} issues, {len(have)} cached, {len(todo)} to pull", flush=True)
    path = RESULTS / f"{name}_scalars.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with path.open("a", encoding="utf-8") as fh:
        for k, (day, cyc) in todo:
            rec = daily_fn(cyc, day)
            if rec is None:
                continue
            rec["k"] = k
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            done += 1
            if done % 100 == 0:
                print(f"  {name} pulled {done}/{len(todo)}", flush=True)
    print(f"{name} done: +{done}", flush=True)


def stage_gefs():
    _run_grib_rung("gefs", gefs_avail, gefs_daily)


def stage_ecmwf():
    _run_grib_rung("ecmwf", ecmwf_avail, ecmwf_daily)


def stage_mos():
    tp = g3.load_tpeak()
    have = _scalar_cache("mos_scalars.jsonl")
    sd = _station_days()
    need = {}
    for (st, dstr), offs in sd.items():
        day = dt.date.fromisoformat(dstr)
        cuts = _band_cutoffs_for(st, day, tp)
        if cuts is None:
            continue
        for T in cuts:
            c0 = T.replace(minute=0, second=0, microsecond=0)
            for back in range(0, 24):
                c = c0 - dt.timedelta(hours=back)
                if c.hour in (0, 6, 12, 18):
                    need.setdefault(f"{st}|{dstr}|{c.isoformat()}", (st, day, c))
                    break
    todo = [(k, v) for k, v in need.items() if k not in have]
    print(f"mos: {len(need)} issues, {len(todo)} to pull", flush=True)
    path = RESULTS / "mos_scalars.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with path.open("a", encoding="utf-8") as fh:
        for k, (st, day, c) in todo:
            mx = mos_maxt(st, c, day)
            rec = {"k": k, "cycle": c.isoformat(), "maxt": mx}
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            done += 1
            if done % 500 == 0:
                print(f"  mos {done}/{len(todo)}", flush=True)
    print(f"mos done: +{done}", flush=True)


# --------------------------------------------------------------- score

def _norm_sf(x, mu, sd):
    return ap.norm_sf(x, mu, sd)


def stage_score():
    tp = g3.load_tpeak()
    gefs = _scalar_cache("gefs_scalars.jsonl")
    ecm = _scalar_cache("ecmwf_scalars.jsonl")
    nbm = {r["key"]: r for r in
           (json.loads(l) for l in (G3RES / "nbm_scalars.jsonl").read_text().splitlines())}
    mos = _scalar_cache("mos_scalars.jsonl")
    gefs_by_day = _by_day(gefs)
    ecm_by_day = _by_day(ecm)
    ecmwf_on = len(ecm) > 0
    mos_ok = sum(1 for r in mos.values() if r.get("maxt") is not None)
    mos_on = len(mos) > 0 and mos_ok / len(mos) > 0.5
    print(f"score: ECMWF rung {'ON' if ecmwf_on else 'OFF (throttle-blocked)'}; "
          f"MOS rung {'ON' if mos_on else f'OFF (coverage {mos_ok}/{len(mos)} — NBS archive sparse)'}",
          flush=True)
    sd = _station_days()
    out = (RESULTS / "scores_g4.jsonl").open("w", encoding="utf-8")
    n = 0
    max_age = C["max_quote_age_minutes"]
    days_by_st = {}
    for (st, dstr) in sd:
        days_by_st.setdefault(st, []).append(dstr)
    for st in CITIES:
        y0, y1 = WIN0.year, WIN1.year
        cli = ap.cli_days(st, y0, y1)
        ms = g3.station_markets(st)
        by_day = {}
        for m in ms:
            d = m["_event_date"]
            if WIN0 <= dt.date.fromisoformat(d) <= WIN1:
                by_day.setdefault(d, []).append(m)
        for i, dstr in enumerate(sorted(by_day)):
            if (st, dstr) not in sd:
                continue
            day = dt.date.fromisoformat(dstr)
            row_cli = cli.get(day)
            cuts = _band_cutoffs_for(st, day, tp)
            if row_cli is None or cuts is None:
                continue
            high = row_cli["high"]
            dms = by_day[dstr]
            prep = {m["ticker"]: g3.ap.prep_candles(m, cuts[-1] + dt.timedelta(hours=1)) for m in dms}
            bounds = {m["ticker"]: g3.strike_bounds(m) for m in dms}
            for off, T in zip(range(C["band_start_offset_hours"], C["band_end_offset_hours"] + 1), cuts):
                ts = int(T.timestamp())
                bins, vol = [], 0.0
                for m in dms:
                    cand, tl, vp = prep[m["ticker"]]
                    j = bisect.bisect_right(tl, ts) - 1
                    if j >= 0:
                        vol += vp[j]
                    q, _ = g3.ap.book_at(cand, tl, ts, max_age)
                    if q is None or (q["ask"] - q["bid"]) * 100 > C["max_spread_cents"]:
                        continue
                    lo, hi, _s = bounds[m["ticker"]]
                    if lo is None and hi is None:
                        continue
                    bins.append((lo, hi, (q["bid"] + q["ask"]) / 2))
                if len(bins) < C["min_valid_strikes"] or vol < C["min_event_volume_contracts"]:
                    continue
                # freshest cached rung issue available at this cutoff
                g_rec = _pick_cached(gefs_by_day, dstr, T)
                e_rec = _pick_cached(ecm_by_day, dstr, T) if ecmwf_on else True
                nb = g3.latest_issue(day, T)
                mrec = _mos_at(mos, st, day, T) if mos_on else None
                if not g_rec or not e_rec or nb is None or nb["key"] not in nbm:
                    continue
                if mos_on and mrec is None:
                    continue
                nmu, nsd = nbm[nb["key"]]["mu"][st], nbm[nb["key"]]["sd"][st]
                bnds, _raw, mono = g3.build_exceedance(bins)
                for b, fmkt in zip(bnds, mono):
                    fmkt = min(1.0, max(0.0, fmkt))
                    rec = {"st": st, "day": dstr, "off": off, "b": b,
                           "z": 1 if high > b else 0,
                           "f_mkt": round(fmkt, 6),
                           "f_nbm": round(_norm_sf(b, nmu, nsd), 6),
                           "f_gefs": round(_norm_sf(b, g_rec["mu"][st], g_rec["sd"][st]), 6)}
                    if mos_on:
                        rec["mos_margin"] = round(mrec - b, 3)
                    if ecmwf_on:
                        rec["ecmwf_margin"] = round(e_rec["maxt"][st] - b, 3)
                    out.write(json.dumps(rec) + "\n")
                    n += 1
        print(f"  {st} scored (rows so far {n})", flush=True)
    out.close()
    print(f"score done: {n} joint-complete rows", flush=True)


def _by_day(cache: dict) -> dict:
    """day -> sorted [(available_dt, rec)] from a rung scalar cache."""
    out: dict = {}
    for k, rec in cache.items():
        dstr = k.split("|", 1)[0]
        out.setdefault(dstr, []).append(
            (dt.datetime.fromisoformat(rec["available"]), rec))
    for dstr in out:
        out[dstr].sort(key=lambda x: x[0])
    return out


def _pick_cached(by_day: dict, dstr: str, cutoff: dt.datetime):
    """Freshest cached issue whose availability <= cutoff (score-time, no fetch)."""
    best = None
    for av, rec in by_day.get(dstr, []):
        if av <= cutoff and (best is None or av > best[0]):
            best = (av, rec)
    return best[1] if best else None


def _mos_at(cache, st, day, T):
    c0 = T.replace(minute=0, second=0, microsecond=0)
    for back in range(0, 24):
        c = c0 - dt.timedelta(hours=back)
        if c.hour in (0, 6, 12, 18):
            r = cache.get(f"{st}|{day.isoformat()}|{c.isoformat()}")
            return r["maxt"] if r and r.get("maxt") is not None else None
    return None


# --------------------------------------------------------------- adjudicate

def _logit(p):
    p = np.clip(p, C["prob_clip_low"], C["prob_clip_high"])
    return np.log(p / (1 - p))


def _ridge_irls(X, z, pen):
    beta = np.zeros(X.shape[1])
    P = np.diag(pen)
    for _ in range(80):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.maximum(p * (1 - p), 1e-9)
        H = (X * W[:, None]).T @ X + P
        step = np.linalg.solve(H, X.T @ (z - p) - P @ beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def _rung_names(rows):
    base = ["gefs", "nbm"]
    if "ecmwf_margin" in rows[0]:
        base.append("ecmwf")
    if "mos_margin" in rows[0]:
        base.append("mos")
    base.append("mkt")
    return base


def _design(rows, sts):
    rungs = _rung_names(rows)
    cols, names = [], []
    for rn in rungs:
        if rn == "gefs":
            cols.append(_logit(np.array([r["f_gefs"] for r in rows])))
        elif rn == "nbm":
            cols.append(_logit(np.array([r["f_nbm"] for r in rows])))
        elif rn == "ecmwf":
            cols.append(np.array([r["ecmwf_margin"] for r in rows]) / 10.0)
        elif rn == "mos":
            cols.append(np.array([r["mos_margin"] for r in rows]) / 10.0)
        elif rn == "mkt":
            cols.append(_logit(np.array([r["f_mkt"] for r in rows])))
        names.append(rn)
    cols.append(np.ones(len(rows)))
    names.append("const")
    for s in sts[1:]:
        cols.append(np.array([1.0 if r["st"] == s else 0.0 for r in rows]))
        names.append(f"st_{s}")
    return np.column_stack(cols), names


def _fit_ci(rows, sts, lam, seed):
    X, names = _design(rows, sts)
    rungs = _rung_names(rows)
    nr = len(rungs)
    z = np.array([float(r["z"]) for r in rows])
    pen = np.array([lam if n in rungs else 0.0 for n in names])
    beta = _ridge_irls(X, z, pen)
    rows_by_day = {}
    for i, r in enumerate(rows):
        rows_by_day.setdefault(r["day"], []).append(i)
    days = sorted(rows_by_day)
    rng = random.Random(seed)
    boot = {n: np.empty(ENC_BOOT) for n in names[:nr]}
    for b in range(ENC_BOOT):
        idx = np.array([i for d in g3._mbb_days(days, rng, C["block_length_days"])
                        for i in rows_by_day.get(d, [])])
        bb = _ridge_irls(X[idx], z[idx], pen)
        for j, n in enumerate(names[:nr]):
            boot[n][b] = bb[j]
        if (b + 1) % 500 == 0:
            print(f"    boot {b+1}/{ENC_BOOT}", flush=True)
    ci = {n: [round(float(np.percentile(boot[n], 2.5)), 4),
              round(float(np.percentile(boot[n], 97.5)), 4)] for n in names[:nr]}
    coef = {n: round(float(beta[j]), 4) for j, n in enumerate(names[:nr])}
    return coef, ci


def _pick_lambda(rows, sts):
    """day-blocked CV deviance over the grid; return the lambda minimizing it."""
    X, names = _design(rows, sts)
    z = np.array([float(r["z"]) for r in rows])
    rungs = _rung_names(rows)
    days = sorted({r["day"] for r in rows})
    folds = 5
    rng = random.Random(12345)
    dd = days[:]
    rng.shuffle(dd)
    assign = {d: i % folds for i, d in enumerate(dd)}
    fold_of = np.array([assign[r["day"]] for r in rows])
    best, bestlam = 1e18, RIDGE_GRID[0]
    for lam in RIDGE_GRID:
        pen = np.array([lam if n in rungs else 0.0 for n in names])
        dev = 0.0
        for f in range(folds):
            tr, te = fold_of != f, fold_of == f
            beta = _ridge_irls(X[tr], z[tr], pen)
            p = 1 / (1 + np.exp(-np.clip(X[te] @ beta, -30, 30)))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            dev += -2 * np.sum(z[te] * np.log(p) + (1 - z[te]) * np.log(1 - p))
        if dev < best:
            best, bestlam = dev, lam
    print(f"  lambda* = {bestlam} (CV deviance {best:.1f})", flush=True)
    return bestlam


def stage_adjudicate():
    rows = [json.loads(l) for l in (RESULTS / "scores_g4.jsonl").read_text().splitlines()]
    if not rows:
        sys.exit("no g4 rows")
    sts = sorted(CITIES)
    n_sd = len({(r["st"], r["day"]) for r in rows})
    lam = _pick_lambda(rows, sts)
    print("full-ladder joint fit...", flush=True)
    coef_all, ci_all = _fit_ci(rows, sts, lam, G4_SEED_ENC)
    # by horizon
    short = [r for r in rows if r["off"] in SHORT_OFFSETS]
    long = [r for r in rows if r["off"] in LONG_OFFSETS]
    coef_s, ci_s = _fit_ci(short, sts, lam, G4_SEED_ENC + 1)
    coef_l, ci_l = _fit_ci(long, sts, lam, G4_SEED_ENC + 2)
    rungs = _rung_names(rows)
    det_set = [n for n in rungs if ci_all[n][0] > 0]
    mkt_short = ci_s["mkt"][0] > 0
    mkt_long = ci_l["mkt"][0] > 0
    if n_sd < C["min_valid_station_days"]:
        verdict = "AUGURY_G4_COLLAPSE"
    elif mkt_short or mkt_long:
        verdict = "AUGURY_G4_MARKET_MEMBER"
    elif ci_all["mkt"][0] <= 0:
        verdict = "AUGURY_G4_MARKET_ENCOMPASSED"
    else:
        verdict = "AUGURY_G4_GAP"
    res = {"verdict": verdict, "n_station_days": n_sd, "n_rows": len(rows),
           "ecmwf_rung": "ecmwf" in rungs, "ladder": rungs,
           "lambda": lam, "determining_set_pooled": det_set,
           "pooled": {"coef": coef_all, "ci95": ci_all},
           "short": {"coef": coef_s, "ci95": ci_s, "market_in_set": mkt_short},
           "long": {"coef": coef_l, "ci95": ci_l, "market_in_set": mkt_long}}
    sha = wjson(RESULTS / "g4_result.json", res)
    print(json.dumps(res, indent=2))
    print(f"g4_result sha256 {sha}", flush=True)


# --------------------------------------------------------------- selftest

def stage_selftest():
    ok = []

    def ck(name, got, want):
        ok.append(got == want)
        if got != want:
            print(f"  FAIL {name}: {got!r} != {want!r}")
    cyc = dt.datetime(2026, 6, 15, 0, tzinfo=dt.timezone.utc)
    bl = _civilday_blocks_gefs(cyc, dt.date(2026, 6, 15))
    ck("gefs.blocks", [f for f, _ in bl], [18, 24])
    st = _civilday_steps_ecmwf(cyc, dt.date(2026, 6, 15))
    ck("ecmwf.steps", [s for s, _ in st], [15, 18, 21, 24])
    ck("members", len(ECMWF_MEMBERS), 21)
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.normal(size=(n, 3))
    z = (rng.random(n) < 1 / (1 + np.exp(-(0.7 * x[:, 0] + 1.1 * x[:, 2])))).astype(float)
    X = np.column_stack([x, np.ones(n)])
    beta = _ridge_irls(X, z, np.zeros(4))
    ck("irls", bool(abs(beta[0] - 0.7) < 0.2 and abs(beta[2] - 1.1) < 0.2), True)
    n_bad = sum(1 for g in ok if not g)
    print(f"g4 selftest: {len(ok)-n_bad}/{len(ok)} passed")
    if n_bad:
        sys.exit(1)


def stage_plan():
    sd = _station_days()
    from collections import Counter
    c = Counter(st for (st, _d) in sd)
    print(json.dumps({"station_days": len(sd), "by_station": dict(sorted(c.items())),
                      "window": f"{WIN0}..{WIN1}"}, indent=2))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["selftest", "plan", "gefs", "ecmwf", "mos",
                                     "score", "adjudicate", "all"])
    p.add_argument("--admitted", action="store_true")
    a = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.stage == "selftest":
        return stage_selftest()
    if a.stage == "plan":
        return stage_plan()
    if not a.admitted:
        sys.exit("binding stage needs --admitted")
    order = [a.stage] if a.stage != "all" else ["gefs", "ecmwf", "mos", "score", "adjudicate"]
    for s in order:
        t0 = time.time()
        {"gefs": stage_gefs, "ecmwf": stage_ecmwf, "mos": stage_mos,
         "score": stage_score, "adjudicate": stage_adjudicate}[s]()
        print(f"[{s}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
