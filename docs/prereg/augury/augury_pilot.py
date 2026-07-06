#!/usr/bin/env python
"""AUGURY G2 pilot runner (KNYC) — docs/prereg/augury/AUGURY_G1_PREREG.md §8.

Stages:
  selftest  offline unit checks (no network)
  smoke     tiny network smokes (Kalshi/IEM/S3/eccodes decode) — non-binding
  tpeak     compute + freeze the t_peak table from CLI 2015-2019 (non-market data)
  seam      series-seam chart, 7 cities (market METADATA only, no prices)
  nbmscan   NBM TMAX cycle-availability map from .idx scans (S3 metadata only)
  pilot     the binding KNYC run (requires --admitted): settlement audit, CDF
            receipts, NBM matched-join demo, bucket calibration, exclusion table
  all       tpeak + seam + nbmscan + pilot

Frozen implementation choices (bind with this file's sha256, recorded in the
G1 Amendment A freeze marker):
  - Constants come from g1_constants.yaml (flat `key: value`); the selftest
    asserts the parsed values match the §1-§8 numbers embedded in EXPECTED.
  - NBM comparator (Amendment A): core TMAX blend mean + "ens std dev" at the
    nearest grid point, 12-hour max window ending 00Z UTC of event-day+1,
    F(θ) = Normal(mu, sigma) survival. qmd is receipted unavailable pre-2023
    (APCP-only); NBP TXN rows are V5-era only. Era-uniform by construction.
  - Availability of a model issue = max(S3 LastModified of .grib2, .idx).
    Nominal cycle time is never used (prereg §2).
  - Market quotes are yes bid/ask CLOSE of the last 1-min candle at or before
    the cutoff; midpoint = (bid+ask)/2 in dollars; spread in cents.
  - Implied exceedance: ladder bins (from floor/cap strike fields) sorted by
    lower edge; P(high > b) = sum of bin midpoints strictly above boundary b;
    monotonized non-increasing by pool-adjacent-violators (unit weights).
  - Strike semantics: two candidate variants (exclusive / inclusive bounds).
    The DISCOVERY sample (seed 20260705, n=50) selects the variant that
    reproduces the API `result`; the AUDIT sample (seed 20260706, n=50,
    disjoint) then verifies the bound variant. 50/50 required.
  - CLI local times -> LST by subtracting 1h inside US DST dates (2007 rule,
    date-level: second Sunday of March through first Sunday of November).
  - t_peak = per station-month MEDIAN of LST minutes of the CLI daily-high
    time over 2015-2019, rounded to the nearest hour.
  - HTTP: >=0.12 s between calls, 4 retries with exponential backoff, cached
    to results/augury/g2-pilot-knyc/cache keyed by sha1(url) (append-only).

Requires: Python 3.10+; numpy NOT required; `eccodes` (pip) only for
smoke/nbm stages. Wrapper: scripts/augury-pilot.mjs (SUNDOG_PYTHON honored).
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results" / "augury" / "g2-pilot-knyc"
CACHE = RESULTS / "cache"
UA = "sundog-augury-g2/0.1 (research; contact via repo)"

MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

# ----------------------------------------------------------------- constants

EXPECTED = {  # selftest cross-check against g1_constants.yaml (prereg §1-§8)
    "max_spread_cents": 4, "max_quote_age_minutes": 60, "min_valid_strikes": 4,
    "min_event_volume_contracts": 250, "block_length_days": 7,
    "bootstrap_resamples": 10000, "min_valid_station_days": 500,
    "settlement_audit_n": 50, "settlement_audit_seed": 20260706,
    "strike_semantics_discovery_seed": 20260705,
    "band_start_offset_hours": -12, "band_end_offset_hours": -2,
    "tpeak_source_start_year": 2015, "tpeak_source_end_year": 2019,
}


def load_constants() -> dict:
    c: dict = {}
    for line in (HERE / "g1_constants.yaml").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip() if not line.lstrip().startswith("#") else ""
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if re.fullmatch(r"-?\d+", v):
            c[k.strip()] = int(v)
        elif re.fullmatch(r"-?\d+\.\d+", v):
            c[k.strip()] = float(v)
        else:
            c[k.strip()] = v
    return c


C = load_constants()

# ----------------------------------------------------------------- http/cache

_last_call = 0.0


def _throttle() -> None:
    global _last_call
    wait = C.get("http_min_interval_seconds", 0.12) - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def http_get(url: str, rng: tuple[int, int] | None = None, ok404: bool = False) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if rng is not None:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1] - 1}")
    for attempt in range(C.get("http_retries", 4) + 1):
        _throttle()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 and ok404:
                return None
            if e.code in (429, 500, 502, 503, 504) and attempt < C.get("http_retries", 4):
                time.sleep(1.5 * 2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < C.get("http_retries", 4):
                time.sleep(1.5 * 2 ** attempt)
                continue
            raise
    return None


def cached_get(url: str, ok404: bool = False) -> bytes | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()
    f = CACHE / key
    miss = CACHE / (key + ".404")
    if f.exists():
        return f.read_bytes()
    if miss.exists():
        return None
    data = http_get(url, ok404=ok404)
    if data is None:
        miss.write_bytes(b"")
        return None
    f.write_bytes(data)
    with (CACHE / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "url": url, "bytes": len(data)}) + "\n")
    return data


def jget(url: str, ok404: bool = False) -> dict | None:
    data = cached_get(url, ok404=ok404)
    return None if data is None else json.loads(data)


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------- kalshi

KB = C["kalshi_base"]


def kalshi_paginate(path: str, params: str) -> list[dict]:
    out, cursor = [], ""
    while True:
        url = f"{KB}{path}?{params}&limit=1000" + (f"&cursor={cursor}" if cursor else "")
        d = jget(url)
        out.extend(d.get("markets", []))
        cursor = d.get("cursor") or ""
        if not cursor:
            return out


def settled_markets(series: str) -> list[dict]:
    live = kalshi_paginate("/markets", f"series_ticker={series}&status=settled")
    hist = kalshi_paginate("/historical/markets", f"series_ticker={series}")
    seen, out = set(), []
    for m in hist + live:
        t = m.get("ticker")
        if t and t not in seen and m.get("result") in ("yes", "no"):
            seen.add(t)
            out.append(m)
    return out


def event_date_of(ticker: str) -> dt.date | None:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(?:-|$)", ticker)
    if not m:
        return None
    return dt.date(2000 + int(m.group(1)), MON[m.group(2)], int(m.group(3)))


def parse_ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def candles_for(market: dict, end_dt: dt.datetime) -> list[dict]:
    """1-min candles from market open to end_dt; historical endpoint first for
    old markets, live fallback (and vice versa). Normalized dicts."""
    t = market["ticker"]
    series = t.split("-")[0]
    start = int(parse_ts(market["open_time"]).timestamp()) - 60
    end = int(min(parse_ts(market["close_time"]), end_dt).timestamp()) + 60
    urls = [
        f"{KB}/historical/markets/{t}/candlesticks?start_ts={start}&end_ts={end}&period_interval=1",
        f"{KB}/series/{series}/markets/{t}/candlesticks?start_ts={start}&end_ts={end}&period_interval=1",
    ]
    if parse_ts(market["close_time"]) > dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=90):
        urls.reverse()
    for u in urls:
        d = jget(u, ok404=True)
        if d and d.get("candlesticks"):
            return [normalize_candle(c) for c in d["candlesticks"]]
    return []


def _fnum(x) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def normalize_candle(c: dict) -> dict:
    def side(name):
        s = c.get(name) or {}
        return _fnum(s.get("close_dollars", s.get("close")))
    return {
        "ts": c["end_period_ts"],
        "bid": side("yes_bid"),
        "ask": side("yes_ask"),
        "vol": _fnum(c.get("volume_fp", c.get("volume"))) or 0.0,
    }


def strike_bounds(m: dict) -> tuple[float | None, float | None, str]:
    """Normalized half-integer thresholds (lo, hi, source): YES iff lo < high < hi
    (None = unbounded). Modern fields: between floor..cap inclusive of both ends;
    greater/less strict (verified: T90 with high 90 settled NO). Legacy markets
    lack the fields; their rules text carries one of three phrasings ("is
    [strictly ]greater than X°" / "is less than X°" / "is between X-Y°" or
    "between X° and Y°"), parsed as the fallback."""
    st = m.get("strike_type")
    fl, cp = _fnum(m.get("floor_strike")), _fnum(m.get("cap_strike"))
    if st == "between" and fl is not None and cp is not None:
        return fl - 0.5, cp + 0.5, "fields"
    if st == "greater" and fl is not None:
        return fl + 0.5, None, "fields"
    if st == "less" and cp is not None:
        return None, cp - 0.5, "fields"
    if st is None and (fl is not None or cp is not None):
        if fl is not None and cp is not None:
            return fl - 0.5, cp + 0.5, "fields"
        if fl is not None:
            return fl + 0.5, None, "fields"
        return None, cp - 0.5, "fields"
    r = m.get("rules_primary") or ""
    mt = re.search(r"is (?:strictly )?greater than (\d+(?:\.\d+)?)", r)
    if mt:
        return float(mt.group(1)) + 0.5, None, "rules"
    mt = re.search(r"is less than (\d+(?:\.\d+)?)", r)
    if mt:
        return None, float(mt.group(1)) - 0.5, "rules"
    mt = re.search(r"is between (\d+(?:\.\d+)?)\s*°?\s*(?:and|-|–|to)\s*(\d+(?:\.\d+)?)", r)
    if mt:
        return float(mt.group(1)) - 0.5, float(mt.group(2)) + 0.5, "rules"
    return None, None, "none"


def outcome_yes(high: float, lo: float | None, hi: float | None) -> bool | None:
    if lo is None and hi is None:
        return None
    return (lo is None or high > lo) and (hi is None or high < hi)


def _two_sided(c: dict) -> bool:
    return (c["bid"] is not None and c["ask"] is not None
            and 0.0 < c["bid"] <= c["ask"] < 1.0)


def book_at(cand: list[dict], ts_list: list[int], ts: int,
            max_age_min: int) -> tuple[dict | None, str]:
    """Standing book at ts (§5). The last candle at-or-before ts is the latest
    known book state: valid if strictly two-sided AND (within the age window,
    OR provably persistent — the next candle carries the identical book,
    sandwiching ts). Returns (candle|None, fail_reason)."""
    j = bisect.bisect_right(ts_list, ts) - 1
    if j < 0:
        return None, "book"
    c = cand[j]
    if not _two_sided(c):
        return None, "book"
    if ts - c["ts"] <= max_age_min * 60:
        return c, ""
    if j + 1 < len(cand) and cand[j + 1]["bid"] == c["bid"] \
            and cand[j + 1]["ask"] == c["ask"]:
        return c, ""
    return None, "stale"


def prep_candles(market: dict, end_dt: dt.datetime):
    cand = sorted(candles_for(market, end_dt), key=lambda c: c["ts"])
    ts_list = [c["ts"] for c in cand]
    vol_pfx, acc = [], 0.0
    for c in cand:
        acc += c["vol"]
        vol_pfx.append(acc)
    return cand, ts_list, vol_pfx


def build_exceedance(bins: list[tuple[float | None, float | None, float]]):
    """bins: (lo, hi, prob). Returns (boundaries, raw, pav) with exceedance
    P(high > b) = sum of probs of bins whose lower edge >= b."""
    bounds = sorted({b for lo, hi, _ in bins for b in (lo, hi) if b is not None})
    raw = []
    for b in bounds:
        raw.append(sum(p for lo, hi, p in bins if (lo if lo is not None else -1e9) >= b - 1e-9))
    return bounds, raw, pav_nonincreasing(raw)


def pav_nonincreasing(vals: list[float]) -> list[float]:
    blocks = [[-v, 1.0] for v in vals]  # negate -> enforce nondecreasing
    out: list[list[float]] = []
    for b in blocks:
        out.append(list(b))
        while len(out) > 1 and out[-2][0] / out[-2][1] > out[-1][0] / out[-1][1]:
            s2, w2 = out.pop()
            out[-1][0] += s2
            out[-1][1] += w2
    res: list[float] = []
    for s, w in out:
        res.extend([-(s / w)] * int(w))
    return res


# ----------------------------------------------------------------- iem / cli

def cli_year(station: str, year: int) -> list[dict]:
    d = jget(f"{C['iem_cli']}?station={station}&year={year}")
    return d.get("results", []) if d else []


def parse_high_time(s) -> int | None:
    if not s or not isinstance(s, str):
        return None
    m = re.fullmatch(r"\s*(\d{1,2})(\d{2})\s*(AM|PM)\s*", s)
    if not m:
        return None
    h, mm, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if h == 12:
        h = 0
    if ap == "PM":
        h += 12
    return h * 60 + mm


def is_us_dst(d: dt.date) -> bool:
    def nth_sunday(y, month, n):
        day = dt.date(y, month, 1)
        day += dt.timedelta(days=(6 - day.weekday()) % 7)
        return day + dt.timedelta(days=7 * (n - 1))
    return nth_sunday(d.year, 3, 2) <= d < nth_sunday(d.year, 11, 1)


def cli_days(station: str, y0: int, y1: int) -> dict[dt.date, dict]:
    out = {}
    for y in range(y0, y1 + 1):
        for row in cli_year(station, y):
            try:
                day = dt.date.fromisoformat(row["valid"])
            except (KeyError, ValueError):
                continue
            high = row.get("high")
            if not isinstance(high, (int, float)):
                continue
            local_min = parse_high_time(row.get("high_time"))
            lst_min = None
            if local_min is not None:
                lst_min = local_min - 60 if is_us_dst(day) else local_min
            out[day] = {"high": float(high), "lst_min": lst_min}
    return out


# ----------------------------------------------------------------- band

def lst_offset(station: str) -> int:
    return int(C[f"lst_offset_{station}"])


def band_cutoffs(station: str, day: dt.date, tpeak_hour: int) -> list[dt.datetime]:
    utc_peak = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc) \
        + dt.timedelta(hours=tpeak_hour - lst_offset(station))
    a, b = C["band_start_offset_hours"], C["band_end_offset_hours"]
    return [utc_peak + dt.timedelta(hours=h) for h in range(a, b + 1)]


# ----------------------------------------------------------------- s3 / nbm

def s3_list(prefix: str, delimiter: str | None = None, max_keys: int = 1000) -> list[dict]:
    url = (f"https://{C['nbm_bucket']}.s3.amazonaws.com/?list-type=2"
           f"&prefix={prefix}&max-keys={max_keys}")
    if delimiter:
        url += f"&delimiter={delimiter}"
    data = cached_get(url)
    root = ET.fromstring(data)
    ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""
    out = []
    for c in root.findall(f"{ns}Contents"):
        out.append({
            "key": c.find(f"{ns}Key").text,
            "last_modified": c.find(f"{ns}LastModified").text,
            "size": int(c.find(f"{ns}Size").text),
        })
    return out


def idx_records(key: str) -> list[dict]:
    data = cached_get(f"https://{C['nbm_bucket']}.s3.amazonaws.com/{key}", ok404=True)
    if data is None:
        return []
    lines = data.decode("utf-8", "replace").strip().splitlines()
    recs = []
    for i, line in enumerate(lines):
        p = line.split(":")
        if len(p) < 6:
            continue
        recs.append({"start": int(p[1]), "var": p[3], "level": p[4],
                     "window": p[5], "extra": ":".join(p[6:]), "end": None})
    for i in range(len(recs) - 1):
        recs[i]["end"] = recs[i + 1]["start"]
    return recs


def tmax_records(cycle: dt.datetime, event_day: dt.date) -> dict | None:
    """core TMAX mean + std-dev records covering event_day's 12h window ending
    00Z of event_day+1. Returns None if this cycle does not carry them."""
    end = dt.datetime(event_day.year, event_day.month, event_day.day,
                      tzinfo=dt.timezone.utc) + dt.timedelta(days=1)
    fhr = int((end - cycle).total_seconds() // 3600)
    if fhr <= 0 or fhr > 200:
        return None
    ymd, hh = cycle.strftime("%Y%m%d"), cycle.strftime("%H")
    key = f"blend.{ymd}/{hh}/core/blend.t{hh}z.core.f{fhr:03d}.co.grib2"
    recs = idx_records(key + ".idx")
    lo, hi = fhr - 12, fhr
    mean = sd = None
    for r in recs:
        if r["var"] == "TMAX" and r["window"] == f"{lo}-{hi} hour max fcst":
            if "std dev" in r["extra"]:
                sd = r
            elif r["extra"].strip(": ") == "":
                mean = r
    if mean is None or sd is None:
        return None
    return {"key": key, "mean": mean, "sd": sd, "fhr": fhr}


def availability_of(key: str) -> dt.datetime | None:
    listed = s3_list(key)
    exact = [x for x in listed if x["key"] in (key, key + ".idx")]
    if not exact:
        return None
    return max(parse_ts(x["last_modified"]) for x in exact)


def latest_tmax_issue(event_day: dt.date, cutoff: dt.datetime) -> dict | None:
    cyc = cutoff.replace(minute=0, second=0, microsecond=0)
    for back in range(0, 48):
        c = cyc - dt.timedelta(hours=back)
        rec = tmax_records(c, event_day)
        if rec is None:
            continue
        avail = availability_of(rec["key"])
        if avail is not None and avail <= cutoff:
            rec.update({"cycle": c.isoformat(), "available": avail.isoformat(),
                        "lag_min": round((cutoff - avail).total_seconds() / 60, 1)})
            return rec
    return None


def decode_point_f(key: str, rec: dict, lat: float, lon: float) -> float:
    import eccodes  # required only on nbm paths
    data = http_get(f"https://{C['nbm_bucket']}.s3.amazonaws.com/{key}",
                    rng=(rec["start"], rec["end"] or rec["start"] + 40_000_000))
    h = eccodes.codes_new_from_message(data)
    try:
        near = eccodes.codes_grib_find_nearest(h, lat, lon)[0]
        kelvin = near.value
    finally:
        eccodes.codes_release(h)
    return kelvin  # caller converts (mean: K->F; sd: K*1.8)


def norm_sf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.5
    return 0.5 * math.erfc((x - mu) / (sigma * math.sqrt(2)))


# ================================================================= stages

def stage_selftest() -> None:
    ok = []

    def check(name, got, want):
        good = got == want
        ok.append((name, good, got, want))
        if not good:
            print(f"  FAIL {name}: got {got!r} want {want!r}")

    check("pav", [round(v, 4) for v in pav_nonincreasing([0.9, 0.95, 0.6, 0.4, 0.45, 0.1])],
          [0.925, 0.925, 0.6, 0.425, 0.425, 0.1])
    bins = [(None, 84.5, 0.1), (84.5, 86.5, 0.25), (86.5, 88.5, 0.4), (88.5, None, 0.3)]
    b, raw, _ = build_exceedance(bins)
    check("exceedance.bounds", b, [84.5, 86.5, 88.5])
    check("exceedance.raw", [round(v, 4) for v in raw], [0.95, 0.7, 0.3])
    briers = [(0.9 - 1) ** 2, (0.6 - 1) ** 2, (0.2 - 0) ** 2]
    check("brier", round(sum(briers) / 3, 4), 0.07)
    check("ht.1219AM", parse_high_time("1219 AM"), 19)
    check("ht.239PM", parse_high_time("239 PM"), 879)
    check("ht.1200PM", parse_high_time("1200 PM"), 720)
    check("dst.jul", is_us_dst(dt.date(2022, 7, 1)), True)
    check("dst.jan", is_us_dst(dt.date(2022, 1, 15)), False)
    check("dst.mar13", is_us_dst(dt.date(2022, 3, 13)), True)
    check("sem.T90.at90", outcome_yes(90, *strike_bounds(
        {"strike_type": "greater", "floor_strike": 90})[:2]), False)
    check("sem.less98.at97", outcome_yes(97, *strike_bounds(
        {"strike_type": "less", "cap_strike": 98})[:2]), True)
    check("sem.between.at104", outcome_yes(104, *strike_bounds(
        {"strike_type": "between", "floor_strike": 104, "cap_strike": 105})[:2]), True)
    check("sem.between.at103", outcome_yes(103, *strike_bounds(
        {"strike_type": "between", "floor_strike": 104, "cap_strike": 105})[:2]), False)
    check("rules.greater", strike_bounds(
        {"rules_primary": "temperature ..., is strictly greater than 50°, then"}),
        (50.5, None, "rules"))
    check("rules.less", strike_bounds(
        {"rules_primary": "Report, is less than 87°, then the market"}),
        (None, 86.5, "rules"))
    check("rules.between.hyphen", strike_bounds(
        {"rules_primary": "Report, is between 89-90°, then the market"}),
        (88.5, 90.5, "rules"))
    check("rules.between.and", strike_bounds(
        {"rules_primary": "Report, is between 85° and 86°, then"}),
        (84.5, 86.5, "rules"))
    _bk = [{"ts": 100, "bid": 0.4, "ask": 0.42, "vol": 1.0},
           {"ts": 8000, "bid": 0.4, "ask": 0.42, "vol": 0.0}]
    check("book.fresh", book_at(_bk, [100, 8000], 200, 60)[0] is not None, True)
    check("book.sandwich", book_at(_bk, [100, 8000], 5000, 60)[0] is not None, True)
    _bk2 = [{"ts": 100, "bid": 0.4, "ask": 0.42, "vol": 1.0},
            {"ts": 8000, "bid": 0.4, "ask": 0.50, "vol": 0.0}]
    check("book.stale", book_at(_bk2, [100, 8000], 5000, 60), (None, "stale"))
    _bk3 = [{"ts": 100, "bid": 0.0, "ask": 0.55, "vol": 1.0}]
    check("book.onesided", book_at(_bk3, [100], 200, 60), (None, "book"))
    check("seam.regex", re.search(
        r"recorded (?:in|at) (.+?),? (?:for|on) ",
        "recorded in Central Park, New York for October 23, 2024 as").group(1),
        "Central Park, New York")
    check("evdate", event_date_of("HIGHNY-22JUL04-T86"), dt.date(2022, 7, 4))
    check("normsf.mid", round(norm_sf(0, 0, 1), 4), 0.5)
    check("normsf.1sd", round(norm_sf(1, 0, 1), 4), 0.1587)
    for k, v in EXPECTED.items():
        check(f"const.{k}", C.get(k), v)
    cuts = band_cutoffs("KNYC", dt.date(2026, 6, 15), 15)  # peak 15 LST -> 20Z
    check("band.n", len(cuts), 11)
    check("band.first", cuts[0].isoformat(), "2026-06-15T08:00:00+00:00")
    check("band.last", cuts[-1].isoformat(), "2026-06-15T18:00:00+00:00")
    n_bad = sum(1 for _, good, *_ in ok if not good)
    receipt = {"checks": len(ok), "failed": n_bad,
               "detail": [{"name": n, "ok": g} for n, g, *_ in ok]}
    write_json(RESULTS / "selftest_receipt.json", receipt)
    print(f"selftest: {len(ok) - n_bad}/{len(ok)} passed")
    if n_bad:
        sys.exit(1)


def stage_smoke() -> None:
    r: dict = {}
    d = jget(f"{KB}/historical/cutoff")
    r["kalshi_cutoff"] = d.get("market_settled_ts")
    m = jget(f"{KB}/markets?series_ticker=KXHIGHNY&status=settled&limit=1")["markets"][0]
    cds = candles_for(m, parse_ts(m["close_time"]))
    r["kalshi_candles"] = {"ticker": m["ticker"], "n": len(cds)}
    assert len(cds) > 200, "candle smoke failed"
    y15 = cli_year("KNYC", 2015)
    r["iem_cli_2015_days"] = len(y15)
    assert len(y15) >= 360, "CLI 2015 coverage failed"
    day = dt.date(2026, 6, 15)
    cutoff = dt.datetime(2026, 6, 15, 16, 0, tzinfo=dt.timezone.utc)
    issue = latest_tmax_issue(day, cutoff)
    assert issue, "no TMAX issue found for smoke day"
    r["nbm_issue"] = {k: issue[k] for k in ("cycle", "available", "lag_min", "fhr", "key")}
    try:
        mu_k = decode_point_f(issue["key"], issue["mean"], C["knyc_lat"], C["knyc_lon"])
        sd_k = decode_point_f(issue["key"], issue["sd"], C["knyc_lat"], C["knyc_lon"])
        mu_f = (mu_k - 273.15) * 1.8 + 32
        sd_f = sd_k * 1.8
        r["nbm_decode"] = {"mu_F": round(mu_f, 2), "sigma_F": round(sd_f, 2)}
        assert 50 < mu_f < 120 and 0 < sd_f < 15, "decoded values implausible"
        r["eccodes"] = "ok"
    except ImportError:
        r["eccodes"] = "MISSING — nbm stages unavailable in this interpreter"
    write_json(RESULTS / "smoke_receipt.json", r)
    print(json.dumps(r, indent=2))


def stage_tpeak() -> None:
    y0, y1 = C["tpeak_source_start_year"], C["tpeak_source_end_year"]
    table: dict = {}
    for st in C["stations"].split(","):
        days = cli_days(st, y0, y1)
        per_month: dict[int, list[int]] = {m: [] for m in range(1, 13)}
        for day, row in days.items():
            if row["lst_min"] is not None:
                per_month[day.month].append(row["lst_min"])
        table[st] = {}
        for m, vals in per_month.items():
            if len(vals) < 20:
                table[st][str(m)] = {"n": len(vals), "hour": None}
                continue
            vals.sort()
            med = statistics.median(vals)
            table[st][str(m)] = {
                "n": len(vals), "median_lst_min": med,
                "hour": int((med + 30) // 60),
                "p25": vals[len(vals) // 4], "p75": vals[3 * len(vals) // 4],
            }
    sha = write_json(RESULTS / "tpeak_table.json", table)
    lines = ["# t_peak table (frozen; CLI 2015-2019 median LST hour of daily high)", ""]
    lines.append("| station | " + " | ".join(str(m) for m in range(1, 13)) + " |")
    lines.append("|---" * 13 + "|")
    for st, months in table.items():
        row = [str(months[str(m)]["hour"]) for m in range(1, 13)]
        lines.append(f"| {st} | " + " | ".join(row) + " |")
    lines += ["", f"sha256(tpeak_table.json) = `{sha}`"]
    (RESULTS / "tpeak_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"tpeak: written, sha256 {sha}")


def stage_seam() -> None:
    cities = {
        "KNYC": ["KXHIGHNY", "HIGHNY"], "KMDW": ["KXHIGHCHI", "HIGHCHI"],
        "KDEN": ["KXHIGHDEN", "HIGHDEN"], "KLAX": ["KXHIGHLAX", "HIGHLAX"],
        "KAUS": ["KXHIGHAUS", "HIGHAUS"], "KMIA": ["KXHIGHMIA", "HIGHMIA"],
        "KPHL": ["KXHIGHPHIL", "HIGHPHIL"],
    }
    def event_exists(series: str, day: dt.date) -> bool:
        t = f"{series}-{day.strftime('%y%b%d').upper()}"
        d = jget(f"{KB}/events/{t}", ok404=True)
        return bool(d and d.get("event"))

    chart: dict = {}
    months = []
    cur = dt.date(2021, 7, 1)
    while cur <= dt.date(2026, 7, 1):
        months.append(cur)
        cur = (cur.replace(day=28) + dt.timedelta(days=6)).replace(day=1)
    for st, series_list in cities.items():
        chart[st] = {}
        for series in series_list:
            present = [d for d in months if event_exists(series, d)]
            if not present:
                chart[st][series] = {"active": False}
                continue
            info: dict = {"active": True, "first_month": present[0].isoformat(),
                          "last_month": present[-1].isoformat(), "stations_by_year": {}}
            for d in present:
                if str(d.year) in info["stations_by_year"]:
                    continue
                ev = f"{series}-{d.strftime('%y%b%d').upper()}"
                ms = (jget(f"{KB}/historical/markets?event_ticker={ev}&limit=3", ok404=True)
                      or {}).get("markets") or \
                     (jget(f"{KB}/markets?event_ticker={ev}&limit=3", ok404=True)
                      or {}).get("markets") or []
                if ms:
                    m = re.search(r"recorded (?:in|at) (.+?),? (?:for|on) ",
                                  ms[0].get("rules_primary", ""))
                    info["stations_by_year"][str(d.year)] = m.group(1) if m else "UNPARSED"
            chart[st][series] = info
    sha = write_json(RESULTS / "seam_chart.json", chart)
    print(f"seam: written, sha256 {sha}")


def stage_nbmscan() -> None:
    days = []
    for y in range(2022, 2027):
        for m in (1, 4, 7, 10):
            d = dt.date(y, m, 1)
            if dt.date(2022, 1, 1) <= d <= dt.date(2026, 6, 15):
                days.append(d)
    days.append(dt.date(2026, 6, 15))
    out: dict = {}
    for day in days:
        row = {}
        for hh in range(24):
            cyc = dt.datetime(day.year, day.month, day.day, hh, tzinfo=dt.timezone.utc)
            rec = tmax_records(cyc, day)
            if rec:
                avail = availability_of(rec["key"])
                lag = None
                if avail:
                    lag = round((avail - cyc).total_seconds() / 60, 1)
                row[f"{hh:02d}"] = {"fhr": rec["fhr"], "issue_lag_min": lag}
        out[day.isoformat()] = row
    sha = write_json(RESULTS / "nbm_cycle_map.json", out)
    print(f"nbmscan: written, sha256 {sha}")


def _nyc_markets() -> list[dict]:
    ms = settled_markets(C["series_current_KNYC"]) + settled_markets(C["series_legacy_KNYC"])
    w0 = dt.date.fromisoformat(C["primary_window_start"])
    w1 = dt.date.fromisoformat(C["primary_window_end"])
    out = []
    for m in ms:
        d = event_date_of(m["ticker"])
        if d and w0 <= d <= w1:
            m["_event_date"] = d.isoformat()
            out.append(m)
    out.sort(key=lambda m: m["ticker"])
    return out


def _audit_one(m: dict, cli: dict) -> dict:
    d = dt.date.fromisoformat(m["_event_date"])
    row = cli.get(d)
    lo, hi, src = strike_bounds(m)
    rec = {"ticker": m["ticker"], "event_date": m["_event_date"],
           "lo": lo, "hi": hi, "bounds_source": src, "api_result": m["result"],
           "expiration_value": m.get("expiration_value")}
    if row is None:
        rec["status"] = "NO_CLI_DAY"
        return rec
    high = row["high"]
    rec["cli_high"] = high
    ev = _fnum(m.get("expiration_value") or None)  # '' on legacy markets = absent
    rec["expiration_status"] = ("absent" if ev is None else
                                "match" if abs(ev - high) < 0.01 else "MISMATCH")
    pred = outcome_yes(high, lo, hi)
    rec["predicted"] = None if pred is None else ("yes" if pred else "no")
    rec["match"] = rec["predicted"] == m["result"]
    rec["status"] = "OK"
    return rec


def stage_pilot(admitted: bool) -> None:
    if not admitted:
        sys.exit("pilot is a binding stage: pass --admitted (see G1 Amendment A)")
    tp_path = RESULTS / "tpeak_table.json"
    if not tp_path.exists():
        sys.exit("tpeak_table.json missing — run tpeak first (freeze order, prereg §3)")
    tpeak = json.loads(tp_path.read_text(encoding="utf-8"))
    print("tpeak sha256:", hashlib.sha256(tp_path.read_text(encoding='utf-8').encode()).hexdigest())
    summary: dict = {}

    markets = _nyc_markets()
    print(f"settled NYC markets in window: {len(markets)}")
    summary["n_markets"] = len(markets)
    y0 = dt.date.fromisoformat(C["primary_window_start"]).year
    y1 = dt.date.fromisoformat(C["primary_window_end"]).year
    cli = cli_days("KNYC", y0, y1)

    # -- artifact 2a: semantics validation sample (seed 20260705)
    rng = random.Random(C["strike_semantics_discovery_seed"])
    disc = rng.sample(markets, min(C["settlement_audit_n"], len(markets)))
    disc_rows = [_audit_one(m, cli) for m in disc]
    disc_ok = sum(1 for r in disc_rows if r.get("match"))
    disc_tickers = {m["ticker"] for m in disc}
    # -- artifact 2b: settlement audit (seed 20260706, disjoint)
    rng2 = random.Random(C["settlement_audit_seed"])
    pool = [m for m in markets if m["ticker"] not in disc_tickers]
    audit = rng2.sample(pool, min(C["settlement_audit_n"], len(pool)))
    rows = [_audit_one(m, cli) for m in audit]
    n_ok = sum(1 for r in rows if r.get("match"))
    exp_counts = {}
    for r in disc_rows + rows:
        s = r.get("expiration_status")
        if s:
            exp_counts[s] = exp_counts.get(s, 0) + 1
    write_json(RESULTS / "settlement_audit.json", {
        "validation_sample": {"n": len(disc_rows), "matches": disc_ok},
        "audit_n": len(rows), "result_matches": n_ok,
        "expiration_status_counts": exp_counts,
        "validation_rows": disc_rows, "rows": rows})
    summary["settlement_audit"] = f"audit {n_ok}/{len(rows)}, validation {disc_ok}/{len(disc_rows)}, exp {exp_counts}"
    print(f"settlement audit: {n_ok}/{len(rows)} (validation {disc_ok}/{len(disc_rows)})")

    # -- group markets by event day
    by_day: dict[str, list[dict]] = {}
    for m in markets:
        by_day.setdefault(m["_event_date"], []).append(m)

    def day_hour(day: dt.date) -> int | None:
        e = tpeak["KNYC"].get(str(day.month), {})
        return e.get("hour")

    # -- artifacts 6+7: bucket calibration + exclusions (the heavy pass)
    max_age = C["max_quote_age_minutes"]
    buckets = [[0, 0] for _ in range(10)]
    excl: dict[str, dict] = {}
    mismatch: dict[str, list[int]] = {}
    n_days = len(by_day)
    for i, (dstr, dms) in enumerate(sorted(by_day.items())):
        day = dt.date.fromisoformat(dstr)
        era = str(day.year) if day.year < 2026 else "2026H1"
        E = excl.setdefault(era, {"days": 0, "cutoffs": 0, "valid_cutoffs": 0,
                                  "strikes": 0, "valid_strikes": 0,
                                  "fail_spread": 0, "fail_stale": 0, "fail_book": 0})
        M = mismatch.setdefault(era, [0, 0])
        h = day_hour(day)
        if h is None:
            continue
        E["days"] += 1
        row = cli.get(day)
        if row and row["lst_min"] is not None:
            M[1] += 1
            in_window = 7 * 60 <= row["lst_min"] <= 19 * 60  # 12Z-00Z at KNYC = 07-19 LST
            if not in_window:
                M[0] += 1
        cuts = band_cutoffs("KNYC", day, h)
        prep = {m["ticker"]: prep_candles(m, cuts[-1] + dt.timedelta(hours=1))
                for m in dms}
        first_valid_mid: dict[str, float] = {}
        for T in cuts:
            ts = int(T.timestamp())
            E["cutoffs"] += 1
            n_valid, vol = 0, 0.0
            for m in dms:
                E["strikes"] += 1
                cand, ts_list, vol_pfx = prep[m["ticker"]]
                j = bisect.bisect_right(ts_list, ts) - 1
                if j >= 0:
                    vol += vol_pfx[j]
                q, why = book_at(cand, ts_list, ts, max_age)
                if q is None:
                    E["fail_stale" if why == "stale" else "fail_book"] += 1
                    continue
                spread_c = (q["ask"] - q["bid"]) * 100
                if spread_c > C["max_spread_cents"]:
                    E["fail_spread"] += 1
                    continue
                n_valid += 1
                E["valid_strikes"] += 1
                if m["ticker"] not in first_valid_mid:
                    first_valid_mid[m["ticker"]] = (q["bid"] + q["ask"]) / 2
            if n_valid >= C["min_valid_strikes"] and vol >= C["min_event_volume_contracts"]:
                E["valid_cutoffs"] += 1
        for m in dms:
            p = first_valid_mid.get(m["ticker"])
            if p is None:
                continue
            b = min(int(p * 10), 9)
            buckets[b][0] += 1
            buckets[b][1] += 1 if m["result"] == "yes" else 0
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{n_days} days")

    top_n, top_y = buckets[9]
    top_rate = 100 * top_y / top_n if top_n else None
    calib = {"buckets": [{"range": f"{b * 10}-{b * 10 + 10}%", "n": n,
                          "yes_rate_pct": round(100 * y / n, 2) if n else None}
                         for b, (n, y) in enumerate(buckets)],
             "top_bucket_rate_pct": round(top_rate, 2) if top_rate else None,
             "top_bucket_n": top_n,
             "pass": (top_rate is not None and
                      C["calibration_top_bucket_low_pct"] <= top_rate
                      <= C["calibration_top_bucket_high_pct"])}
    write_json(RESULTS / "bucket_calibration.json", calib)
    summary["bucket_calibration"] = f"top bucket {calib['top_bucket_rate_pct']}% " \
                                    f"(n={top_n}) pass={calib['pass']}"
    for era, E in excl.items():
        E["strike_valid_rate"] = round(E["valid_strikes"] / E["strikes"], 4) if E["strikes"] else None
        E["cutoff_valid_rate"] = round(E["valid_cutoffs"] / E["cutoffs"], 4) if E["cutoffs"] else None
    write_json(RESULTS / "exclusion_table.json",
               {"eras": excl,
                "maxt_window_mismatch": {e: {"outside": m[0], "with_time": m[1]}
                                         for e, m in mismatch.items()}})
    summary["exclusions"] = {e: E["cutoff_valid_rate"] for e, E in excl.items()}

    # -- artifact 4: CDF receipts on pinned days
    receipts = []
    for dstr in C["cdf_receipt_days"].split(","):
        day = dt.date.fromisoformat(dstr)
        dms = by_day.get(dstr, [])
        h = day_hour(day)
        if not dms or h is None:
            receipts.append({"day": dstr, "status": "NO_MARKETS"})
            continue
        T = band_cutoffs("KNYC", day, h)[6]  # mid-band (t_peak - 6h)
        ts = int(T.timestamp())
        bins = []
        for m in dms:
            cand, ts_list, _ = prep_candles(m, T + dt.timedelta(hours=1))
            q, _why = book_at(cand, ts_list, ts, max_age)
            if q is None or (q["ask"] - q["bid"]) * 100 > C["max_spread_cents"]:
                continue
            lo, hi, _src = strike_bounds(m)
            if lo is None and hi is None:
                continue
            bins.append((lo, hi, (q["bid"] + q["ask"]) / 2))
        if len(bins) < C["min_valid_strikes"]:
            receipts.append({"day": dstr, "status": "TOO_FEW_VALID", "n": len(bins)})
            continue
        bounds, raw, mono = build_exceedance(bins)
        receipts.append({"day": dstr, "cutoff": T.isoformat(), "status": "OK",
                         "boundaries": bounds,
                         "raw": [round(v, 4) for v in raw],
                         "isotonic": [round(v, 4) for v in mono],
                         "violations": sum(1 for a, b in zip(raw, raw[1:]) if b > a + 1e-9)})
    write_json(RESULTS / "cdf_receipts.json", receipts)
    summary["cdf_receipts"] = f"{sum(1 for r in receipts if r['status'] == 'OK')}/{len(receipts)} OK"

    # -- artifact 5: NBM matched-join demo
    demo = []
    for dstr in C["nbm_join_demo_days"].split(","):
        day = dt.date.fromisoformat(dstr)
        h = day_hour(day)
        if h is None:
            continue
        for T in band_cutoffs("KNYC", day, h)[::5]:  # 3 of 11 cutoffs
            issue = latest_tmax_issue(day, T)
            if issue is None:
                demo.append({"day": dstr, "cutoff": T.isoformat(), "status": "NO_ISSUE"})
                continue
            try:
                mu_f = (decode_point_f(issue["key"], issue["mean"],
                                       C["knyc_lat"], C["knyc_lon"]) - 273.15) * 1.8 + 32
                sd_f = decode_point_f(issue["key"], issue["sd"],
                                      C["knyc_lat"], C["knyc_lon"]) * 1.8
                demo.append({"day": dstr, "cutoff": T.isoformat(), "status": "OK",
                             "cycle": issue["cycle"], "available": issue["available"],
                             "lag_min": issue["lag_min"],
                             "mu_F": round(mu_f, 2), "sigma_F": round(sd_f, 2)})
            except ImportError:
                demo.append({"day": dstr, "cutoff": T.isoformat(),
                             "status": "ECCODES_MISSING"})
    write_json(RESULTS / "nbm_join_demo.json", demo)
    summary["nbm_join"] = f"{sum(1 for r in demo if r['status'] == 'OK')}/{len(demo)} OK"

    sha = write_json(RESULTS / "pilot_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"pilot summary sha256 {sha}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["selftest", "smoke", "tpeak", "seam",
                                      "nbmscan", "pilot", "all"])
    ap.add_argument("--admitted", action="store_true",
                    help="required for the binding pilot stage")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.stage in ("selftest",):
        stage_selftest()
    elif a.stage == "smoke":
        stage_smoke()
    elif a.stage == "tpeak":
        stage_tpeak()
    elif a.stage == "seam":
        stage_seam()
    elif a.stage == "nbmscan":
        stage_nbmscan()
    elif a.stage == "pilot":
        stage_pilot(a.admitted)
    elif a.stage == "all":
        stage_tpeak()
        stage_seam()
        stage_nbmscan()
        stage_pilot(a.admitted)


if __name__ == "__main__":
    main()
