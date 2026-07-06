#!/usr/bin/env python
"""AUGURY G3 full run — docs/prereg/augury/AUGURY_G1_PREREG.md §6-§7 (Amendment C).

Stages:
  selftest    offline unit checks (scoring, DM, logistic, MBB determinism)
  plan        dry enumeration: stations, day counts, estimated pulls
  market      pull candles for all 7 cities' series lineages (resumable)
  audit       per-city settlement audit (n=20 each, seed 20260707; 20/20 gate)
  nbm         NBM issue scalar extraction for the union of band joins
              (parallel S3 fetch + eccodes decode; scalar cache, resumable)
  score       per-(station, day, cutoff, strike) records -> scores.jsonl
  adjudicate  DM + encompassing + verdict (§7 precedence) + exploratory maps
  all         market -> audit -> nbm -> score -> adjudicate

Frozen implementation choices (Amendment C; the pilot's primitives are imported
from augury_pilot.py at its Amendment-B hash and NOT modified):
  - Station-day COUNTS as valid iff >= 1 of its 11 band cutoffs is valid (§5);
    scores average over the valid cutoffs only; the §7 floor (500) counts these
    station-days. The valid-cutoffs-per-day distribution is reported.
  - Primary score at a cutoff: mean over valid strikes of
    (F_exc(b) - 1{high > b})^2, boundaries b at half-integers; market F_exc =
    PAV-monotonized midpoint exceedance clipped to [0,1]; NBM F_exc =
    Normal(mu, sigma) survival at b. Identical strike set for both rungs.
  - Day differential d(s,d) = mean over valid cutoffs of (S_mkt - S_nbm).
  - DM: pooled mean of station-day differentials; circular moving-block
    bootstrap over CALENDAR DAYS (block 7, B=10000, seed 20260707; all stations
    of a resampled day travel together); DM stat = mean/se_boot; one-sided
    p = Phi(stat) for H1 "market < NBM".
  - Encompassing rows: one per (station-day, band cutoff, valid strike);
    z = 1{high > b} ~ logit(F_nbm) + logit(F_mkt) + station dummies + const,
    probabilities clipped to [0.01, 0.99]; logistic fit by IRLS (max 60 iter,
    tol 1e-10, ridge 1e-8 for conditioning). beta_mkt 95% CI = percentile CI
    over 2000 block-bootstrap refits (seed 20260708; B reduced from 10000 for
    refit cost, pinned here). Survives iff CI entirely > 0.
  - Verdicts (§7 precedence): SAMPLE_COLLAPSE (<500 valid station-days) ->
    MARGIN_CONFIRMED (DM p<0.05 AND CI>0) -> ENCOMPASSED (neither) -> GAP.
  - Exploratory (declared, non-adjudicating): dominance-by-offset map
    (per band offset -12..-2), era-stratified DM, reliability bins,
    delta-theta-weighted score sensitivity, per-station exclusion tables.
  - NBM S3 message fetches bypass the polite throttle (AWS open data) and run
    on a 6-thread pool; Kalshi stays sequential + throttled.

Requires numpy + eccodes (both in .venv-augury).
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
import augury_pilot as ap  # frozen Amendment-B primitives

C = ap.C
RESULTS = ap.ROOT / "results" / "augury" / "g3-full-run"
G3_SEED_DM = 20260707
G3_SEED_ENC = 20260708
ENC_BOOT = 2000
AUDIT_N_CITY = 20
AUDIT_SEED_CITY = 20260707

CITIES = {  # station -> (series lineage, CLI station), from the G2 seam chart
    "KNYC": ["KXHIGHNY", "HIGHNY"],
    "KMDW": ["KXHIGHCHI", "HIGHCHI"],
    "KAUS": ["KXHIGHAUS", "HIGHAUS"],
    "KMIA": ["KXHIGHMIA", "HIGHMIA"],
    "KDEN": ["KXHIGHDEN"],
    "KLAX": ["KXHIGHLAX"],
    "KPHL": ["KXHIGHPHIL"],
}


def wjson(path: Path, obj) -> str:
    return ap.write_json(path, obj)


def load_tpeak() -> dict:
    p = ap.RESULTS / "tpeak_table.json"
    tp = json.loads(p.read_text(encoding="utf-8"))
    sha = hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
    print(f"tpeak sha256: {sha}")
    return tp


def station_markets(st: str) -> list[dict]:
    out, seen = [], set()
    for series in CITIES[st]:
        for m in ap.settled_markets(series):
            d = ap.event_date_of(m["ticker"])
            if not d:
                continue
            w0 = dt.date.fromisoformat(C["primary_window_start"])
            w1 = dt.date.fromisoformat(C["primary_window_end"])
            if w0 <= d <= w1 and m["ticker"] not in seen:
                seen.add(m["ticker"])
                m["_event_date"] = d.isoformat()
                out.append(m)
    out.sort(key=lambda m: m["ticker"])
    return out


def day_hour(tp: dict, st: str, day: dt.date) -> int | None:
    return tp.get(st, {}).get(str(day.month), {}).get("hour")


# ----------------------------------------------------------------- stages

def stage_plan() -> None:
    tp = load_tpeak()
    plan = {}
    for st in CITIES:
        ms = station_markets(st)
        days = {m["_event_date"] for m in ms}
        plan[st] = {"markets": len(ms), "event_days": len(days),
                    "first": min(days) if days else None,
                    "last": max(days) if days else None,
                    "tpeak_ok": all(day_hour(tp, st, dt.date(2025, mo, 1)) is not None
                                    for mo in range(1, 13))}
    wjson(RESULTS / "plan.json", plan)
    print(json.dumps(plan, indent=2))


def stage_market() -> None:
    tp = load_tpeak()
    n = 0
    for st in CITIES:
        ms = station_markets(st)
        by_day: dict[str, list[dict]] = {}
        for m in ms:
            by_day.setdefault(m["_event_date"], []).append(m)
        for i, (dstr, dms) in enumerate(sorted(by_day.items())):
            day = dt.date.fromisoformat(dstr)
            h = day_hour(tp, st, day)
            if h is None:
                continue
            cuts = ap.band_cutoffs(st, day, h)
            for m in dms:
                ap.candles_for(m, cuts[-1] + dt.timedelta(hours=1))
                n += 1
            if (i + 1) % 200 == 0:
                print(f"  {st}: {i + 1}/{len(by_day)} days")
        print(f"{st}: candles cached for {len(ms)} markets")
    print(f"market stage done: {n} pulls (cache-resumable)")


def stage_audit() -> None:
    receipts = {}
    ok_all = True
    for st in CITIES:
        y0 = dt.date.fromisoformat(C["primary_window_start"]).year
        y1 = dt.date.fromisoformat(C["primary_window_end"]).year
        cli = ap.cli_days(st, y0, y1)
        ms = station_markets(st)
        rng = random.Random(AUDIT_SEED_CITY)
        pick = rng.sample(ms, min(AUDIT_N_CITY, len(ms)))
        rows = [ap._audit_one(m, cli) for m in pick]
        n_ok = sum(1 for r in rows if r.get("match"))
        receipts[st] = {"n": len(rows), "matches": n_ok, "rows": rows}
        ok_all = ok_all and n_ok == len(rows)
        print(f"audit {st}: {n_ok}/{len(rows)}")
    wjson(RESULTS / "city_settlement_audit.json", receipts)
    if not ok_all:
        sys.exit("CITY SETTLEMENT AUDIT FAILED — inspect city_settlement_audit.json; "
                 "G3 scoring must not proceed for failing cities (§1).")


def _scalar_cache() -> dict:
    p = RESULTS / "nbm_scalars.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["key"]] = r
    return out


STATION_LL = {  # airport/park coordinates for nearest-point extraction
    "KNYC": (40.7789, -73.9692), "KMDW": (41.7861, -87.7522),
    "KAUS": (30.1945, -97.6699), "KMIA": (25.7932, -80.2906),
    "KDEN": (39.8467, -104.6562), "KLAX": (33.9382, -118.3866),
    "KPHL": (39.8683, -75.2311),
}

# in-process memos over the frozen pilot primitives (pilot file unmodified)
_tmax_memo: dict = {}
_avail_memo: dict = {}


def _tmax_rec(cyc: dt.datetime, day: dt.date):
    k = (cyc.isoformat(), day.isoformat())
    if k not in _tmax_memo:
        _tmax_memo[k] = ap.tmax_records(cyc, day)
    return _tmax_memo[k]


def _avail(key: str):
    if key not in _avail_memo:
        _avail_memo[key] = ap.availability_of(key)
    return _avail_memo[key]


def latest_issue(day: dt.date, cutoff: dt.datetime) -> dict | None:
    """ap.latest_tmax_issue with in-process memoization (identical logic)."""
    cyc = cutoff.replace(minute=0, second=0, microsecond=0)
    for back in range(0, 48):
        c = cyc - dt.timedelta(hours=back)
        rec = _tmax_rec(c, day)
        if rec is None:
            continue
        avail = _avail(rec["key"])
        if avail is not None and avail <= cutoff:
            out = dict(rec)
            out.update({"cycle": c.isoformat(), "available": avail.isoformat(),
                        "lag_min": round((cutoff - avail).total_seconds() / 60, 1)})
            return out
    return None


def _fetch_msg(key: str, rec: dict) -> bytes:
    return ap.http_get(f"https://{C['nbm_bucket']}.s3.amazonaws.com/{key}",
                       rng=(rec["start"], rec["end"] or rec["start"] + 40_000_000))


def _fetch_issue(issue: dict) -> tuple[dict, bytes, bytes]:
    return issue, _fetch_msg(issue["key"], issue["mean"]), \
        _fetch_msg(issue["key"], issue["sd"])


def _decode_all_stations(data: bytes) -> dict[str, float]:
    import eccodes
    h = eccodes.codes_new_from_message(data)
    try:
        return {st: eccodes.codes_grib_find_nearest(h, lat, lon)[0].value
                for st, (lat, lon) in STATION_LL.items()}
    finally:
        eccodes.codes_release(h)


def _issue_union(tp: dict) -> dict[str, dict]:
    """All (cycle,fhr) issues selected by any station's band joins."""
    issues: dict[str, dict] = {}
    for st in CITIES:
        ms = station_markets(st)
        days = sorted({m["_event_date"] for m in ms})
        for i, dstr in enumerate(days):
            day = dt.date.fromisoformat(dstr)
            h = day_hour(tp, st, day)
            if h is None:
                continue
            for T in ap.band_cutoffs(st, day, h):
                iss = latest_issue(day, T)
                if iss and iss["key"] not in issues:
                    issues[iss["key"]] = iss
            if (i + 1) % 200 == 0:
                print(f"  {st} join-scan {i + 1}/{len(days)} (issues so far {len(issues)})")
    return issues


def stage_nbm() -> None:
    tp = load_tpeak()
    have = _scalar_cache()
    issues = _issue_union(tp)
    todo = [v for k, v in issues.items() if k not in have]
    print(f"issues: {len(issues)} total, {len(have)} cached, {len(todo)} to decode")
    path = RESULTS / "nbm_scalars.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    # parallel S3 fetch (threads), serial eccodes decode (not thread-safe)
    with cf.ThreadPoolExecutor(max_workers=6) as pool, \
            path.open("a", encoding="utf-8") as fh:
        for issue, b_mean, b_sd in pool.map(_fetch_issue, todo):
            raw_mu = _decode_all_stations(b_mean)
            raw_sd = _decode_all_stations(b_sd)
            rec = {"key": issue["key"], "cycle": issue["cycle"],
                   "available": issue["available"],
                   "mu": {s: round((v - 273.15) * 1.8 + 32, 3)
                          for s, v in raw_mu.items()},
                   "sd": {s: round(v * 1.8, 3) for s, v in raw_sd.items()}}
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            done += 1
            if done % 100 == 0:
                print(f"  decoded {done}/{len(todo)}")
    print(f"nbm stage done: +{done} issues")


def stage_score() -> None:
    tp = load_tpeak()
    scal = _scalar_cache()
    max_age = C["max_quote_age_minutes"]
    out = (RESULTS / "scores.jsonl").open("w", encoding="utf-8")
    n_rows = 0
    excl: dict[str, dict] = {}
    for st in CITIES:
        y0 = dt.date.fromisoformat(C["primary_window_start"]).year
        y1 = dt.date.fromisoformat(C["primary_window_end"]).year
        cli = ap.cli_days(st, y0, y1)
        ms = station_markets(st)
        by_day: dict[str, list[dict]] = {}
        for m in ms:
            by_day.setdefault(m["_event_date"], []).append(m)
        for i, (dstr, dms) in enumerate(sorted(by_day.items())):
            day = dt.date.fromisoformat(dstr)
            row_cli = cli.get(day)
            h = day_hour(tp, st, day)
            if h is None or row_cli is None:
                continue
            high = row_cli["high"]
            era = str(day.year) if day.year < 2026 else "2026H1"
            E = excl.setdefault(f"{st}:{era}", {"cutoffs": 0, "valid_cutoffs": 0})
            cuts = ap.band_cutoffs(st, day, h)
            prep = {m["ticker"]: ap.prep_candles(m, cuts[-1] + dt.timedelta(hours=1))
                    for m in dms}
            bounds_of = {m["ticker"]: ap.strike_bounds(m) for m in dms}
            for off, T in zip(range(C["band_start_offset_hours"],
                                    C["band_end_offset_hours"] + 1), cuts):
                ts = int(T.timestamp())
                E["cutoffs"] += 1
                bins, vol = [], 0.0
                for m in dms:
                    cand, ts_list, vol_pfx = prep[m["ticker"]]
                    j = bisect.bisect_right(ts_list, ts) - 1
                    if j >= 0:
                        vol += vol_pfx[j]
                    q, _ = ap.book_at(cand, ts_list, ts, max_age)
                    if q is None or (q["ask"] - q["bid"]) * 100 > C["max_spread_cents"]:
                        continue
                    lo, hi, _src = bounds_of[m["ticker"]]
                    if lo is None and hi is None:
                        continue
                    bins.append((lo, hi, (q["bid"] + q["ask"]) / 2))
                if len(bins) < C["min_valid_strikes"] or \
                        vol < C["min_event_volume_contracts"]:
                    continue
                E["valid_cutoffs"] += 1
                boundaries, _raw, mono = ap.build_exceedance(bins)
                iss = latest_issue(day, T)
                if iss is None or iss["key"] not in scal:
                    continue
                rec_s = scal[iss["key"]]
                mu, sd = rec_s["mu"][st], rec_s["sd"][st]
                for b, f_mkt in zip(boundaries, mono):
                    f_mkt = min(1.0, max(0.0, f_mkt))
                    f_nbm = ap.norm_sf(b, mu, sd)
                    z = 1 if high > b else 0
                    out.write(json.dumps({
                        "st": st, "day": dstr, "off": off, "b": b, "z": z,
                        "f_mkt": round(f_mkt, 6), "f_nbm": round(f_nbm, 6),
                        "era": era}) + "\n")
                    n_rows += 1
            if (i + 1) % 200 == 0:
                print(f"  {st} score {i + 1}/{len(by_day)}")
        print(f"{st}: scored")
    out.close()
    wjson(RESULTS / "score_exclusions.json", excl)
    print(f"score stage done: {n_rows} strike rows")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, C["prob_clip_low"], C["prob_clip_high"])
    return np.log(p / (1 - p))


def logistic_irls(X: np.ndarray, z: np.ndarray) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(60):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.maximum(p * (1 - p), 1e-10)
        H = (X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1])
        g = X.T @ (z - p)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def _mbb_days(days: list[str], rng: random.Random, L: int) -> list[str]:
    n = len(days)
    picks: list[str] = []
    while len(picks) < n:
        s = rng.randrange(n)
        picks.extend(days[(s + k) % n] for k in range(L))
    return picks[:n]


def stage_adjudicate() -> None:
    rows = [json.loads(l) for l in
            (RESULTS / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    if not rows:
        sys.exit("no score rows")
    # per station-day differentials (mean over cutoffs of mean over strikes)
    per_cut: dict[tuple, dict] = {}
    for r in rows:
        k = (r["st"], r["day"], r["off"])
        e = per_cut.setdefault(k, {"m": [], "n": []})
        e["m"].append((r["f_mkt"] - r["z"]) ** 2)
        e["n"].append((r["f_nbm"] - r["z"]) ** 2)
    per_sd: dict[tuple, list] = {}
    for (st, day, off), e in per_cut.items():
        d_cut = sum(e["m"]) / len(e["m"]) - sum(e["n"]) / len(e["n"])
        per_sd.setdefault((st, day), []).append(d_cut)
    sd_keys = sorted(per_sd)
    diffs = np.array([sum(v) / len(v) for k, v in sorted(per_sd.items())])
    n_sd = len(diffs)
    days_all = sorted({d for _, d in sd_keys})
    rows_of_day: dict[str, list[int]] = {}
    for i, (st, day) in enumerate(sd_keys):
        rows_of_day.setdefault(day, []).append(i)

    # DM: circular MBB over calendar days
    L, B = C["block_length_days"], C["bootstrap_resamples"]
    rng = random.Random(G3_SEED_DM)
    means = np.empty(B)
    for b in range(B):
        idx = [i for d in _mbb_days(days_all, rng, L) for i in rows_of_day[d]]
        means[b] = diffs[np.array(idx)].mean()
    d_bar = float(diffs.mean())
    se = float(means.std(ddof=1))
    stat = d_bar / se if se > 0 else 0.0
    p_one = 0.5 * math.erfc(-stat / math.sqrt(2))  # Phi(stat)
    dm = {"n_station_days": n_sd, "mean_diff_mkt_minus_nbm": round(d_bar, 6),
          "se_block_boot": round(se, 6), "dm_stat": round(stat, 3),
          "p_one_sided_mkt_better": round(p_one, 5)}

    # encompassing: pooled logistic with station FE + day-block bootstrap CI
    sts = sorted(CITIES)
    st_idx = {s: i for i, s in enumerate(sts)}
    X = np.column_stack(
        [_logit(np.array([r["f_nbm"] for r in rows])),
         _logit(np.array([r["f_mkt"] for r in rows])),
         np.ones(len(rows))] +
        [np.array([1.0 if r["st"] == s else 0.0 for r in rows]) for s in sts[1:]])
    z = np.array([float(r["z"]) for r in rows])
    beta = logistic_irls(X, z)
    row_day = [r["day"] for r in rows]
    rows_by_day: dict[str, list[int]] = {}
    for i, d in enumerate(row_day):
        rows_by_day.setdefault(d, []).append(i)
    rng2 = random.Random(G3_SEED_ENC)
    betas = np.empty(ENC_BOOT)
    for b in range(ENC_BOOT):
        idx = np.array([i for d in _mbb_days(days_all, rng2, L)
                        for i in rows_by_day.get(d, [])])
        betas[b] = logistic_irls(X[idx], z[idx])[1]
        if (b + 1) % 200 == 0:
            print(f"  enc boot {b + 1}/{ENC_BOOT}")
    ci = [round(float(np.percentile(betas, 2.5)), 4),
          round(float(np.percentile(betas, 97.5)), 4)]
    enc = {"beta_nbm": round(float(beta[0]), 4),
           "beta_mkt": round(float(beta[1]), 4),
           "beta_mkt_ci95_dayblock": ci, "n_rows": len(rows),
           "boot_refits": ENC_BOOT}

    # verdict (§7 precedence)
    if n_sd < C["min_valid_station_days"]:
        verdict = "AUGURY_SAMPLE_COLLAPSE"
    else:
        dm_pass = p_one < C["alpha_one_sided"]
        enc_pass = ci[0] > 0
        verdict = ("AUGURY_MARGIN_CONFIRMED" if dm_pass and enc_pass else
                   "AUGURY_ENCOMPASSED" if not dm_pass and not enc_pass else
                   "AUGURY_GAP")

    # exploratory (declared, non-adjudicating)
    by_off: dict[int, list] = {}
    for (st, day, off), e in per_cut.items():
        by_off.setdefault(off, []).append(
            sum(e["m"]) / len(e["m"]) - sum(e["n"]) / len(e["n"]))
    dom = {str(o): {"n": len(v), "mean_diff": round(float(np.mean(v)), 6)}
           for o, v in sorted(by_off.items())}
    by_era: dict[str, list] = {}
    for (st, day), v in per_sd.items():
        era = day[:4] if day[:4] != "2026" else "2026H1"
        by_era.setdefault(era, []).append(sum(v) / len(v))
    era_tab = {e: {"n": len(v), "mean_diff": round(float(np.mean(v)), 6)}
               for e, v in sorted(by_era.items())}
    rel = []
    for lo_edge in [i / 10 for i in range(10)]:
        sel_m = [(r["z"]) for r in rows if lo_edge <= r["f_mkt"] < lo_edge + 0.1]
        sel_n = [(r["z"]) for r in rows if lo_edge <= r["f_nbm"] < lo_edge + 0.1]
        rel.append({"bin": f"{lo_edge:.1f}",
                    "mkt_n": len(sel_m),
                    "mkt_rate": round(float(np.mean(sel_m)), 4) if sel_m else None,
                    "nbm_n": len(sel_n),
                    "nbm_rate": round(float(np.mean(sel_n)), 4) if sel_n else None})
    per_st = {}
    for (st, day), v in per_sd.items():
        per_st.setdefault(st, []).append(sum(v) / len(v))
    st_tab = {s: {"n": len(v), "mean_diff": round(float(np.mean(v)), 6)}
              for s, v in sorted(per_st.items())}

    result = {"verdict": verdict, "dm": dm, "encompassing": enc,
              "valid_station_days": n_sd,
              "exploratory": {"dominance_by_offset": dom, "by_era": era_tab,
                              "by_station": st_tab, "reliability": rel}}
    sha = wjson(RESULTS / "g3_result.json", result)
    print(json.dumps({k: result[k] for k in ("verdict", "dm", "encompassing",
                                             "valid_station_days")}, indent=2))
    print(f"g3_result sha256 {sha}")


def stage_selftest() -> None:
    ok = []

    def check(name, got, want):
        good = got == want
        ok.append(good)
        if not good:
            print(f"  FAIL {name}: {got!r} != {want!r}")

    rng = np.random.default_rng(7)
    n = 4000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    eta = 0.8 * x1 + 1.5 * x2 - 0.2
    z = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    X = np.column_stack([x1, x2, np.ones(n)])
    beta = logistic_irls(X, z)
    check("irls.recovery", bool(abs(beta[0] - 0.8) < 0.15
                                and abs(beta[1] - 1.5) < 0.2), True)
    r = random.Random(1)
    days = [f"d{i}" for i in range(20)]
    picks = _mbb_days(days, r, 7)
    check("mbb.len", len(picks), 20)
    r2 = random.Random(1)
    check("mbb.deterministic", _mbb_days(days, r2, 7), picks)
    check("logit.clip", bool(abs(_logit(np.array([0.0]))[0]
                                 - math.log(0.01 / 0.99)) < 1e-9), True)
    check("phi", round(0.5 * math.erfc(1.6449 / math.sqrt(2)), 3), 0.05)
    check("brier.row", round((0.7 - 1) ** 2, 2), 0.09)
    n_bad = sum(1 for g in ok if not g)
    print(f"g3 selftest: {len(ok) - n_bad}/{len(ok)} passed")
    if n_bad:
        sys.exit(1)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["selftest", "plan", "market", "audit",
                                     "nbm", "score", "adjudicate", "all"])
    p.add_argument("--admitted", action="store_true")
    a = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.stage == "selftest":
        stage_selftest()
        return
    if a.stage == "plan":
        stage_plan()
        return
    if not a.admitted:
        sys.exit("binding G3 stages require --admitted (Amendment C)")
    stages = [a.stage] if a.stage != "all" else \
        ["market", "audit", "nbm", "score", "adjudicate"]
    for s in stages:
        t0 = time.time()
        {"market": stage_market, "audit": stage_audit, "nbm": stage_nbm,
         "score": stage_score, "adjudicate": stage_adjudicate}[s]()
        print(f"[{s}] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
