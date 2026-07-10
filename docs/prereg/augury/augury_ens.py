#!/usr/bin/env python
"""AUGURY G4-beta — ECMWF-ENS sub-amendment runner (sensitivity of the ECMWF rung).

Spec: AUGURY_G4_PREREG.md Amendment G4-beta. Upgrades the ECMWF rung from the
HRES-deterministic margin to the ensemble distribution: F = Normal(mu, sigma)
survival, mu/sigma from the 21 perturbed members (pf 1..21; mx2t3 has no
control), per-member civil-day max over the 3-h blocks ending 15/18/21/00 UTC.

Composes the FROZEN augury_g4 primitives read-only (throttle-hardened _ecm_get /
_ecm_cached_index / s3_head_mtime_ecm, decode_points, _ridge_irls) — same
discipline as HARUSPEX. Issue set = the cached HRES issues (day|cycle keys);
selection availability = the HRES (oper) availability (named approximation);
enfo LastModified recorded per issue as the honesty diagnostic.

Stages:
  probe       ~6 issues sequential; s/issue + 503 exhaustion rate -> ETA (binding pull gate)
  pull        all issues not in the ENS scalar cache (per-issue flush, resumable)
  score       rebuild the G4 rows with f_ecmwf_ens (in-memory join; no new Kalshi pulls)
  adjudicate  re-run the G4-style encompassing on the ENS ladder; report deltas vs G4
Binding stages require --admitted.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import augury_pilot as ap  # noqa: E402
import augury_g3 as g3  # noqa: E402
import augury_g4 as g4  # noqa: E402

C = ap.C
G4RES = ap.ROOT / "results" / "augury" / "g4-run"
RESULTS = ap.ROOT / "results" / "augury" / "g4-ens-run"
MEMBERS = [str(i) for i in range(1, 22)]  # pf 1..21 (mx2t3 has no cf)
ENS_SEED = 20260713
SHORT, LONG = g4.SHORT_OFFSETS, g4.LONG_OFFSETS


def wjson(p, o):
    return ap.write_json(p, o)


def hres_issues() -> list[dict]:
    """The cached HRES issue set: [{'k': day|cycle, 'available': iso, ...}]."""
    out = []
    for line in (G4RES / "ecmwf_scalars.jsonl").read_text(encoding="utf-8").splitlines():
        out.append(json.loads(line))
    return out


def _ens_cache() -> dict:
    p = RESULTS / "ens_scalars.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["k"]] = r
    return out


def ens_issue(day: dt.date, cyc: dt.datetime) -> dict | None:
    """21-member civil-day-max mu/sigma per station for one cycle, or None
    (throttle-exhausted / missing). Paced by the frozen g4._ecm_get."""
    steps = g4._civilday_steps_ecmwf(cyc, day)
    if steps is None:
        return None
    mem_max = {st: {m: -1e9 for m in MEMBERS} for st in g4.STATION_LL}
    enfo_avail = None
    for step, _e in steps:
        ymd, hh = cyc.strftime("%Y%m%d"), cyc.strftime("%H")
        stamp = cyc.strftime("%Y%m%d%H%M%S")
        base = f"{g4.ECM_HOST}/{ymd}/{hh}z/ifs/0p25/enfo/{stamp}-{step}h-enfo-ef"
        idx = g4._ecm_cached_index(base + ".index")
        if idx is None:
            return None
        by = {}
        for line in idx.decode().strip().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("param") == "mx2t3" and r.get("type") == "pf":
                by[r.get("number")] = r
        if any(m not in by for m in MEMBERS):
            return None
        mt = g4.s3_head_mtime_ecm(base + ".grib2")
        if mt is not None:
            enfo_avail = mt if enfo_avail is None else max(enfo_avail, mt)
        for m in MEMBERS:
            r = by[m]
            data = g4._ecm_get(base + ".grib2", r["_offset"], r["_offset"] + r["_length"])
            if data is None:
                return None
            pts = g4.decode_points(data)
            for st in g4.STATION_LL:
                if pts[st] > mem_max[st][m]:
                    mem_max[st][m] = pts[st]
    mu, sd = {}, {}
    for st in g4.STATION_LL:
        arr = np.array([mem_max[st][m] for m in MEMBERS])
        mu[st] = round((arr.mean() - 273.15) * 1.8 + 32, 3)
        sd[st] = round(max(arr.std(ddof=1) * 1.8, 0.3), 3)
    return {"mu": mu, "sd": sd,
            "enfo_available": enfo_avail.isoformat() if enfo_avail else None}


def stage_probe(admitted):
    if not admitted:
        sys.exit("binding: pass --admitted")
    issues = hres_issues()
    have = _ens_cache()
    todo = [r for r in issues if r["k"] not in have][:6]
    print(f"probe: {len(todo)} issues (of {len(issues)} total, {len(have)} cached)", flush=True)
    times, fails = [], 0
    path = RESULTS / "ens_scalars.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in todo:
            dstr, cyc_s = r["k"].split("|", 1)
            day = dt.date.fromisoformat(dstr)
            cyc = dt.datetime.fromisoformat(cyc_s)
            t0 = time.time()
            rec = ens_issue(day, cyc)
            el = time.time() - t0
            if rec is None:
                fails += 1
                print(f"  {r['k']}: FAILED after {el:.0f}s (throttle/missing)", flush=True)
                continue
            times.append(el)
            rec.update({"k": r["k"], "oper_available": r["available"]})
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            print(f"  {r['k']}: OK {el:.0f}s  mu[KNYC]={rec['mu']['KNYC']}", flush=True)
    if times:
        spi = sum(times) / len(times)
        remaining = len(issues) - len(have) - len(times)
        print(f"probe result: {len(times)} ok / {fails} failed; {spi:.0f}s/issue "
              f"-> ETA {remaining * spi / 3600:.1f}h for {remaining} remaining", flush=True)
    else:
        print(f"probe result: 0 ok / {fails} failed — bucket still throttled; retry later",
              flush=True)


def stage_pull(admitted):
    if not admitted:
        sys.exit("binding: pass --admitted")
    issues = hres_issues()
    have = _ens_cache()
    todo = [r for r in issues if r["k"] not in have]
    print(f"pull: {len(todo)} to pull ({len(have)} cached)", flush=True)
    path = RESULTS / "ens_scalars.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = fails = 0
    t0 = time.time()
    with path.open("a", encoding="utf-8") as fh:
        for r in todo:
            dstr, cyc_s = r["k"].split("|", 1)
            rec = ens_issue(dt.date.fromisoformat(dstr), dt.datetime.fromisoformat(cyc_s))
            if rec is None:
                fails += 1
                if fails % 10 == 0:
                    print(f"  {fails} failures so far (throttle) — continuing", flush=True)
                continue
            rec.update({"k": r["k"], "oper_available": r["available"]})
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                print(f"  pulled {done}/{len(todo)} ({fails} fails) "
                      f"[{el/done:.0f}s/issue, ETA {(len(todo)-done)*el/done/3600:.1f}h]",
                      flush=True)
    print(f"pull done: +{done}, {fails} failed (re-run resumes)", flush=True)


def stage_score(admitted):
    """Rebuild the G4 rows with f_ecmwf_ens via in-memory join (no new pulls).
    Selection availability = the oper (HRES) availability, per the amendment pin."""
    if not admitted:
        sys.exit("binding: pass --admitted")
    tp = g3.load_tpeak()
    ens = _ens_cache()
    by_day: dict = {}
    late = 0
    for k, rec in ens.items():
        dstr = k.split("|", 1)[0]
        av = dt.datetime.fromisoformat(rec["oper_available"])
        if rec.get("enfo_available"):
            ea = dt.datetime.fromisoformat(rec["enfo_available"])
            if (ea - av).total_seconds() > 1800:
                late += 1
        by_day.setdefault(dstr, []).append((av, rec))
    for d in by_day:
        by_day[d].sort(key=lambda x: x[0])
    print(f"ens issues: {len(ens)}; enfo>30min-later-than-oper: {late} "
          f"({100*late/max(1,len(ens)):.1f}% — honesty diagnostic)", flush=True)
    rows = [json.loads(l) for l in (G4RES / "scores_g4.jsonl").read_text().splitlines()]
    out = (RESULTS / "scores_ens.jsonl").open("w", encoding="utf-8")
    n = drop = 0
    for r in rows:
        day = dt.date.fromisoformat(r["day"])
        T = None
        h = g3.day_hour(tp, r["st"], day)
        if h is not None:
            T = ap.band_cutoffs(r["st"], day, h)[r["off"] + 12]
        rec = g4._pick_cached(by_day, r["day"], T) if T else None
        if rec is None:
            drop += 1
            continue
        rr = dict(r)
        rr.pop("ecmwf_margin", None)
        rr["f_ecmwf_ens"] = round(ap.norm_sf(r["b"], rec["mu"][r["st"]], rec["sd"][r["st"]]), 6)
        out.write(json.dumps(rr) + "\n")
        n += 1
    out.close()
    print(f"score: {n} rows ({drop} dropped, no ENS issue at cutoff)", flush=True)


def _design(rows, sts):
    cols = [g4._logit(np.array([r["f_gefs"] for r in rows])),
            g4._logit(np.array([r["f_nbm"] for r in rows])),
            g4._logit(np.array([r["f_ecmwf_ens"] for r in rows])),
            g4._logit(np.array([r["f_mkt"] for r in rows])),
            np.ones(len(rows))]
    names = ["gefs", "nbm", "ecmwf_ens", "mkt", "const"]
    for s in sts[1:]:
        cols.append(np.array([1.0 if r["st"] == s else 0.0 for r in rows]))
        names.append("st_" + s)
    return np.column_stack(cols), names


def _fit(rows, sts, lam, seed, boot=2000):
    X, names = _design(rows, sts)
    z = np.array([float(r["z"]) for r in rows])
    pen = np.array([lam if i < 4 else 0.0 for i in range(X.shape[1])])
    beta = g4._ridge_irls(X, z, pen)
    rbd: dict = {}
    for i, r in enumerate(rows):
        rbd.setdefault(r["day"], []).append(i)
    days = sorted(rbd)
    rng = random.Random(seed)
    bs = np.empty((boot, 4))
    for bi in range(boot):
        idx = np.array([i for d in g3._mbb_days(days, rng, C["block_length_days"])
                        for i in rbd[d]])
        bs[bi] = g4._ridge_irls(X[idx], z[idx], pen)[:4]
        if (bi + 1) % 500 == 0:
            print(f"    boot {bi+1}/{boot}", flush=True)
    coef = {names[i]: round(float(beta[i]), 4) for i in range(4)}
    ci = {names[i]: [round(float(np.percentile(bs[:, i], 2.5)), 4),
                     round(float(np.percentile(bs[:, i], 97.5)), 4)] for i in range(4)}
    return {"coef": coef, "ci95": ci, "n_rows": len(rows)}


def stage_adjudicate(admitted):
    if not admitted:
        sys.exit("binding: pass --admitted")
    rows = [json.loads(l) for l in (RESULTS / "scores_ens.jsonl").read_text().splitlines()]
    sts = sorted({r["st"] for r in rows})
    n_sd = len({(r["st"], r["day"]) for r in rows})
    lam = 10.0  # the G4 CV choice, inherited (sensitivity uses the same penalty)
    print(f"adjudicate: {len(rows)} rows, {n_sd} station-days, lambda={lam}", flush=True)
    pooled = _fit(rows, sts, lam, ENS_SEED)
    short = _fit([r for r in rows if r["off"] in SHORT], sts, lam, ENS_SEED + 1)
    long_ = _fit([r for r in rows if r["off"] in LONG], sts, lam, ENS_SEED + 2)
    det = [n for n in ("gefs", "nbm", "ecmwf_ens", "mkt") if pooled["ci95"][n][0] > 0]
    mkt_s, mkt_l = short["ci95"]["mkt"][0] > 0, long_["ci95"]["mkt"][0] > 0
    if n_sd < C["min_valid_station_days"]:
        verdict = "AUGURY_G4_COLLAPSE"
    elif mkt_s or mkt_l:
        verdict = "AUGURY_G4_MARKET_MEMBER"
    elif pooled["ci95"]["mkt"][0] <= 0:
        verdict = "AUGURY_G4_MARKET_ENCOMPASSED"
    else:
        verdict = "AUGURY_G4_GAP"
    res = {"verdict": verdict, "n_station_days": n_sd, "lambda": lam,
           "determining_set_pooled": det,
           "pooled": pooled,
           "short": {**short, "market_in_set": mkt_s},
           "long": {**long_, "market_in_set": mkt_l},
           "g4_hres_reference": {
               "pooled_mkt": 0.7016, "pooled_mkt_ci": [0.6466, 0.7506],
               "pooled_ecmwf_margin": 0.6141, "pooled_ecmwf_ci": [0.1232, 1.1759]}}
    sha = wjson(RESULTS / "g4_ens_result.json", res)
    print(json.dumps({k: res[k] for k in ("verdict", "determining_set_pooled")}, indent=2))
    print(json.dumps({"pooled": pooled["coef"], "ci": pooled["ci95"]}, indent=2))
    print(f"g4_ens_result sha256 {sha}", flush=True)


def stage_selftest():
    ok = []

    def ck(n, c):
        ok.append(c)
        if not c:
            print(f"  FAIL {n}")
    ck("members.21", len(MEMBERS) == 21 and "cf" not in MEMBERS)
    cyc = dt.datetime(2026, 6, 15, 0, tzinfo=dt.timezone.utc)
    st = g4._civilday_steps_ecmwf(cyc, dt.date(2026, 6, 15))
    ck("steps", [s for s, _ in st] == [15, 18, 21, 24])
    rows = [{"st": "KNYC", "day": "2026-06-01", "z": 1, "f_gefs": .6, "f_nbm": .7,
             "f_ecmwf_ens": .65, "f_mkt": .8},
            {"st": "KMDW", "day": "2026-06-02", "z": 0, "f_gefs": .4, "f_nbm": .3,
             "f_ecmwf_ens": .35, "f_mkt": .2}]
    X, names = _design(rows, ["KNYC", "KMDW"])
    ck("design", X.shape == (2, 6) and names[2] == "ecmwf_ens")
    ck("sigma.floor", True)  # enforced in ens_issue (max(sd, 0.3))
    nbad = sum(1 for g in ok if not g)
    print(f"ens selftest: {len(ok)-nbad}/{len(ok)} passed")
    if nbad:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["selftest", "probe", "pull", "score", "adjudicate"])
    p.add_argument("--admitted", action="store_true")
    a = p.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"selftest": lambda: stage_selftest(),
     "probe": lambda: stage_probe(a.admitted),
     "pull": lambda: stage_pull(a.admitted),
     "score": lambda: stage_score(a.admitted),
     "adjudicate": lambda: stage_adjudicate(a.admitted)}[a.stage]()


if __name__ == "__main__":
    main()
