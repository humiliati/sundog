#!/usr/bin/env python
"""A1 Leg 0 (Slate 3, S3-A1) — the INTERNAL KILL GATE for the Atlas-walls-vs-Taivaanvahti test.

A frozen analysis pipeline is validated on SYNTHETIC citizen halo-report populations carrying the full
selection stack (Finnish-municipality latitudes, diurnal/seasonal effort bias, +-30-min time smear,
misclassification, habit-mix climatology drift): it must RECOVER blind-injected walls (P1 edge at
{28.0, 32.196, 36.0} deg; P2 metamorphosis midpoint at {26.7, 29.71, 32.7} deg) and must NOT hallucinate
them in wall-free controls — before ANY live data is touched. Pre-registration (frozen first):
docs/atlas/A1_TAIVAANVAHTI_PREREG.md. Kill gates K0a/K0b/K0c/K0-P2 + ABORT-McD per its section 3.

The estimator takes NO Atlas constant as input (wall-value-BLIND); the only bias correction permitted is
a single constant learned identically across all three injected positions (blindness-preserving rule).

NOT public-eligible. Attribution: McDowell 1979 (sole precedent); NOAA/Meeus solar position; the Atlas
phase-diagram lane (atlas_forward_sweep / atlas_bifurcation_set / atlas_orientation_boundaries).
Run: python scripts/atlas_report_edge_pipeline.py          (first run builds the ~21-min flux cache)
"""
import sys
import os
from pathlib import Path
import numpy as np
from scipy import optimize, interpolate

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_SEED = 20260612
CACHE = Path(__file__).resolve().parent / "_cache_a1_cza_flux.npz"
WALL = 32.196                     # derived CZA wall (used ONLY for injection truth / verdicts, never by the estimator)
P1_INJECT = (28.0, 32.196, 36.0)
P2_INJECT = (26.7, 29.71, 32.7)
NDAYS = 5114                      # 2012-01-01 .. 2025-12-31 (2012-01-01 is a Sunday)
H_BINS = np.arange(5.0, 46.0, 1.0)

# pinned municipality table (lat, lon, weight) — prereg s2(d)
MUNI = np.array([
    [60.17, 24.94, 0.20], [60.21, 24.66, 0.09], [61.50, 23.76, 0.08], [60.29, 25.04, 0.08],
    [65.01, 25.47, 0.07], [60.45, 22.27, 0.07], [62.24, 25.75, 0.05], [60.98, 25.66, 0.04],
    [62.89, 27.68, 0.04], [61.49, 21.80, 0.03], [62.60, 29.76, 0.03], [66.50, 25.73, 0.03],
    [63.00, 26.00, 0.19]])
M_LAT, M_LON, M_W = MUNI[:, 0], MUNI[:, 1], MUNI[:, 2]


# ---- solar position (NOAA/Meeus low-precision; GEOMETRIC elevation, no refraction) ----------------- #
def _solar(lat_deg, lon_deg, day_idx, utc_hour):
    """Vectorized solar elevation (deg) + declination (deg) + EoT (min). day_idx counts from 2012-01-01;
    n = days since J2000.0 (2000-01-01 12:00 UTC): JD(2012-01-01 00:00) - 2451545.0 = 4382.5."""
    n = 4382.5 + np.asarray(day_idx, float) + np.asarray(utc_hour, float) / 24.0
    L = (280.460 + 0.9856474 * n) % 360.0
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 4e-7 * n)
    decl = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    eot = 4.0 * (((L - np.degrees(ra)) + 180.0) % 360.0 - 180.0)          # minutes
    gmst = (280.46061837 + 360.98564736629 * n) % 360.0
    H = np.radians(((gmst + np.asarray(lon_deg, float) - np.degrees(ra)) + 180.0) % 360.0 - 180.0)
    la = np.radians(np.asarray(lat_deg, float))
    elev = np.degrees(np.arcsin(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(H)))
    return elev, np.degrees(decl), eot


def solar_elevation(lat_deg, lon_deg, day_idx, utc_hour):
    return _solar(lat_deg, lon_deg, day_idx, utc_hour)[0]


def almanac_checks(verbose=True):
    """Prereg s2(a) pins: exact identities + almanac ranges. Returns ok, rows."""
    rows, ok = [], True
    d_solstice = (np.datetime64("2025-06-21") - np.datetime64("2012-01-01")).astype(int)
    d_equinox = (np.datetime64("2025-03-20") - np.datetime64("2012-01-01")).astype(int)
    _, decl_s, _ = _solar(60.0, 25.0, d_solstice, 2.7)
    g1 = 23.42 <= float(decl_s) <= 23.45
    rows.append(("declination @ June-2025 solstice in [23.42, 23.45]", g1, f"{float(decl_s):.4f}"))
    _, decl_e, _ = _solar(60.0, 25.0, d_equinox, 12.0)
    g2 = abs(float(decl_e)) < 0.5
    rows.append(("|declination| < 0.5 on 2025-03-20", g2, f"{float(decl_e):.4f}"))
    worst = 0.0
    for city, la, lo in (("Helsinki", 60.17, 24.94), ("Rovaniemi", 66.50, 25.73)):
        for dt in ("2025-06-21", "2025-03-20", "2025-09-23", "2025-12-21"):
            d = (np.datetime64(dt) - np.datetime64("2012-01-01")).astype(int)
            hrs = np.arange(9.0, 15.0, 1.0 / 60.0)
            el, dc, _ = _solar(la, lo, d, hrs)
            i = int(np.argmax(el))
            worst = max(worst, abs(float(el[i]) - (90.0 - la + float(dc[i]))))
    g3 = worst < 0.05
    rows.append(("noon identity |max elev - (90 - lat + decl)| < 0.05 (2 cities x 4 dates)", g3, f"worst {worst:.4f}"))
    feb = [float(_solar(60.0, 25.0, (np.datetime64(f"2025-02-{dd:02d}") - np.datetime64("2012-01-01")).astype(int), 12.0)[2]) for dd in range(5, 21)]
    nov = [float(_solar(60.0, 25.0, (np.datetime64(d0) - np.datetime64("2012-01-01")).astype(int), 12.0)[2]) for d0 in
           [f"2025-10-{dd:02d}" for dd in range(25, 32)] + [f"2025-11-{dd:02d}" for dd in range(1, 11)]]
    g4 = -15.5 <= min(feb) <= -13.5 and 15.5 <= max(nov) <= 17.5
    rows.append(("EoT extremes: mid-Feb in [-15.5,-13.5], early-Nov in [15.5,17.5] min", g4, f"{min(feb):.2f}, {max(nov):.2f}"))
    d = (np.datetime64("2025-06-21") - np.datetime64("2012-01-01")).astype(int)
    el = solar_elevation(59.81, 23.0, d, np.arange(9.0, 15.0, 1.0 / 60.0))
    g5 = abs(float(np.max(el)) - 53.6) <= 0.3
    rows.append(("max Finnish solar elevation (Hanko, solstice) = 53.6 +- 0.3", g5, f"{float(np.max(el)):.2f}"))
    for nm, g, det in rows:
        ok &= g
        if verbose:
            print(f"    [{'PASS' if g else 'FAIL'}] {nm}  ({det})")
    return ok, rows


# ---- the frozen flux curve (cached; prereg s2(b)) -------------------------------------------------- #
FLUX_GRID = np.concatenate([np.arange(3.0, 28.5, 1.0), np.arange(28.5, 35.6, 0.5), np.arange(36.0, 40.1, 1.0)])


def flux_curve():
    if CACHE.exists():
        z = np.load(CACHE)
        return z["e"], z["f"]
    import atlas_orientation_boundaries as ob
    print(f"    (building the frozen flux cache: {len(FLUX_GRID)} seeded raytracer calls, ~20 min)")
    f = np.array([ob.cza_flux(float(e)) for e in FLUX_GRID])
    np.savez(CACHE, e=FLUX_GRID, f=f)
    return FLUX_GRID, f


_PHI = None


def phi(h, wall=WALL):
    """Continuous flux curve, h-TRANSLATED so its derived wall sits at `wall` (prereg s2(c))."""
    global _PHI
    if _PHI is None:
        e, f = flux_curve()
        _PHI = interpolate.PchipInterpolator(e, np.maximum(f, 0.0), extrapolate=False)
    hs = np.asarray(h, float) - (wall - WALL)
    out = _PHI(np.clip(hs, FLUX_GRID[0], FLUX_GRID[-1]))
    out = np.where((hs < FLUX_GRID[0]) | (hs > 35.5), 0.0, np.nan_to_num(out))
    return np.maximum(out, 0.0)


PHI_MAX = None


# ---- synthetic report generator (prereg s2(d)) ----------------------------------------------------- #
def _candidates(rng, m):
    mu = rng.choice(len(M_W), size=m, p=M_W)
    day = rng.integers(0, NDAYS, size=m)
    hloc = rng.uniform(6.0, 20.0, size=m)
    wkd = (day + 6) % 7                                     # 2012-01-01 Sunday -> weekday 6 (Mon=0)
    w_eff = np.exp(-((hloc - 13.0) / 4.0) ** 2) * np.where(wkd >= 5, 1.4, 1.0)
    h_true = solar_elevation(M_LAT[mu], M_LON[mu], day, hloc - 2.0)
    ok = h_true > 2.0
    return mu, day, hloc, np.where(ok, w_eff, 0.0), h_true


def _recorded(rng, mu, day, hloc):
    sm = rng.uniform(-0.5, 0.5, size=len(hloc))             # +-30 min, recorded clock
    return solar_elevation(M_LAT[mu], M_LON[mu], day, hloc - 2.0 + sm)


def _draw(rng, weights, n):
    s = weights.sum()
    if s <= 0:
        return np.zeros(0, int)
    return rng.choice(len(weights), size=n, p=weights / s)


def gen_p1(rng, w_inj, gamma=1.0, eps=0.05, n_cza=1000, ring_mult=3, drift=0.0, wall_free=False):
    """Recorded-h samples: (h_cza_labeled, h_ring). drift = +-0.3 linear plate-fraction trend; wall_free
    replaces the flux factor by 1 (the no-wall control class)."""
    global PHI_MAX
    if PHI_MAX is None:
        PHI_MAX = float(np.max(phi(np.arange(3, 36, 0.1))))
    n_true = int(round(n_cza * (1.0 - eps))); n_cont = n_cza - n_true
    mu, day, hloc, w_eff, h_true = _candidates(rng, 8 * max(n_true, 1))
    if wall_free:
        w = w_eff
    else:
        w = w_eff * (phi(h_true, w_inj) / PHI_MAX) ** gamma
    if drift != 0.0:
        w = w * np.clip(1.0 + drift * (h_true - 23.5) / 21.5, 0.1, None)
    pick = _draw(rng, w, n_true)
    h_cza = _recorded(rng, mu[pick], day[pick], hloc[pick])
    mu2, day2, hloc2, w2, _ = _candidates(rng, 3 * (ring_mult * n_cza + n_cont))
    pick2 = _draw(rng, w2, ring_mult * n_cza + n_cont)
    h_other = _recorded(rng, mu2[pick2], day2[pick2], hloc2[pick2])
    h_ring, h_cont = h_other[:ring_mult * n_cza], h_other[ring_mult * n_cza:]
    return np.concatenate([h_cza, h_cont]), h_ring


def gen_p2(rng, m_inj, w_t=1.5, eps_m=0.05, n_rep=2000, no_transition=False):
    """Column-display reports with circumscribed-vs-tangent labels from TRUE h; recorded h smeared."""
    mu, day, hloc, w_eff, h_true = _candidates(rng, 4 * n_rep)
    pick = _draw(rng, w_eff, n_rep)
    p_circ = (np.full(n_rep, 0.5) if no_transition else
              eps_m / 2.0 + (1.0 - eps_m) / (1.0 + np.exp(-(h_true[pick] - m_inj) / w_t)))
    lab = rng.random(n_rep) < p_circ
    h_rec = _recorded(rng, mu[pick], day[pick], hloc[pick])
    return h_rec, lab


# ---- frozen estimators (NO Atlas constant enters; prereg s2(f)/(i)) -------------------------------- #
def _bin_counts(h_a, h_b):
    ka, _ = np.histogram(h_a, H_BINS); kb, _ = np.histogram(h_b, H_BINS)
    return ka.astype(float), kb.astype(float), 0.5 * (H_BINS[:-1] + H_BINS[1:])


def _ll_binom(k, n, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))


def fit_p1(h_cza, h_ring):
    """Scaled-logistic binomial MLE: p(h) = a * expit((h0-h)/w). Returns (h0, a, w, LR_vs_flat)."""
    k, m, c = _bin_counts(h_cza, h_ring)
    n = k + m
    use = n > 0
    k, n, c = k[use], n[use], c[use]

    def nll(t):
        a, h0, w = t
        return -_ll_binom(k, n, a / (1.0 + np.exp((c - h0) / w)))
    best = None
    for h0_start in (20.0, 28.0, 36.0):
        r = optimize.minimize(nll, x0=np.array([0.4, h0_start, 1.5]), method="L-BFGS-B",
                              bounds=[(0.01, 0.99), (10.0, 45.0), (0.15, 6.0)])
        if best is None or r.fun < best.fun:
            best = r
    a_flat = float(k.sum() / n.sum())
    ll_flat = _ll_binom(k, n, np.full_like(c, a_flat))
    return float(best.x[1]), float(best.x[0]), float(best.x[2]), float(2.0 * (-best.fun - ll_flat))


def fit_p2(h_rec, lab):
    """3-param binomial MLE: f(h) = e/2 + (1-e)*expit((h-m)/w). Returns (m, w, e, LR_vs_flat)."""
    k, m_, c = _bin_counts(h_rec[lab], h_rec[~lab])
    n = k + m_
    use = n > 0
    k, n, c = k[use], n[use], c[use]

    def nll(t):
        mm, w, e = t
        return -_ll_binom(k, n, e / 2.0 + (1.0 - e) / (1.0 + np.exp(-(c - mm) / w)))
    best = None
    for m_start in (24.0, 30.0, 34.0):
        r = optimize.minimize(nll, x0=np.array([m_start, 1.5, 0.05]), method="L-BFGS-B",
                              bounds=[(15.0, 45.0), (0.2, 6.0), (0.0, 0.4)])
        if best is None or r.fun < best.fun:
            best = r
    a_flat = float(k.sum() / n.sum())
    ll_flat = _ll_binom(k, n, np.full_like(c, a_flat))
    return float(best.x[0]), float(best.x[1]), float(best.x[2]), float(2.0 * (-best.fun - ll_flat))


def isotonic_maxdrop(h_cza, h_ring):
    """Secondary robustness column: PAV decreasing fit to the per-bin ratio; changepoint = max drop."""
    k, m, c = _bin_counts(h_cza, h_ring)
    n = k + m
    use = n > 0
    r, w, cc = (k[use] / n[use]), n[use], c[use]
    y, wt = list(-r), list(w)                               # PAV increasing on -r = decreasing on r
    blocks = [[y[i] * wt[i], wt[i], i, i] for i in range(len(y))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] / out[-2][1] > out[-1][0] / out[-1][1]:
            s2 = out.pop(); s1 = out.pop()
            out.append([s1[0] + s2[0], s1[1] + s2[1], s1[2], s2[3]])
    fit = np.empty(len(y))
    for s, wsum, i0, i1 in out:
        fit[i0:i1 + 1] = -(s / wsum)
    d = -np.diff(fit)
    return float(cc[int(np.argmax(d))] + 0.5) if len(d) and np.max(d) > 0 else float("nan")


def _rng(*branch):
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *branch]))


# ---- Leg-0 driver ---------------------------------------------------------------------------------- #
def main(n_rep=200):
    print("=" * 100)
    print("A1 Leg 0 — synthetic kill gate for the Taivaanvahti phase-boundary pipeline")
    print("  prereg: docs/atlas/A1_TAIVAANVAHTI_PREREG.md (frozen 2026-06-12, amendment A1 logged)")
    print("=" * 100)

    print("\n[APPARATUS] solar-position identity/almanac pins:")
    ok_alm, _ = almanac_checks()
    if not ok_alm:
        print("\nVERDICT: ABORT — solar-position module fails its pins; fix apparatus.")
        return 2
    e_grid, f_grid = flux_curve()
    print(f"    flux cache: {len(e_grid)} points, max={f_grid.max():.0f} at e={e_grid[np.argmax(f_grid)]:.1f}")

    # ABORT-McD: McDowell-shape calibration (true wall, gamma=1, eps=0, no drift, UNSMEARED truth view)
    rng = _rng(0, 0, 0, 0)
    mu, day, hloc, w_eff, h_true = _candidates(rng, 200000)
    w = w_eff * (phi(h_true) / np.max(phi(np.arange(3, 36, 0.1)))) ** 1.0
    pick = _draw(rng, w, 20000)
    hs = h_true[pick]
    mode = float(0.5 * (H_BINS[:-1] + H_BINS[1:])[np.argmax(np.histogram(hs, H_BINS)[0])])
    tail = float(np.mean(hs > 33.0))
    ok_mcd = 18.0 <= mode <= 29.0 and tail < 0.02
    print(f"\n[ABORT-McD] McDowell shape: mode={mode:.1f} deg (window [18,29]), unsmeared tail>33 = {tail:.2%} (<2%)"
          f"  [{'PASS' if ok_mcd else 'FAIL'}]")
    if not ok_mcd:
        print("\nVERDICT: ABORT-McD — generator does not reproduce the qualitative McDowell shape; redesign.")
        return 2

    # ---------------- P1 recovery cells ---------------- #
    print(f"\n[P1] blind-injection recovery ({n_rep} replicates/cell, N_CZA=1000, ring x3):")
    cells = [(g, e) for g in (0.5, 1.0, 2.0) for e in (0.01, 0.05, 0.10)]
    raw = {}
    for ci, (g, e) in enumerate(cells):
        for pi, w_inj in enumerate(P1_INJECT):
            ests = [fit_p1(*gen_p1(_rng(1, ci, pi, r), w_inj, g, e))[0] for r in range(n_rep)]
            raw[(ci, pi)] = np.array(ests)
    pos_bias = [np.median(np.concatenate([raw[(ci, pi)] for ci in range(len(cells))]) - P1_INJECT[pi])
                for pi in range(3)]
    spread = max(abs(pos_bias[i] - pos_bias[j]) for i in range(3) for j in range(3))
    bias_ok = spread <= 0.4
    b_hat = float(np.median([raw[(ci, pi)][r] - P1_INJECT[pi] for ci in range(len(cells))
                             for pi in range(3) for r in range(n_rep)])) if bias_ok else 0.0
    print(f"    per-position median bias: {[f'{b:+.3f}' for b in pos_bias]}  max pairwise spread {spread:.3f}"
          f" -> constant correction {'PERMITTED b=' + format(b_hat, '+.3f') if bias_ok else 'FORBIDDEN (scored raw)'}")
    worst = 0.0
    print(f"    {'gamma':>6} {'eps':>5} " + " ".join(f"w={w:>7}" for w in P1_INJECT) + "   (median |err| after rule)")
    for ci, (g, e) in enumerate(cells):
        meds = [float(np.median(np.abs(raw[(ci, pi)] - b_hat - P1_INJECT[pi]))) for pi in range(3)]
        worst = max(worst, *meds)
        print(f"    {g:>6} {e:>5.0%} " + " ".join(f"{m:>9.3f}" for m in meds))
    extra_rows = {}
    for name, kw in (("drift +30%", dict(gamma=1.0, eps=0.05, drift=+0.3)),
                     ("drift -30%", dict(gamma=1.0, eps=0.05, drift=-0.3)),
                     ("ring x1", dict(gamma=1.0, eps=0.05, ring_mult=1)),
                     ("ring x6", dict(gamma=1.0, eps=0.05, ring_mult=6))):
        meds = []
        for pi, w_inj in enumerate(P1_INJECT):
            ests = np.array([fit_p1(*gen_p1(_rng(2, hash(name) % 97, pi, r), w_inj, **kw))[0] for r in range(n_rep)])
            meds.append(float(np.median(np.abs(ests - b_hat - P1_INJECT[pi]))))
        extra_rows[name] = meds
        worst = max(worst, *meds)
        print(f"    {name:>12} " + " ".join(f"{m:>9.3f}" for m in meds))
    k0a = worst > 1.0
    print(f"    worst-cell median recovery error = {worst:.3f} deg  [{'K0a FIRES (>1.0)' if k0a else 'PASS'}]")

    # ---------------- wall-free controls: threshold (A) + specificity (B) ---------------- #
    print("\n[P1] wall-free controls (disjoint seed batches):")
    lr_a = np.array([fit_p1(*gen_p1(_rng(3, 1000 + r, 0, 0), WALL, 1.0, 0.05, wall_free=True))[3] for r in range(n_rep)])
    thresh = float(np.percentile(lr_a, 95))
    lr_b = np.array([fit_p1(*gen_p1(_rng(4, 5000 + r, 0, 0), WALL, 1.0, 0.05, wall_free=True))[3] for r in range(n_rep)])
    spec = float(np.mean(lr_b <= thresh))
    k0b = spec < 0.95
    print(f"    batch-A 95th-pct LR threshold = {thresh:.2f}; batch-B specificity = {spec:.1%}"
          f"  [{'K0b FIRES (<95%)' if k0b else 'PASS'}]")

    # ---------------- power tiers ---------------- #
    print("\n[P1] detection power (gamma=1, eps=5%, all 3 positions pooled):")
    power = {}
    for n_cza in (300, 1000, 3000):
        det = []
        for pi, w_inj in enumerate(P1_INJECT):
            det += [fit_p1(*gen_p1(_rng(5, n_cza, pi, r), w_inj, 1.0, 0.05, n_cza=n_cza))[3] > thresh
                    for r in range(n_rep // 2)]
        power[n_cza] = float(np.mean(det))
        print(f"    N_CZA={n_cza:>5}: power = {power[n_cza]:.1%}")
    k0c = power[1000] < 0.80

    # consistency columns (report-only, gamma=1/eps=5 cell): isotonic + exposure normalization
    iso = [isotonic_maxdrop(*gen_p1(_rng(6, 0, 1, r), WALL, 1.0, 0.05)) for r in range(50)]
    iso_med = float(np.nanmedian(np.abs(np.array(iso) - b_hat - WALL)))
    print(f"\n[P1] consistency columns (report-only): isotonic max-drop median |err| = {iso_med:.2f} deg")

    # ---------------- Leg 0-P2 ---------------- #
    print(f"\n[P2] metamorphosis-midpoint recovery ({n_rep} replicates/cell, N=2000):")
    worst2 = 0.0
    for wt in (1.5, 0.8, 2.5):
        meds = []
        for pi, m_inj in enumerate(P2_INJECT):
            ests = np.array([fit_p2(*gen_p2(_rng(7, int(wt * 10), pi, r), m_inj, w_t=wt))[0] for r in range(n_rep)])
            meds.append(float(np.median(np.abs(ests - m_inj))))
        worst2 = max(worst2, *meds)
        print(f"    w_t={wt:>4}: " + " ".join(f"m={m_inj}: {m:>6.3f}" for m_inj, m in zip(P2_INJECT, meds)))
    lr2a = np.array([fit_p2(*gen_p2(_rng(8, 1000 + r, 0, 0), 29.71, no_transition=True))[3] for r in range(n_rep)])
    t2 = float(np.percentile(lr2a, 95))
    lr2b = np.array([fit_p2(*gen_p2(_rng(9, 5000 + r, 0, 0), 29.71, no_transition=True))[3] for r in range(n_rep)])
    spec2 = float(np.mean(lr2b <= t2))
    k0p2 = worst2 > 1.5 or spec2 < 0.95
    print(f"    worst-cell median |err| = {worst2:.3f} (gate 1.5); no-transition specificity = {spec2:.1%}"
          f"  [{'K0-P2 FIRES' if k0p2 else 'PASS'}]")

    # ---------------- verdict ---------------- #
    print("\n" + "=" * 100)
    fired = [nm for nm, k in (("K0a", k0a), ("K0b", k0b), ("K0c", k0c), ("K0-P2", k0p2)) if k]
    if fired:
        print(f"VERDICT: {'/'.join(fired)} FIRED — the pipeline cannot resolve the walls at citizen-report")
        print("  fidelity (the lane's resolution-floor null, banked as a SUCCESS). NO live pull."
              + ("  (P1 alone may proceed if only K0-P2 fired — see prereg s3.)" if fired == ["K0-P2"] else ""))
        return 1
    print("VERDICT: LEG-0 PASS (P1 AND P2) — blind wall/midpoint recovery within gates under the full")
    print(f"  selection stack (worst P1 cell {worst:.3f} deg <= 1.0; worst P2 cell {worst2:.3f} <= 1.5;")
    print(f"  specificity {spec:.1%}/{spec2:.1%}; power@1000 {power[1000]:.1%}). The live leg (Leg 1) is now")
    print("  UNLOCKED pending owner sign-off; protocol + verdict spaces are pinned in the prereg s4.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
