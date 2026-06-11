#!/usr/bin/env python
"""HS10 LATTICE-FRAG -- the jittered-lattice recovery horizon obeys the 1/eps Debye-Waller law.

Prereg: docs/atlas/SHADOW_LATTICE_JITTER_DW_PREREG.md (frozen 2026-06-11, commit 77e38e7c, BEFORE
this file existed). Population xi = +-1 + eps*N(0,1) on the frozen S0 v2 band-pass apparatus;
analytic charFun (convolution => product, a theorem, NOT the claim):

    Re phi_eps(s) = cos(s) * exp(-(eps*s)^2 / 2)            (lattice recurrence x DW damping)

HONEST SCOPE: the qualitative wash for eps>0 is a corollary of the proved resistance_general /
absCont_resists. ALL falsifiable content is the QUANTITATIVE BRIDGE from the frozen apparatus's
readout to the DW envelope:

  K1 (bridge RMS):       RMS(R_emp - R_pred) over the masked grid, pooled over main eps, > 0.10
  K2 (horizon exponent): OLS slope of log lam*(eps) vs log eps outside [-1.25, -0.75], or a
                         missing crossing inside any main-eps masked window.

R(lam,eps) is a PAIRED (common-random-numbers) fringe-amplitude ratio: at fixed lam the
(xc, xd, lat, gz, obs-noise) streams are identical for every eps (eps is only a multiplier), so
the lattice recurrence factor cos(2*pi*lam*t) and the apparatus's band dephasing cancel. The
prediction is the analytic charFun pushed through the SAME estimator on the SAME draws
(forward-generated; the estimator is linear in the signal and the K-average is unbiased for
Re phi, so E[emp] = pred exactly -- residuals are pure finite-(n,K)+noise variance).

DEVIATION D1 (instrument v2, after the 2026-06-11 run-1 GATE ABORT -- see the receipt). Run 1's
band-AGGREGATE amplitude ratio a_hat(lam,eps)/a_hat(lam,0) cancels the recurrence factor only
approximately: near band-straddling zeros of cos(2*pi*lam*t) (lam ~ 0.5, 1.5, 2.5, ...) the
DW reweighting DISPLACES the numerator's zero from the denominator's, the ratio excurses
structurally (R_pred itself dips), and the theta-crossing detector measured null positions
(lam* flat ~1.5 for eps in {0.15,0.2,0.3}) -- G4 fired, structurally, not by power. v2 takes the
ratio PER-T (where the cancellation is EXACT: rho_t = f_t(eps)/f_t(0) -> DW(2*pi*lam*t*eps))
and aggregates with pred-derived weights W(lam,t) = (env_f(t)^2 * f_pred_t(lam,0))^2, which kill
the per-t nulls quadratically -- the observable is smooth in lam and needs no denominator-floor
mask (the prereg's DENOM_FLOOR is superseded for the v2 observable; Nyquist window retained).
Kill thresholds, theta, grids, seeds, n: all UNCHANGED from the prereg.

Gates (abort = bug, not result): G1 eps=0 denominator control (band-aggregate a_hat(lam,0) tracks
pred(lam,0), RMS <= 0.10, the banked band-dephasing structure -- v1 estimator, still the right
control for the generator/estimator bridge); G2 regression (existing pops byte-unchanged --
pinned in the frozen test); G3 pairing determinism + R(lam->0)=1; G4 split-half power at the
crossings. Secondary (NO kill): the unchanged frozen cont/disc recovery sweep per main eps.

NOT public-eligible. Attribution: Debye 1913 / Waller 1923; Lukacs, "Characteristic Functions".
Run:  python scripts/shadow_lattice_jitter_dw.py            (~15-25 min; recovery sweep dominates)
      python scripts/shadow_lattice_jitter_dw.py --no-recovery   (ratio leg only, ~1 min)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pvnp_phase5_lossiness_crossover as h     # noqa: E402  (frozen S0 apparatus)
import shadow_charfun_populations as scp        # noqa: E402  (frozen pop module, +lattice_jitter)

# ---- FROZEN constants (prereg section 2) ------------------------------------ #
SEED_BASE = 20260608                  # the sweep_pop convention; per-lam seed = base + 1000*lam + 7
N = 600                               # per-cell ensemble (G4 power knob; kill thresholds frozen)
EPS_MAIN = [0.15, 0.20, 0.30, 0.50]
EPS_SANITY = [0.01, 0.03]
THETA = 0.5                           # ratio-horizon threshold
RMS_KILL = 0.10                       # K1
SLOPE_LO, SLOPE_HI = -1.25, -0.75     # K2
DENOM_FLOOR = 0.10                    # |pred(lam,0)| mask floor (recurrence nulls)
LAM_STEP = 0.05
LAM_CAP = 5.5
NYQ_BUDGET = 8.75                     # 63/4 Nyquist - xc_hi(7): lam*(1+4*eps) <= 8.75
SANITY_MIN = 0.8                      # no-early-collapse floor for EPS_SANITY
G1_RMS = 0.10
G3_TOL = 0.02
G4_SE = 0.03


def lam_max(eps):
    return min(NYQ_BUDGET / (1.0 + 4.0 * eps), LAM_CAP)


UNION_GRID = np.round(np.arange(LAM_STEP, LAM_CAP + 1e-9, LAM_STEP), 4)


def cell(lam, eps, n=N):
    """One (lam, eps) cell: per-sample band amplitudes, empirical and analytic-through-estimator.
    Uses the FROZEN gen path (scp.gen_s0_pop) with the frozen per-lam seed, so the stream is
    identical for every eps at fixed lam (paired). Returns (a_emp_i, a_pred_i) arrays."""
    c = h.S0
    rng = np.random.default_rng(SEED_BASE + int(round(lam * 1000)) + 7)
    sig, xc, xd = scp.gen_s0_pop(n, lam, rng, h.NOISE, "lattice_jitter", eps)
    t = np.linspace(-1, 1, c["T"])
    env_g = np.exp(-t ** 2 / (2 * c["w"] ** 2))
    env_f = np.exp(-(np.abs(t) - c["t0_f"]) ** 2 / (2 * c["w_f"] ** 2))
    bump = np.exp(-(t - c["t0"]) ** 2 / (2 * c["w_b"] ** 2))
    # frozen estimator: least-squares amplitude of the sample's own band template
    T_i = np.cos(2 * np.pi * xc[:, None] * t[None, :]) * env_f[None, :]          # (n,T)
    a_emp = (sig * T_i).sum(1) / (T_i ** 2).sum(1)
    # analytic prediction through the SAME estimator on the SAME draws (xi/noise-expectation)
    rephi = scp.charfun_re("lattice_jitter", 2 * np.pi * lam * t, eps)           # (T,)
    parity = xd[:, None] * np.sin(2 * np.pi * c["f_p"] * t)[None, :]
    sig_pred = (c["D"] * bump[None, :]
                + c["A"] * np.cos(2 * np.pi * xc[:, None] * t[None, :]) * rephi[None, :] * env_f[None, :]
                + c["C"] * parity * env_g[None, :])
    a_pred = (sig_pred * T_i).sum(1) / (T_i ** 2).sum(1)
    return a_emp, a_pred


def cell_pert(lam, eps, n=N):
    """v2 (D1): PER-T band amplitudes. f_t = sum_i sig_i(t)*T_i(t) / sum_i T_i(t)^2 estimates the
    per-t fringe coefficient; per-t the recurrence factor cancels EXACTLY in the eps/0 ratio.
    Returns (f_emp, f_pred, f_emp_h1, f_emp_h2), each (T,)."""
    c = h.S0
    rng = np.random.default_rng(SEED_BASE + int(round(lam * 1000)) + 7)
    sig, xc, xd = scp.gen_s0_pop(n, lam, rng, h.NOISE, "lattice_jitter", eps)
    t = np.linspace(-1, 1, c["T"])
    env_g = np.exp(-t ** 2 / (2 * c["w"] ** 2))
    env_f = np.exp(-(np.abs(t) - c["t0_f"]) ** 2 / (2 * c["w_f"] ** 2))
    bump = np.exp(-(t - c["t0"]) ** 2 / (2 * c["w_b"] ** 2))
    T_i = np.cos(2 * np.pi * xc[:, None] * t[None, :]) * env_f[None, :]          # (n,T)
    rephi = scp.charfun_re("lattice_jitter", 2 * np.pi * lam * t, eps)           # (T,)
    parity = xd[:, None] * np.sin(2 * np.pi * c["f_p"] * t)[None, :]
    sig_pred = (c["D"] * bump[None, :]
                + c["A"] * np.cos(2 * np.pi * xc[:, None] * t[None, :]) * rephi[None, :] * env_f[None, :]
                + c["C"] * parity * env_g[None, :])

    def per_t(S, idx):
        return (S[idx] * T_i[idx]).sum(0) / np.maximum((T_i[idx] ** 2).sum(0), 1e-300)

    full = np.arange(n)
    return (per_t(sig, full), per_t(sig_pred, full),
            per_t(sig, full[0::2]), per_t(sig, full[1::2]))


def ratio_sweep(eps_list):
    """Per-t amplitudes over the union grid for eps=0 and each eps in eps_list (paired per lam)."""
    out = {0.0: {}}
    for eps in eps_list:
        out[eps] = {}
    for lam in UNION_GRID:
        for eps in [0.0] + list(eps_list):
            f_emp, f_pred, f_h1, f_h2 = cell_pert(lam, eps)
            out[eps][float(lam)] = dict(f_emp=f_emp, f_pred=f_pred, f_h1=f_h1, f_h2=f_h2)
    return out


ENV_F = np.exp(-(np.abs(np.linspace(-1, 1, h.S0["T"])) - h.S0["t0_f"]) ** 2 / (2 * h.S0["w_f"] ** 2))


def _agg(f_num, f_den, W):
    """Weighted per-t ratio aggregate: sum W*(f_num/f_den)/sum W, nulls killed by W."""
    rho = np.where(np.abs(f_den) > 1e-12, f_num / np.where(f_den == 0, 1, f_den), 0.0)
    return float((W * rho).sum() / W.sum())


def masked_grid(sweep, eps):
    """(lam, R_emp, R_pred, se_half) at masked points: lam <= lam_max(eps) (Nyquist; v2 needs no
    denominator floor -- the weights kill per-t nulls)."""
    rows = []
    for lam in UNION_GRID:
        lam = float(lam)
        if lam > lam_max(eps) + 1e-9:
            continue
        d0, d = sweep[0.0][lam], sweep[eps][lam]
        W = (ENV_F ** 2 * d0["f_pred"]) ** 2                                     # (T,) pred-derived
        r_emp = _agg(d["f_emp"], d0["f_emp"], W)
        r_pred = _agg(d["f_pred"], d0["f_pred"], W)
        se = abs(_agg(d["f_h1"], d0["f_h1"], W) - _agg(d["f_h2"], d0["f_h2"], W)) / 2.0
        rows.append((lam, r_emp, r_pred, se))
    return rows


def first_crossing(rows, theta=THETA):
    """First downward crossing of R_emp below theta, linear interpolation between adjacent masked
    points (prereg: adjacent IN THE MASKED SUBSEQUENCE; a crossing inside an excluded gap is
    interpolated across it). None if no crossing."""
    for (l0, r0, _, _), (l1, r1, _, _) in zip(rows, rows[1:]):
        if r0 >= theta > r1:
            return l0 + (l1 - l0) * (r0 - theta) / max(r0 - r1, 1e-12)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-recovery", action="store_true", help="skip the (slow) secondary recovery sweep")
    ap.add_argument("--out", default="results/shadow/hs10_lattice_jitter_dw.json")
    args = ap.parse_args()
    t0 = time.time()

    print("=" * 78)
    print("HS10 LATTICE-FRAG -- jittered-lattice DW horizon (prereg-frozen kill criteria)")
    print(f"  xi = +-1 + eps*N(0,1); Re phi = cos(s)*exp(-(eps*s)^2/2); paired ratio, n={N}")
    print("  instrument v2 (D1): per-t paired ratio, pred-weighted aggregation (see header)")
    print(f"  K1: pooled masked RMS(R_emp - R_pred) <= {RMS_KILL}")
    print(f"  K2: log-log slope of lam*(eps) in [{SLOPE_LO}, {SLOPE_HI}], all main-eps crossings present")
    print("=" * 78)

    sweep = ratio_sweep(EPS_MAIN + EPS_SANITY)

    # ---- G1: eps=0 denominator control (v1 band-aggregate, unnormalized, full grid) ---- #
    g1_res = []
    for lam in UNION_GRID:
        a_emp, a_pred = cell(float(lam), 0.0)
        g1_res.append(float(a_emp.mean() - a_pred.mean()))
    g1_rms = float(np.sqrt(np.mean(np.square(g1_res))))
    g1_ok = g1_rms <= G1_RMS
    print(f"\n[G1] eps=0 denominator control: RMS(a_hat - pred) over {len(UNION_GRID)} lams "
          f"= {g1_rms:.4f}  (<= {G1_RMS})  -> {'PASS' if g1_ok else 'ABORT'}")

    # ---- G3: pairing determinism + R(lam->0) ~ 1 (v2 observable) ---- #
    a1, _, _, _ = cell_pert(1.0, 0.30)
    a2, _, _, _ = cell_pert(1.0, 0.30)
    det_ok = bool(np.array_equal(a1, a2))
    r0_ok, r0_vals = True, []
    for eps in EPS_MAIN:
        r = masked_grid(sweep, eps)[0][1]                  # R_emp at lam = LAM_STEP
        r0_vals.append(r)
        r0_ok &= abs(r - 1.0) <= G3_TOL
    print(f"[G3] determinism={'PASS' if det_ok else 'ABORT'}; "
          f"R(lam={LAM_STEP}) = {[f'{v:.4f}' for v in r0_vals]} (|R-1| <= {G3_TOL}) "
          f"-> {'PASS' if r0_ok else 'ABORT'}")

    # ---- main-eps tables, crossings, G4 power ---- #
    print(f"\n{'eps':>6} {'lam_max':>8} {'masked':>7} {'rms':>7} {'lam*':>7} {'SE@lam*':>8}")
    rms_sq, rms_n = 0.0, 0
    lam_stars, g4_ok = {}, True
    for eps in EPS_MAIN:
        rows = masked_grid(sweep, eps)
        res = [r_emp - r_pred for (_, r_emp, r_pred, _) in rows]
        rms_sq += float(np.sum(np.square(res)))
        rms_n += len(res)
        rms_eps = float(np.sqrt(np.mean(np.square(res))))
        ls = first_crossing(rows)
        lam_stars[eps] = ls
        se_star = None
        if ls is not None:
            near = sorted(rows, key=lambda r: abs(r[0] - ls))[:3]
            se_star = max(se for (_, _, _, se) in near)
            g4_ok &= se_star <= G4_SE
        print(f"{eps:>6} {lam_max(eps):>8.2f} {len(rows):>7} {rms_eps:>7.4f} "
              f"{'--' if ls is None else f'{ls:.3f}':>7} "
              f"{'--' if se_star is None else f'{se_star:.4f}':>8}")
    pooled_rms = float(np.sqrt(rms_sq / max(rms_n, 1)))
    print(f"[G4] split-half SE at crossings <= {G4_SE} -> {'PASS' if g4_ok else 'ABORT'}")

    # ---- sanity eps: no early collapse ---- #
    sanity_ok = True
    for eps in EPS_SANITY:
        rows = masked_grid(sweep, eps)
        mn = min(r_emp for (_, r_emp, _, _) in rows)
        ok = mn >= SANITY_MIN
        sanity_ok &= ok
        print(f"[sanity] eps={eps}: min masked R_emp = {mn:.4f} (>= {SANITY_MIN}) -> {'PASS' if ok else 'FAIL'}")

    # ---- K1 ---- #
    k1_fired = pooled_rms > RMS_KILL
    print(f"\n[K1] pooled masked RMS(R_emp - R_pred) = {pooled_rms:.4f}  (kill if > {RMS_KILL})  "
          f"-> {'KILL FIRED' if k1_fired else 'pass'}")

    # ---- K2 ---- #
    missing = [eps for eps in EPS_MAIN if lam_stars[eps] is None]
    slope, intercept, r2 = None, None, None
    if not missing:
        lx = np.log(np.array(EPS_MAIN))
        ly = np.log(np.array([lam_stars[e] for e in EPS_MAIN]))
        slope, intercept = np.polyfit(lx, ly, 1)
        yhat = slope * lx + intercept
        ss_res = float(np.sum((ly - yhat) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        k2_fired = not (SLOPE_LO <= slope <= SLOPE_HI)
        print(f"[K2] lam*(eps): " + ", ".join(f"{e}->{lam_stars[e]:.3f}" for e in EPS_MAIN))
        print(f"     log-log slope = {slope:.4f} (R^2 = {r2:.4f}; kill if outside "
              f"[{SLOPE_LO}, {SLOPE_HI}]; const C = lam*eps = {float(np.exp(intercept)):.3f})  "
              f"-> {'KILL FIRED' if k2_fired else 'pass'}")
    else:
        k2_fired = True
        print(f"[K2] missing crossing(s) for eps = {missing} -> KILL FIRED (saturation/flattening)")

    gates_ok = g1_ok and det_ok and r0_ok and g4_ok
    killed = k1_fired or k2_fired

    print("\n" + "=" * 78)
    if not gates_ok:
        print("RESULT: GATE ABORT (apparatus bug/power; fix and rerun -- NOT a verdict).")
        verdict = "gate_abort"
    elif killed:
        print("RESULT: KILLED (informative). The apparatus readout does NOT reduce to the DW")
        print("  envelope (K1) and/or the horizon is not 1/eps (K2) -- the bridge any future")
        print("  'wash means resist' interpretation leans on fails at the stated tolerance.")
        verdict = "killed"
    else:
        print("RESULT: CONFIRMED (clean confirmatory success). The SURVIVE clause now carries a")
        print("  quantitative atomicity tolerance: the jittered-lattice recovery horizon tracks")
        print(f"  the DW envelope (pooled RMS {pooled_rms:.3f} <= {RMS_KILL}) with a 1/eps law "
              f"(slope {slope:.2f}).")
        verdict = "confirmed"
    print(f"  sanity(no early collapse @ eps {EPS_SANITY}) = {'PASS' if sanity_ok else 'FAIL'}")
    print("=" * 78)

    # ---- secondary (NO kill): frozen recovery sweep per main eps ---- #
    recovery = {}
    if not args.no_recovery:
        print("\nSECONDARY (no kill): frozen cont/disc recovery sweep, pop='lattice_jitter'")
        print(f"   {'lam':>6} " + " ".join(f"{l:>5}" for l in h.LAMBDAS))
        for eps in EPS_MAIN:
            cont, disc = scp.sweep_pop("lattice_jitter", n=N, eps=eps)
            lc = h.half_life(cont, h.LAMBDAS)
            recovery[eps] = dict(cont=cont, disc=disc, half_life=lc)
            print(f"   eps={eps:<4} cont " + " ".join(f"{v:5.2f}" for v in cont)
                  + f"   lam*_c={lc}  minDisc={min(disc):.2f}")
        print("   (disc ~ banked throughout: jitter keeps a finite centered mean -- determination")
        print("    is the theorem; the cont horizon closing with eps is the readout corollary.)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        verdict=verdict, pooled_rms=pooled_rms, k1_fired=bool(k1_fired), k2_fired=bool(k2_fired),
        slope=None if slope is None else float(slope), r2=None if r2 is None else float(r2),
        lam_stars={str(k): v for k, v in lam_stars.items()},
        gates=dict(g1_rms=g1_rms, g1=bool(g1_ok), g3_det=bool(det_ok), g3_r0=bool(r0_ok),
                   g4=bool(g4_ok), sanity=bool(sanity_ok)),
        eps_main=EPS_MAIN, eps_sanity=EPS_SANITY, n=N, theta=THETA, seed_base=SEED_BASE,
        recovery={str(k): v for k, v in recovery.items()},
        wall_s=round(time.time() - t0, 1)), indent=2))
    print(f"\nwrote {out}  ({round(time.time() - t0, 1)}s)")
    return 1 if (not gates_ok or killed) else 0


if __name__ == "__main__":
    sys.exit(main())
