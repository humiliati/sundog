#!/usr/bin/env python3
"""v3 scaling-ladder REGIME SELECTION helper (planning only; not a frozen harness).

The v2 frozen run found LB beats Stern at [128,64] w=12 (Stern's success heuristic was
optimistic). v3 tests whether Stern overtakes LB as w grows and whether the find-vs-check
gap scales. Constraint: Prange's N explodes with w (the slow baseline), so the scaled
regimes measure ONLY LB + Stern (small N); Prange is measured at the anchor + predicted by
formula at scale. This scans candidate regimes for: (a) where analytic C_Stern crosses
below C_LB, (b) LB+Stern N staying measurable, (c) Prange N (to confirm it is NOT).

Uses the v2-validated cost model: per_iter = base(m) + enum, base(m)=3.95*m^3.07 (fit at
m=40,60; EXTRAPOLATED here for selection only — the v3 slate re-calibrates via a fresh
two-size smoke). enum analytic. NOTE the v2 measured Stern ~1.64x its analytic C (heuristic
optimism), so the analytic crossover below is itself optimistic for Stern — the true
crossover is likely at larger w; that gap is the v3 measurement.
"""
import math

LN2 = math.log(2)
ALPHA, BETA = 3.9519351266355742, 3.0714390585016966  # v2 base(m) fit
V2_MS_PER_ITER_AT_M64 = 9.84  # measured Prange wall, m=64; scale ~ m^3


def base(m):
    return ALPHA * m ** BETA


def enum_lb(k, m, p=2):
    return math.comb(k, p) * (p + 1) * m


def enum_stern(k, m, l, p=2):
    half = math.comb(k // 2, p)
    return 2 * half * (p * l) + (half * half / 2 ** l) * (3 * p + 1) * m


def p_succ(kind, n, k, w, p=2, l=8):
    C = math.comb; m = n - k
    if kind == "prange":
        return C(m, w) / C(n, w)
    if kind == "lb":
        return C(k, p) * C(m, w - p) / C(n, w)
    if kind == "stern":
        if w - 2 * p < 0 or m - l < w - 2 * p:
            return 0.0
        return C(k // 2, p) ** 2 * C(m - l, w - 2 * p) / C(n, w)
    raise ValueError


def C_ops(kind, n, k, w, l=8, p=2):
    m = n - k
    ps = p_succ(kind, n, k, w, p, l)
    if ps <= 0:
        return math.inf, math.inf, math.inf
    N = 1 / ps
    if kind == "prange":
        per = base(m)
    elif kind == "lb":
        per = base(m) + enum_lb(k, m, p)
    else:
        per = base(m) + enum_stern(k, m, l, p)
    return N * LN2 * per, N, per


def best_stern(n, k, w, p=2):
    m = n - k
    best = (math.inf, None, None, None)
    for l in range(2, min(m - (w - 2 * p), 22) + 1):  # sweep l, pick cost-optimal
        c, N, per = C_ops("stern", n, k, w, l, p)
        if c < best[0]:
            best = (c, N, per, l)
    return best  # (C, N, per, l*)


def runtime_hr(n, k, w, l_star):
    """Rough LB+Stern-only wall: sum over the two attackers of T*N*rho*ms(m), ms~m^3."""
    m = n - k; T = 64; rho = 3.46
    ms = V2_MS_PER_ITER_AT_M64 * (m / 64) ** 3
    Nlb = 1 / p_succ("lb", n, k, w); Nst = best_stern(n, k, w)[1]
    iters = T * rho * (Nlb + Nst)
    return iters * ms / 1000 / 3600


print(f"{'regime':>14} {'w':>3} | {'N_prange':>10} {'N_lb':>8} {'N_st':>8} {'l*':>3} | "
      f"{'C_LB':>9} {'C_Stern':>9} {'St/LB':>6} | {'gap@best':>9} | {'LB+St hr':>8}")
print("-" * 110)
candidates = [
    (128, 12), (128, 14), (128, 16), (128, 18), (128, 20),
    (160, 16), (160, 20), (160, 24),
    (192, 18), (192, 24), (192, 28),
    (224, 22), (224, 28),
    (256, 24), (256, 32),
]
for n, w in candidates:
    k = n // 2; m = n - k
    Cpr, Npr, _ = C_ops("prange", n, k, w)
    Clb, Nlb, _ = C_ops("lb", n, k, w)
    Cst, Nst, per_st, l_star = best_stern(n, k, w)
    verifier = 2 * m * n + n + m
    Cbest = min(Clb, Cst)
    gap = Cbest / verifier
    ratio = Cst / Clb
    hr = runtime_hr(n, k, w, l_star)
    flag = "  <-- Stern<LB" if ratio < 1.0 else ""
    npr_s = f"{Npr:.1e}" if Npr < 1e9 else f"{Npr:.0e}!!"
    print(f"[{n:>3},{k:>3}]{'':>3} {w:>3} | {npr_s:>10} {Nlb:>8.0f} {Nst:>8.0f} {l_star:>3} | "
          f"{Clb:>9.2e} {Cst:>9.2e} {ratio:>6.2f} | {gap:>9.0f} | {hr:>8.2f}{flag}")

print("\nLegend: St/LB<1 => analytic says Stern beats LB (but v2 showed Stern analytic is")
print("~1.64x optimistic at w=12, so true crossover is at LARGER w than analytic shows).")
print("N_prange marked !! when > ~1e6 (Prange unmeasurable -> formula-only at that regime).")
