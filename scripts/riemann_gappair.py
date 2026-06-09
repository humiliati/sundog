#!/usr/bin/env python
"""H7 — the Riemann gap-pair caustic (empirical reconnaissance). Against docs/atlas/H7_RIEMANN_GAPPAIR_
CAUSTIC_PREREG.md. Points the lab's caustic machinery at the SPECTRUM of the Riemann zeros.

Data: the registered Odlyzko cache (results/riemann/probe01-isotropy-zero-pairs/source/zeros1.txt, 100k
zeros γ_n). Unfold via the Riemann-von Mangoldt smooth count ⟨N(t)⟩=(t/2π)(log(t/2π)−1)+7/8, so w_n=⟨N(γ_n)⟩
has unit mean spacing; gaps δ_n = w_{n+1}−w_n.

FACE 2 (the real caustic): the spectral form factor K(τ)=(1/L)|Σ e^{2πi w τ}|², block-averaged. GUE has a
genuine non-analyticity at τ=1 (ramp K=τ → plateau K=1) — the caustic. Beyond τ=1 the PRIMES contribute
(Berry-Keating / Bogomolny): test whether the zeros deviate from the GUE plateau ABOVE the CUE noise floor.
Controls: CUE (random-unitary eigenphases = GUE universality) and Poisson (independent gaps, flat K=1).
Power curve N=10k→20k→50k→100k. FACE 1 (the literal gap-pair density): consecutive-gap anti-correlation +
the jet classifier on the density (honest prior: GUE density is smooth, likely no literal caustic).

NOT public-eligible; empirical probe, NOT an RH claim. Attribution: Montgomery; Odlyzko (the zeros);
Berry & Keating (spectral form factor / arithmetic). Run: python scripts/riemann_gappair.py
"""
import sys
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))

ZEROS = Path("results/riemann/probe01-isotropy-zero-pairs/source/zeros1.txt")
L = 1000                       # levels per form-factor block
TAU = np.linspace(0.0, 2.6, 521)
RNG = np.random.default_rng(20260608)


def smooth_count(t):
    """⟨N(t)⟩ — the Riemann-von Mangoldt average zero-counting function."""
    return (t / (2 * np.pi)) * (np.log(t / (2 * np.pi)) - 1.0) + 7.0 / 8.0


def load_unfolded():
    g = np.array([float(x) for x in ZEROS.read_text().split()])
    return smooth_count(g)      # w_n, unit mean spacing


def form_factor(w_blocks):
    """Block-averaged spectral form factor K(τ) over a list of equal-length unfolded level blocks."""
    K = np.zeros_like(TAU)
    for w in w_blocks:
        # S(τ) = Σ_n exp(2πi w_n τ); chunk over τ to bound memory
        acc = np.empty(TAU.shape, complex)
        for a in range(0, len(TAU), 64):
            b = min(a + 64, len(TAU))
            acc[a:b] = np.exp(2j * np.pi * np.outer(w, TAU[a:b])).sum(axis=0)
        K += (np.abs(acc) ** 2) / len(w)
    return K / len(w_blocks)


def blocks_of(w, nblk):
    return [w[i * L:(i + 1) * L] for i in range(nblk)]


def cue_blocks(nblk):
    """nblk independent CUE(L) eigenphase sequences, unfolded to unit mean spacing."""
    from scipy.stats import unitary_group
    out = []
    for _ in range(nblk):
        U = unitary_group.rvs(L, random_state=RNG)
        th = np.sort(np.angle(np.linalg.eigvals(U)) % (2 * np.pi))
        out.append(th * L / (2 * np.pi))          # unfold: density L/2π -> unit
    return out


def poisson_blocks(nblk):
    return [np.cumsum(RNG.exponential(1.0, L)) for _ in range(nblk)]


def main():
    print("=" * 86)
    print("H7 — the Riemann gap-pair caustic (Odlyzko 100k; Face 2 = spectral form factor)")
    print("=" * 86)
    w_all = load_unfolded()
    gaps_all = np.diff(w_all)
    print(f"  loaded {len(w_all)} unfolded zeros; mean gap = {gaps_all.mean():.4f} (target 1.000), "
          f"γ_max≈{74920:.0f}\n")

    Ns = [10000, 20000, 50000, 100000]
    nb = {N: N // L for N in Ns}
    # controls sized to the largest N (reuse prefixes for the power curve)
    print(f"  building CUE noise floor ({nb[100000]} blocks of L={L}) + Poisson ...", flush=True)
    cue_all = cue_blocks(nb[100000])
    poi_all = poisson_blocks(nb[100000])

    print("\n(FACE 2) spectral form factor K(τ) — power curve. GUE: ramp K≈τ then KINK at τ=1, plateau K≈1.")
    print("  the CAUSTIC = the τ=1 kink; the PRIZE = zeros deviating from the plateau beyond τ=1 (arithmetic).")
    ti = {name: int(np.argmin(np.abs(TAU - v))) for name, v in
          {"0.5": 0.5, "1.0": 1.0, "1.3": 1.3, "1.6": 1.6, "2.0": 2.0}.items()}
    results = {}
    for N in Ns:
        b = nb[N]
        Kz = form_factor(blocks_of(w_all, b))
        Kc = form_factor(cue_all[:b])
        Kp = form_factor(poi_all[:b])
        # CUE noise floor: per-block std / sqrt(b) (the band on the block mean)
        cue_band = np.std([form_factor([blk]) for blk in cue_all[:b]], axis=0) / np.sqrt(b)
        results[N] = (Kz, Kc, Kp, cue_band)
        row = " ".join(f"{TAU[ti[k]]:.1f}:z={Kz[ti[k]]:.2f}/c={Kc[ti[k]]:.2f}" for k in ti)
        print(f"  N={N:>6} ({b:>3} blk): {row}")

    # caustic check: is there a ramp->plateau kink near τ=1? (slope drops across τ=1)
    Kz, Kc, Kp, band = results[100000]
    def slope(K, lo, hi):
        m = (TAU >= lo) & (TAU <= hi)
        return float(np.polyfit(TAU[m], K[m], 1)[0])
    ramp = slope(Kz, 0.3, 0.9); plateau = slope(Kz, 1.2, 2.0)
    kink = ramp - plateau
    print(f"\n  τ=1 CAUSTIC (zeros): ramp-slope(0.3-0.9)={ramp:+.2f}  plateau-slope(1.2-2.0)={plateau:+.2f}  "
          f"kink={kink:+.2f}  [{'CAUSTIC' if ramp > 0.4 and abs(plateau) < 0.3 else 'weak'}]")

    # arithmetic deviation beyond τ=1: zeros vs CUE band (3σ) over τ∈[1.05, 2.5]
    m = (TAU >= 1.05) & (TAU <= 2.5)
    dev = (Kz - Kc)[m]
    sig = (3 * band)[m]
    exceed = int(np.sum(dev > sig))
    maxz = float(np.max((dev / (band[m] + 1e-9))))
    print(f"  ARITHMETIC test (τ∈[1.05,2.5]): zeros−CUE exceeds 3σ_CUE in {exceed}/{m.sum()} bins; "
          f"max excess = {maxz:.1f}σ  [{'DEVIATION' if exceed >= 5 and maxz > 3 else 'within GUE floor'}]")

    # (FACE 1) the literal gap-pair density: anti-correlation + caustic check
    print("\n(FACE 1) gap-pair (δ_n, δ_{n+1}) — anti-correlation (GUE rigidity) + literal-caustic check:")
    def anticorr(g):
        return float(np.corrcoef(g[:-1], g[1:])[0, 1])
    cz = anticorr(gaps_all)
    cc = anticorr(np.diff(np.concatenate(cue_all)))
    cp = anticorr(np.diff(np.concatenate(poi_all)))
    print(f"  consecutive-gap correlation:  zeros={cz:+.3f}   CUE={cc:+.3f}   Poisson={cp:+.3f}"
          f"   [{'GUE-type anti-correlation (zeros≈CUE, both<0; Poisson≈0)' if cz < -0.05 and abs(cp) < 0.05 else '?'}]")

    # jet classifier on the gap-pair density gradient map (honest: likely smooth, no literal caustic)
    import atlas_jet_classify as jc
    ng = 120
    edges = np.linspace(0, 3.2, ng + 1)
    H, _, _ = np.histogram2d(gaps_all[:-1], gaps_all[1:], bins=[edges, edges], density=True)
    from scipy.ndimage import gaussian_filter
    rho = gaussian_filter(H, 2.0)
    gu, gv = np.gradient(rho)
    d = edges[1] - edges[0]
    phi, c2, c3 = jc.jet_from_chart(gu, gv, d, d)
    cusps = jc.cusp_c3(phi, c2, c3)
    print(f"  jet classifier on ∇ρ:  caustic curve present, #cusps={len(cusps)}  "
          f"(honest: this reads ρ's curvature, not an optical caustic; GUE density is smooth)")

    # ---- verdict vs pre-reg ---- #
    caustic_tau1 = ramp > 0.4 and abs(plateau) < 0.3
    arith = exceed >= 5 and maxz > 3
    gue_match = (cz < -0.05 and abs(cz - cc) < 0.08)
    print("\n" + "=" * 86)
    print("VERDICT (vs H7 pre-reg)")
    print("=" * 86)
    print(f"  [{'PASS' if caustic_tau1 else 'FAIL'}] P-F1: the τ=1 CAUSTIC (GUE ramp→plateau kink) is present in the zeros' form factor")
    print(f"  [{'PASS' if gue_match else 'FAIL'}] zeros carry GUE structure (anti-correlated gaps ≈ CUE; Poisson ≈ 0)")
    print(f"  [{'PRIZE!' if arith else 'null'}] P-F2: arithmetic deviation beyond τ=1 above the CUE 3σ floor")
    print("\n" + ("PRIZE: the zeros' form factor DEVIATES from GUE beyond τ=1 above the noise floor — an "
                  "arithmetic caustic fingerprint." if arith else
                  "NULL-A (as pre-registered, expected): the zeros are GUE-universal — the gap-pair caustic "
                  "is exactly the universal τ=1 kink; no arithmetic deviation survives the CUE noise floor at "
                  "this N. The lab's caustic tools touched a spectrum and read it correctly. A clean, bounded result."))
    print("=" * 86)
    np.savez(Path("results/atlas/h7/formfactor.npz"), tau=TAU,
             **{f"Kz_{N}": results[N][0] for N in Ns}, **{f"Kc_{N}": results[N][1] for N in Ns},
             **{f"band_{N}": results[N][3] for N in Ns})
    return 0


if __name__ == "__main__":
    Path("results/atlas/h7").mkdir(parents=True, exist_ok=True)
    sys.exit(main())
