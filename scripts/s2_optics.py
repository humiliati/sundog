#!/usr/bin/env python
"""S2 forward optics — real-physics models for the halo physical leg of PHASE5 §3.12.

Two halo/diffraction-family legs (see docs/atlas/S2_LITPASS_E_G.md, lit-pass 2026-06-07):
  - CORONA (continuous-resists / SIZE shadow): Fraunhofer Airy diffraction I = [2 J1(x)/x]^2,
    x = (2 pi a / lam) sin(theta); ring angle theta ∝ lam/a. Size washes out under population
    size-spread (corona -> iridescence -> featureless aureole).
  - HALO (discrete-determines shadow):
      * ICE-PHASE via min-deviation halo radius delta_min = 2 arcsin(n sin(A/2)) - A
        (apex 60 deg -> 21.8 deg; apex 90 deg -> 45.6 deg). Radius is SIZE-INDEPENDENT -> the
        habit class survives population averaging.  [robust primary]
      * HANDEDNESS via the Fresnel x birefringent-retarder Mueller chain; the SIGN of Stokes-V
        carries a shared ray-path/c-axis parity that survives averaging.  [predicted/novel]

BINDING RULE (PHASE5 §3.12): the physics equations here are FIXED by cited literature and may NOT
be tuned to produce a crossover. Only power knobs in the harness (noise, n, K, lam grid, size
range) are calibrated on throwaway seed 999.

HONESTY GATE: ice Ih is NOT optically active (achiral space group P6_3/mmc; optically positive
uniaxial). All halo circular polarization is ray-path parity x c-axis birefringent retardance,
NEVER molecular chirality (Können & Tinbergen, Appl. Opt. 30:3382, 1991).

Refs: Airy disk / Born & Wolf (corona Airy pattern); Berry, Appl. Opt. 33:4563 (1994) (halo
diffraction: 22 deg edge is a zero-contrast STEP, parhelia 0.178 contrast); Mishchenko & Macke,
Appl. Opt. 38:1626 (1999) (halo existence floor x = 2 pi a/lam >~ 100, a >~ 10 um); Warren &
Brandt, JGR 113:D14220 (2008) (n_ice ~ 1.31 visible, normal dispersion); ice Delta_n = +0.0014.
"""
import numpy as np
from scipy.special import j1, airy

# ---- fixed cited constants ------------------------------------------------- #
N_ICE = 1.31          # Warren & Brandt 2008, visible mean refractive index
DN_ICE = 0.0014       # n_e - n_o, positive uniaxial @ ~590 nm
LAM_LIGHT = 0.55      # um, representative visible wavelength
A_FLOOR_UM = 10.0     # Mishchenko-Macke halo/diffraction existence floor (a >~ 10 um @ visible)

# Halo-class radii (deg) for the ice-phase discrete leg.
# 22 deg = hexagonal Ih prism (apex 60 deg), exact min-deviation; the alternate class (~28 deg)
# stands for pyramidal/odd-radius ice (label; the physics point is a distinct, SIZE-INDEPENDENT
# radius). Both are size-independent geometric invariants of the habit class.
HALO_R_HEX = 22.0
HALO_R_ALT = 28.0


# ===== CORONA — continuous-resists / SIZE shadow ============================ #
def corona_intensity(theta, a, lam=LAM_LIGHT):
    """Fraunhofer Airy intensity I/I0 = [2 J1(x)/x]^2, x = (2 pi a/lam) sin(theta).
    theta [rad]; a particle radius [um]; lam [um]. (Babinet: obstacle == aperture.)"""
    x = (2.0 * np.pi * a / lam) * np.sin(theta)
    xs = np.where(np.abs(x) < 1e-12, 1.0, x)
    core = np.where(np.abs(x) < 1e-12, 1.0, 2.0 * j1(xs) / xs)
    return core ** 2


def corona_profile(thetas, sizes, lam=LAM_LIGHT):
    """Ensemble-averaged corona radial intensity profile over a population of `sizes` [um].
    thetas: (T,) [rad]; sizes: (K,) [um]. Returns mean_i [2 J1(x_i)/x_i]^2 -> (T,)."""
    thetas = np.asarray(thetas, float)
    sizes = np.asarray(sizes, float)
    x = (2.0 * np.pi / lam) * sizes[:, None] * np.sin(thetas)[None, :]   # (K,T)
    xs = np.where(np.abs(x) < 1e-12, 1.0, x)
    core = np.where(np.abs(x) < 1e-12, 1.0, 2.0 * j1(xs) / xs)
    return (core ** 2).mean(0)                                          # (T,)


def airy_first_dark_deg(a, lam=LAM_LIGHT):
    """Angular radius (deg) of the first Airy dark ring: sin(theta) = 1.2197 * lam/(2a)."""
    return np.degrees(np.arcsin(np.clip(1.2197 * lam / (2.0 * a), -1, 1)))


# ===== FOLD-AIRY — continuous-resists / SIZE shadow (PHASE-encoded) ========= #
# A fold caustic (A2; the parhelion/sun-dog minimum-deviation edge) is dressed by the Airy function:
# I ∝ Ai(-s)^2, s = kappa*(theta - theta_edge). The fringe scale kappa ∝ crystal size, so larger
# crystals -> finer supernumerary fringes, while the EDGE position theta_edge is SIZE-INDEPENDENT.
# Hence size lives ONLY in the fringes (phase-encoded): it washes by destructive interference under
# population size-spread, with no surviving envelope (cf. corona, whose envelope leaks the mean size).
# Refs: Berry & Upstill, Prog. Opt. XVIII (1980); Berry, Appl. Opt. 33:4563 (1994) — parhelion
# supernumeraries (genuine fold fringes, faint, max contrast ~0.178).
SUPERNUM_S = (1.0188, 3.2482, 4.8201, 6.1633)   # |s| at Ai^2 maxima (zeros of Ai'); 1st supernumerary etc.


def fold_airy_intensity(theta_deg, theta_edge_deg, kappa):
    """Fold diffraction dressing I = Ai(-s)^2, s = kappa*(theta - theta_edge). Lit side (s>0)
    oscillates (supernumeraries); dark side decays. kappa ∝ size. theta_edge size-independent."""
    s = kappa * (np.asarray(theta_deg, float) - theta_edge_deg)
    return airy(-s)[0] ** 2


def fold_airy_profile(thetas_deg, kappas, theta_edge_deg):
    """Ensemble-averaged fold-Airy profile over a population of fringe-scales kappas (∝ size).
    Returns mean_i Ai(-kappa_i*(theta - theta_edge))^2 -> (T,). The supernumerary fringes decohere
    (wash) as the kappa population spreads; the edge stays put -> phase-encoded size washout."""
    thetas = np.asarray(thetas_deg, float)
    kappas = np.asarray(kappas, float)
    s = kappas[:, None] * (thetas[None, :] - theta_edge_deg)        # (K,T)
    return (airy(-s)[0] ** 2).mean(0)                              # (T,)


def supernumerary_deg(order, kappa, theta_edge_deg):
    """Angular position (deg) of the n-th supernumerary maximum: theta = theta_edge + s_n/kappa.
    Position - edge ∝ 1/kappa ∝ 1/size (the monotonic size readout)."""
    return theta_edge_deg + SUPERNUM_S[order - 1] / kappa


# ===== HALO RADIUS — discrete-determines / ICE-PHASE shadow ================= #
def halo_min_deviation(apex_deg, n=N_ICE):
    """Minimum-deviation angle [deg] of a prism, apex A: delta_min = 2 arcsin(n sin(A/2)) - A.
    apex 60 -> 21.84 deg (22 deg halo); apex 90 -> 45.6 deg (46 deg halo). NaN if beyond TIR."""
    A = np.radians(apex_deg)
    arg = n * np.sin(A / 2.0)
    if np.any(np.asarray(arg) > 1.0):
        return np.nan
    return np.degrees(2.0 * np.arcsin(arg) - A)


def halo_radial_profile(thetas_deg, radius_deg, width_deg):
    """Halo as an orientation/diffraction-broadened bright ring (Gaussian) at `radius_deg`.
    The PEAK LOCATION encodes the discrete habit class and is SIZE-INDEPENDENT."""
    thetas_deg = np.asarray(thetas_deg, float)
    return np.exp(-0.5 * ((thetas_deg - radius_deg) / width_deg) ** 2)


def halo_pol_dop(radius_deg):
    """Predicted linear degree-of-polarization of a minimum-deviation REFRACTION halo of angular
    radius R, as a function of R ALONE:
        DoP(R) = (1 - cos^4(R/2)) / (1 + cos^4(R/2)),
    radially oriented (E in the scattering plane), with U = 0 (mirror symmetry) and no net circular
    V (refraction halo). Derivation: at minimum deviation the internal/external incidence difference
    equals half the deviation, theta_i - theta_t = R/2, and the Fresnel transmittance ratio through the
    two faces is T_p/T_s = 1/cos^2(theta_i - theta_t) per face -> cos^4(R/2) for the pair (the n- and
    apex-dependence cancels via Snell, leaving R alone). Generalizes Können & Tinbergen 1991's 22-deg
    Fresnel-floor (R=21.84 -> 3.7%) to ANY refraction-halo radius -> the unifying polarization
    observable across the Atlas's classified halos (incl. the pyramidal 9/18/20/23/24/35-deg family).
    Independent of refractive index and habit GIVEN the observed radius R."""
    f = np.cos(np.radians(np.asarray(radius_deg, float) / 2.0)) ** 4
    return (1.0 - f) / (1.0 + f)


# ===== HANDEDNESS — discrete-determines / STOKES-V shadow ================== #
def mueller_fresnel(theta_i_deg, n1, n2):
    """Mueller matrix of Fresnel transmission (partial linear diattenuator) at n1->n2 interface.
    Returns None on total internal reflection. s/p frame."""
    ti = np.radians(theta_i_deg)
    st = (n1 / n2) * np.sin(ti)
    if abs(st) >= 1.0:
        return None
    tt = np.arcsin(st)
    ci, ct = np.cos(ti), np.cos(tt)
    ts = 2.0 * n1 * ci / (n1 * ci + n2 * ct)        # Fresnel amplitude transmission, s
    tp = 2.0 * n1 * ci / (n2 * ci + n1 * ct)        # Fresnel amplitude transmission, p
    A, B = ts ** 2, tp ** 2                          # intensity transmittances (s, p)
    g = np.sqrt(A * B)
    return 0.5 * np.array([
        [A + B, A - B, 0.0, 0.0],
        [A - B, A + B, 0.0, 0.0],
        [0.0,   0.0,   2 * g, 0.0],
        [0.0,   0.0,   0.0,   2 * g],
    ]), np.degrees(tt)


def mueller_fresnel_reflect(theta_i_deg, n1, n2):
    """Mueller matrix of partial Fresnel REFLECTION at an n1->n2 dielectric interface (sub-critical).
    Returns None in total internal reflection (use mueller_tir there). s/p frame, matching
    mueller_fresnel's block layout + 0.5 prefactor. The intensity reflectances R_s=r_s^2, R_p=r_p^2 are
    already energy-correct (reflection keeps the medium, so NO (n2 ct)/(n1 ci) geometric factor).

    Sub-critical reflection coefficients are REAL, so the s-p phase is 0 or pi (sin delta = 0): a single
    partial reflection produces NO circular polarization -> V arises only from the TIR phase +
    birefringence. The sign of r_s*r_p carries the Brewster phase flip, and -> +1 at the critical angle
    (R_s,R_p -> 1), matching mueller_tir's identity there (the continuity anchor). Carry the SIGN of the
    amplitude coefficients, never |r|: that is exactly where the V handedness would be lost."""
    ti = np.radians(theta_i_deg)
    st = (n1 / n2) * np.sin(ti)
    if abs(st) >= 1.0:
        return None                                  # total internal reflection -> use mueller_tir
    tt = np.arcsin(st)
    ci, ct = np.cos(ti), np.cos(tt)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)   # Fresnel amplitude reflection, s
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)   # Fresnel amplitude reflection, p
    Rs, Rp = rs ** 2, rp ** 2                         # intensity reflectances (energy-correct)
    g = np.sign(rs * rp) * np.sqrt(Rs * Rp)          # signed; sin(delta)=0 sub-critical -> no V
    return 0.5 * np.array([
        [Rs + Rp, Rs - Rp, 0.0,   0.0],
        [Rs - Rp, Rs + Rp, 0.0,   0.0],
        [0.0,     0.0,     2 * g, 0.0],
        [0.0,     0.0,     0.0,   2 * g],
    ])


def mueller_retarder(delta, phi):
    """Linear-retarder Mueller matrix; retardance delta [rad], fast-axis azimuth phi [rad]."""
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    C, S = np.cos(delta), np.sin(delta)
    return np.array([
        [1.0, 0.0,            0.0,            0.0],
        [0.0, c2 ** 2 + s2 ** 2 * C, c2 * s2 * (1 - C), -s2 * S],
        [0.0, c2 * s2 * (1 - C), s2 ** 2 + c2 ** 2 * C,  c2 * S],
        [0.0, s2 * S,         -c2 * S,         C],
    ])


def tir_retardance(theta_i_deg, n1, n2):
    """Fresnel total-internal-reflection s-p phase retardance [rad] at an n1->n2 interface (n1>n2).
    On TIR (sin theta_i > n2/n1) the reflected s and p amplitudes are unit-modulus but acquire a
    relative phase delta = delta_s - delta_p; the reflection acts as a PURE retarder (the Fresnel-rhomb
    effect; Born & Wolf 1.5.4 / Hecht 4.7) and converts linear -> circular. This is the primary
    linear->circular mechanism on the TIR-rich features (parhelic circle, subhelic/46-deg grazing) and
    is exactly what the transmission-only mueller_fresnel chain (returns None on TIR) was BLIND to.
    Returns 0.0 below the critical angle (no pure TIR retardance) and -> 0 at grazing.
        tan(delta/2) = cos(theta) * sqrt(sin^2 theta - (n2/n1)^2) / sin^2 theta
    Analytic anchors: max retardance delta_max = 2*arctan(cos^2 thetac / (2 sin thetac)) at
    sin^2 theta = 2 sin^2 thetac/(1+sin^2 thetac); ice (1.31->1) delta_max=30.56 deg @ 59.1 deg;
    glass (1.51->1) delta_max=45.9 deg, crossing 45 deg at the two Fresnel-rhomb angles ~48.6/54.6."""
    ti = np.radians(theta_i_deg)
    s = np.sin(ti)
    sc = n2 / n1                                     # sin of the critical angle
    if s <= sc:                                      # not in TIR -> partial reflection, no pure retarder
        return 0.0
    num = np.cos(ti) * np.sqrt(s ** 2 - sc ** 2)
    return 2.0 * np.arctan2(num, s ** 2)


def mueller_tir(theta_i_deg, n1, n2, phi):
    """Mueller matrix of one total-internal-reflection bounce: a PURE retarder (no diattenuation, energy
    conserving, |r_s|=|r_p|=1) with retardance tir_retardance(theta) and fast-axis azimuth phi [rad]
    (the orientation of the reflection s-axis in the working frame). Identity below the critical angle."""
    return mueller_retarder(tir_retardance(theta_i_deg, n1, n2), phi)


def ray_stokes(theta_i_deg, phi, L_um, parity=1, n=N_ICE, dn=DN_ICE, lam=LAM_LIGHT):
    """Stokes (I,Q,U,V) after entry-Fresnel x birefringent-retarder x exit-Fresnel on unpolarized
    input (1,0,0,0). `parity` (+/-1) flips the c-axis-projection retardance sign -> flips Stokes-V
    sign: the handedness signal. Returns (4,) Stokes vector; V = S[3]."""
    ent = mueller_fresnel(theta_i_deg, 1.0, n)
    if ent is None:
        return np.array([0.0, 0.0, 0.0, 0.0])
    Me, tt_deg = ent
    ext = mueller_fresnel(tt_deg, n, 1.0)            # exit face, internal angle back to air
    if ext is None:
        return np.array([0.0, 0.0, 0.0, 0.0])
    Mx, _ = ext
    delta = parity * 2.0 * np.pi * dn * L_um / lam   # parity flips retardance sign (c-axis parity)
    Mr = mueller_retarder(delta, phi)
    return Mx @ Mr @ Me @ np.array([1.0, 0.0, 0.0, 0.0])


def stokes_V_over_I(theta_i_deg, phi, L_um, parity=1, **kw):
    """Circular-polarization fraction V/I for one ray (the handedness observable)."""
    S = ray_stokes(theta_i_deg, phi, L_um, parity=parity, **kw)
    return S[3] / S[0] if S[0] != 0 else 0.0


def _fresnel_mueller_batch(theta_i_deg, n1, n2):
    """Vectorized Fresnel-transmission Mueller matrices over an array of incidence angles.
    Returns (M (m,4,4), exit_angle_deg (m,), valid (m,))."""
    ti = np.radians(np.atleast_1d(np.asarray(theta_i_deg, float)))
    st = (n1 / n2) * np.sin(ti)
    valid = np.abs(st) < 1.0
    sts = np.clip(st, -0.999999, 0.999999)
    tt = np.arcsin(sts)
    ci, ct = np.cos(ti), np.cos(tt)
    ts = 2.0 * n1 * ci / (n1 * ci + n2 * ct)
    tp = 2.0 * n1 * ci / (n2 * ci + n1 * ct)
    A, B = ts ** 2, tp ** 2
    g = np.sqrt(np.clip(A * B, 0, None))
    M = np.zeros((ti.size, 4, 4))
    M[:, 0, 0] = 0.5 * (A + B); M[:, 0, 1] = 0.5 * (A - B)
    M[:, 1, 0] = 0.5 * (A - B); M[:, 1, 1] = 0.5 * (A + B)
    M[:, 2, 2] = g; M[:, 3, 3] = g
    M[~valid] = 0.0
    return M, np.degrees(tt), valid


def ray_stokes_batch(theta_i_deg, phi, L_um, parity=1, n=N_ICE, dn=DN_ICE, lam=LAM_LIGHT):
    """Vectorized ray_stokes over an array of incidence angles. Returns (m,4) Stokes vectors.
    Same physics as ray_stokes (entry-Fresnel x retarder x exit-Fresnel on unpolarized input)."""
    Me, tt_deg, v1 = _fresnel_mueller_batch(theta_i_deg, 1.0, n)
    Mx, _, v2 = _fresnel_mueller_batch(tt_deg, n, 1.0)
    delta = parity * 2.0 * np.pi * dn * L_um / lam
    Mr = mueller_retarder(delta, phi)
    Sin = np.array([1.0, 0.0, 0.0, 0.0])
    S = np.einsum('mij,jk,mkl,l->mi', Mx, Mr, Me, Sin)
    S[~(v1 & v2)] = 0.0
    return S


if __name__ == "__main__":
    # quick sanity print (not the test suite — see test_s2_optics.py)
    print("halo 22deg (apex 60):", round(float(halo_min_deviation(60)), 3))
    print("halo 46deg (apex 90):", round(float(halo_min_deviation(90)), 3))
    print("corona 1st dark, a=15um:", round(float(airy_first_dark_deg(15)), 3), "deg")
    print("V/I phi=45,L=100um,par=+1:", round(stokes_V_over_I(40, np.pi / 4, 100, +1), 5))
    print("V/I phi=45,L=100um,par=-1:", round(stokes_V_over_I(40, np.pi / 4, 100, -1), 5))
