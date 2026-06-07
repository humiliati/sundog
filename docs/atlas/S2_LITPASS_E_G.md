# S2 Lit-Pass — Track E (size shadow) + Track G (handedness shadow)

> **Stage-0 deliverable for the S2 halo physical leg** (`docs/proof/PHASE5_CROSS_SUBSTRATE.md` §3.2 S2).
> Real lit-pass (2026-06-07), every claim confidence-tagged. Purpose: settle the smallest defensible
> per-shadow forward model and decide the **size-legibility kill-gate**. **Outcome: the naïve "size off
> the refraction-halo fringe" is physically dead; the corona is the live home for the size shadow.**
> Confidence tags: `web-confirmed` (literature/URL), `standard-physics` (textbook, model-knowledge),
> `SYNTHESIS` (Sundog framing, NOT established — flagged), `OPEN` (unresolved).

## Verdict in one line
The **discrete-determines** legs are physically solid (ice-phase rock-solid; handedness real-but-novel).
The **continuous-resists** (size) leg **cannot** use refraction-halo supernumerary fringes (Berry 1994:
zero contrast, "rarely if ever seen") — it must use the **corona** (pure-diffraction Airy size law,
web-confirmed) or fall back. Kill-gate fires a **substrate decision**, not a full stop.

---

## Track E — the SIZE shadow (continuous, should resist)

### E-dead: refraction-halo supernumerary fringes — NOT legible
- **22° halo edge = a STEP singularity** (randomly-oriented crystals), not a fold caustic. Its universal
  diffraction function `h(q) = ½ − sin(2q)/(2πq) − Si(2q)/π` (`scipy.special.sici`) has **zero-contrast
  shoulders, no intensity maxima**. `web-confirmed` — Berry, *Supernumerary ice-crystal halos?*, Appl.
  Opt. 33(21):4563 (1994), Eq.(21), §4. There is **no readable size-fringe**, even monodisperse.
- **Parhelia (sun dogs)** have a square-root (fold-type) edge with genuine Fresnel-integral
  supernumeraries, but **max contrast only 0.178**, quenched for crystals > ~0.18 mm @600 nm.
  `web-confirmed` (Berry 1994 §3, Eqs.11–13).
- **Observational status:** a size-dependent readable halo fringe is **not** an established phenomenon;
  mainstream halo sims (HaloRay, Greenler) are pure geometric-optics + overall edge broadening only.
  `web-confirmed` (Berry 1994; Können 1983 reinterprets "supernumerary halos" as pyramidal odd-radius
  halos — geometric, not diffractive).
- **Existence floor:** halos require size parameter `x = 2πa/λ ≳ 100`, i.e. `a ≳ 10 µm` @600 nm, just to
  form. `web-confirmed` (Mishchenko & Macke, Appl. Opt. 38:1626, 1999).
- **Second quench channel:** observed halo edge widths *exceed* the Fraunhofer prediction for the
  measured size distribution — face-angle disorder smears independently of size. `web-confirmed`
  (Können & Tinbergen, Appl. Opt. 30:3382, 1991).

### E-live: the CORONA (pure-diffraction Airy size law) — the legible size shadow
- A corona is the Fraunhofer/Airy diffraction pattern of small particles; **size reads off the ring
  positions**, `sin θ_n = C_n·λ/(2a)`, `C_dark = {1.2197, 2.2331, 3.2383, 4.2411}` (J₁ zeros /π),
  `C_bright = {1.6347, 2.6793, 3.6987}`. `web-confirmed` (Airy disk; atoptics corona). Smaller particle
  → larger corona. Intensity `I(θ)=[2J₁(x)/x]²`, `x = (2πa/λ)·sinθ` (`scipy.special.j1`).
- **Washout = lossiness:** distinct rings require a **near-monodisperse** population; broad size spread
  washes the corona into a featureless aureole (the established corona→iridescence→aureole transition).
  `web-confirmed` (montana.edu/jshaw/corona; Cowley/Laven/Vollmer, Phys. Educ. 40:51, 2005). **This is
  exactly the continuous-resists crossover, on real, web-confirmed optics.**
- Perceived red rings (Mie/colour): `θ_deg ≈ 16/r, 31/r, 47/r` (r in µm). `web-confirmed` (Cowley/Laven).
  Use Airy/Bessel constants for the physics core; the 16/31/47 lookup only for matching perceived rings.
- **`SYNTHESIS` flag:** the closed-form monodispersity threshold `σ_a/a < 1/(2n)` (n-th ring survival)
  is a Sundog derivation from fringe-shift scaling, **not** a quoted literature constant. The
  *qualitative* requirement (narrow distribution) is web-confirmed; the `1/(2n)` factor must be
  validated against an explicit size-integral, not cited.

### E-weak: refraction-halo edge BROADENING (FWHM ∝ λ/d) — legible but disorder-dominated
- The geometric edge *position* is size-independent; only the diffraction *broadening* width encodes
  size (`FWHM ~ λ/d`, Fraunhofer). `standard-physics`. But the size-dependent part is **subdominant** to
  fixed broadenings (solar disk ~0.5°, dispersion ~0.6°, face-angle disorder) — real risk it fails the
  `cont(0) ≥ 0.70` preflight. A single-number, confounded observable.

### Track-E minimal model (corona, the recommended size shadow)
```python
from scipy.special import j1
def corona_intensity(theta, a, lam):      # a=radius[µm], lam[µm], theta[rad]
    x = (2*np.pi*a/lam)*np.sin(theta)
    return np.where(x==0, 1.0, 2*j1(x)/x)**2   # Airy [2J1/x]^2
# size readout: a = C*lam/(2*sin theta_ring); population washout: integrate over a size pdf.
```

---

## Track G — the DISCRETE shadow (should be determined)

### G-solid: ICE-PHASE via halo radius (the robust primary anchor)
- 22° (hexagonal prism, apex 60°) vs ~28° (pyramidal odd-radius) — **pure geometry**, min-deviation
  `δ_min(n, A)`. Size-independent, survives population averaging trivially. The discrete *class* = crystal
  habit. `web-confirmed` (Tape; Können pyramidal reinterpretation). The safest discrete-determines leg.

### G-real-but-novel: HANDEDNESS via Stokes-V (the deep target — flagged)
- **Ice Ih is NOT optically active** (achiral space group `P6₃/mmc`, optically positive uniaxial). All
  halo circular polarization is **ray-path parity × c-axis birefringent retardance**, never molecular
  chirality. `web-confirmed` (Können & Tinbergen 1991). *Every docstring/write-up must say this.*
- **Minimal Mueller chain** (numpy, no special functions): `S_out = M_Fresnel_exit · M_retarder(δ,φ) ·
  M_Fresnel_entry · (1,0,0,0)`. Entry Fresnel = partial linear diattenuator (makes Q≠0); retarder rotates
  Q→V. `V/I ≈ −(A−B)/(A+B)·sin(2φ)·sin(δ)`, `δ = 2π·Δn·L/λ`. **V sign-flips under φ→φ+90° (fast-axis
  parity) and Δn→−Δn (c-axis-projection parity)** — the handedness signal. `standard-physics`,
  numerically verified (V/I = −0.0117 at φ=45° → +0.0117 at φ=135°).
- Constants: `n_ice ≈ 1.31` (Warren&Brandt 2008, normal dispersion, ~1.319@400nm→1.306@700nm);
  `Δn = n_e−n_o = +0.0014` @590nm; half-wave parity-max at `L ~ λ/(2Δn) ~ 210 µm`. `web-confirmed`.
- **`SYNTHESIS` + `OPEN` flags (load-bearing honesty):**
  - **No published observation of nonzero Stokes-V in a visible ice halo.** Können's entire program
    measures *linear* polarization only. `web-confirmed` (absence). The handedness leg is a **NOVEL
    PREDICTED observable**, not an established one.
  - "**Net-V over a display = population handedness imbalance**" is **`SYNTHESIS`** — no citation;
    established ice-cloud V-use is shape/phase discrimination, not L/R population imbalance.
  - **Cancellation risk:** mirror-image crystal orientations (equally populated in an unbiased swarm)
    give equal-and-opposite V → **net V ≈ 0** unless a shared parity/orientation asymmetry is imposed.
    Birefringent-V also sign-averages to ~0 over a polydisperse population (δ sweeps half-waves over
    ~200 µm). So the handedness leg in the forward model **requires an imposed shared `x_d` parity**
    (legitimate — it is the law's "shared discrete variable" — but it is a *predicted*, not *observed*,
    physical effect, and must be framed that way).

> **SHARPENED 2026-06-07 (measured-sky recon → `S2_MEASURED_SKY_SCOPE.md`).** The handedness claim
> splits in two, with opposite fates — use this framing, not the old flat "net-V = handedness, novel":
> - **Per-feature V (DEFENSIBLE, MEASURABLE):** Stokes-V on TIR features (parhelic circle,
>   subhelic/anthelic arcs) is physically sound — the *rainbow* TIR circular polarization is the
>   web-confirmed precedent; ice birefringence adds to it. A spatially-resolved measurement would
>   likely *confirm* a ~1% per-feature V (a genuine first; no halo Stokes-V is published).
> - **Net-V = population handedness (DISFAVORED, QUARANTINE):** no mechanism breaks ice handedness
>   symmetry (achiral; E/B not pseudoscalar; Können's `U=0` is the exact net-V-cancels analog). An
>   integrated measurement most likely returns ~0 → *falsifies* this framing. Do not assert it.
> - Pre-registered measured-sky target = an antisymmetric `±V(θ,φ)` integrating to ~0 (confirms the
>   mechanism + demonstrates cancellation in one shot).
> - **Linear-pol physics now validated against MEASURED SKY:** `scripts/s2_konnen_validate.py` shows
>   the Mueller chain reproduces Können 1991 (Fresnel floor 3.65%≈3.71%, birefringent split
>   0.106°≈0.11°, ~100% inner ledge, U=0). The *same code* generates the V prediction — a credibility
>   gate the chain passes — but V itself stays forward-model (Können measured none).

### Track-G minimal model (handedness Mueller core + ice-phase geometry)
```python
def stokes_after_prism(theta_i, n, L, dn, phi, lam):  # returns (I,Q,U,V)
    # entry/exit Fresnel diattenuator Mueller + retarder Mueller(delta=2*pi*dn*L/lam, axis phi)
    ...   # V/I = -(A-B)/(A+B)*sin(2phi)*sin(delta) to leading order; sign = handedness
def halo_radius(n, apex_deg):                          # min-deviation -> 22 deg (60) vs ~28 deg (pyramidal)
    A = np.radians(apex_deg); return np.degrees(2*np.arcsin(n*np.sin(A/2)) - A)
```

---

## Confidence-tagged cite spine
| ref | role | tag |
| --- | --- | --- |
| Berry, *Supernumerary ice-crystal halos?*, Appl.Opt. 33:4563 (1994) | 22° halo step `h(q)`; parhelia 0.178; "rarely if ever seen" | web-confirmed |
| Mishchenko & Macke, Appl.Opt. 38:1626 (1999) | halo existence floor `x≳100` (`a≳10µm`) | web-confirmed |
| Können & Tinbergen, Appl.Opt. 30:3382 (1991) | ice optically positive; halo = LINEAR pol; width > Fraunhofer | web-confirmed |
| Cowley, Laven & Vollmer, Phys.Educ. 40:51 (2005) | corona size law, red-ring 16/31/47 | web-confirmed |
| Airy-disk / Born&Wolf | corona `[2J₁/x]²`, J₁-zero ring constants | web-confirmed / standard |
| Berry & Upstill, Prog.Opt. XVIII (1980) | fold→Airy dressing, `K^{−2/3}` fringe scaling (rainbow case) | web-confirmed |
| Warren & Brandt, JGR 113:D14220 (2008) | ice `n(λ)` visible, normal dispersion | web-confirmed |
| ice `Δn=+0.0014` (Hobbs; IceCube optics) | birefringent retardance source | web-confirmed |
| monodispersity `σ_a/a < 1/(2n)` | size-fringe survival threshold | **SYNTHESIS** |
| net-V = population handedness | handedness-leg framing | **SYNTHESIS** |
| halo Stokes-V observation | — none found — | **OPEN** |

## GO / NO-GO (owner-gated)
- **Discrete-determines leg: GO.** Ice-phase (halo radius) is the robust primary; handedness (Stokes-V)
  is GO as the deep target **but must be pre-registered as a NOVEL PREDICTED observable** (flagged
  novelty + cancellation caveat), not cited as established.
- **Continuous-resists (size) leg: the kill-gate fires a substrate choice** —
  - **CORONA (recommended):** clean, web-confirmed size→ring→aureole washout; faithful to "size off the
    Airy dressing"; **cost: the continuous shadow lives on a different atmospheric-ice phenomenon than
    the discrete shadow** (relaxes S0/S1's "one shadow, two variables" to "two members of the
    halo/diffraction atlas" — still a physical instantiation, consistent with §3.2's two-observable S2).
  - **EDGE-BROADENING (one-substrate but weak):** keeps size + discrete on the same refraction halo, but
    the size signal is disorder-dominated — real risk of `cont(0)<0.70` preflight void.
  - **DISCRETE-ONLY (FALLBACK B):** drop continuous-physical (declare *owed*); S2 = ice-phase +
    handedness determines-only demo. Banks the discrete physical anchor §4 leans on; explicitly a
    *partial* physical leg.
- **Owner decision required** before Stage 1 (pre-register the slate): which continuous-leg substrate.
