// scripts/lib/yang-mills-u1-2d-core.mjs
//
// U(1) 2D lattice gauge-theory core for the Yang-Mills Phase 1 U(1) 2D
// gauge-invariance smoke runner. Implements:
//
//   - state representation (link phases on a periodic 2D lattice)
//   - staple-based single-link Metropolis sweep
//   - mean plaquette + Wilson loop real-part evaluator
//   - v1 signature vocabulary (W11, W12, W13, W22 mean+variance,
//     position-and-orientation averaged)
//   - v1 held-out target vocabulary (W14, W23, W33 mean+variance) for
//     format compatibility in Phase 1 (not scored)
//   - random U(1) site-gauge transform
//   - raw-link representation as a flat Float64Array of link phases
//   - Sokal automatic-windowing tau_int estimator
//   - deterministic seeded mulberry32 RNG with substream factory
//
// All routines are pure JS. No external lattice library is permitted by
// the P0 lock.
//
// References (per lit-pass / P0 lock):
//   - Creutz 1980 (heatbath SU(2))
//   - Kennedy-Pendleton 1985 (heatbath improvement)
//   - Brown-Woch 1987 (overrelaxation)
//   - Sokal lecture notes (autocorrelation windowing)
//
// This module is for U(1) only. SU(2) lives in a future companion module.

const TWO_PI = 2 * Math.PI;

// ---------------------------------------------------------------- RNG ---

// mulberry32 — small fast deterministic 32-bit state PRNG. Repro across
// machines; one-line state; good enough for Monte Carlo proposals here.
export function mulberry32(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Deterministic substream: hash (masterSeed, ...labels) → 32-bit seed.
// Used so every (config, transform) pair gets an independent stream and
// the result is bit-for-bit reproducible across runs.
export function deriveSubstreamSeed(masterSeed, ...labels) {
  let h = (masterSeed >>> 0) ^ 0x9e3779b9;
  for (const label of labels) {
    let v;
    if (typeof label === "number") {
      v = label >>> 0;
    } else {
      v = 0;
      const s = String(label);
      for (let i = 0; i < s.length; i++) {
        v = Math.imul(v ^ s.charCodeAt(i), 0x01000193) >>> 0;
      }
    }
    h = Math.imul(h ^ v, 0x85ebca6b) >>> 0;
    h ^= h >>> 13;
    h = Math.imul(h, 0xc2b2ae35) >>> 0;
    h ^= h >>> 16;
  }
  return h >>> 0;
}

// ---------------------------------------------------------- state ------

// State layout: theta[mu * L * L + x * L + y] for mu in {0, 1}.
// mu = 0 → link in +x direction from site (x, y); mu = 1 → +y direction.
// Periodic boundary in both directions.
export function createU1Lattice(L, seed) {
  const links = new Float64Array(2 * L * L);
  if (seed !== undefined && seed !== null) {
    const rng = mulberry32(seed >>> 0);
    for (let i = 0; i < links.length; i++) {
      links[i] = rng() * TWO_PI;
    }
  }
  return { L, links };
}

export function cloneU1Lattice(state) {
  return { L: state.L, links: new Float64Array(state.links) };
}

function linkIndex(L, mu, x, y) {
  return mu * L * L + x * L + y;
}

function wrap(i, L) {
  return ((i % L) + L) % L;
}

// ------------------------------------------------ plaquette / loops ----

// Plaquette phase at base site (x, y): theta_x(x,y) + theta_y(x+1,y)
//                                    - theta_x(x,y+1) - theta_y(x,y).
export function plaquettePhase(state, x, y) {
  const L = state.L;
  const t = state.links;
  const xp = (x + 1) % L;
  const yp = (y + 1) % L;
  return (
    t[linkIndex(L, 0, x, y)] +
    t[linkIndex(L, 1, xp, y)] -
    t[linkIndex(L, 0, x, yp)] -
    t[linkIndex(L, 1, x, y)]
  );
}

export function meanPlaquette(state) {
  const L = state.L;
  let sum = 0;
  for (let x = 0; x < L; x++) {
    for (let y = 0; y < L; y++) {
      sum += Math.cos(plaquettePhase(state, x, y));
    }
  }
  return sum / (L * L);
}

// Wilson loop nx × ny anchored at (x, y), running +x then +y then -x then -y.
// Returns Re(exp(i * phi_loop)).
export function wilsonLoopReal(state, x, y, nx, ny) {
  const L = state.L;
  const t = state.links;
  let phi = 0;
  for (let i = 0; i < nx; i++) {
    phi += t[linkIndex(L, 0, wrap(x + i, L), y)];
  }
  for (let j = 0; j < ny; j++) {
    phi += t[linkIndex(L, 1, wrap(x + nx, L), wrap(y + j, L))];
  }
  for (let i = 0; i < nx; i++) {
    phi -= t[linkIndex(L, 0, wrap(x + nx - 1 - i, L), wrap(y + ny, L))];
  }
  for (let j = 0; j < ny; j++) {
    phi -= t[linkIndex(L, 1, x, wrap(y + ny - 1 - j, L))];
  }
  return Math.cos(phi);
}

// Mean + variance of Re(loop) across all base positions AND both orientations
// (nx × ny and ny × nx). In 2D U(1) the two orientations are equivalent in
// expectation by symmetry, but we average explicitly so the signature is
// rotation-symmetric by construction.
function loopMeanVar(state, nx, ny) {
  const L = state.L;
  let n = 0;
  let sum = 0;
  let sumSq = 0;
  const accumulate = (a, b) => {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        const v = wilsonLoopReal(state, x, y, a, b);
        sum += v;
        sumSq += v * v;
        n++;
      }
    }
  };
  accumulate(nx, ny);
  if (nx !== ny) accumulate(ny, nx);
  const mean = sum / n;
  const variance = sumSq / n - mean * mean;
  return { mean, variance };
}

// v1 signature vocabulary: W11, W12, W13, W22 mean + variance.
export function computeSignatureV1(state) {
  const W11 = loopMeanVar(state, 1, 1);
  const W12 = loopMeanVar(state, 1, 2);
  const W13 = loopMeanVar(state, 1, 3);
  const W22 = loopMeanVar(state, 2, 2);
  return {
    W11_mean: W11.mean,
    W11_var: W11.variance,
    W12_mean: W12.mean,
    W12_var: W12.variance,
    W13_mean: W13.mean,
    W13_var: W13.variance,
    W22_mean: W22.mean,
    W22_var: W22.variance,
  };
}

// v1 held-out target vocabulary: W14, W23, W33 mean + variance. Not
// scored in Phase 1; emitted for format compatibility only.
export function computeHeldoutV1(state) {
  const W14 = loopMeanVar(state, 1, 4);
  const W23 = loopMeanVar(state, 2, 3);
  const W33 = loopMeanVar(state, 3, 3);
  return {
    W14_mean: W14.mean,
    W14_var: W14.variance,
    W23_mean: W23.mean,
    W23_var: W23.variance,
    W33_mean: W33.mean,
    W33_var: W33.variance,
  };
}

// Raw-link "vector" — flat Float64Array of all link phases. Used by the
// CTRL_RAW diagnostic to confirm raw links are NOT invariant under gauge
// transforms.
export function computeRawLinkVector(state) {
  return new Float64Array(state.links);
}

// --------------------------------------------------- Metropolis -------

// Staple phase for link mu at (x, y). For U(1) Wilson action, the
// effective single-link "force" is
//     S_link(theta) = -beta * sum_{staples} cos(theta + phi_staple)
// so given the staple phase sum we can compute the contribution at the
// proposed link phase.
//
// In 2D there are exactly two plaquettes touching every link:
//   - one with this link as the "+" segment
//   - one with this link as the "-" segment in the neighboring plaquette
//
// For link mu=0 at (x,y) (i.e. the +x link from (x,y)):
//   "forward" plaquette: theta_x(x,y) + theta_y(x+1,y) - theta_x(x,y+1) - theta_y(x,y)
//   "backward" plaquette: theta_x(x,y-1) + theta_y(x+1,y-1) - theta_x(x,y) - theta_y(x,y-1)
//
// Removing theta_x(x,y) from each, the staple phase that the link couples
// to is:
//   phi_staple_fwd = theta_y(x+1,y) - theta_x(x,y+1) - theta_y(x,y)
//   phi_staple_bwd = theta_y(x+1,y-1) - theta_x(x,y-1) - theta_y(x,y-1)
// and the link contribution to the action is
//   -beta * [ cos(theta + phi_staple_fwd) + cos(theta - phi_staple_bwd) ]
// (sign on bwd flips because the link appears with the opposite sign in
// that plaquette).
//
// Same shape, swapped axes, for mu = 1.

function staplePhasesForLink(state, mu, x, y) {
  const L = state.L;
  const t = state.links;
  if (mu === 0) {
    // forward plaquette at (x, y)
    const xp = (x + 1) % L;
    const yp = (y + 1) % L;
    const ym = (y - 1 + L) % L;
    const phiFwd =
      t[linkIndex(L, 1, xp, y)] -
      t[linkIndex(L, 0, x, yp)] -
      t[linkIndex(L, 1, x, y)];
    // backward plaquette at (x, y-1): plaquette phase is
    //   t_x(x,y-1) + t_y(x+1,y-1) - t_x(x,y) - t_y(x,y-1)
    // removing t_x(x,y) gives this link sitting with a MINUS sign, so its
    // contribution to the action under proposal theta is
    //   -beta * cos( -theta + [ t_x(x,y-1) + t_y(x+1,y-1) - t_y(x,y-1) ] )
    // which equals
    //   -beta * cos( theta - [ t_x(x,y-1) + t_y(x+1,y-1) - t_y(x,y-1) ] )
    // so we pass that bracketed phase as phiBwd and the caller uses
    //   cos(theta - phiBwd).
    const phiBwd =
      t[linkIndex(L, 0, x, ym)] +
      t[linkIndex(L, 1, xp, ym)] -
      t[linkIndex(L, 1, x, ym)];
    return { phiFwd, phiBwd };
  } else {
    // mu = 1 (link in +y direction)
    const xp = (x + 1) % L;
    const yp = (y + 1) % L;
    const xm = (x - 1 + L) % L;
    // forward plaquette at (x, y): t_x(x,y) + t_y(x+1,y) - t_x(x,y+1) - t_y(x,y)
    // removing t_y(x,y) (link with MINUS sign), contribution is
    //   -beta * cos(theta - [ t_x(x,y) + t_y(x+1,y) - t_x(x,y+1) ])
    // Wait — that's plaquette where t_y(x,y) has minus. The "forward"
    // plaquette here is actually the (x,y) plaquette using the +y link
    // with sign -1, so we need to be careful about which neighbor
    // plaquette we call forward vs backward.
    //
    // Symmetric formulation: for any link, sum over both touching
    // plaquettes of [link_sign * (sum of the OTHER three links signed
    // appropriately)]. We compute the two staple phases such that the
    // link contribution to the action is
    //   -beta * [ cos(theta + phiPlus) + cos(theta - phiMinus) ]
    // where phiPlus is the staple where this link enters with sign +1
    // and phiMinus is the staple where it enters with sign -1.
    //
    // For mu = 1 at (x, y): the +1-sign plaquette is the (x-1, y) one
    //   t_x(x-1,y) + t_y(x,y) - t_x(x-1,y+1) - t_y(x-1,y)
    // staple = t_x(x-1,y) - t_x(x-1,y+1) - t_y(x-1,y)
    // and the -1-sign plaquette is the (x, y) one
    //   t_x(x,y) + t_y(x+1,y) - t_x(x,y+1) - t_y(x,y)
    // staple = t_x(x,y) + t_y(x+1,y) - t_x(x,y+1)
    const phiPlus =
      t[linkIndex(L, 0, xm, y)] -
      t[linkIndex(L, 0, xm, yp)] -
      t[linkIndex(L, 1, xm, y)];
    const phiMinus =
      t[linkIndex(L, 0, x, y)] +
      t[linkIndex(L, 1, xp, y)] -
      t[linkIndex(L, 0, x, yp)];
    return { phiFwd: phiPlus, phiBwd: phiMinus };
  }
}

// Single-link Metropolis update for link (mu, x, y) at coupling beta with
// proposal width halfWidth (full window is 2*halfWidth wide, centered on
// current value). Returns true if accepted.
export function metropolisLinkUpdate(state, mu, x, y, beta, halfWidth, rng) {
  const L = state.L;
  const idx = linkIndex(L, mu, x, y);
  const theta = state.links[idx];

  // For mu = 0 we use the (phiFwd, phiBwd) convention where the link
  // appears with +1 in fwd and -1 in bwd, so action contribution at
  // proposal phi is -beta * [ cos(phi + phiFwd) + cos(phi - phiBwd) ].
  // For mu = 1 the staple helper already returns (phiPlus, phiMinus)
  // matching the same convention.
  const staples = staplePhasesForLink(state, mu, x, y);
  const phiPlus = staples.phiFwd;
  const phiMinus = staples.phiBwd;

  const proposal = theta + (rng() * 2 - 1) * halfWidth;

  const sOld =
    -beta * (Math.cos(theta + phiPlus) + Math.cos(theta - phiMinus));
  const sNew =
    -beta * (Math.cos(proposal + phiPlus) + Math.cos(proposal - phiMinus));
  const dS = sNew - sOld;

  if (dS <= 0 || rng() < Math.exp(-dS)) {
    // Wrap into [0, 2*pi) for numerical hygiene.
    let v = proposal % TWO_PI;
    if (v < 0) v += TWO_PI;
    state.links[idx] = v;
    return true;
  }
  return false;
}

// One full sweep = update every link once in a fixed lexicographic order.
// Returns the accepted-update fraction for the sweep.
export function metropolisSweep(state, beta, halfWidth, rng) {
  const L = state.L;
  let accepted = 0;
  let total = 0;
  for (let mu = 0; mu < 2; mu++) {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        if (metropolisLinkUpdate(state, mu, x, y, beta, halfWidth, rng))
          accepted++;
        total++;
      }
    }
  }
  return accepted / total;
}

// ---------------------------------------------- gauge transform -------

// Apply a U(1) site gauge transformation. alphas is a Float64Array of
// length L*L of site phases. Link update:
//   theta_mu(x) -> alpha(x) + theta_mu(x) - alpha(x + mu)
export function applyU1GaugeTransform(state, alphas) {
  const L = state.L;
  const out = new Float64Array(state.links.length);
  for (let mu = 0; mu < 2; mu++) {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        const xn = mu === 0 ? (x + 1) % L : x;
        const yn = mu === 1 ? (y + 1) % L : y;
        const a0 = alphas[x * L + y];
        const a1 = alphas[xn * L + yn];
        let v = (a0 + state.links[linkIndex(L, mu, x, y)] - a1) % TWO_PI;
        if (v < 0) v += TWO_PI;
        out[linkIndex(L, mu, x, y)] = v;
      }
    }
  }
  return { L, links: out };
}

// Build a random alpha array using the provided RNG (uniform on [0, 2*pi)).
export function randomAlphas(L, rng) {
  const a = new Float64Array(L * L);
  for (let i = 0; i < a.length; i++) {
    a[i] = rng() * TWO_PI;
  }
  return a;
}

export function identityAlphas(L) {
  return new Float64Array(L * L);
}

// ----------------------------------------- residuals --------------------

// Max absolute residual between two signature objects (same keys).
export function signatureMaxAbsResidual(sigA, sigB) {
  let maxAbs = 0;
  for (const k of Object.keys(sigA)) {
    const d = Math.abs(sigA[k] - sigB[k]);
    if (d > maxAbs) maxAbs = d;
  }
  return maxAbs;
}

// Normalized L2 residual between two raw-link vectors. Returns
// ||a - b||_2 / max(||a||_2, ||b||_2, eps). Uses circular link distance
// so a small physical change does not get amplified by wrap from 2pi - eps
// to 0.
export function rawLinkNormalizedL2(rawA, rawB) {
  let num = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < rawA.length; i++) {
    let d = rawA[i] - rawB[i];
    // Wrap into (-pi, pi].
    d = d - TWO_PI * Math.round(d / TWO_PI);
    num += d * d;
    normA += rawA[i] * rawA[i];
    normB += rawB[i] * rawB[i];
  }
  const denom = Math.max(Math.sqrt(normA), Math.sqrt(normB), 1e-30);
  return Math.sqrt(num) / denom;
}

// ----------------------------------------- tau_int (Sokal) -------------

// Sokal automatic-windowing integrated autocorrelation time. Returns
// { tauInt, window, mean, variance } for the series.
//
//   rho(t) = ( <x_s x_{s+t}> - <x>^2 ) / Var(x)
//   tauHat(W) = 0.5 + sum_{t=1..W} rho(t)
//   pick smallest W such that W >= c * tauHat(W); default c = 6.
export function estimateTauIntSokal(series, c = 6) {
  const n = series.length;
  if (n < 8) {
    return { tauInt: NaN, window: 0, mean: NaN, variance: NaN };
  }
  let mean = 0;
  for (let i = 0; i < n; i++) mean += series[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = series[i] - mean;
    variance += d * d;
  }
  variance /= n;
  if (variance <= 0) {
    return { tauInt: 0.5, window: 0, mean, variance };
  }
  const maxLag = Math.floor(n / 4);
  let tauHat = 0.5;
  let chosenWindow = maxLag;
  for (let t = 1; t <= maxLag; t++) {
    let cov = 0;
    for (let i = 0; i < n - t; i++) {
      cov += (series[i] - mean) * (series[i + t] - mean);
    }
    cov /= n - t;
    tauHat += cov / variance;
    if (t >= c * tauHat) {
      chosenWindow = t;
      return { tauInt: tauHat, window: chosenWindow, mean, variance };
    }
  }
  return { tauInt: tauHat, window: chosenWindow, mean, variance };
}
