// scripts/lib/yang-mills-su2-2d-core.mjs
//
// SU(2) 2D lattice gauge-theory core for the Yang-Mills Phase 1 SU(2)
// gauge-invariance smoke runner. This module is intentionally self-contained
// and does not refactor the existing U(1) runner path.

const TWO_PI = 2 * Math.PI;
const UNITARITY_TOL = 1e-10;

// ---------------------------------------------------------------- RNG ---

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

function randNormal(rng) {
  let u1 = 0;
  while (u1 <= 0) u1 = rng();
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1));
  return r * Math.cos(TWO_PI * u2);
}

function randomUnit3(rng) {
  const x = randNormal(rng);
  const y = randNormal(rng);
  const z = randNormal(rng);
  const n = Math.hypot(x, y, z);
  if (n <= 0) return [1, 0, 0];
  return [x / n, y / n, z / n];
}

export function haarQuaternion(rng) {
  const q0 = randNormal(rng);
  const q1 = randNormal(rng);
  const q2 = randNormal(rng);
  const q3 = randNormal(rng);
  const n = Math.hypot(q0, q1, q2, q3);
  if (n <= 0) return [1, 0, 0, 0];
  return [q0 / n, q1 / n, q2 / n, q3 / n];
}

// ------------------------------------------------------------- SU(2) ---

// Quaternion convention:
//   q = (q0, q1, q2, q3)
//   M(q) = [[ q0 + i q3,  q2 + i q1 ],
//           [-q2 + i q1,  q0 - i q3 ]]
// With this convention, multiplication has the minus-cross-product sign.
export function qMul(a, b) {
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  return [
    a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
    a0 * b1 + b0 * a1 - (a2 * b3 - a3 * b2),
    a0 * b2 + b0 * a2 - (a3 * b1 - a1 * b3),
    a0 * b3 + b0 * a3 - (a1 * b2 - a2 * b1),
  ];
}

export function qConj(q) {
  return [q[0], -q[1], -q[2], -q[3]];
}

function qNorm(q) {
  return Math.hypot(q[0], q[1], q[2], q[3]);
}

function qNormalize(q) {
  const n = qNorm(q);
  if (n <= 0) return [1, 0, 0, 0];
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function qAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
}

function qScale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s, a[3] * s];
}

export function qUnitarityResidual(q) {
  return Math.abs(1 - (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]));
}

function assertUnit(q, context) {
  const residual = qUnitarityResidual(q);
  if (residual > UNITARITY_TOL) {
    throw new Error(`${context} SU(2) unit residual ${residual}`);
  }
}

// ------------------------------------------------------------ state ----

// State layout: links[((mu * L * L + x * L + y) * 4) + component].
// mu = 0 is +x, mu = 1 is +y.
export function createSU2Lattice(L, seed) {
  const links = new Float64Array(2 * L * L * 4);
  const rng = mulberry32(seed >>> 0);
  for (let mu = 0; mu < 2; mu++) {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        setLink({ L, links }, mu, x, y, haarQuaternion(rng));
      }
    }
  }
  return { L, links };
}

export function cloneSU2Lattice(state) {
  return { L: state.L, links: new Float64Array(state.links) };
}

function linkBase(L, mu, x, y) {
  return (mu * L * L + x * L + y) * 4;
}

function wrap(i, L) {
  return ((i % L) + L) % L;
}

export function getLink(state, mu, x, y) {
  const L = state.L;
  const b = linkBase(L, mu, wrap(x, L), wrap(y, L));
  return [
    state.links[b],
    state.links[b + 1],
    state.links[b + 2],
    state.links[b + 3],
  ];
}

function setLink(state, mu, x, y, q) {
  const L = state.L;
  const n = qNormalize(q);
  const b = linkBase(L, mu, wrap(x, L), wrap(y, L));
  state.links[b] = n[0];
  state.links[b + 1] = n[1];
  state.links[b + 2] = n[2];
  state.links[b + 3] = n[3];
}

export function maxLinkUnitarityResidual(state) {
  let maxResidual = 0;
  for (let i = 0; i < state.links.length; i += 4) {
    const q = [
      state.links[i],
      state.links[i + 1],
      state.links[i + 2],
      state.links[i + 3],
    ];
    maxResidual = Math.max(maxResidual, qUnitarityResidual(q));
  }
  return maxResidual;
}

// ----------------------------------------------------- loops ----------

export function plaquetteQuaternion(state, x, y) {
  const L = state.L;
  const ux = getLink(state, 0, x, y);
  const uyXp = getLink(state, 1, wrap(x + 1, L), y);
  const uxYp = getLink(state, 0, x, wrap(y + 1, L));
  const uy = getLink(state, 1, x, y);
  return qMul(qMul(qMul(ux, uyXp), qConj(uxYp)), qConj(uy));
}

export function meanPlaquette(state) {
  const L = state.L;
  let sum = 0;
  for (let x = 0; x < L; x++) {
    for (let y = 0; y < L; y++) {
      sum += plaquetteQuaternion(state, x, y)[0];
    }
  }
  return sum / (L * L);
}

export function wilsonLoopTraceHalf(state, x, y, nx, ny) {
  const L = state.L;
  let acc = [1, 0, 0, 0];
  for (let i = 0; i < nx; i++) {
    acc = qMul(acc, getLink(state, 0, wrap(x + i, L), y));
  }
  for (let j = 0; j < ny; j++) {
    acc = qMul(acc, getLink(state, 1, wrap(x + nx, L), wrap(y + j, L)));
  }
  for (let i = 0; i < nx; i++) {
    acc = qMul(
      acc,
      qConj(getLink(state, 0, wrap(x + nx - 1 - i, L), wrap(y + ny, L))),
    );
  }
  for (let j = 0; j < ny; j++) {
    acc = qMul(
      acc,
      qConj(getLink(state, 1, x, wrap(y + ny - 1 - j, L))),
    );
  }
  return acc[0];
}

function loopMeanVar(state, nx, ny) {
  const L = state.L;
  let n = 0;
  let sum = 0;
  let sumSq = 0;
  const accumulate = (a, b) => {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        const v = wilsonLoopTraceHalf(state, x, y, a, b);
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

export function computeRawMatrixVector(state) {
  const out = new Float64Array((state.links.length / 4) * 8);
  let j = 0;
  for (let i = 0; i < state.links.length; i += 4) {
    const q0 = state.links[i];
    const q1 = state.links[i + 1];
    const q2 = state.links[i + 2];
    const q3 = state.links[i + 3];
    out[j++] = q0;
    out[j++] = q3;
    out[j++] = q2;
    out[j++] = q1;
    out[j++] = -q2;
    out[j++] = q1;
    out[j++] = q0;
    out[j++] = -q3;
  }
  return out;
}

// ----------------------------------------------------- staples --------

function stapleForLink(state, mu, x, y) {
  const L = state.L;
  if (mu === 0) {
    const termForward = qMul(
      qMul(getLink(state, 1, x + 1, y), qConj(getLink(state, 0, x, y + 1))),
      qConj(getLink(state, 1, x, y)),
    );
    const termBackward = qMul(
      qMul(qConj(getLink(state, 1, x + 1, y - 1)), qConj(getLink(state, 0, x, y - 1))),
      getLink(state, 1, x, y - 1),
    );
    return qAdd(termForward, termBackward);
  }

  const termPlus = qMul(
    qMul(qConj(getLink(state, 0, x - 1, y + 1)), qConj(getLink(state, 1, x - 1, y))),
    getLink(state, 0, x - 1, y),
  );
  const termMinus = qMul(
    qMul(getLink(state, 0, x, y + 1), qConj(getLink(state, 1, x + 1, y))),
    qConj(getLink(state, 0, x, y)),
  );
  return qAdd(termPlus, termMinus);
}

function sampleA0CreutzKP(alpha, rng) {
  if (alpha <= 1e-12) {
    return { q: haarQuaternion(rng), attempts: 0, fallback: true };
  }

  // Exact rejection sampler for density proportional to
  // sqrt(1 - a0^2) * exp(alpha * a0), a0 in [-1, 1].
  // The exponential proposal is the Creutz/KP-style biased a0 draw; the
  // sqrt term is the SU(2) Haar correction.
  const expNeg2a = Math.exp(-2 * alpha);
  for (let attempts = 1; attempts <= 10000; attempts++) {
    const u = rng();
    const a0 = 1 + Math.log(expNeg2a + u * (1 - expNeg2a)) / alpha;
    const acceptProb = Math.sqrt(Math.max(0, 1 - a0 * a0));
    if (rng() <= acceptProb) {
      const r = Math.sqrt(Math.max(0, 1 - a0 * a0));
      const n = randomUnit3(rng);
      return { q: [a0, r * n[0], r * n[1], r * n[2]], attempts, fallback: false };
    }
  }
  return { q: haarQuaternion(rng), attempts: 10000, fallback: true };
}

export function heatbathLinkUpdate(state, mu, x, y, beta, rng, stats) {
  const staple = stapleForLink(state, mu, x, y);
  const k = qNorm(staple);
  stats.heatbathLinkUpdates++;
  if (k <= 1e-14) {
    stats.heatbathFallbackCount++;
    setLink(state, mu, x, y, haarQuaternion(rng));
    return;
  }

  const v = qScale(staple, 1 / k);
  const draw = sampleA0CreutzKP(beta * k, rng);
  stats.heatbathRejectionAttempts += draw.attempts;
  if (draw.fallback) stats.heatbathFallbackCount++;
  const next = qMul(draw.q, qConj(v));
  assertUnit(next, "heatbath next link");
  setLink(state, mu, x, y, next);
}

export function overrelaxLinkUpdate(state, mu, x, y) {
  const staple = stapleForLink(state, mu, x, y);
  const k = qNorm(staple);
  if (k <= 1e-14) return;
  const v = qScale(staple, 1 / k);
  const current = getLink(state, mu, x, y);
  const next = qMul(qMul(qConj(v), qConj(current)), qConj(v));
  assertUnit(next, "overrelax next link");
  setLink(state, mu, x, y, next);
}

export function combinedSweep(state, beta, overrelaxPerHeatbath, rng, stats) {
  const L = state.L;
  for (let mu = 0; mu < 2; mu++) {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        heatbathLinkUpdate(state, mu, x, y, beta, rng, stats);
      }
    }
  }
  for (let k = 0; k < overrelaxPerHeatbath; k++) {
    for (let mu = 0; mu < 2; mu++) {
      for (let x = 0; x < L; x++) {
        for (let y = 0; y < L; y++) {
          overrelaxLinkUpdate(state, mu, x, y);
        }
      }
    }
  }
  const unitResidual = maxLinkUnitarityResidual(state);
  if (unitResidual > UNITARITY_TOL) {
    throw new Error(`post-sweep unit residual ${unitResidual}`);
  }
}

// ------------------------------------------------ gauge transform -----

export function randomGaugeQuaternions(L, rng) {
  const gauges = new Float64Array(L * L * 4);
  for (let x = 0; x < L; x++) {
    for (let y = 0; y < L; y++) {
      const q = haarQuaternion(rng);
      const b = (x * L + y) * 4;
      gauges[b] = q[0];
      gauges[b + 1] = q[1];
      gauges[b + 2] = q[2];
      gauges[b + 3] = q[3];
    }
  }
  return gauges;
}

export function identityGaugeQuaternions(L) {
  const gauges = new Float64Array(L * L * 4);
  for (let i = 0; i < gauges.length; i += 4) gauges[i] = 1;
  return gauges;
}

function getGauge(gauges, L, x, y) {
  const b = (wrap(x, L) * L + wrap(y, L)) * 4;
  return [gauges[b], gauges[b + 1], gauges[b + 2], gauges[b + 3]];
}

export function applySU2GaugeTransform(state, gauges) {
  const L = state.L;
  const out = { L, links: new Float64Array(state.links.length) };
  for (let mu = 0; mu < 2; mu++) {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        const xn = mu === 0 ? x + 1 : x;
        const yn = mu === 1 ? y + 1 : y;
        const left = getGauge(gauges, L, x, y);
        const right = qConj(getGauge(gauges, L, xn, yn));
        const transformed = qMul(qMul(left, getLink(state, mu, x, y)), right);
        assertUnit(transformed, "gauge transformed link");
        setLink(out, mu, x, y, transformed);
      }
    }
  }
  return out;
}

// ------------------------------------------------ residuals -----------

export function signatureMaxAbsResidual(sigA, sigB) {
  let maxAbs = 0;
  for (const k of Object.keys(sigA)) {
    const d = Math.abs(sigA[k] - sigB[k]);
    if (d > maxAbs) maxAbs = d;
  }
  return maxAbs;
}

export function rawMatrixNormalizedL2(rawA, rawB) {
  let num = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < rawA.length; i++) {
    const d = rawA[i] - rawB[i];
    num += d * d;
    normA += rawA[i] * rawA[i];
    normB += rawB[i] * rawB[i];
  }
  const denom = Math.max(Math.sqrt(normA), Math.sqrt(normB), 1e-30);
  return Math.sqrt(num) / denom;
}

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
