// scripts/lib/yang-mills-su2-3d-core.mjs
//
// SU(2) 3D lattice gauge-theory core for the Yang-Mills Phase 1 SU(2)
// gauge-invariance smoke runner. This module is intentionally self-contained
// until all three Phase 1 receipts are green.

const TWO_PI = 2 * Math.PI;

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
  if (n <= 0 || !Number.isFinite(n)) return [1, 0, 0, 0];
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function qAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
}

function qScale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s, a[3] * s];
}

export function qUnitarityFrobeniusResidual(q) {
  const normSq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  return Math.SQRT2 * Math.abs(1 - normSq);
}

// ------------------------------------------------------------ state ----

// State layout: links[((mu * L^3 + (x * L + y) * L + z) * 4) + component].
// mu = 0 is +x, mu = 1 is +y, mu = 2 is +z.
export function createSU2Lattice(L, seed) {
  const links = new Float64Array(3 * L * L * L * 4);
  const rng = mulberry32(seed >>> 0);
  const state = { L, links };
  for (let mu = 0; mu < 3; mu++) {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        for (let z = 0; z < L; z++) {
          setLink(state, mu, x, y, z, haarQuaternion(rng));
        }
      }
    }
  }
  return state;
}

export function cloneSU2Lattice(state) {
  return { L: state.L, links: new Float64Array(state.links) };
}

function wrap(i, L) {
  return ((i % L) + L) % L;
}

function linkBase(L, mu, x, y, z) {
  return (mu * L * L * L + (wrap(x, L) * L + wrap(y, L)) * L + wrap(z, L)) * 4;
}

function dirX(mu) {
  return mu === 0 ? 1 : 0;
}

function dirY(mu) {
  return mu === 1 ? 1 : 0;
}

function dirZ(mu) {
  return mu === 2 ? 1 : 0;
}

export function getLink(state, mu, x, y, z) {
  const b = linkBase(state.L, mu, x, y, z);
  return [
    state.links[b],
    state.links[b + 1],
    state.links[b + 2],
    state.links[b + 3],
  ];
}

function setLink(state, mu, x, y, z, q) {
  const n = qNormalize(q);
  const b = linkBase(state.L, mu, x, y, z);
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
    const residual = qUnitarityFrobeniusResidual(q);
    if (!Number.isFinite(residual)) return Number.POSITIVE_INFINITY;
    if (residual > maxResidual) maxResidual = residual;
  }
  return maxResidual;
}

// ----------------------------------------------------- loops ----------

export function plaquetteQuaternion(state, x, y, z, mu, nu) {
  const mx = dirX(mu), my = dirY(mu), mz = dirZ(mu);
  const nx = dirX(nu), ny = dirY(nu), nz = dirZ(nu);
  const uMu = getLink(state, mu, x, y, z);
  const uNuMu = getLink(state, nu, x + mx, y + my, z + mz);
  const uMuNu = getLink(state, mu, x + nx, y + ny, z + nz);
  const uNu = getLink(state, nu, x, y, z);
  return qMul(qMul(qMul(uMu, uNuMu), qConj(uMuNu)), qConj(uNu));
}

const ORIENTATIONS = Object.freeze([
  ["xy", 0, 1],
  ["xz", 0, 2],
  ["yz", 1, 2],
]);

export function meanPlaquetteByOrientation(state) {
  const L = state.L;
  const out = {};
  for (const [name, mu, nu] of ORIENTATIONS) {
    let sum = 0;
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        for (let z = 0; z < L; z++) {
          sum += plaquetteQuaternion(state, x, y, z, mu, nu)[0];
        }
      }
    }
    out[name] = sum / (L * L * L);
  }
  return out;
}

export function meanPlaquette(state) {
  const byOrientation = meanPlaquetteByOrientation(state);
  return (byOrientation.xy + byOrientation.xz + byOrientation.yz) / 3;
}

export function orientationRelativeSpread(byOrientation) {
  const values = [byOrientation.xy, byOrientation.xz, byOrientation.yz];
  let maxSpread = 0;
  for (let i = 0; i < values.length; i++) {
    for (let j = i + 1; j < values.length; j++) {
      const denom = Math.max(Math.abs(values[i]), Math.abs(values[j]), 1e-12);
      maxSpread = Math.max(maxSpread, Math.abs(values[i] - values[j]) / denom);
    }
  }
  return maxSpread;
}

function wilsonLoopTraceHalfInPlane(state, x, y, z, mu, nu, nMu, nNu) {
  let acc = [1, 0, 0, 0];
  const mx = dirX(mu), my = dirY(mu), mz = dirZ(mu);
  const nx = dirX(nu), ny = dirY(nu), nz = dirZ(nu);
  for (let i = 0; i < nMu; i++) {
    acc = qMul(acc, getLink(state, mu, x + i * mx, y + i * my, z + i * mz));
  }
  for (let j = 0; j < nNu; j++) {
    acc = qMul(
      acc,
      getLink(
        state,
        nu,
        x + nMu * mx + j * nx,
        y + nMu * my + j * ny,
        z + nMu * mz + j * nz,
      ),
    );
  }
  for (let i = 0; i < nMu; i++) {
    acc = qMul(
      acc,
      qConj(
        getLink(
          state,
          mu,
          x + (nMu - 1 - i) * mx + nNu * nx,
          y + (nMu - 1 - i) * my + nNu * ny,
          z + (nMu - 1 - i) * mz + nNu * nz,
        ),
      ),
    );
  }
  for (let j = 0; j < nNu; j++) {
    acc = qMul(
      acc,
      qConj(
        getLink(
          state,
          nu,
          x + (nNu - 1 - j) * nx,
          y + (nNu - 1 - j) * ny,
          z + (nNu - 1 - j) * nz,
        ),
      ),
    );
  }
  return acc[0];
}

function loopMeanVar(state, nA, nB) {
  const L = state.L;
  let n = 0;
  let sum = 0;
  let sumSq = 0;
  const accumulate = (mu, nu, a, b) => {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        for (let z = 0; z < L; z++) {
          const v = wilsonLoopTraceHalfInPlane(state, x, y, z, mu, nu, a, b);
          sum += v;
          sumSq += v * v;
          n++;
        }
      }
    }
  };
  for (const [, mu, nu] of ORIENTATIONS) {
    accumulate(mu, nu, nA, nB);
    if (nA !== nB) accumulate(mu, nu, nB, nA);
  }
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

function stapleForLink(state, mu, x, y, z) {
  let staple = [0, 0, 0, 0];
  const mx = dirX(mu), my = dirY(mu), mz = dirZ(mu);
  for (let nu = 0; nu < 3; nu++) {
    if (nu === mu) continue;
    const nx = dirX(nu), ny = dirY(nu), nz = dirZ(nu);
    const termForward = qMul(
      qMul(
        getLink(state, nu, x + mx, y + my, z + mz),
        qConj(getLink(state, mu, x + nx, y + ny, z + nz)),
      ),
      qConj(getLink(state, nu, x, y, z)),
    );
    const termBackward = qMul(
      qMul(
        qConj(getLink(state, nu, x + mx - nx, y + my - ny, z + mz - nz)),
        qConj(getLink(state, mu, x - nx, y - ny, z - nz)),
      ),
      getLink(state, nu, x - nx, y - ny, z - nz),
    );
    staple = qAdd(qAdd(staple, termForward), termBackward);
  }
  return staple;
}

function sampleA0CreutzKP(alpha, rng) {
  if (alpha <= 1e-12) {
    return { q: haarQuaternion(rng), attempts: 0, fallback: true };
  }

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

export function heatbathLinkUpdate(state, mu, x, y, z, beta, rng, stats) {
  const staple = stapleForLink(state, mu, x, y, z);
  const k = qNorm(staple);
  stats.heatbathLinkUpdates++;
  if (k <= 1e-14) {
    stats.heatbathFallbackCount++;
    setLink(state, mu, x, y, z, haarQuaternion(rng));
    return;
  }

  const v = qScale(staple, 1 / k);
  const draw = sampleA0CreutzKP(beta * k, rng);
  stats.heatbathRejectionAttempts += draw.attempts;
  if (draw.fallback) stats.heatbathFallbackCount++;
  setLink(state, mu, x, y, z, qMul(draw.q, qConj(v)));
}

export function overrelaxLinkUpdate(state, mu, x, y, z) {
  const staple = stapleForLink(state, mu, x, y, z);
  const k = qNorm(staple);
  if (k <= 1e-14) return;
  const v = qScale(staple, 1 / k);
  const current = getLink(state, mu, x, y, z);
  setLink(state, mu, x, y, z, qMul(qMul(qConj(v), qConj(current)), qConj(v)));
}

export function combinedSweep(state, beta, overrelaxPerHeatbath, rng, stats) {
  const L = state.L;
  for (let mu = 0; mu < 3; mu++) {
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        for (let z = 0; z < L; z++) {
          heatbathLinkUpdate(state, mu, x, y, z, beta, rng, stats);
        }
      }
    }
  }
  for (let k = 0; k < overrelaxPerHeatbath; k++) {
    for (let mu = 0; mu < 3; mu++) {
      for (let x = 0; x < L; x++) {
        for (let y = 0; y < L; y++) {
          for (let z = 0; z < L; z++) {
            overrelaxLinkUpdate(state, mu, x, y, z);
          }
        }
      }
    }
  }
}

// ------------------------------------------------ gauge transform -----

export function randomGaugeQuaternions(L, rng) {
  const gauges = new Float64Array(L * L * L * 4);
  for (let x = 0; x < L; x++) {
    for (let y = 0; y < L; y++) {
      for (let z = 0; z < L; z++) {
        const q = haarQuaternion(rng);
        const b = ((x * L + y) * L + z) * 4;
        gauges[b] = q[0];
        gauges[b + 1] = q[1];
        gauges[b + 2] = q[2];
        gauges[b + 3] = q[3];
      }
    }
  }
  return gauges;
}

export function identityGaugeQuaternions(L) {
  const gauges = new Float64Array(L * L * L * 4);
  for (let i = 0; i < gauges.length; i += 4) gauges[i] = 1;
  return gauges;
}

function getGauge(gauges, L, x, y, z) {
  const b = ((wrap(x, L) * L + wrap(y, L)) * L + wrap(z, L)) * 4;
  return [gauges[b], gauges[b + 1], gauges[b + 2], gauges[b + 3]];
}

export function applySU2GaugeTransform(state, gauges) {
  const L = state.L;
  const out = { L, links: new Float64Array(state.links.length) };
  for (let mu = 0; mu < 3; mu++) {
    const mx = dirX(mu), my = dirY(mu), mz = dirZ(mu);
    for (let x = 0; x < L; x++) {
      for (let y = 0; y < L; y++) {
        for (let z = 0; z < L; z++) {
          const left = getGauge(gauges, L, x, y, z);
          const right = qConj(getGauge(gauges, L, x + mx, y + my, z + mz));
          const transformed = qMul(qMul(left, getLink(state, mu, x, y, z)), right);
          setLink(out, mu, x, y, z, transformed);
        }
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
