// scripts/lib/yang-mills-su2-3d-smearing.mjs
//
// APE smearing helpers for the Yang-Mills SU(2) 3D Phase 2 v1 aggregation
// runner. This intentionally leaves the Phase 1 / Phase 2 v0 SU(2) core
// untouched.

import {
  getLink,
  qMul,
  qConj,
  qUnitarityFrobeniusResidual,
} from "./yang-mills-su2-3d-core.mjs";

export const APE_SMEARING_LOCK = Object.freeze({
  algorithm: "APE",
  alpha: 0.5,
  iterations: 10,
  dimension: 3,
  stapleNormalization: 1 / 4,
});

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

function qAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
}

function qScale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s, a[3] * s];
}

function qNorm(q) {
  return Math.hypot(q[0], q[1], q[2], q[3]);
}

function qProjectSU2(q) {
  const n = qNorm(q);
  if (!Number.isFinite(n) || n <= 0) {
    return { q: [1, 0, 0, 0], detDrift: Number.POSITIVE_INFINITY, unitarityResidual: 0 };
  }
  const projected = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
  const normSq =
    projected[0] * projected[0] +
    projected[1] * projected[1] +
    projected[2] * projected[2] +
    projected[3] * projected[3];
  return {
    q: projected,
    detDrift: Math.abs(normSq - 1),
    unitarityResidual: qUnitarityFrobeniusResidual(projected),
  };
}

function setRawLink(out, mu, x, y, z, q) {
  const b = linkBase(out.L, mu, x, y, z);
  out.links[b] = q[0];
  out.links[b + 1] = q[1];
  out.links[b + 2] = q[2];
  out.links[b + 3] = q[3];
}

function stapleForLink(state, mu, x, y, z) {
  let staple = [0, 0, 0, 0];
  const mx = dirX(mu), my = dirY(mu), mz = dirZ(mu);
  for (let nu = 0; nu < 3; nu++) {
    if (nu === mu) continue;
    const nx = dirX(nu), ny = dirY(nu), nz = dirZ(nu);
    const termForward = qMul(
      qMul(
        getLink(state, nu, x, y, z),
        getLink(state, mu, x + nx, y + ny, z + nz),
      ),
      qConj(getLink(state, nu, x + mx, y + my, z + mz)),
    );
    const termBackward = qMul(
      qMul(
        qConj(getLink(state, nu, x - nx, y - ny, z - nz)),
        getLink(state, mu, x - nx, y - ny, z - nz),
      ),
      getLink(state, nu, x + mx - nx, y + my - ny, z + mz - nz),
    );
    staple = qAdd(qAdd(staple, termForward), termBackward);
  }
  return staple;
}

export function apeSmearSU2_3D(state, options = {}) {
  const alpha = options.alpha ?? APE_SMEARING_LOCK.alpha;
  const iterations = options.iterations ?? APE_SMEARING_LOCK.iterations;
  if (alpha !== APE_SMEARING_LOCK.alpha) throw new Error(`APE alpha drift: ${alpha}`);
  if (iterations !== APE_SMEARING_LOCK.iterations) throw new Error(`APE iteration drift: ${iterations}`);

  const L = state.L;
  let current = { L, links: new Float64Array(state.links) };
  const iterationHealth = [];
  const stapleWeight = alpha * APE_SMEARING_LOCK.stapleNormalization;
  const linkWeight = 1 - alpha;

  for (let iteration = 1; iteration <= iterations; iteration++) {
    const next = { L, links: new Float64Array(current.links.length) };
    let maxDetDrift = 0;
    let maxUnitarityResidual = 0;
    for (let mu = 0; mu < 3; mu++) {
      for (let x = 0; x < L; x++) {
        for (let y = 0; y < L; y++) {
          for (let z = 0; z < L; z++) {
            const u = getLink(current, mu, x, y, z);
            const staple = stapleForLink(current, mu, x, y, z);
            const mixed = qAdd(qScale(u, linkWeight), qScale(staple, stapleWeight));
            const projected = qProjectSU2(mixed);
            if (projected.detDrift > maxDetDrift) maxDetDrift = projected.detDrift;
            if (projected.unitarityResidual > maxUnitarityResidual) {
              maxUnitarityResidual = projected.unitarityResidual;
            }
            setRawLink(next, mu, x, y, z, projected.q);
          }
        }
      }
    }
    iterationHealth.push({ iteration, maxDetDrift, maxUnitarityResidual });
    current = next;
  }

  return { state: current, iterationHealth };
}
