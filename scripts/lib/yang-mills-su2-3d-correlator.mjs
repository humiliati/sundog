// scripts/lib/yang-mills-su2-3d-correlator.mjs
//
// Connected 2-point Wilson-loop correlator helpers for Yang-Mills Phase 2 v2.

import {
  getLink,
  qMul,
  qConj,
} from "./yang-mills-su2-3d-core.mjs";

export const CORRELATOR_LOOP_CLASSES = Object.freeze([
  { id: "W11", nMu: 1, nNu: 1 },
  { id: "W12", nMu: 1, nNu: 2 },
  { id: "W13", nMu: 1, nNu: 3 },
  { id: "W22", nMu: 2, nNu: 2 },
]);

export const DISPLACEMENT_CLASS_REPRESENTATIVES = Object.freeze([
  { id: "r1", representative: [1, 0, 0] },
  { id: "r2", representative: [1, 1, 0] },
  { id: "r3", representative: [1, 1, 1] },
  { id: "r4", representative: [2, 0, 0] },
  { id: "r5", representative: [2, 1, 0] },
]);

const ORIENTATIONS = Object.freeze([
  ["xy", 0, 1],
  ["xz", 0, 2],
  ["yz", 1, 2],
]);

function wrap(i, L) {
  return ((i % L) + L) % L;
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

function siteIndex(L, x, y, z) {
  return (wrap(x, L) * L + wrap(y, L)) * L + wrap(z, L);
}

function signedValues(v) {
  if (v === 0) return [0];
  return [-v, v];
}

function permutations3(values) {
  const out = [];
  const used = new Set();
  const push = (a) => {
    const key = a.join(",");
    if (!used.has(key)) {
      used.add(key);
      out.push(a);
    }
  };
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      if (j === i) continue;
      for (let k = 0; k < 3; k++) {
        if (k === i || k === j) continue;
        push([values[i], values[j], values[k]]);
      }
    }
  }
  return out;
}

function cubicClass(rep) {
  const vectors = [];
  const seen = new Set();
  for (const perm of permutations3(rep)) {
    for (const sx of signedValues(perm[0])) {
      for (const sy of signedValues(perm[1])) {
        for (const sz of signedValues(perm[2])) {
          const v = [sx, sy, sz];
          const key = v.join(",");
          if (!seen.has(key)) {
            seen.add(key);
            vectors.push(v);
          }
        }
      }
    }
  }
  vectors.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  return vectors;
}

export function displacementClasses() {
  return DISPLACEMENT_CLASS_REPRESENTATIVES.map((d) => ({
    id: d.id,
    representative: d.representative,
    classSize: cubicClass(d.representative).length,
    vectors: cubicClass(d.representative),
  }));
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

export function loopPositionArray(state, loopClass) {
  const L = state.L;
  const values = new Float64Array(L * L * L);
  let sum = 0;
  for (let x = 0; x < L; x++) {
    for (let y = 0; y < L; y++) {
      for (let z = 0; z < L; z++) {
        let v = 0;
        for (const [, mu, nu] of ORIENTATIONS) {
          v += wilsonLoopTraceHalfInPlane(state, x, y, z, mu, nu, loopClass.nMu, loopClass.nNu);
        }
        v /= ORIENTATIONS.length;
        values[siteIndex(L, x, y, z)] = v;
        sum += v;
      }
    }
  }
  return { values, mean: sum / values.length };
}

export function computeCorrelatorSignatureV5(state) {
  const L = state.L;
  const classes = displacementClasses();
  const signature = [];
  const named = {};
  for (const loopClass of CORRELATOR_LOOP_CLASSES) {
    const field = loopPositionArray(state, loopClass);
    const meanSq = field.mean * field.mean;
    for (const displacementClass of classes) {
      let productSum = 0;
      let count = 0;
      for (const [dx, dy, dz] of displacementClass.vectors) {
        for (let x = 0; x < L; x++) {
          for (let y = 0; y < L; y++) {
            for (let z = 0; z < L; z++) {
              const a = field.values[siteIndex(L, x, y, z)];
              const b = field.values[siteIndex(L, x + dx, y + dy, z + dz)];
              productSum += a * b;
              count++;
            }
          }
        }
      }
      const value = productSum / count - meanSq;
      const key = `${loopClass.id}_${displacementClass.id}`;
      named[key] = value;
      signature.push(value);
    }
  }
  return { vector: new Float64Array(signature), named };
}

export function correlatorSignatureKeys() {
  const out = [];
  for (const loopClass of CORRELATOR_LOOP_CLASSES) {
    for (const displacementClass of DISPLACEMENT_CLASS_REPRESENTATIVES) {
      out.push(`${loopClass.id}_${displacementClass.id}`);
    }
  }
  return out;
}
