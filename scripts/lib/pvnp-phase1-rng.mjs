// scripts/lib/pvnp-phase1-rng.mjs
//
// Deterministic seeded PRNG for the SUNDOG_V_P_V_NP Phase 1 toy verifier.
//
// Spec reference:
//   docs/pvnp/PHASE1_V0_SLATE.md §Split Lock
//
// The seed namespace is the env-id string (e.g. "pvnp-v0-cal-0001"). All
// random draws for that environment derive deterministically from the
// SHA-256 of that namespace, mixed with a counter for the draw index.
//
// This is a sfc32 PRNG seeded from 128 bits of namespace-derived entropy.
// Pure JS, no Node-crypto required at draw time (we only use crypto once,
// when constructing the RNG, for namespace → 128-bit seed). The PRNG itself
// is portable.

import { createHash } from "node:crypto";

function namespaceToSeed(namespace) {
  const hash = createHash("sha256").update(namespace, "utf8").digest();
  // First 16 bytes → 4 × uint32
  return [
    hash.readUInt32BE(0),
    hash.readUInt32BE(4),
    hash.readUInt32BE(8),
    hash.readUInt32BE(12),
  ];
}

// sfc32 (Small Fast Counter, 32-bit). 128-bit state, period ≈ 2^128.
// Returns a function () => float in [0, 1).
function sfc32(a, b, c, d) {
  return function next() {
    a |= 0; b |= 0; c |= 0; d |= 0;
    const t = ((a + b) | 0) + d | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = ((c << 21) | (c >>> 11));
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}

export function makeRng(namespace) {
  const [a, b, c, d] = namespaceToSeed(namespace);
  return sfc32(a, b, c, d);
}

// Uniform on [lo, hi).
export function uniform(rng, lo, hi) {
  return lo + (hi - lo) * rng();
}

// Gaussian via Box-Muller. Mean 0, std 1.
export function gaussian(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Pick from a discrete list with uniform probability.
export function choice(rng, items) {
  return items[Math.floor(rng() * items.length)];
}

// Integer on [lo, hi] inclusive.
export function intRange(rng, lo, hi) {
  return lo + Math.floor(rng() * (hi - lo + 1));
}
