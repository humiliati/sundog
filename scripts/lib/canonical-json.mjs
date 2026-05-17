// scripts/lib/canonical-json.mjs
//
// Shared canonical-JSON serializer for the structural-failure-coincidence
// program. Pins the algorithm that produces the byte string fed into
// SHA-256 for `calib_sha256` (and other canonical hashes used by the
// prereg artifacts), so that a Node implementation and the browser-side
// H0 measurement tool can be byte-identical by spec, not by accident.
//
// Spec reference:
//   docs/prereg/structural-failure-coincidence/P2_CUT3_H0_2_SCHEMA.md
//   (Finding-1 cross-runtime canonicalization pin, 2026-05-16 audit-notes).
//
// Cross-runtime test vector (pinned and verified):
//   results/structural-failure/cut3-prereg/h0-canonicalization-test-vector.json
//
// ALGORITHM (exact). For an input value, produce a UTF-8 string by:
//
//   value is an Array:
//     "[" + canonicalize(v0) + "," + canonicalize(v1) + ... + "]"
//     (order preserved; no spaces)
//
//   value is a non-null object:
//     keys = Object.keys(value).sort()   // lexicographic sort by code unit
//     "{" + JSON.stringify(k0) + ":" + canonicalize(v0)
//         + "," + JSON.stringify(k1) + ":" + canonicalize(v1)
//         + ... + "}"
//     (no spaces; recursive; keys sorted)
//
//   value is null, boolean, number, or string:
//     JSON.stringify(value)
//
// CONSEQUENCES of the algorithm (pinned, do not change):
//   - Top-level and ALL nested objects have keys sorted lexicographically.
//   - Arrays preserve insertion order.
//   - Numbers are formatted by the platform's JSON.stringify; ECMA-262
//     and Node's V8 produce identical output for finite IEEE-754 doubles
//     (this is the load-bearing cross-runtime assumption; see test vector).
//   - Strings are JSON-stringified (UTF-8 native passthrough for non-ASCII
//     except for control chars / surrogates handled by JSON.stringify).
//   - undefined values in objects → key is OMITTED (per JSON.stringify
//     semantics). Sidecars MUST NOT carry undefined values; use null.
//   - The output contains NO whitespace.
//
// SELF-PIN convention for sidecars (the calib_sha256 contract):
//   1. Take the sidecar object with `calib_sha256` set to the empty
//      string "" (NOT null, NOT absent).
//   2. canonicalize(...) it via the algorithm above.
//   3. SHA-256 the UTF-8 bytes of the resulting canonical string.
//   4. Hex-encode (lowercase) → that hex IS calib_sha256.
//   5. Write that hex back into the sidecar's `calib_sha256` field.
//
// Any other implementation (browser, alternative language, future
// Node port) MUST produce byte-identical canonical strings for the
// pinned cross-runtime test vector. That equality is the §0 acceptance
// criterion for the restored Node checker (Finding-1 audit-notes).

import { createHash } from "node:crypto";

export function canonicalize(value) {
  if (Array.isArray(value)) {
    return "[" + value.map(canonicalize).join(",") + "]";
  }
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return (
      "{" +
      keys
        .map((k) => JSON.stringify(k) + ":" + canonicalize(value[k]))
        .join(",") +
      "}"
    );
  }
  return JSON.stringify(value);
}

export function sha256Hex(str) {
  return createHash("sha256").update(str, "utf8").digest("hex");
}

// Self-pin a sidecar: returns the calib_sha256 hex string.
// Does NOT mutate the sidecar; caller writes the returned value into
// sidecar.calib_sha256 themselves (so the mutation is explicit, not
// hidden inside this helper).
export function computeSidecarSelfPin(sidecar) {
  const copy = { ...sidecar, calib_sha256: "" };
  const canon = canonicalize(copy);
  return sha256Hex(canon);
}

// Verify a sidecar's stored calib_sha256 against a fresh recompute.
// Returns { ok, recomputed }.
export function verifySidecarSelfPin(sidecar) {
  const recomputed = computeSidecarSelfPin(sidecar);
  return { ok: recomputed === sidecar.calib_sha256, recomputed };
}
