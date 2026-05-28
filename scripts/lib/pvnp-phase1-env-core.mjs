// scripts/lib/pvnp-phase1-env-core.mjs
//
// Environment generator for the SUNDOG_V_P_V_NP Phase 1 toy verifier.
//
// Spec references:
//   docs/pvnp/PHASE1_TOY_VERIFIER_SPEC.md §Object Under Test
//   docs/pvnp/PHASE1_V0_SLATE.md §Environment Parameters
//
// Each environment lives on the unit square D = [0, 1]^2 with:
//
//   start_region:  thin left strip x ∈ [0.05, 0.10]
//   goal_region:   thin right strip x ∈ [0.90, 0.95]
//   basin:         hidden unsafe region (one of 4 families)
//   field F:       signed distance to basin (positive outside, negative inside)
//   probes:        local field samples available to policy/signature, subject
//                  to a registered noise tier (none / gaussian / dropout-delay)
//
// PRIVACY CONVENTION: each generated env carries a `hidden_state` field that
// holds basin parameters, latent field parameters, and any decoy parameters.
// The verifier-side loader (loadVerifierEnvironments) strips `hidden_state`
// before passing envs to verifier or signature code. The evaluator
// (lib/pvnp-phase1-evaluator-core.mjs) is the only module that reads
// `hidden_state`. The privilege-leak audit greps verifier modules for these
// token names and requires zero matches.

import { makeRng, uniform, gaussian, choice } from "./pvnp-phase1-rng.mjs";

export const BASIN_FAMILIES = Object.freeze([
  "circle",
  "ellipse",
  "crescent",
  "decoy_doublet",
]);

export const PROBE_NOISE_TIERS = Object.freeze([
  "none",
  "gaussian",
  "dropout_delay",
]);

export const START_REGION = Object.freeze({
  x_min: 0.05, x_max: 0.10, y_min: 0.20, y_max: 0.80,
});

export const GOAL_REGION = Object.freeze({
  x_min: 0.90, x_max: 0.95, y_min: 0.20, y_max: 0.80,
});

export const DOMAIN = Object.freeze({ x_min: 0, x_max: 1, y_min: 0, y_max: 1 });

// Promise-domain bounds. Outside these → env is out-of-promise; verifier
// must return quarantine. We use these to construct the falsifier split.
export const PROMISE_BOUNDS = Object.freeze({
  basin_min_diameter: 0.12,
  basin_max_diameter: 0.30,
  basin_max_curvature: 32.0,     // inverse of min radius of curvature
  field_max_lipschitz: 4.0,
  probe_noise_max_std: 0.05,
  probe_dropout_max_rate: 0.15,
  probe_delay_max_steps: 2,
});

// -- Basin geometry helpers ------------------------------------------------

// Signed distance to the basin: positive outside, negative inside, zero on
// boundary. Magnitude is meters (i.e., domain units).
export function signedDistanceToBasin(x, y, basin) {
  switch (basin.family) {
    case "circle": {
      const dx = x - basin.cx;
      const dy = y - basin.cy;
      return Math.hypot(dx, dy) - basin.r;
    }
    case "ellipse": {
      const c = Math.cos(basin.angle);
      const s = Math.sin(basin.angle);
      const u = ((x - basin.cx) * c + (y - basin.cy) * s) / basin.a;
      const v = (-(x - basin.cx) * s + (y - basin.cy) * c) / basin.b;
      // Approximate signed distance for an ellipse (not exact; conservative
      // bound for small eccentricities, fine for the toy).
      const norm = Math.hypot(u, v);
      const local = norm - 1;
      const scale = Math.min(basin.a, basin.b);
      return local * scale;
    }
    case "crescent": {
      // Crescent = big disk minus shifted small disk. Signed distance is
      // outer signed distance unless inside the cutout, then -inner.
      const outerD = Math.hypot(x - basin.cx, y - basin.cy) - basin.r_outer;
      const innerD = Math.hypot(x - basin.cx - basin.shift, y - basin.cy)
        - basin.r_inner;
      if (outerD < 0 && innerD < 0) {
        // inside outer but also inside the cutout → safe interior of crescent
        return -innerD; // positive: distance to crescent (basin) boundary
      }
      if (outerD < 0) {
        // inside the basin
        return Math.max(outerD, -innerD);
      }
      return outerD;
    }
    case "decoy_doublet": {
      // Two close-together circles fused as one basin.
      const d1 = Math.hypot(x - basin.cx1, y - basin.cy1) - basin.r1;
      const d2 = Math.hypot(x - basin.cx2, y - basin.cy2) - basin.r2;
      return Math.min(d1, d2);
    }
    default:
      throw new Error(`Unknown basin family: ${basin.family}`);
  }
}

// Generate basin parameters within the promise envelope. The basin must
// lie strictly inside the domain with a margin from start/goal regions.
function sampleBasinParams(rng, family, options = {}) {
  const minDiam = options.minDiameter ?? 0.14;
  const maxDiam = options.maxDiameter ?? 0.26;

  // Allowed basin-center band: middle of the domain, away from start/goal.
  const cxBand = [0.30, 0.75];
  const cyBand = [0.15, 0.85];

  switch (family) {
    case "circle": {
      const r = uniform(rng, minDiam / 2, maxDiam / 2);
      const cx = uniform(rng, cxBand[0], cxBand[1]);
      const cy = uniform(rng, cyBand[0], cyBand[1]);
      return { family, cx, cy, r };
    }
    case "ellipse": {
      const a = uniform(rng, minDiam / 2, maxDiam / 2);
      const b = uniform(rng, minDiam / 2.4, maxDiam / 2);
      const angle = uniform(rng, 0, Math.PI);
      const cx = uniform(rng, cxBand[0], cxBand[1]);
      const cy = uniform(rng, cyBand[0], cyBand[1]);
      return { family, cx, cy, a, b, angle };
    }
    case "crescent": {
      const r_outer = uniform(rng, minDiam / 2, maxDiam / 2);
      const r_inner = r_outer * uniform(rng, 0.55, 0.75);
      const shift = r_outer * uniform(rng, 0.30, 0.55);
      const cx = uniform(rng, cxBand[0] + 0.05, cxBand[1] - 0.05);
      const cy = uniform(rng, cyBand[0] + 0.05, cyBand[1] - 0.05);
      return { family, cx, cy, r_outer, r_inner, shift };
    }
    case "decoy_doublet": {
      const r1 = uniform(rng, minDiam / 2.4, maxDiam / 2.4);
      const r2 = r1 * uniform(rng, 0.85, 1.15);
      const cx1 = uniform(rng, cxBand[0], cxBand[1]);
      const cy1 = uniform(rng, cyBand[0], cyBand[1]);
      const sep = uniform(rng, r1 * 0.9, r1 * 1.6);
      const angle = uniform(rng, 0, 2 * Math.PI);
      const cx2 = cx1 + sep * Math.cos(angle);
      const cy2 = cy1 + sep * Math.sin(angle);
      return { family, cx1, cy1, r1, cx2, cy2, r2 };
    }
    default:
      throw new Error(`Unknown basin family: ${family}`);
  }
}

// Sample probe noise tier params. Stays inside promise envelope by default.
// For falsifier split, callers may pass `inPromise: false` to push tiers
// beyond bounds.
function sampleNoiseTierParams(rng, tier, options = {}) {
  const inPromise = options.inPromise !== false;
  switch (tier) {
    case "none":
      return { tier, std: 0, dropout_rate: 0, delay_steps: 0 };
    case "gaussian": {
      const std = inPromise
        ? uniform(rng, 0.005, 0.04)
        : uniform(rng, 0.08, 0.18);
      return { tier, std, dropout_rate: 0, delay_steps: 0 };
    }
    case "dropout_delay": {
      const dropout = inPromise
        ? uniform(rng, 0.02, 0.12)
        : uniform(rng, 0.20, 0.40);
      const delay = inPromise
        ? Math.floor(uniform(rng, 0, 2.999))
        : Math.floor(uniform(rng, 3, 5.999));
      return { tier, std: 0.01, dropout_rate: dropout, delay_steps: delay };
    }
    default:
      throw new Error(`Unknown probe noise tier: ${tier}`);
  }
}

// Generate the full slate for a split. Returns an array of env objects.
// Cycles basin families and probe-noise tiers in a balanced round-robin
// pattern (deterministic) so every split has uniform coverage.
export function generateSplit({ split, count, seedPrefix, inPromise = true }) {
  const envs = [];
  for (let i = 0; i < count; i += 1) {
    const envIndex = i + 1;
    const id = `${seedPrefix}-${String(envIndex).padStart(4, "0")}`;
    const rng = makeRng(id);
    const family = BASIN_FAMILIES[i % BASIN_FAMILIES.length];
    const tier = PROBE_NOISE_TIERS[Math.floor(i / BASIN_FAMILIES.length) % PROBE_NOISE_TIERS.length];

    const basin = sampleBasinParams(rng, family, {
      minDiameter: inPromise
        ? PROMISE_BOUNDS.basin_min_diameter
        : Math.max(0.05, PROMISE_BOUNDS.basin_min_diameter * 0.55),
      maxDiameter: inPromise
        ? PROMISE_BOUNDS.basin_max_diameter
        : PROMISE_BOUNDS.basin_max_diameter * 1.25,
    });
    const noise = sampleNoiseTierParams(rng, tier, { inPromise });

    envs.push({
      id,
      split,
      seed_namespace: id,
      domain: DOMAIN,
      start_region: START_REGION,
      goal_region: GOAL_REGION,
      horizon: 128,
      max_action_step: 0.025,
      basin_family: family,
      probe_noise_tier: noise.tier,
      probe_noise_params: { std: noise.std, dropout_rate: noise.dropout_rate, delay_steps: noise.delay_steps },
      decoy_signature_present: false,
      promise_compliance: inPromise ? "in_promise" : "out_of_promise",
      hidden_state: {
        basin_params: basin,
        latent_field: { kind: "signed_distance_to_basin" },
        decoy_params: null,
      },
    });
  }
  return envs;
}

// Strip the privileged fields from an env so verifier/signature code never
// sees them. Returns a shallow copy with `hidden_state` removed.
export function redactForVerifier(env) {
  const { hidden_state: _hidden, ...rest } = env;
  return rest;
}

// Load envs as written and split into public + hidden sets. Convenience for
// runners that need both views.
export function splitPublicAndHidden(envs) {
  const publicEnvs = envs.map(redactForVerifier);
  const hidden = envs.map((e) => ({ id: e.id, hidden_state: e.hidden_state }));
  return { publicEnvs, hidden };
}
