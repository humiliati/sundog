// Semantic verifier layer — interface + reference implementations (Phase 13 §13).
//
// A verifier is layer 2, behind the lexical gate, on the hosted-draft path. By
// CONTRACT it is reject-only: it may block (-> static-template fallback) but can
// NEVER approve, green-light, upgrade a tier, or raise confidence. `reject:false`
// means "no objection", not "endorsed". The layer is strictly subtractive.
//
// Interface:
//   verifier = { name, meta: {version, threshold, hash}, verify }
//   verify({ draft, route }) -> { reject: boolean, reason: string|null, score?: number }
//   route carries { id, evidenceTier, failureMode, boundaries[], answerTemplate, earned }
//
// `meta` is the provenance the harness records in every receipt (non-negotiable
// for model-backed verifiers, whose verdicts drift silently on model upgrade).

// Baseline: never objects. With this plugged in, the semantic layer contributes
// nothing and the harness reproduces the lexical-gate-only numbers — the control.
export const nullVerifier = {
  name: "null",
  meta: { version: "n/a", threshold: null, hash: "n/a" },
  verify() {
    return { reject: false, reason: null };
  }
};

// Wilson score-interval upper bound for `failures`/`n` at confidence z (default 95%).
// Required by the harness so an observed 0/n FPR is never reported as a bare "0%".
export function wilsonUpper(failures, n, z = 1.96) {
  if (n === 0) return 1;
  const phat = failures / n;
  const denom = 1 + (z * z) / n;
  const center = phat + (z * z) / (2 * n);
  const margin = z * Math.sqrt((phat * (1 - phat)) / n + (z * z) / (4 * n * n));
  return (center + margin) / denom;
}

// NLI-entailment verifier SCAFFOLD (front-runner candidate). Requires an injected
// `infer(premise, hypothesis) -> { label: 'entailment'|'neutral'|'contradiction', score }`
// — i.e. the actual NLI model. Buildable/testable now with a mock `infer`; a real
// measurement needs a real model (infra). Tests BOTH directions because NLI is
// unreliable on negation and our boundaries are phrased negatively:
//   (i)  does the draft ENTAIL a forbidden upgrade?  (premise = draft, hyp = upgrade)
//   (ii) does the draft CONTRADICT the honest boundary stance?
// Reject if either fires above threshold.
export function makeNliVerifier({ infer, threshold = 0.6, version = "unknown", hash = "unknown" }) {
  if (typeof infer !== "function") {
    throw new Error("makeNliVerifier requires infer(premise, hypothesis) -> {label, score} (the NLI model).");
  }
  return {
    name: "nli-entailment",
    meta: { version, threshold, hash },
    async verify({ draft, route }) {
      const reasons = [];
      // (i) does the draft ENTAIL a forbidden upgrade?
      for (const hyp of forbiddenUpgradeHypotheses(route)) {
        const r = await infer(draft, hyp);
        if (r && r.label === "entailment" && r.score >= threshold) {
          reasons.push(`entails_forbidden:${hyp}`);
        }
      }
      // (ii) does the draft CONTRADICT the honest boundary stance?
      for (const hyp of boundaryPropositions(route)) {
        const r = await infer(draft, hyp);
        if (r && r.label === "contradiction" && r.score >= threshold) {
          reasons.push(`contradicts_boundary:${hyp}`);
        }
      }
      return { reject: reasons.length > 0, reason: reasons[0] || null, score: reasons.length };
    }
  };
}

// Initial hypothesis derivations for the NLI candidate. These are deliberately
// simple and WILL be refined once a real model is measuring them — the both-
// directions structure is the load-bearing part, not the exact wording.
const TAG_UPGRADE = {
  "identity-success": "Sundog made an original new-physics discovery.",
  "exact-separation": "Sundog proved an original new theorem.",
  "marginal": "Sundog solved the problem.",
  "numerical": "Sundog produced a real result from the run.",
  "bounded-null": "Sundog found a real separating result.",
  "vacuous": "Sundog produced an informative result.",
  "bounded-positive": "Sundog resolved the complexity problem.",
  "convergence-to-null": "Sundog solved the benchmark with a high-dimensional body.",
  "conditional": "Sundog produced a general, unconditional predictor.",
  "deflationary": "The controller senses the hidden field directly.",
  "explainer-tier": "Sundog proved the result itself.",
  "build-gate-partial": "The reasoner works and is shipping."
};

export function forbiddenUpgradeHypotheses(route) {
  const out = [];
  if (route.failureMode && TAG_UPGRADE[route.failureMode]) out.push(TAG_UPGRADE[route.failureMode]);
  return out;
}

export function boundaryPropositions(route) {
  // Convert imperative boundaries ("Do not claim X") into declarative honest
  // stances ("Sundog does not claim X") so contradiction has a proposition to bite.
  return (route.boundaries || []).map((b) =>
    b.replace(/^do not\s+/i, "Sundog does not ").replace(/\.$/, "") + "."
  );
}
