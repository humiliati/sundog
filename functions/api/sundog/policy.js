import { jsonResponse, errorResponse, withCors, handleOptions } from "./_lib.js";

/**
 * GET /api/sundog/policy
 *
 * Returns the photo-data policy as JSON (clients embed it inline before
 * showing the consent checkbox) and also serves the human-readable
 * Markdown via /docs/PHOTO_DATA_POLICY.md (separate static asset).
 *
 * Keeping the policy text in JSON form here lets the upload widget show the
 * consent dialog without a separate fetch round-trip to the Markdown file.
 */

const POLICY_BODY = {
  document: "Sundog Photo Data Policy",
  applies_to: "POST /api/sundog/upload",
  summary: [
    "We render the atlas on your photo entirely in your browser.",
    "If you check 'Share my photo with the Sundog project', we POST the photo (without EXIF) and the JSON pose to our Cloudflare R2 bucket.",
    "You receive a deletion token at submission time. Keep it; it is the only way to remove your data.",
    "We never persist your IP address or the raw EXIF metadata from your photo.",
  ],
  data_we_collect_only_if_you_share: [
    "The re-encoded JPEG/WebP/PNG of your photo (EXIF stripped client-side before upload).",
    "The JSON pose you produced: sun pixel, R₂₂, parhelion offset, inverse-inferred altitude.",
    "Country code derived from your IP at the Cloudflare edge (e.g. 'US') — never the IP itself.",
    "A SHA-256 hash of your User-Agent string, truncated to 32 hex chars. The raw UA is not persisted.",
    "ISO timestamp of submission.",
  ],
  data_we_use_it_for: [
    "Improving the Halo Atlas geometric model (calibration, edge-case detection).",
    "Building public illustration assets (with separate consent for any individual photo being published).",
  ],
  retention: {
    period: "12 months from submission, unless aggregated into a derived dataset.",
    note: "Aggregated derived datasets that have already incorporated a submission may persist statistical features beyond the deletion request, but the original photo and pose are removed.",
  },
  deletion: {
    how: "POST /api/sundog/delete with { token: '<your-deletion-token>' } or visit your deletion URL.",
    sla: "Within 24 hours under normal conditions.",
    storage: "We store a SHA-256 hash of your deletion token, never the token itself. A leaked metadata file cannot be used to delete other submissions.",
  },
  not_collected: [
    "Raw IP address (only the country code, derived at edge).",
    "Raw User-Agent (only its SHA-256 hash, truncated).",
    "EXIF metadata from the photo (GPS, camera ID, timestamp): the upload widget re-encodes through canvas before any POST.",
    "Login or account info — there is no account system.",
    "Cookies (we do not set any).",
  ],
  contact: "Reach out via the project's GitHub: https://github.com/humiliati/sundog",
};

export const onRequestOptions = ({ request }) => handleOptions(request);

export async function onRequestGet({ env, request }) {
  const origin = request.headers.get("origin") || "*";
  const version = env.POLICY_VERSION || "unknown";
  return withCors(
    jsonResponse({ policy_version: version, body: POLICY_BODY }),
    origin,
  );
}

export const onRequest = (ctx) => {
  if (ctx.request.method === "GET") return onRequestGet(ctx);
  if (ctx.request.method === "OPTIONS") return handleOptions(ctx.request);
  return withCors(errorResponse(405, "method_not_allowed", "Use GET."));
};
