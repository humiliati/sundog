import {
  jsonResponse, errorResponse, withCors, handleOptions,
  readJsonBody, decodeImageBase64, makeSubmissionId, makeDeletionToken,
  sha256Hex, checkRateLimit, httpError,
} from "./_lib.js";

/**
 * POST /api/sundog/upload
 *
 * Request:
 *   {
 *     image: { data: "<base64 jpeg/webp/png>", mime: "image/jpeg", byte_length: N },
 *     pose:  { anchor_sun_px, anchor_22_halo_radius_px, parhelion_offset_px, ... },
 *     consent: { share_for_training: true, agreed_at, agreed_to_policy_version },
 *     client:  { ua, page, exif_stripped }
 *   }
 *
 * Response (201):
 *   { submission_id, deletion_token, deletion_url, inferred_h_deg, policy_version }
 *
 * Side effects: writes 3 R2 objects per submission:
 *   submissions/{YYYY-MM-DD}/{id}.jpg
 *   submissions/{YYYY-MM-DD}/{id}.pose.json
 *   submissions/{YYYY-MM-DD}/{id}.meta.json
 *
 * Rate limit (KV): 5 successful submissions / IP / hour.
 */

const ALLOWED_MIME = new Set(["image/jpeg", "image/png", "image/webp"]);

export const onRequestOptions = ({ request }) => handleOptions(request);

export async function onRequestPost(ctx) {
  const { request, env } = ctx;
  const origin = request.headers.get("origin") || "*";

  try {
    if (!env.sundog_uploads) throw httpError(503, "r2_not_bound", "R2 bucket not configured.");
    if (!env.sundog_rate_limit)      throw httpError(503, "kv_not_bound", "KV namespace not configured.");

    const maxImgBytes = Number(env.MAX_IMAGE_BYTES || 10_485_760);
    const rateLimit = Number(env.RATE_LIMIT_PER_IP_PER_HOUR || 5);
    const policyVersion = env.POLICY_VERSION || "unknown";
    const publicOrigin = env.PUBLIC_ORIGIN || "https://sundog.cc";

    // ---- body parse + validation ----------------------------------------
    const body = await readJsonBody(request, 24 * 1024 * 1024);
    const { image, pose, consent, client } = body || {};
    if (!image?.data || !image?.mime) throw httpError(400, "missing_image", "image.data and image.mime required.");
    if (!ALLOWED_MIME.has(image.mime)) throw httpError(415, "unsupported_image", `Allowed: ${[...ALLOWED_MIME].join(", ")}`);
    if (!consent?.share_for_training) throw httpError(403, "consent_required", "Submission requires consent.share_for_training === true.");
    if (consent.agreed_to_policy_version !== policyVersion) {
      throw httpError(409, "policy_version_mismatch",
        `Client agreed to ${consent.agreed_to_policy_version}; server is on ${policyVersion}. Re-show the policy.`);
    }
    if (!pose || typeof pose !== "object") throw httpError(400, "missing_pose", "pose object required.");

    const imageBytes = decodeImageBase64(image.data);
    if (imageBytes.byteLength > maxImgBytes) {
      throw httpError(413, "image_too_large", `Image exceeds ${maxImgBytes} bytes after decode.`);
    }

    // ---- rate-limit validated submissions before storage writes ----------
    const ip = request.headers.get("cf-connecting-ip") || "0.0.0.0";
    const rl = await checkRateLimit(env, ip, rateLimit);
    if (!rl.allowed) {
      return withCors(jsonResponse(
        { error: { code: "rate_limited", message: `Limit ${rateLimit} uploads/hour. Try again in ~${rl.resetSeconds}s.` } },
        { status: 429, headers: { "retry-after": String(rl.resetSeconds) } },
      ), origin);
    }

    // ---- generate identifiers --------------------------------------------
    const submissionId = makeSubmissionId();
    const deletionToken = makeDeletionToken();
    const deletionTokenHash = await sha256Hex(deletionToken);
    const now = new Date();
    const dayPrefix = now.toISOString().slice(0, 10);    // YYYY-MM-DD
    const baseKey = `submissions/${dayPrefix}/${submissionId}`;

    // ---- extract just what we want to persist ----------------------------
    const inferred = pose?.inferred?.sun_altitude_deg ?? pose?.inferred_h_deg ?? pose?.sunAltitudeDeg ?? null;
    const country = request.cf?.country || "??";
    const uaHash = (await sha256Hex(client?.ua || "")).slice(0, 32);

    const meta = {
      submission_id: submissionId,
      ts: now.toISOString(),
      consent: {
        share_for_training: true,
        policy_version: policyVersion,
        agreed_at: consent.agreed_at || now.toISOString(),
      },
      inferred_h_deg: typeof inferred === "number" ? Number(inferred.toFixed(2)) : null,
      anchors: pose?.anchors || null,
      client: {
        ip_country: country,
        ua_hash: uaHash,
        page: client?.page || null,
        exif_stripped: client?.exif_stripped === true,
      },
      deletion_token_hash: deletionTokenHash,
      image: {
        byte_length: imageBytes.byteLength,
        mime: image.mime,
      },
      keys: {
        image:  `${baseKey}.${image.mime === "image/png" ? "png" : image.mime === "image/webp" ? "webp" : "jpg"}`,
        pose:   `${baseKey}.pose.json`,
        meta:   `${baseKey}.meta.json`,
      },
    };

    // ---- write to R2 -----------------------------------------------------
    const imageExt = image.mime === "image/png" ? "png" : image.mime === "image/webp" ? "webp" : "jpg";
    const imageKey = `${baseKey}.${imageExt}`;
    await env.sundog_uploads.put(imageKey, imageBytes, {
      httpMetadata: { contentType: image.mime },
      customMetadata: { submissionId, policyVersion },
    });
    await env.sundog_uploads.put(`${baseKey}.pose.json`, JSON.stringify(pose, null, 2), {
      httpMetadata: { contentType: "application/json" },
    });
    await env.sundog_uploads.put(`${baseKey}.meta.json`, JSON.stringify(meta, null, 2), {
      httpMetadata: { contentType: "application/json" },
    });

    // ---- index the deletion-token hash for O(1) lookup at delete time ----
    // Store the metadata's R2 key under a `del:` prefix, keyed by the
    // SHA-256(token), so the deletion handler can find this submission
    // without scanning all metadata.
    await env.sundog_rate_limit.put(`del:${deletionTokenHash}`, baseKey, {
      expirationTtl: 60 * 60 * 24 * 400, // 400 days — long enough that legit users can delete
      metadata: { submission_id: submissionId },
    });

    return withCors(jsonResponse(
      {
        submission_id: submissionId,
        deletion_token: deletionToken,
        deletion_url: `${publicOrigin}/api/sundog/delete?token=${deletionToken}`,
        inferred_h_deg: meta.inferred_h_deg,
        policy_version: policyVersion,
      },
      { status: 201 },
    ), origin);
  } catch (err) {
    if (err && err.status) {
      return withCors(errorResponse(err.status, err.code || "error", err.message), origin);
    }
    console.error("upload error:", err);
    return withCors(errorResponse(500, "internal_error", "Unexpected error processing upload."), origin);
  }
}

export const onRequest = (ctx) => {
  if (ctx.request.method === "POST") return onRequestPost(ctx);
  if (ctx.request.method === "OPTIONS") return handleOptions(ctx.request);
  return withCors(errorResponse(405, "method_not_allowed", "Use POST."));
};
