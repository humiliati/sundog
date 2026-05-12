# Phase 5 Backend Plan — Photo Upload + Inverse Inference

Planning doc for the Cloudflare Workers + R2 backend that turns user-submitted
sundog photos into atlas-overlaid renders and (with consent) training-data
captures. Companion to the Phase 5 entry in
[`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md).

This doc is **plan-only** — no infrastructure shipped yet. The intent is to
have the architecture, schema, consent UX, and privacy policy nailed down
before we provision R2 buckets or expose endpoints.

## Why Cloudflare Workers + R2

The project already deploys via `wrangler` (`package.json`, `wrangler.toml`).
Workers gives us a low-overhead path for:

- an HTTP endpoint that accepts uploads close to the user (low latency,
  global edge);
- R2 buckets for the raw photos with no egress fees back to our own
  derived-dataset pipelines;
- KV / D1 for the structured metadata (inverse-inferred altitude, JSON
  pose, opt-in flag, deletion token);
- rate limiting and IP-country attribution without leaving the CF
  ecosystem.

No new vendor; one configuration file extension.

## Endpoint surface

```
POST /api/sundog/upload         — accept upload, return submission_id + deletion_token
GET  /api/sundog/policy         — human-readable + JSON privacy policy
POST /api/sundog/delete         — caller-driven deletion via token
GET  /api/sundog/health         — Workers / R2 ping for monitoring
```

### Request shape (`/api/sundog/upload`)

```json
{
  "image": {
    "data": "<base64 jpeg/webp/png>",
    "mime": "image/jpeg",
    "byte_length": 287104
  },
  "pose": {
    "geometryModel": "halo_atlas",
    "sunAltitudeDeg": 25.0,
    "anchor_sun_px": [400, 356],
    "anchor_22_halo_radius_px": 145,
    "parhelion_offset_px": 160,
    "parhelicCurvature": 0.05,
    "...": "other atlas pose fields"
  },
  "consent": {
    "share_for_training": true,
    "agreed_at": "2026-05-12T18:00:00Z",
    "agreed_to_policy_version": "2026-05-12"
  },
  "client": {
    "ua": "Mozilla/5.0 ...",
    "page": "sundog.html",
    "exif_stripped": true
  }
}
```

### Response shape (success)

```json
{
  "submission_id": "01J5K7H9N0Q1R2S3T4U5V6W7X8",
  "deletion_token": "<opaque ulid>",
  "deletion_url": "https://sundog.cc/api/sundog/delete?token=<...>",
  "inferred_h_deg": 25.0,
  "policy_version": "2026-05-12"
}
```

## R2 layout

```
sundog-uploads/
  submissions/
    2026-05-12/
      01J5K7H9N0Q1R2S3T4U5V6W7X8.jpg     ← original (post EXIF strip)
      01J5K7H9N0Q1R2S3T4U5V6W7X8.pose.json
      01J5K7H9N0Q1R2S3T4U5V6W7X8.meta.json
```

- `*.jpg` — re-encoded image (EXIF stripped), at most 4096 px on the
  long edge. We re-compress at quality 0.85 to drop file size.
- `*.pose.json` — the atlas pose the user marked up.
- `*.meta.json` — submission metadata; lives separately so we can scan
  metadata without touching binary blobs.

## Metadata schema (`*.meta.json`)

```json
{
  "submission_id": "01J5K7H9N0Q1R2S3T4U5V6W7X8",
  "ts": "2026-05-12T18:00:00.000Z",
  "consent": {
    "share_for_training": true,
    "policy_version": "2026-05-12"
  },
  "inferred_h_deg": 25.0,
  "anchors": {
    "sun_px": [400, 356],
    "r22_px": 145,
    "parhelion_offset_px": 160
  },
  "client": {
    "ip_country": "US",
    "ua_hash": "<sha256 of UA, not raw UA>"
  },
  "deletion_token_hash": "<sha256 of token>",
  "image": {
    "byte_length": 287104,
    "mime": "image/jpeg",
    "dimensions": [800, 560],
    "exif_stripped": true
  }
}
```

Note: we keep the deletion-token **hash**, not the token itself, so a
compromised metadata leak can't be used to delete other people's data.

## EXIF / privacy guarantees

- **EXIF is stripped client-side**, before any bytes leave the browser.
  Implementation: read the file via `<input type="file">`, decode through
  `<canvas>`, re-encode with `canvas.toBlob('image/jpeg', 0.85)` — the
  resulting blob has no EXIF. We surface this to the user before they
  click "Share" with a list of the metadata that was present and is
  being removed (timestamp, GPS, camera ID, etc.).
- **IP is hashed at the edge for country attribution only.** The raw IP
  is never persisted; `cf.country` is what lands in metadata.
- **Deletion is honored within 24 hours.** The deletion endpoint
  removes both R2 objects (`*.jpg`, `*.pose.json`, `*.meta.json`) and
  emits an audit log row. Aggregated derived datasets that have
  already incorporated the submission will not be retroactively
  scrubbed — this is stated in the policy.

## Consent UX (frontend obligations)

Phase 5 frontend must:

1. Default the consent checkbox to **off**.
2. Show the user a preview of the rendered atlas overlay *before*
   asking for consent. Rendering happens locally; consent gates only
   the POST.
3. Display the policy version and link to `/api/sundog/policy`.
4. After successful upload, store the `deletion_url` in the user's
   browser localStorage (under `sundog.submissions`) and show them a
   "manage your submissions" link.
5. Tell the user explicitly that aggregated derived datasets may
   incorporate their submission before deletion is processed.

## Rate limiting

- 5 successful uploads / IP / hour via Workers KV with a TTL.
- Above the limit, return `429` with a JSON body explaining the cap.
- Cap exists to prevent storage abuse, not to gatekeep legitimate
  use. Lift it if real users hit it.

## What this plan does NOT cover

- Training-data ingestion pipeline downstream of R2 (a separate workflow
  decides which submissions become labeled training examples).
- Model retraining schedule.
- Human-in-the-loop review queue for submitted photos (recommended as a
  Phase 5+ extension once we see real volume).
- Anti-spam beyond the rate limit (image-hash dedupe, NSFW filter, etc.
  — extensions, not core).

## Open questions before implementation

1. **D1 vs. metadata-in-R2 as the queryable store?** The proposal above
   keeps metadata as JSON next to the image in R2. D1 (Cloudflare's
   SQLite) would give us better queryability for "show me all submissions
   with inferred h between 20° and 30°." Decide once we have ~50
   submissions.
2. **Domain — does this live on `sundog.cc` or a subdomain like
   `api.sundog.cc`?** Same-origin is simpler for the upload UX; subdomain
   gives us cleaner monitoring and lets us swap implementations without
   touching the main page.
3. **Worker bundle size budget** — current build is well under the 1MB
   limit; just confirm the EXIF/canvas helper code stays in the browser
   bundle, not the Worker.

## Implementation order when Phase 5 starts

1. Worker scaffold + health endpoint (sanity check the deployment).
2. R2 bucket + minimal upload happy path (no rate limit, no metadata).
3. Metadata writer + deletion endpoint.
4. Rate limiter via KV.
5. Policy doc + JSON endpoint.
6. Frontend integration — three-click measurement UX + client-side
   EXIF strip + consent gate.
7. End-to-end test: upload → R2 inspect → delete → R2 inspect again.
