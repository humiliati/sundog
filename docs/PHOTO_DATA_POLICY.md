# Sundog Photo Data Policy

**Policy version: 2026-05-12**

This policy covers the optional "Share my photo with the Sundog project for
atlas-model training" checkbox in the photo-upload widget at
[`/sundog`](../sundog) §6. It does NOT cover any other interaction
with the site. If you don't check that box, none of the data described
below ever leaves your browser.

## What we collect, only if you opt in

When you check the share checkbox and submit, your browser POSTs the
following to `/api/sundog/upload`:

- **Your photo**, re-encoded as JPEG/WebP/PNG through `canvas.toBlob` *before*
  the request leaves your browser. This strips EXIF metadata — GPS
  coordinates, camera identifier, original capture timestamp, and any other
  embedded fields — so the upload payload contains pixel data only.
- **The JSON pose** you produced by clicking the three measurement points:
  the sun pixel, the 22° halo edge, and a parhelion. Includes the
  inverse-inferred sun altitude.
- **Country code** derived at the Cloudflare edge from your IP address
  (e.g. `"US"`). We never persist the IP itself.
- **A SHA-256 hash** of your browser's User-Agent string, truncated to 32
  hex characters. The raw User-Agent is never persisted.
- **Submission timestamp** (ISO 8601 UTC).

## What we do with it

- **Improve the Halo Atlas geometric model.** Calibration sweeps, edge-case
  detection (e.g. photos where the auto sun-detection picks a parhelion
  instead of the sun), and validating new primitives against a broader
  photograph cohort.
- **Build public illustration assets.** If we use a specific photo in a
  publication, blog post, or social asset, we ask for separate explicit
  consent for that individual photo. The shared training data is *not*
  blanket publication consent.

## What we do NOT collect

- **Your IP address.** Only the country code derived at the edge. The raw
  IP is never persisted in metadata, logs, or any other store.
- **Your User-Agent.** Only its SHA-256 hash, truncated. This is sufficient
  for aggregate analysis (e.g. "how many submissions came from mobile
  browsers") but not for individual identification.
- **EXIF metadata.** Stripped client-side in your browser before upload.
- **Account information.** There is no account system. There are no
  cookies set by the upload widget.

## How to delete

When you submit, you receive a **deletion token** — a 64-character
hexadecimal string. We give it to you exactly once. We persist only its
SHA-256 hash, so a leaked metadata file cannot be used to delete other
people's submissions.

To delete your data:

1. **Click the deletion URL** that was shown next to your submission ID.
   The URL is `https://sundog.cc/api/sundog/delete?token=<your-token>`.
2. Or **POST** `{ "token": "<your-token>" }` to `/api/sundog/delete`.

We honor deletion within **24 hours** under normal conditions. The three R2
objects associated with your submission (image, pose JSON, metadata) are
removed, and the KV index entry is deleted so the token can't be replayed.

The Sundog project's browser saves a copy of your deletion URL in
`localStorage` after a successful submission (under the key
`sundog.submissions`), so you can revisit `/sundog` and click "manage
my submissions" to find your token again. Clearing your browser storage
will lose this — keep a copy elsewhere if you want long-term recovery.

## Retention

- **Original photo + pose**: 12 months from submission, unless you delete
  earlier.
- **Aggregated derived datasets**: if your submission has already been
  incorporated into a derived training set or a published statistic by the
  time you request deletion, the original photo is removed but the
  aggregated statistical features may persist. We make a best effort to
  flag your submission ID in the derived dataset so future revisions of
  the dataset omit it.

## Rate limiting

Uploads are rate-limited to **5 successful submissions per IP address per
hour**, enforced via a hashed IP key in Cloudflare KV with hourly TTL. The
raw IP is hashed and the hash is truncated before storage — the rate limit
counter cannot be used to identify the submitter.

## Storage location

R2 bucket `sundog-uploads` on the project's Cloudflare account. Object
keys are prefixed by submission date:

```
submissions/{YYYY-MM-DD}/{ULID}.jpg          ← re-encoded image
submissions/{YYYY-MM-DD}/{ULID}.pose.json    ← JSON pose
submissions/{YYYY-MM-DD}/{ULID}.meta.json    ← metadata
```

The `ULID` portion is a timestamp-prefixed, alphabetically-sortable
identifier — it gives us natural ordering by submission time but is
unguessable.

## Changes to this policy

This is policy version **`2026-05-12`**. If we change the policy in a way
that requires re-consenting, the version string updates and the upload
endpoint returns HTTP `409 policy_version_mismatch` when older clients
try to submit — they will be re-shown the updated policy before they can
submit. Older submissions remain governed by the policy version in their
metadata (`consent.policy_version`).

## Contact

The project lives at [github.com/humiliati/sundog](https://github.com/humiliati/sundog).
For policy questions or expedited deletion, file an issue or reach out via
the contact links on `/`.
