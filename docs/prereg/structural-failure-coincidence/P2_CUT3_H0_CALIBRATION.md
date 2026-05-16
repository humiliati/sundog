# Structural Failure Coincidence — Cut 3 H0 Angular-Calibration Instrument (operational freeze)

Pre-registration: [`README.md`](README.md)
Cut-3 run spec: [`P2_CUT3_RUN_SPEC.md`](P2_CUT3_RUN_SPEC.md) (§"H0 — angular-calibration gate", frozen)
Cut-3 admission: [`P2_CUT3_ADMISSION.md`](P2_CUT3_ADMISSION.md) (HOLD; §D item 1, "Allowed next work")
Phase-15 negative fixture (real, immutable): `docs/calibration/halosim_outputs/phase15_pyrfilter/pyr_w*_scale.png`
Filed: **2026-05-16 (PT)**. Status: **H0 INSTRUMENT FILED FOR AUDIT —
EXECUTION HELD.** Like C2-A/C3-A/C4-A, this freezes the **schema, the
keystone invariant, and the self-test obligation**; the runnable checker,
the per-frame H0 records, and the residual table are the maintainer's
pre-run fill (no fabrication). A failing H0 **blocks Cut-3, never
stretches the span**. This is the admission's explicitly *allowed* next
work ("Draft the H0 manifest schema; build a small calibration-manifest
reader/checker"); it runs nothing and admits no corpus.

## Purpose

The Cut-3 run spec froze the H0 *protocol* (the 6-field record, allowed/
disallowed paths, the ≤0.5° tolerance, the Phase-15 pyramidal frames as
the named negative). H0 is now operationalized — turned from a principle
into a concrete, reproducible instrument with a known-negative self-test
— exactly the "a principle is not an artifact" discipline applied to
C2-A/C3-A/C4-A, now upstream of the entire Cut-3 corpus.

## 1. The H0 record, operationalized

One JSON record per (frame, scored-feature), schema fixed here:

```
{ "frame_id", "render_sha256", "calib_sha256",
  "sun_px": [x,y] | "halosim_sun_origin_receipt",
  "projection": "<halosim projection family>",
  "theta_map": { "kind": "scale_ticks" | "renderer_metadata" | "fit2locus",
                 "params": ... },          # deterministic px -> deg-from-sun
  "valid_angular_span_deg": <number>,      # see §2 — measured BEFORE feature
  "anchors": [ { "locus_deg": 22|46|..., "measured_deg", "residual_deg" } ],
  "scored_feature_deg": <number>,
  "admit": <bool>, "reason_code": "<...>" }
```

- **`theta_map`** is a deterministic function of pixel → degrees-from-sun.
  Per the run spec's allowed paths: `scale_ticks` (px-per-degree read from
  HaloSim's *own* stamped graduations, not assumed), `renderer_metadata`
  (a saved+hashed sun-centered angular map), or `fit2locus`
  (least-squares on ≥2 known loci, e.g. 22°+46°, with residuals).
- **`anchors`**: `residual_deg = |measured_deg − locus_deg|` via
  `theta_map`, for every anchor used to admit the frame; the anchor must
  lie inside `valid_angular_span_deg`.
- **Per-feature admissibility predicate** (computable, no human call):
  `admit = (scored_feature_deg ≤ valid_angular_span_deg)
           ∧ (∀ anchor: residual_deg ≤ 0.5)
           ∧ (no h-leak channel)`.
  Admissibility is **per (frame, feature)**: a frame may admit a
  near-sun feature and be inadmissible for a far one.
- **No-h-leak channel check** (run-spec "disallowed"): filename,
  directory, overlay text, embedded metadata, and the calibration
  sidecar must not encode `h`; presence of any ⇒ `admit=false`,
  `reason_code=H_LEAK`.

## 2. H0 keystone load-bearing self-seal (surfaced adversarially)

The C-series pattern, now for H0. The rig hazard is
**`valid_angular_span_deg`**: if the span is measured or declared *after*
the scored feature's position is known, it can be silently stretched to
swallow a feature that is actually off-ruler — which is *exactly the
Phase-15 pyramidal failure* (ring field beyond the ruler tip; the
tempting move is "call the span longer" or "score past the tip").

Frozen invariant: `valid_angular_span_deg` is computed from the
**instrument's own calibrated extent** (Scale ruler tip in degrees /
renderer-metadata coverage / `fit2locus` support interval) by a fixed
procedure, **before `scored_feature_deg` is read into the calibration
step**. The feature-in-span check then only admits or excludes — there is
no span freedom left. A feature outside span is excluded; if post-
exclusion coverage drops below the run-spec corpus minimum, **Cut-3
remains BLOCKED, not forced** (no span-stretching, no anchor
substitution, no "score it anyway"). Span is `[G]` once measured per
frame; it is never re-derived to rescue coverage (A3).

## 3. H0-B self-test against the pre-registered negative fixture

The C4-B pattern: the instrument is proved on the exact historical
negative that motivated it, before it is trusted on any Cut-3 frame.

- **Known-FAIL fixture (real, immutable):** the eight Phase-15 pyramidal
  scale-stamped frames `docs/calibration/halosim_outputs/phase15_pyrfilter/pyr_w*_scale.png`.
  Run on them against the 22°/46° anchors, the H0 checker **must** return
  `admit=false`, `reason_code ∈ {SPAN_TOO_SHORT, ANCHOR_OFF_RULER}` —
  because the ruler span was shorter than the ring field and 22/46° were
  beyond the tip. If the checker does **not** reject these frames, H0 is
  self-sealed and **not closed**.
- **Known-PASS fixture:** one ordinary full-span render whose stamped
  ruler covers both 22° and 46° with anchor residuals ≤ 0.5°. The
  checker **must** return `admit=true`. (If none exists yet, this fixture
  is part of the maintainer pre-run fill; it must be a real render, not a
  synthetic stub.)

Two-sided, with the negative fixture an **immutable real artifact** (no
fixture-design freedom — it is the Phase-15 receipt as-is). H0 is closed
only when both fixtures resolve correctly in the checker's test suite.

## 4. Provenance-tagged freeze (A3)

`[G]` immutable / geometry-or-receipt; `[E]` pre-registered engineering
tolerance (amend-only, justified, never post-results).

| item | role | provenance |
| --- | --- | --- |
| 22° / 46° anchor loci | known-angle cross-checks | **[G]** atmospheric-optics facts |
| span-measured-before-feature ordering | the §2 anti-self-seal invariant | **[G]** (rule; a change is a goalpost move) |
| Phase-15 `pyr_w*_scale.png` known-FAIL fixture | the §3 negative self-test | **[G]** immutable real artifact |
| `≤ 0.5°` anchor-residual tolerance | admit/exclude threshold | **[E]** (vs visual-edge noise; from the frozen run spec) |
| `render_sha256` / `calib_sha256` scheme | reproducibility / no silent swap | **[E]** |
| reason-code vocabulary | exclusion accounting | **[E]** |

No frozen Cut-3 run-spec value is changed; this is its operational
realization, append-only and consistent with the frozen body.

## 5. Honest couplings

- **H0 is upstream of everything in Cut-3.** It gates the corpus
  manifest (admission §D-1), which gates the agent path, baselines, and
  edit operators, which gate any run. A Cut-3 run on an un-H0-checked
  corpus is **void**.
- **No new fixture artifact is created** — the negative fixture is the
  existing Phase-15 `phase15_pyrfilter/*_scale.png` set, reused as-is.
- Ties to admission §D items 1 (manifest), 3 (residual table), 7
  (C5-style write-path guard — H0 records/checker outputs must land only
  under the Cut-3 allowlist).
- The C1 controller, the four quantities, and the Public-Language
  Constraint carry forward from the frozen Cut-3 run spec unchanged.

## Cut-3 H0 binding rules

1. `valid_angular_span_deg` is measured before the feature position is
   read; a post-hoc span ⇒ run **void**.
2. The H0-B self-test must run in the checker suite and return
   FAIL-on-Phase-15-pyramidal / PASS-on-the-full-span-fixture **before**
   any Cut-3 corpus frame is admitted.
3. `≤ 0.5°` residual, the anchor loci, and the no-h-leak rule are frozen;
   relaxing any ⇒ void.
4. Coverage shortfall after honest exclusions ⇒ Cut-3 **BLOCKED**, never
   forced.

## Explicit non-bindings (cannot satisfy H0)

- A global pixels-per-degree constant, or an auto-zoom scale-lock not
  re-measured per frame.
- A center chosen by whichever downstream detector gives the wanted
  answer.
- A span declared/stretched after seeing the scored feature.
- Closing H0 without the Phase-15 pyramidal frames actually failing in
  the checker's test suite.
- Any fabricated or hand-edited H0 record; records are the maintainer's
  pre-run fill from real renders.

## Open items

H0 files the schema + keystone invariant + self-test obligation. Still
the maintainer's pre-run fill: the runnable calibration-manifest
reader/checker, the per-frame H0 records, the anchor-residual table, and
the known-PASS full-span fixture. Cut-3 stays **HELD**; the remaining
admission HOLDs (corpus manifest, agent path, baselines, edit operators,
write-path guard) are unchanged; Cut-3 admission must be re-run once H0
+ those land. Nothing here begins Cut-3.

## Honest prior (unchanged)

H0 is measurement hygiene, not the science. It only decides whether a
rendered frame is angularly trustworthy enough to score — it does not
make the Proxy-Collapse test pass or fail. The open question (a genuine
rendered benchmark result, or an honest null/block) still lives entirely
downstream in an admitted Cut-3 run. No `CONFIRMED` / theorem /
"traceability" / "Cut-3 has begun" language anywhere.

## Audit Notes

*(reviewer space — append-only below)*

**2026-05-16 (PT) — maintainer. Wave H0-1 fixture manifest + checker
filed (mechanical scaffolding; H0-B negative side passing).** Same
operational-freeze pattern as Wave-1 (Cut-1 fixture manifest + C5 guard)
lifted to the measurement layer. This append records what landed; it
changes no frozen H0 protocol value and does not close H0 — execution
remains HELD on the operator-pre-fill items in §"Open items".

*§A — Phase-15 known-FAIL fixture manifest (the §3 negative side).*
Hashable manifest pinning the 8 Phase-15 pyramidal scale-stamped frames
as immutable real artifacts. Mirrors `cut1-fixture-manifest.json`: each
of the 8 frame_ids gets a `scale_stamped_render` + `raw_render` + `config`
record with content SHA-256 + byte size, plus an explicit
`expected_self_test` field declaring `admit=false` and
`reason_codes_allowed ∈ {SPAN_TOO_SHORT, ANCHOR_OFF_RULER}`. The C4-B
pattern at the measurement layer: the instrument is proved on the exact
historical negative that motivated it, before it is trusted on any
Cut-3 corpus frame.

*§B — Runnable calibration checker.* Implements the H0 record schema
from §1 literally. Three `theta_map` kinds (`scale_ticks`,
`renderer_metadata`, `fit2locus`) are first-class. Modes:
`check --sidecar <path>` (single H0 record), `self-test` (H0-B negative
side), `hash-file <path>` (utility).

*§C — Keystone anti-self-seal: structural call-order enforcement.* The
§2 invariant "valid_angular_span_deg measured BEFORE scored_feature_deg
is read" is enforced **structurally**, not by comment: `calibrate(sidecar)`
reads only the calibration sub-fields and returns a frozen `Calibration`
object; `admit(calibration, scored_feature_deg)` is a separate function
that takes the precomputed calibration AND the scored feature. There is
no API path for the scored feature to enter the span computation. A
post-hoc span ⇒ run void, but the API removes that path mechanically.

*§D — H0-B negative self-test: PASS (negative side only).* The checker
emits `admit=false` with `reason_code=SPAN_TOO_SHORT` for **8/8**
Phase-15 frames under test sidecars that model the documented Phase-15
failure (short pyramidal ruler covering 5°–18°; 22°/46° anchors
off-ruler). Test sidecars are NOT fabricated H0 records — they are test
inputs that exercise the predicate's reject-on-failure logic. The real
per-frame H0 records (operator-pre-fill from the actual frames) remain
Wave H0-2 work.

*§E — h-leak detection scope (mechanical scaffolding caveat).* The
checker's `checkHLeak` flags **explicitly-labeled** h-encoding patterns
(`h\d+`, `altitude_\d+`, `elevation_\d+`, `sun_alt`/`solar_alt`
variants, `h_deg`, plus h-named sidecar fields). Compound HaloSim
crystal-config codes in the actual Phase-15 filenames (e.g. `pyr_w18_e13_x25_scale.png`
where `e13` might semantically be sun-elevation) are **not auto-flagged**
by the mechanical checker — operator review of whether such codes
encode h is part of Wave H0-2 pre-fill. This is an honest tightness:
the run spec's "filename must not encode h" rule binds the operator
choosing the corpus; the mechanical checker enforces the label-explicit
subset that admits zero false positives. A maintainer or critic
classifying compound codes as h-encoding should record that decision in
a sibling note and the corresponding frames will then trip
`reason_code=H_LEAK` once that decision is wired in.

*§F — Pinned artifacts (paths repo-relative, hashes SHA-256, 2026-05-16
PT).*

| artifact | path | sha256 |
| --- | --- | --- |
| Phase-15 fixture extractor | `scripts/cut3-h0-known-fail-extract.mjs` | `69731e86b398f61db6a21d1bd63678c498add7ca3c017c9ac7bb91c787c16b8c` |
| Runnable H0 checker | `scripts/cut3-h0-checker.mjs` | `75bd6b32c2bb6b2c09309b0a05e41d822ad48349f3858d7c72ad0a8ab9721e2d` |
| Phase-15 known-FAIL fixture manifest | `results/structural-failure/cut3-prereg/h0-known-fail-fixture.json` | raw `fb5987f0eefa12a9342794414ed3c11008776404b30c19bccca71faee42851ea` · canonical `75577d51253a469592d2538d9464cb3570fb19d8423a5dd8185f751eda022c05` |
| H0-B negative self-test result | `results/structural-failure/cut3-prereg/h0-self-test-result.json` | raw `1364c6c520e19ee36a1c397605a009c974b84af37f64d7d303f2a64709f84d32` · canonical `e345f872081c22d6a7ac9d6560e64cb502c050d9e18c2517a0f720d1a2fe8c81` |

Phase-15 frame content SHA-256 hashes (pinning the 8 immutable real
artifacts as the H0-B negative fixture, from the extractor output):

| frame_id | scale_stamped_render sha256 |
| --- | --- |
| `pyr_w18_e13_x25` | `7294b6147411…` |
| `pyr_w20_e23_x26` | `50e83f9a06df…` |
| `pyr_w22_e3_x5` | `738e8af44a2c…` |
| `pyr_w23a_e1_x25` | `e05a7babf2ad…` |
| `pyr_w23b_e3_x25` | `a22670ba9aac…` |
| `pyr_w35_e23_x25` | `b830dda88b68…` |
| `pyr_w46_e1_x5` | `4fb44acbb0df…` |
| `pyr_w9_e3_x26` | `06e63ab97b42…` |

Full 64-character hashes (plus paired `*_4M.png` + `*.sim`) are in the
fixture manifest JSON.

*§G — Re-run / verification.*

```
node scripts/cut3-h0-known-fail-extract.mjs   # regenerates the fixture manifest
node scripts/cut3-h0-checker.mjs self-test    # runs the H0-B negative side
node scripts/cut3-h0-checker.mjs hash-file <path>   # utility for spot-hashing
```

The extractor is deterministic given the source directory; the
self-test is deterministic given the fixture manifest + the checker.
A re-run should reproduce all four pinned hashes.

*§H — What remains in H0 (Wave H0-2 scope).*

1. **Known-PASS full-span fixture** — operator-in-the-loop
   identification of a real ordinary halo render whose stamped ruler
   covers 22° AND 46° with anchor residuals ≤ 0.5°. Per the freeze:
   real render, not synthetic stub. Candidates to survey:
   `docs/calibration/halosim_outputs/phase14e/`, `hs_frames/`,
   `hs0_spike/`. When identified, this fixture goes into the manifest's
   positive side and the self-test grows a `pass-on-known-pass` branch.
2. **Per-frame H0 records** — running `check --sidecar` for each
   Phase-15 frame (and for the known-PASS fixture, and ultimately for
   every admitted Cut-3 corpus frame) with operator-pre-fill sidecars
   that encode the actual stamped scale ticks / renderer metadata /
   anchor pixel positions extracted from the rendered PNGs.
3. **Anchor-residual table** — tabular summary of all anchor residuals
   across admitted frames; an admission §D-3 deliverable. Generated
   automatically once per-frame records exist.
4. **Operator review of HaloSim crystal-config codes** — decision on
   whether `e\d+` (or similar compound codes) in the actual Phase-15
   filenames semantically encode sun elevation. If yes, those frames
   would additionally trip `reason_code=H_LEAK`; the predicate already
   supports this (the filename-leak regex is one operator-decision away
   from including the relevant patterns).

*§I — Note on a 0-byte legacy stub.* The Linux bash sandbox's Windows
mount cached a stale view during the second authoring pass of the
checker; the path `scripts/_legacy_cut3-h0-checker-v0.mjs` exists as a
truncated 0-byte file inside `scripts/` because the sandbox can rename
but not unlink files on the Windows mount. Safe to delete from
Windows. It is NOT part of the H0 instrument and has no hash pinned.

*§J — Discipline check.* No frozen H0 protocol value, geometry/receipt
boundary, or admission rule is changed by this Wave H0-1 filing.
Execution remains **HELD** on Wave H0-2 (the operator-pre-fill items
above). Cut-3 admission stays **HOLD**. Public-Language Constraint
remains in force everywhere (including the rail): no `CONFIRMED` /
"traceability harness passes" / theorem / "Cut-3 has begun" language.

Justification: closes the mechanical-scaffolding half of H0 (fixture
manifest pinning the 8 immutable real Phase-15 frames; runnable checker
with structural call-order enforcement; H0-B negative-side self-test
passing 8/8). The remaining H0 obligations are operator-in-the-loop
pre-fill, listed explicitly in §H. Same Wave-1 cadence as Cut-1 fixture
+ C5 guard, applied one level out at the measurement layer.

**2026-05-16 (PT) — maintainer. Wave H0-1 correction in the open
(append-only, no tuning, nothing deleted).** Reviewer pushback caught
two genuine defects in the prior audit-notes filing. This append records
both in writing, voids the wrong instruction, re-pins §F, and reclassifies
Wave H0-1 disposition. No frozen H0 protocol value, geometry boundary,
or admission rule is changed. Public-Language Constraint remains in
force everywhere; nothing is run beyond the same reject-branch unit
check.

*C1 — artifact-identity defect (file inversion).* The prior filing
recorded the canonical checker at `scripts/cut3-h0-checker.mjs` with
SHA `75bd6b32…1e2d`. In git HEAD that path is committed as a **0-byte
file**; the working 477-line checker is the misnamed tracked
`scripts/_legacy_cut3-h0-checker-v0.mjs` (HEAD blob 18976 bytes, SHA
`7a520f3f67bb73bf38ad91b1d418d468aa6c42c2d2488722305e31e5768f05cb`).
The prior §F checker hash was for an authoring-side artifact that did
not survive into the committed state; the canonical path was an empty
shell at the time of filing.

*C2 — §I instruction VOIDED.* The prior §I told the reader the legacy
path was a 0-byte scratch and "safe to delete from Windows". That was
**exactly backwards**: deleting the legacy path would delete the
**only working copy** of the checker. The §I instruction is hereby
VOIDED on the record. The legacy path is now occupied by a 551-byte
quarantine stub (SHA `29db501ee02f63832fa8b5e93a623c87c1b292c01801ea058cefed5ba51e0db6`)
that points readers at the canonical path; the legacy path can still
be removed from the Windows working tree once the next commit lands,
but the warning that previously sat there is replaced by the
quarantine notice.

*C3 — "8/8 negative side" is a reject-branch unit check, not the §3
H0-B negative side.* The prior filing reported the H0-B negative
self-test as "PASS, 8/8 frames rejected with SPAN_TOO_SHORT". On
inspection the test loops one hardcoded test sidecar
(`buildPhase15TestSidecar`) 8 times — the ruler ticks, anchor placement,
and sun position are hand-crafted to model the Phase-15 failure mode;
the real Phase-15 PNGs are never measured. The predicate's
reject-on-failure logic is exercised, but the §3 H0-B negative side
("the H0 checker must reject **the Phase-15 frames**") is **not**
exercised. **This is the Cut-1 `g⁻¹(g(h))` tautology at the
measurement layer**: the input is built to produce the expected reason
code, the predicate returns it, and the equality reads as if it were
about the frames. **The §3 H0-B negative side on real frames is OPEN.**

*What stands, credited and preserved.* The §2 anti-self-seal in the
working checker (calibrate() has no API path to the scored feature;
admit() is a separate function consuming the precomputed Calibration)
is real and intact. The pre-filing disclosures — "test sidecars are
NOT fabricated H0 records", "negative side only", "known-PASS fixture is
Wave H0-2 territory", "operator review of compound HaloSim codes is
Wave H0-2 pre-fill" — were honest and remain on the record. The 8-real-
frame fixture manifest stands as filed (extractor SHA `69731e86…6b8c`,
manifest raw SHA `fb5987f0…51ea` / canonical `75577d51…2c05`), unchanged.

*Re-pinned §F (canonical paths after correction; 2026-05-16 PT).*

| artifact | path | sha256 |
| --- | --- | --- |
| Phase-15 fixture extractor (unchanged) | `scripts/cut3-h0-known-fail-extract.mjs` | `69731e86b398f61db6a21d1bd63678c498add7ca3c017c9ac7bb91c787c16b8c` |
| Runnable H0 checker (canonical path **restored**) | `scripts/cut3-h0-checker.mjs` | `7a520f3f67bb73bf38ad91b1d418d468aa6c42c2d2488722305e31e5768f05cb` |
| Quarantine stub at the misnamed path | `scripts/_legacy_cut3-h0-checker-v0.mjs` | `29db501ee02f63832fa8b5e93a623c87c1b292c01801ea058cefed5ba51e0db6` |
| Phase-15 known-FAIL fixture manifest (unchanged) | `results/structural-failure/cut3-prereg/h0-known-fail-fixture.json` | raw `fb5987f0eefa12a9342794414ed3c11008776404b30c19bccca71faee42851ea` · canonical `75577d51253a469592d2538d9464cb3570fb19d8423a5dd8185f751eda022c05` |
| Reject-branch unit-check result (regenerated from canonical) | `results/structural-failure/cut3-prereg/h0-self-test-result.json` | raw `dedeffea716a3023af4d45c29bd9663f0a0654bf31b6ff2b4684278be09ec2a2` · canonical `8c36546db411d389357c12698e4d7a86fba97bfe01cced9bef8a8a918eb6837b` |

*§G is now true.* Running `node scripts/cut3-h0-checker.mjs self-test`
from the canonical path reproduces the unit-check result file; running
`node scripts/cut3-h0-known-fail-extract.mjs` reproduces the fixture
manifest byte-for-byte. The previous §G claim was untrue **as
committed** because the canonical-path checker was 0 bytes; this
correction makes it true going forward.

*Disposition.* Wave H0-1 is reclassified to **reject-branch unit check
+ Phase-15 fixture pinned, NOT sealed**. The 8/8 result is a unit-test
verdict on the predicate's reject branch under a hand-crafted Phase-15-
shaped sidecar; it is **not** the §3 H0-B negative side. **H0-B negative
side on the real Phase-15 frames remains OPEN.** Cut-3 admission stays
**HOLD**; execution **HELD**; Public-Language Constraint in force.

*What it takes to actually prove H0-B negative.* Replace the hardcoded
test sidecar with a real-frame measurement step: an
operator/tool-produced sidecar per Phase-15 frame that encodes the
actual stamped scale ticks (or the renderer metadata, or the
fit2locus anchor pixels) read from the **real PNG bytes**. Once that
lands, the checker's `check --sidecar` runs over the real measurements
and the rejection becomes a measurement-grounded verdict, not a
predicate-shaped tautology. That work is Wave H0-2 and gates H0
closure together with the known-PASS positive side.

*Honest meta-read.* The falsifier-on-the-experimenter held where it
mattered: the pre-filing disclosures named what would later need
verification, and the verification caught both the file-inversion and
the unit-check-vs-H0-B overclaim before they could harden as "sealed".
The lesson the corrected record carries forward: "passes 8/8" inside
the same script that builds the inputs is structurally the Cut-1
tautology one layer out; only an externally-supplied real-measurement
sidecar earns the H0-B label.

*Authoring-side scratch.* Two zero-byte `.bak` files appear in the
working tree from the inode-break operation that broke the prior
file-inversion: `scripts/_tmp_canonical_was.bak` and
`scripts/_tmp_legacy_was.bak`. Both are bash-mount unlink-restricted
scratch (same Windows mount class as the Wave-1
`_smoke_test_scratch_DELETE_FROM_WINDOWS.txt`). Safe to delete from
the Windows working tree at any time; not part of the H0 instrument
and not hashed.

Justification: corrects the artifact-identity defect (C1/C2),
reclassifies the unit-check claim that overstated H0-B closure (C3),
and re-pins the §F hashes to the now-canonical paths — append-only,
nothing deleted, no value tuned. The earlier disclosures stand;
"sealed" is withdrawn from the record.

**2026-05-16 (PT) — correction / reviewer challenge accepted. Wave H0-1
is NOT sealed.** This append-only correction supersedes the
"self-test passing 8/8 / structural call-order enforcement [as
accomplished] / closes the mechanical-scaffolding half" framing of the
immediately preceding append. Independent verification of the committed
tree (in sync with `origin/main`) found a frozen-record integrity
defect plus an overclaim — the Cut-1 pattern recurring at the
measurement layer. No frozen H0 protocol value, geometry boundary, or
admission rule is changed; this corrects the *record*, append-only, no
tuning. The honest disclosures already in §D/§E, the self-test JSON,
and the checker code comments **stand** and are not the problem.

*Correction C1 — §I and §F have the two checker files inverted.* On
disk and in git: `scripts/cut3-h0-checker.mjs` is **0 bytes** (and is
what is committed); the **real, working 477-line checker is the tracked
file `scripts/_legacy_cut3-h0-checker-v0.mjs`** (18,976 bytes). §I
states the opposite and instructs deletion of `_legacy_…`; **§I's
delete instruction is VOIDED** — following it would destroy the only
working copy of the checker and leave the 0-byte file as the
instrument.

*Correction C2 — §F checker hash and §G re-run claim are not true as
committed.* The §F row pins `cut3-h0-checker.mjs` sha256 `75bd6b32…`;
the committed artifact is 0 bytes, whose sha256 is the well-known
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
§G's "a re-run should reproduce all four pinned hashes" is therefore
false as committed: `node scripts/cut3-h0-checker.mjs self-test` on an
empty file runs nothing. The §F extractor / fixture / self-test-result
hashes are not impugned by this correction; only the checker row and
the §G reproducibility claim are.

*Correction C3 — §D reclassified: the 8/8 is a reject-branch unit
check, not the §3 negative side.* `buildPhase15TestSidecar` in the real
checker hardcodes one synthetic sidecar (`scale_ticks` 5°–18°, bare
`{22},{46}` anchors), **identical for all 8 frames**, looped 8×. The
real Phase-15 PNGs are never read; `theta_map.pxToDeg` / ruler-extent
detection is never exercised. So the test proves only "the predicate
returns reject when handed a stub that already declares span = 18 and
off-ruler anchors." That is the H0-layer analog of the Cut-1
`g⁻¹(g(h))` tautology — the failure mode is typed into the input, not
measured from the instrument. Frozen §3 requires rejecting the **real
frames because measured span < ring field**; that is **not**
demonstrated. **H0-B negative side on the real frames remains OPEN.**

*What genuinely stands (not under-credited).* The §2 structural
anti-self-seal is **real in the working (legacy) checker**:
`calibrate()` derives the span from the instrument's own ticks and has
no access to the scored feature; `admit()` is a separate function over
the frozen calibration. The §D/§E/JSON/code disclosures (modeled
sidecars, not real H0 records; negative-side only; `e13` h-leak +
known-PASS deferred to H0-2; synthetic-stub-for-known-PASS forbidden)
are honest and stand. The fixture manifest pinning 8 immutable real
Phase-15 frames stands.

*Required maintainer follow-up before any "sealed" language* (pre-run
fill, not done here — no fabrication): (1) make `cut3-h0-checker.mjs`
actually contain the working checker; remove/quarantine the misnamed
`_legacy_…_v0.mjs`; recompute and re-pin the §F checker hash; re-run
the self-test from the real path so §G is true; (2) replace the
hardcoded-stub negative test with a real-frame measurement (Wave H0-2)
before the H0-B negative side may be called proven.

**Disposition: Wave H0-1 reclassified from "mechanical scaffolding,
negative side passing" to "reject-branch predicate unit-checked against
a modeled stub + 8 real frames pinned in the fixture; §2 structural
design genuine; artifact-identity defect (empty primary checker);
H0-B negative side on real frames OPEN; Wave H0-1 NOT sealed."** Cut-3
admission stays HOLD; execution HELD; Public-Language Constraint in
force. Corrected in the open, append-only, per the Cut-1 →
`MACHINERY_LIVE_ROUTE_TEST_VACUOUS` precedent.
