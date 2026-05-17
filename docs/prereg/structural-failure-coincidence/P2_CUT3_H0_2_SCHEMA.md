# Structural Failure Coincidence — Cut 3 H0-2 Measured-Sidecar + Residual-Table Schema

Pre-registration: [`README.md`](README.md)
Cut-3 run spec: [`P2_CUT3_RUN_SPEC.md`](P2_CUT3_RUN_SPEC.md) (§"H0 — angular-calibration gate", frozen)
Cut-3 admission: [`P2_CUT3_ADMISSION.md`](P2_CUT3_ADMISSION.md) (HOLD)
H0 instrument **+ Wave-H0-1 NOT-sealed correction**: [`P2_CUT3_H0_CALIBRATION.md`](P2_CUT3_H0_CALIBRATION.md) (corrections C1–C3, 2026-05-16)
Phase-15 fixture (real, immutable): `results/structural-failure/cut3-prereg/h0-known-fail-fixture.json`
Filed & frozen: **2026-05-16 (PT)**. Status: **FROZEN ON SIGN-OFF —
body frozen; append-only below the Audit Notes rule.** Maintainer
sign-off recorded 2026-05-16 (PT): the three judgment calls confirmed
and red-lines A (§3 clause ordering) + B (`known_pass_selection_basis`)
integrated into the frozen body. Scope-locked: defines the H0-2 schema
only — produces no sidecars, runs no checker, emits no records or table,
admits no corpus. Gated on §0 (H0-1 correction). Trail pointers added at
sign-off (admission filing-log · rail · prereg README).

## §0 — Hard gate on the H0-1 correction (precondition, non-negotiable)

H0-2 consumes the H0 checker. That checker is presently the **corrected
artifact-identity defect** recorded in
[`P2_CUT3_H0_CALIBRATION.md`](P2_CUT3_H0_CALIBRATION.md) (2026-05-16
corrections C1/C2): `scripts/cut3-h0-checker.mjs` is committed **0-byte**;
the working logic is the misnamed tracked `_legacy_cut3-h0-checker-v0.mjs`;
§F's checker hash and §G's re-run claim are untrue as committed.

**No H0-2 step may proceed** — no sidecar consumed, no checker run, no
H0 record or residual table emitted — until the H0-1 correction's
required maintainer follow-up is done **and independently verified**:

1. the working checker is restored to its named path
   `scripts/cut3-h0-checker.mjs`;
2. the misnamed `_legacy_cut3-h0-checker-v0.mjs` is removed/quarantined;
3. the §F checker hash is recomputed and re-pinned;
4. the self-test is re-run from the real path so §G is true.

H0-2 does **not** un-correct, re-seal, or paper over H0-1. It is
strictly downstream of a *fixed* H0-1. Any H0-2 artifact produced before
§0 is satisfied is **void**.

## §1 — The 5-field discipline, lifted (justified by Cut-3/H0 alone)

`BOUNDARY_MAP.md`'s row discipline (eligible regime · abstain/switch/fail
regime · exact source receipt · traceable-agent prediction ·
mere-correlate prediction), mapped onto a per-(frame, feature) H0-2 row:

| BOUNDARY_MAP field | H0-2 row field | carries |
| --- | --- | --- |
| eligible regime | `eligible_span_deg` | where the angular map is defined |
| abstain/switch/fail | `fail_or_abstain_reason` | `SPAN_TOO_SHORT` / `ANCHOR_OFF_RULER` / `H_LEAK` / `FEATURE_OUTSIDE_SPAN` |
| exact source receipt | `source_receipts` | PNG sha256 + sidecar sha256 + manifest hash |
| traceable-agent prediction | `expected_checker_verdict` | the expected `admit`+`reason_code` (auditable: does checker output match?) |
| mere-correlate prediction | `operator_decision` | the human call (compound HaloSim code is/ isn't h-leak; fixture class) |

This discipline is adopted **because it makes the H0 measurement
auditable and carries operator judgment explicitly rather than hidden in
pattern-matching**. That merit is entirely internal to Cut-3/H0 and
needs no downstream justification (see §4 on Phase 5).

## §2 — The four schema artifacts (frozen shapes; all under `results/structural-failure/cut3-prereg/`)

**A. Per-frame measured sidecar — `h0-sidecars/<frame_id>.json`.**
Operator/tool pre-fill. **Carries no `admit`/`reason_code`/expected-
verdict field** — verdicts are the checker's job (artifact B), never the
sidecar's.

```
{ "schema_version": 1, "sidecar_kind": "h0-measured-sidecar",
  "frame_id", "frame_path",
  "render_sha256"  (from h0-known-fail-fixture.json / extractor),
  "config_path", "config_sha256",
  "measurement_method": "operator_manual" | "halosim_metadata" | "hybrid",
  "measured_at_pt", "measured_by",
  "sun_px": [x,y], "projection",
  "theta_map_kind": "scale_ticks" | "renderer_metadata" | "fit2locus",
  "scale_ticks": [ { "px_radius", "deg", "source": "operator-eye|halosim-metadata" } ],
  "anchors": [ { "locus_deg": 22|46, "px_radius": int|null,
                 "off_ruler": bool, "measurement_note" } ],
  "scored_feature_deg": number|null,
  "operator_decisions": {
    "compound_code_is_h_leak": "yes"|"no"|"deferred",
    "compound_code_basis": "<written one-line basis>",
    "known_pass_selection_basis": "<written basis if fixture_class == known_pass_fullspan, else null>",
    "fixture_class": "phase15_known_fail"|"known_pass_fullspan"|"corpus_candidate" },
  "calib_sha256": "<sha256 of this file's canonical JSON with calib_sha256
                    set to \"\" — deterministic self-pin>" }
```

**B. Per-frame H0 record — `h0-records/<frame_id>.json`.** The output of
the **restored** `cut3-h0-checker.mjs check --sidecar <A>` (§0), in the
canonical §1 H0-record format, plus a provenance block:
`{ source_sidecar_sha256 (== A.calib_sha256), checker_sha256,
checker_runtime_pt }`.

**C. Anchor-residual table — `h0-anchor-residual-table.{json,csv}`.** One
row per (frame, anchor): `frame_id, frame_class, anchor_locus_deg,
measured_deg, residual_deg, off_ruler, within_tolerance (≤0.5°), admit,
reason_code, operator_decision_codes, png_sha256, sidecar_sha256,
h0_record_sha256`. Emitted canonical-JSON **and** CSV (same column
order). `expected_checker_verdict` for the Phase-15 known-fail set is
**inherited from the already-frozen `h0-known-fail-fixture.json`
`expected_self_test`** (not re-set here); for a known-PASS candidate it
is the instrument-criterion expectation recorded as the §3 selection
rationale *before* the checker runs; for any future corpus candidate it
is `operator-blind`/null. It is never hand-written to match checker
output.

**D. H0-2 manifest — `h0-2-manifest.json`.** Indexes and hash-pins every
sidecar, record, and the residual table; mirrors
`h0-known-fail-fixture.json` / `cut1-fixture-manifest.json`. One place to
verify the whole H0-2 substrate is byte-pinned and reproducible.

## §3 — H0-2 keystone load-bearing self-seal (surfaced adversarially)

The schema's own hazard, the H0-2 analog of Cut-1's `g⁻¹(g(h))` and
Wave-H0-1's hardcoded stub: **the operator pre-fills the sidecar so the
frame comes out the wanted way** — fudging `scale_ticks` /
`anchor px_radius` / `scored_feature_deg`, or picking/measuring a
known-PASS candidate *because it passes* rather than because the
instrument criterion is met. That is verdict-before-measurement.

Frozen invariant — **measure-before-verdict, structurally enforced by
file separation** (the artifact-A / artifact-B split is the §"calibrate
vs admit" separation one level out):

1. The measured sidecar (A) contains **no verdict field**; it is frozen
   and self-pinned (`calib_sha256`) *before* the checker runs.
2. **Primary mechanical detector (load-bearing):** the H0 record (B)
   pins `source_sidecar_sha256 == A.calib_sha256`. If A is altered after
   B exists (to make it admit), A's self-pin changes and B's
   `source_sidecar_sha256` no longer matches ⇒ **void**, mechanically
   detectable by the manifest (D) check.
3. **Additional ordering check (defense in depth):** B carries
   `checker_runtime_pt`; the invariant
   `A.measured_at_pt < B.checker_runtime_pt` must hold. Secondary to
   clause 2 — the SHA-mismatch is the load-bearing test.
4. **Known-PASS sub-hazard:** the H0-calibration freeze forbids a
   synthetic stub for the known-PASS side. The known-PASS render must be
   a *real full-span* render selected **by the instrument criterion**
   (stamped ruler genuinely covers 22° and 46°), with the selection
   basis written in `operator_decisions.known_pass_selection_basis`
   *before* the checker is run on it. "It happened to pass" is not a
   selection basis.

A measured sidecar whose numbers were adjusted to obtain a verdict, or a
known-PASS chosen by its result, is the H0-2 self-seal and voids the
artifact.

## §4 — Honest couplings

**Upstream (binding):** gated on §0 (H0-1 correction); consumes the
restored canonical checker; reuses the frozen Phase-15 fixture
(`h0-known-fail-fixture.json`) as-is; inherits the frozen run-spec
`≤0.5°` tolerance and 22°/46° loci.

**Downstream (severable note — NOT a dependency):**

> Phase 5 (`COARSE_GRAINING_PROOF_ROADMAP.md`), if and when it lands,
> *may* consume the residual table (C) because the column schema is
> compatible. This is a convenience of shared form, **not a
> dependency**. H0-2's measurement rules are fixed by Cut-3/H0 and do
> **not** change if Phase 5 changes, slips, or is abandoned. **No
> Phase-5 need may relax an H0-2 measurement rule.** This paragraph is
> excisable without affecting any H0-2 rule.

The coupling direction is pinned: a measurement-hygiene gate is never
made contingent on a downstream claim track. (Same principle as P1's
`debunked.md` vocabulary bridge: record compatibility, refuse the
dependency.)

## §5 — Provenance-tagged freeze (A3)

`[G]` immutable / inherited; `[E]` pre-registered engineering tolerance
(amend-only, justified, never post-results).

| item | provenance |
| --- | --- |
| §0 H0-1 correction gate | **[G]** (precondition; cannot be waived) |
| §1 5-field per-(frame,feature) discipline | **[G]** |
| §3 measure-before-verdict invariant + file-separation enforcement | **[G]** |
| 22°/46° loci, `≤0.5°` tolerance | **[G]** (inherited from frozen run spec) |
| known-PASS = real full-span render, no synthetic stub | **[G]** (inherited from H0 freeze) |
| `schema_version`, file-path layout, CSV column order, timestamp format | **[E]** |
| canonical-JSON self-pin convention (`calib_sha256` with field = `""`) | **[E]** |

## §6 — Landing order (Wave H0-2; nothing here is run)

0. **§0 gate cleared and verified** (H0-1 C1/C2 fixed).
1. This schema doc frozen on sign-off.
2. `scripts/cut3-h0-residual-table.mjs` — deterministic given sidecars.
3. 8 Phase-15 measured sidecars (operator pre-fill, §3-blind).
4. 1 known-PASS measured sidecar (operator survey + pre-fill, §3.4).
5. restored checker over each → 9 H0 records.
6. residual-table generator → table (C) + manifest (D).
7. audit-notes append on `P2_CUT3_H0_CALIBRATION.md` pinning all hashes
   and recording the H0-B **two-sided** disposition on *real measured*
   frames — only then may the H0-B negative side be considered proven on
   real frames and the two-sided self-test closed.
8. Phase 5 *may* inherit table C (severable, §4).

## Binding rules

1. §0 is a hard gate; any H0-2 artifact before it ⇒ void.
2. Measured sidecar (A) carries no verdict; frozen + self-pinned before
   the checker runs; `measured_at_pt < checker_runtime_pt`.
3. Known-PASS is a real full-span render chosen by the instrument
   criterion with a written basis; never a synthetic stub; never chosen
   by its result.
4. `expected_checker_verdict` is inherited/derived, never hand-set to
   match output.
5. No frozen run-spec/H0 value changed; no Phase-5 need relaxes any
   H0-2 rule.

## Explicit non-bindings (cannot satisfy H0-2)

- Any H0-2 step before the §0 gate.
- A sidecar measured or adjusted after seeing the checker verdict.
- A known-PASS fixture that is synthetic, or chosen because it passed.
- A hand-written `expected_checker_verdict`.
- A binding (non-severable) coupling to Phase 5, or any Phase-5-driven
  relaxation of an H0-2 rule.
- Fabricated sidecars/records (operator pre-fill must be real
  measurement from the real PNGs).

## Open items

Frozen 2026-05-16 (PT) on maintainer sign-off (three judgment calls
confirmed; red-lines A+B integrated). Trail pointers added at sign-off
(admission filing-log · rail · prereg README); the body above is now
append-only below the Audit Notes rule. Wave H0-2 then executes §6 in
order, gated on §0. Nothing is run by filing or freezing this schema.

## Public-Language guard

Producing this schema does **not** seal H0, does not close the H0-B
two-sided self-test, and does not begin Cut-3. No `CONFIRMED` /
"traceability" / theorem / "Cut-3 has begun or passed" language. H0
closes only when the two-sided self-test resolves on real measured
frames per §6.7. Cut-3 admission remains **HOLD**; execution **HELD**.

## Audit Notes

Append-only below this rule. Each entry: timestamp (date + zone),
author, one-line justification. The body above is frozen at sign-off
(2026-05-16 PT).

**2026-05-16 (PT) — maintainer. Sign-off + freeze.** The three judgment
calls confirmed: (1) `expected_checker_verdict` provenance
(Phase-15 inherited from the frozen `h0-known-fail-fixture.json`,
instrument-criterion for known-PASS, operator-blind/null for corpus);
(2) DRAFT-freezes-on-sign-off status with trail pointers unwired until
now; (3) §6.7 — H0-B negative side claimable only after real measured
sidecars + records exist (H0-2 is the path to proving it, not a rescue
of the Wave-H0-1 BLOCK). Red-line A integrated: §3 file-separation
enforcement reordered so the `source_sidecar_sha256 == calib_sha256`
mismatch is the primary load-bearing detector (clause 2) and the
`measured_at_pt < checker_runtime_pt` ordering is demoted to
defense-in-depth (clause 3). Red-line B integrated: dedicated
`known_pass_selection_basis` field added to the §2-A
`operator_decisions` block; §3 clause 4 prose now references it. Both
additive and semantics-preserving. Body frozen; §0 H0-1-correction gate
remains the hard precondition; Cut-3 admission HOLD, execution HELD,
Public-Language Constraint in force; nothing run.

**2026-05-16 (PT) — maintainer. Wave H0-2 §6.2 generator filed.** The
residual-table generator (artifact C + manifest D producer) is filed at
`scripts/cut3-h0-residual-table.mjs` with §3 anti-self-seal applied at
the generator layer: **no verdict synthesis** (admit/reason_code/
measured_deg/residual_deg copied byte-for-byte from artifact B; the
generator never invents a verdict) and **consistency-fail-loud** (the
three pin checks per (sidecar, record) pair — sidecar self-pin recompute,
record↔sidecar SHA equality per red-line A, timestamp ordering as
defense-in-depth — each fail emits a row with `consistency: false` and
the specific failure code; the generator never silently corrects). The
PRIMARY mechanical detector at this layer is the
`record.provenance.source_sidecar_sha256 == sidecar.calib_sha256` SHA
equality check, mirroring the schema's §3 red-line A primacy. Orphan
sidecars/records emit rows with `reason_code = ORPHAN_RECORD` /
`ORPHAN_SIDECAR` and no synthesized admit.

*§G.1 — Schema-mechanical self-test: 11/11 PASS (plumbing only).* Run:
`node scripts/cut3-h0-residual-table.mjs self-test`. Exercises eleven
predicates of the generator's plumbing — happy-path two-row emission +
byte-for-byte admit copying, SHA-mismatch detection emitting
`RECORD_SIDECAR_SHA_MISMATCH`, timestamp-ordering-violation detection,
orphan-sidecar/orphan-record handling, CSV column order matching §2-C
verbatim, and the no-verdict-synthesis invariant on both happy and
orphan paths — using synthetic conformant inputs. **This is a unit
check of the generator's parsing/dispatch/emit logic only.** PASS here
makes no substantive claim about H0-2; per §6.7, H0-2 closure requires
operator-pre-fill on real Phase-15 frames + a known-PASS fixture.
Synthetic self-test inputs are written under
`results/structural-failure/cut3-prereg/h0-residual-selftest-tmp/` and
are explicitly NOT H0 records.

*§G.2 — Empty-state outputs pinned.* Running `generate` against the
currently-empty `h0-sidecars/` and `h0-records/` directories (Wave H0-2
operator pre-fill has not yet produced any) emits scaffolding artifacts
showing 0 rows and 0 inputs. These exist as evidence the generator is
deployed and ready; they will be byte-replaced on the first real run
once operator pre-fill lands.

*Pinned artifacts (paths repo-relative, hashes SHA-256, 2026-05-16 PT).*

| artifact | path | sha256 |
| --- | --- | --- |
| §6.2 generator script | `scripts/cut3-h0-residual-table.mjs` | `d93ac6e1bde9751347eaa5060351dc87cdf8b79ad302e918e842a393151a0c46` |
| Self-test result (11/11 plumbing) | `results/structural-failure/cut3-prereg/h0-residual-table-self-test-result.json` | raw `dbe44a4ce8e7f13d9d753988facb2102e3f232abdf2f9583c47b67ff1fd5481f` · canonical `1819462a17fdb6f084321aad6883a519b2358bdd1f9da5c0ab776a88b371e276` |
| Residual table (empty-state, JSON) | `results/structural-failure/cut3-prereg/h0-anchor-residual-table.json` | raw `33b1f83a48a506fb594598e4fb49a35e8daa1ca1be7270348cdef24445cd21fb` · canonical `d1b559b88ec2e7aaf265dedb19f9da29d2da0b3f4eafb13baed7eee11d15bbc6` |
| Residual table (empty-state, CSV header only) | `results/structural-failure/cut3-prereg/h0-anchor-residual-table.csv` | `892dce86ad29c3fdbdb21539590024ee951cc3db8a046f38788d0a9bbb7c2b31` |
| Manifest (empty-state) | `results/structural-failure/cut3-prereg/h0-2-manifest.json` | raw `6467086c8bc934af89c7a5be34eb212282f7e9cfb4132824e80ae05d0b61612d` · canonical `172a1b006e67cce9f32b566f9cb23b3548952d6938922f3b44ae6ea8e3753d9a` |

*§G.3 — CLI surface.* `generate` (deterministic given the input
directories), `validate --sidecar <path>` / `validate --record <path>`
(schema-mechanical pre-flight that an operator can run on a single
candidate file before committing to the corpus), `self-test` (the §G.1
plumbing check), and `hash-file <path>` (utility).

*§G.4 — Re-run determinism.* Re-running the generator against the same
inputs produces byte-identical outputs. Verified locally: self-test
re-run hash `dbe44a4c…481f` (unchanged); generate re-run manifest hash
`6467086c…612d` (unchanged).

*§G.5 — Authoring-side scratch.* The self-test tmp directory at
`results/structural-failure/cut3-prereg/h0-residual-selftest-tmp/`
contains eight synthetic JSON files (four sidecars, four records) used
by the self-test. These are unit-test fixtures, not H0 records; the
real `generate` mode reads from the canonical `h0-sidecars/` and
`h0-records/` paths and never consumes anything from the tmp directory.
Safe to leave; reproduced by every self-test run.

*§G.6 — Discipline.* No frozen schema body edited. §0 H0-1-correction
gate continues to apply (already satisfied on git main per commit
`bf6aa2a` / current canonical checker SHA `7a520f3f…f05cb`). H0-2
landing order now stands at §6.3–§6.4 (operator pre-fill on real
Phase-15 frames + known-PASS fixture identification + measurement).
Cut-3 admission **HOLD**; execution **HELD**; Public-Language
Constraint in force; H0-B negative side on real frames remains OPEN.

Justification: closes §6.2 of the schema's landing order with the
generator filed, plumbing self-test passing, empty-state outputs pinned,
and §3 anti-self-seal applied at the aggregator layer. Real-table
generation is gated on operator pre-fill landing.

**2026-05-16 (PT) — maintainer. Optional measurement-helper tool filed
(convenience UX, NOT load-bearing).** A polished single-file
HTML/canvas measurement helper for §6.3 operator pre-fill is filed at
`tools/h0-measurement/index.html` (SHA-256
`7e9f279e41d11780af58ce98a47c7990e01c59884b38f48343a21f9c83a52d77`).
Workflow: load any halo PNG → click sun center → drag radial line to
22° → click → drag to 46° → click → tool computes `sun_px` + radial
`px_radius` per anchor + live `calib_sha256` self-pin → operator
downloads a §2-A-conformant sidecar JSON. Both `fit2locus` (clicks at
22°/46° loci, for halo photos without a stamped ruler) and
`scale_ticks` (click each stamped degree mark for HaloSim-style
renders) modes are supported.

*Discipline alignment.* The tool implements §3-blind structurally: it
does **not** import the H0 checker, does not call the predicate, does
not display admit/reason_code anywhere in its UI. Its only output is
the measured sidecar JSON. The checker runs separately, after all eight
sidecars are sealed, per the schema's §6.5. The tool is therefore a
pure-measurement UX layer and could be entirely replaced (or augmented
with an alternative tool, or skipped in favor of manual JSON authoring)
without affecting any frozen schema rule.

*Public-Language Constraint baked into the tool's UI.* The header
carries the disclosure: "No theorem claim. H0 is measurement hygiene —
it decides whether a rendered frame is angularly trustworthy enough to
score, not whether any downstream claim passes. Cut-3 admission remains
HOLD; execution HELD." This is intentional — if the tool is later moved
to `public/` for a public-facing demo, the PLC framing is already
in-page and cannot be silently stripped. For now the tool lives under
`tools/` (which `scripts/copy-site-docs.mjs` does not copy to `dist/`)
and remains internal-facing; promotion to `public/` is a separate
decision that should be filed on its own.

*Scope deliberately limited.* The tool does not auto-detect sun
position, halo arcs, or scale ticks from PNG pixels — measurement is
operator-eye on the canvas. Auto-detection would introduce its own
tautology hazards (a tool that "finds" the 22° halo at a hand-tuned
expected location). The honest design keeps the operator's eye as the
measurement substrate; the tool just records click coordinates and
computes `px_radius` deterministically from `sun_px`.

*Status.* This is an optional convenience for §6.3. §6.3 (8 Phase-15
measured sidecars) and §6.4 (1 known-PASS measured sidecar) remain the
operator pre-fill items. The tool does not produce them automatically;
it makes producing them faster and less error-prone. Cut-3 admission
remains **HOLD**; execution **HELD**; H0-B negative side on real frames
remains **OPEN**; Public-Language Constraint in force.

**2026-05-16 (PT) — Critic / maintainer. Finding-1 fix: cross-runtime
canonicalization pinned with shared library + test vector +
acceptance criterion.** Reviewer pass on the measurement-tool filing
caught a real load-bearing gap: §2-A says "canonical JSON" in prose
without pinning the algorithm precisely. Red-line A's primary mismatch
detector (`record.provenance.source_sidecar_sha256 ==
sidecar.calib_sha256`) is only as strong as the future restored Node
checker reproducing the browser tool's canonical string byte-for-byte.
The three existing implementations (Node guard's `canonicalize`,
residual-table generator's `canonicalJSON`, browser tool's
`canonicalJSON`) happen to be identical algorithms, but only by happy
coincidence — a future implementer who writes `JSON.stringify(obj)`
(top-level only, no sort) would silently break the detector.

Pinning fix, three artifacts:

1. **Shared canonicalizer library** at `scripts/lib/canonical-json.mjs`
   (SHA-256 `e2e32a03ad0735d63f2f28a97e2c3556204fbb27317fe7bdf60a8c395058c730`).
   Exports `canonicalize(value)`, `sha256Hex(str)`,
   `computeSidecarSelfPin(sidecar)`, `verifySidecarSelfPin(sidecar)`.
   Header documents the algorithm exactly (recursive structure for
   arrays/objects, `Object.keys().sort()`, `JSON.stringify` for leaves,
   no whitespace, UTF-8 output, the `calib_sha256 = ""` self-pin
   convention). Both the Node guard and the residual-table generator
   already use byte-identical algorithms; the restored Node checker
   (§0/C1 follow-up) MUST adopt this library. The browser tool inlines
   the same algorithm verbatim (cannot import Node modules); its
   correctness is contracted by the test vector below.

2. **Cross-runtime test vector** at
   `results/structural-failure/cut3-prereg/h0-canonicalization-test-vector.json`
   (raw `0c470f30074141405c54ff4b7c3e5958df8b25dd9b86d2565e9268421baa6d33`,
   canonical `6cf64a11f736a3c55ee1ad065f70069a90dd20588fa3408b26afce7e992880a6`).
   Pins one synthetic conformant sidecar + its expected canonical
   string (922 UTF-8 bytes) + its expected `calib_sha256`
   (`9c52ab15d5293341cfd45e52dba615d01b6698ee2c5345f59942e25f711ac8b6`).
   The test sidecar is constructed to exercise: nested objects
   (`operator_decisions`), arrays of objects in insertion order
   (`anchors`), mixed leaf types, UTF-8 in strings, and **deliberately
   unsorted top-level keys at fixture-construction time** so the
   sort step is non-trivial. Generated by
   `scripts/cut3-h0-make-test-vector.mjs` (SHA-256
   `bbebf6ccfc5294316ca1da6bb4d6865c056f969e904b33f7e95d3b6aa3dc3541`);
   deterministic and re-derivable.

3. **Cross-runtime parity verified in this filing.** The same input
   sidecar, run through both the Node `canonicalize` (via the shared
   lib) AND the browser tool's inlined `canonicalJSON` (re-executed in
   Node for the verification), produces byte-identical 922-byte
   canonical strings and byte-identical `calib_sha256`
   `9c52ab15…c8b6`. The two implementations agree on this vector.

*New §0 acceptance criterion (additive; does NOT relax §0 — §0's C1
defect is still satisfied per the H0-1 correction).* The restored Node
H0 checker, when wired in, MUST pass
`verifySidecarSelfPin(pinned_sidecar_with_calib_sha256).ok === true`
on the test vector's pinned sidecar. This becomes part of the §0
verification: §0 is satisfied iff (a) the canonical-path file at
`scripts/cut3-h0-checker.mjs` is the working checker (already
verified), (b) the `_legacy_` path is quarantined (already verified),
AND (c) the checker reproduces the test vector's `calib_sha256`
byte-for-byte. (c) is a new criterion attached to the same §0 gate;
it makes red-line A's primary detector load-bearing in writing, not
just by happy coincidence.

*Pinned algorithm summary* (the authoritative version lives in the
library header):

> Recursive serialization. Arrays: `"[" + canonicalize(v0) + "," +
> canonicalize(v1) + ... + "]"`, insertion order, no whitespace.
> Objects: `keys = Object.keys(value).sort()` then
> `"{" + JSON.stringify(k0) + ":" + canonicalize(v0) + "," + ... + "}"`
> recursively. Leaves (`null` / boolean / number / string):
> `JSON.stringify(value)`. UTF-8 output. The self-pin step sets
> `calib_sha256 = ""` (empty string, NOT null and NOT absent) prior to
> canonicalization, SHA-256s the UTF-8 bytes, hex-encodes lowercase.

*Scope of this finding's fix.* No frozen schema body edited. No
existing artifact's `calib_sha256` changed (the algorithm was already
identical across all three implementations; this filing pins the
agreement so future implementations can't drift). No Public-Language
language altered; no Cut-3 admission move. The fix is append-only and
purely about contracting the existing parity in writing + giving any
future Node checker re-implementation a one-line acceptance test it
must pass.

*(reviewer space continues — append-only)*
