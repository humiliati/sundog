# Yang-Mills Bounded-Null External Review Packet

> Minimal packet for an external sanity check by a lattice gauge theorist.
> This is not a public page and not a Clay-problem claim. It exists to make
> it easy for a reviewer to say "yes, this null is correctly characterized,"
> "no, this overstates X," or "you missed standard issue Y."

> **2026-05-31 status addendum:** this packet was drafted before the v4/v5
> powered-target audits. Both receipts landed
> `YM-P2-UNDERPOWERED no_powered_target_in_envelope`: v4 found no powered
> Wilson-loop target in the 12^3 envelope, and v5 found the symmetric
> Polyakov targets disjoint but underpowered. Stage 2 was not scored in either
> audit. Do not send this packet as-is; update the ask from "four-probe bounded
> null" to "four prior nulls plus two underpowered-envelope audits, with v6
> finite-temperature Polyakov as the only registered continuation" if external
> review is still pursued.

**Date:** 2026-05-29
**Status:** draft reviewer packet
**Primary synthesis:** [`receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md)

## Reviewer Snapshot

Repository map:

- Primary working packet: `docs/yang-mills/`.
- Main ledger: [`../SUNDOG_V_YANG_MILLS.md`](../SUNDOG_V_YANG_MILLS.md).
- Lit-pass: [`../YANG_MILLS_LITPASS_MEMO.md`](../YANG_MILLS_LITPASS_MEMO.md).
- P0 domain lock: [`../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md).
- P0 amendment 1 (APE smearing): [`../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md).
- Current conclusion: [`receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md).

Why we looked at Yang-Mills:

> Pure SU(N) lattice gauge theory was used as a bounded stress test for whether
> Sundog's "lossy gauge-invariant signature preserves a held-out observable
> label in rank space beyond controls" discipline can earn a finite-lattice
> certificate on a real gauge substrate. It did not, on the registered cell.
> The review question is whether that bounded null is framed honestly given
> standard lattice gauge theory practice.

Current status:

- Phase 1 instrumentation closed across U(1) 2D / SU(2) 2D / SU(2) 3D
  (gauge-randomization smokes, all `P1-A smoke_pass`).
- Phase 2 ran four pre-registered relative-locality probes (v0 / v1 / v2 / v3)
  on the SU(2) 3D primary cell; all four landed `YM-P2-NEG-A no_rank_local_structure`.
- A bounded-null synthesis receipt is filed; the lane is paused.
- No Yang-Mills, confinement, mass-gap, continuum, or Clay-problem claim is
  live.
- The public surface is blocked until an external sanity check confirms or
  corrects the bounded-null framing.

Core mapping:

| Sundog concept | Yang-Mills instantiation | Current read |
| --- | --- | --- |
| Lossy gauge-invariant signature | small Wilson-loop summaries: bare mean/var (vocab v1, 8-dim), APE-smeared mean/var with frozen `(α, N_sm) = (0.5, 10)` (vocab v4, 8-dim), bare connected 2-point correlator at 5 cubic-symmetry displacements (vocab v5, 20-dim) | three signature classes registered; all gauge-invariant by construction (`CTRL_GAUGE_RAND` to ≤ 1e-12 / 1e-13 / 5e-13) |
| Held-out observable label | larger-Wilson-loop class: per-config exponential-decay slope `γ_held` over `{W14, W23, W33}` (target vocab v1); per-config spatial variance `σ²_W33` over all 12³ × 3 = 5184 (position × orientation) samples (target vocab v2) | two target classes registered; one area-law, one non-area-law; both gauge-invariant |
| Relative-locality certificate | within-β k-NN bin-purity@5 against per-β tertile labels on either γ_held or σ²_W33 | discrimination ratio inside ±0.04 of chance baseline 1/3 across all four (signature × target) probes |
| Leakage controls battery (P0 lock) | seven entries: CTRL_META, CTRL_RAW, CTRL_RAND, CTRL_RAND_STRAT, CTRL_PERM, CTRL_GAUGE_RAND (all scored); CTRL_FINITE_SIZE declared but deferred to Phase 4 | all health gates passed across all four probes; one bug caught by `CTRL_GAUGE_RAND` (APE staple orientation, see methodology note below) before any score was interpreted |
| Pre-registration discipline | every probe filed a dated spec before its runner ran; smearing parameters frozen at the amendment that admitted them; v4 fallback pre-stated as PAUSE-and-synthesize before v3 ran | followed verbatim across all four probes; PAUSE executed on schedule |

Ways to falsify or weaken this packet:

- Show that one of the three signature classes (vocab v1 / v4 / v5) is so
  thin a representation of small-loop structure that the bounded null says
  nothing about whether richer small-loop signatures would or would not
  resolve γ_held / σ²_W33.
- Show that 32 configurations per β at 12³ is below the standard threshold
  for resolving any held-out tertile structure with k-NN scoring, regardless
  of signature design — making the bounded null an "N too small" finding
  rather than a signature-class finding.
- Identify a standard lattice-gauge artifact at 12³ (finite-volume, β
  near a crossover, etc.) that would have made any positive result on this
  cell a known false signal in the first place.
- Recommend that the bounded-null surface language be narrowed from
  "small-loop signature does not resolve this target" to "small-loop
  signature × small ensemble does not resolve this target."
- Point out that the held-out target γ_held (LS slope over three points)
  has a known noise structure that we mischaracterized as a useful tertile
  label, and propose a more standard area-law summary.
- Flag a known correlation between σ²_W33 and per-config mean W33 that
  would make σ²_W33 effectively redundant with γ_held, invalidating the
  claim that the two targets are independent information classes.

## One-Sentence Ask

Please sanity-check whether our bounded-null conclusion is correctly framed:
across four pre-registered (signature × target) probes on the registered
SU(2) 3D 12³ × β slate `{2.0, 2.4, 2.8}` × 32-configs-per-β cell, the
small-loop gauge-invariant signature space did not earn a relative-locality
certificate, and the named null is the durable result rather than an artifact
of N, β slate, signature class, target choice, or pre-registration design.

## What We Are Not Asking

- Not asking for a review of, or position on, the Clay Yang-Mills existence
  and mass-gap problem.
- Not asking whether this is evidence for or against confinement, a mass gap,
  or continuum behavior.
- Not asking for endorsement of Sundog, the broader apparatus, or any public
  presentation.
- Not asking the reviewer to debug the entire repository.
- Not asking for a lit-pass adjudication of recent claimed Clay solutions
  (Jacobsen withdrawn, Glimm 2025, Odusanya 2026 — quarantined in
  [`../YANG_MILLS_LITPASS_MEMO.md`](../YANG_MILLS_LITPASS_MEMO.md) and
  not load-bearing on this packet).

## What We Are Asking

Please check the following calls. Adapted from the five locked reviewer
questions in the P0 lock, with two added bounded-null-specific questions.

1. **Signature/target disjointness.** Is `γ_held` (LS slope over
   `(W14, W23, W33)` with `1e-10` ε floor) independent enough from the v1
   signature `{W11, W12, W13, W22}` mean/variances that **any positive read**
   would have been the natural relative-locality result rather than trivial
   signature-into-target leakage? Same question for `σ²_W33` per-config
   spatial variance vs the v1 signature. We want this audited from the
   gauge-theory side, not the receipt side.

2. **Leakage-control sufficiency.** Are the seven leakage controls
   (`CTRL_META`, `CTRL_RAW`, `CTRL_RAND`, `CTRL_RAND_STRAT`, `CTRL_PERM`,
   `CTRL_GAUGE_RAND`, plus the deferred `CTRL_FINITE_SIZE`) sufficient for
   a small-lattice non-Abelian setting, or is there a standard
   gauge-theory failure mode at 12³ they would miss? In particular: would
   an absent control plausibly have flipped any of the four probes from
   NEG-A to a positive?

3. **Autocorrelation / ensemble health.** Is the locked rule `τ_int ≤ 16
   combined sweeps, thinning ≥ 2 · τ_int, 2000 burn-in sweeps minimum`
   strict enough at β ∈ `{2.0, 2.4, 2.8}` and 12³? Observed:
   τ_int(plaquette) ≈ 0.5–1.0 across the three β with 1 HB + 4 OR per
   combined sweep; thinning interval 32 sweeps; 32 configurations per β.
   Are 32-config-per-β samples large enough for any meaningful k-NN
   rank-locality scoring against tertile labels?

4. **β-slate coverage.** Is the SU(2) 3D β slate `{2.0, 2.4, 2.8}` a
   reasonable choice for spanning confinement-to-perturbative regimes at
   8³ and 12³, or is there a missing β value (e.g. β ≈ 2.2 or 2.6) at
   which the signature space plausibly **would** preserve held-out
   structure that the registered slate misses? If yes, please name it.

5. **Standard lattice artifacts.** Are there standard finite-volume,
   action-improvement, or β-near-crossover artifacts on a 12³ SU(2)
   Wilson-action ensemble that would trivialize either a positive read OR
   the bounded-null read? In particular, is the bare-signature 1×1
   plaquette ⟨W11⟩ ≈ 0.50–0.62 across the registered β consistent with
   standard published numbers? (We did not pre-register a published-range
   gate; we report the value and ask whether it is in family.)

6. **Probe-ladder completeness.** The lane tested four (signature ×
   target) combinations: `{v1 bare, v4 smeared, v5 correlator} × {γ_held}`
   plus `{v1 bare} × {σ²_W33}`. Pre-stated v4 fallback candidates that
   were NOT run (and remain admissible only with fresh motivation):
   `σ²_W14`, `σ²_W23`, Polyakov-loop target (would need P0 amendment 2),
   smeared signature × σ²_W33. Are any of these "should have done before
   declaring bounded null" rather than "optional with external motivation"?
   If a missing combination is a known textbook discriminator on small
   ensembles, please name it.

7. **Pre-registration discipline as a methodological claim.** The lane's
   pre-registration practice — dated probe spec before each runner, frozen
   smearing parameters with anti-scope-creep clause, pre-stated PAUSE-and-
   synthesize default before the trigger probe — is itself a claim we make
   about how to file null results honestly. Is this recognized as a
   defensible safeguard against signature/target shopping on this kind of
   substrate? Or are there gauge-theory-specific concerns (e.g.
   non-uniqueness of "natural" small-loop signatures, β-slate aliasing,
   smearing-parameter sensitivity) that this discipline papers over and
   that we should flag explicitly in the synthesis?

A short reply is enough. The most useful possible answers are:

```text
Yes, the bounded-null framing is basically right.
No, this overstates X.
I would quarantine Y until you address Z.
Re-run with W; the current null is dominated by N=32 noise.
This is standard/known as A; cite B.
The probe ladder missed obvious combination C; that should be tested.
```

## Methodology Note: CTRL_GAUGE_RAND Bug Catch

During v1 (APE-smearing) implementation, an early draft of the smearing
routine used the heatbath-staple traversal order (matrix product
sequence optimized for Boltzmann conditional sampling of a single link
given its three-link staple sum) instead of the gauge-equivariant
plaquette traversal required for APE smearing to commute with gauge
transformations. The bug surfaced immediately when `CTRL_GAUGE_RAND`
re-ran the smearing pipeline on a Haar-randomized ensemble and found
the smeared signature was **not** invariant — `YM-P1-NEG-A gauge_leakage`
fired before any rank-locality score could be interpreted.

The corrected implementation re-ran and produced the clean v1 receipt
numerics (det drift 6.66e-16, post-smearing unitarity 9.42e-16, gauge
residual 1.44e-15). The receipt records the bug catch as a methodology
event: it is the kind of catch the seven-entry leakage controls battery
is meant to produce. We mention this here so a reviewer can audit the
discipline working as designed rather than just the four green checkmarks.

## Reviewer Time Budget

Suggested review paths:

- **10 minutes:** read the bounded-null synthesis receipt's "Synthesis"
  section, the four-receipt matrix, and the PAUSE Disposition.
- **25 minutes:** additionally inspect the v3 probe spec's pre-stated v4
  fallback table (the discipline that produced the PAUSE), and the
  per-probe pass/fail tables in the v0 and v3 binding specs.
- **60 minutes:** additionally inspect P0 amendment 1 (smearing parameter
  freeze), the v1 probe spec's hypothesis-falsification logic, and at
  least one of the per-β v0 ensemble manifests (for the lattice / β /
  generator / health-gate parameters at the most concrete level).

## Core Claim To Audit

From the synthesis:

> Inside the registered `SU(2) 3D`, `12³`, β-slate `{2.0, 2.4, 2.8}`,
> 32-configurations-per-β envelope, the Sundog small-loop
> gauge-invariant signature space — tested across vocab v1 (bare
> mean+var, 8-dim), vocab v4 (APE-smeared mean+var with frozen
> `(α, N_sm) = (0.5, 10)`, 8-dim), and vocab v5 (bare connected 2-point
> correlator at five frozen cubic-symmetry displacement classes, 20-dim)
> — does not preserve within-β rank-local structure for either of the
> two pre-registered held-out targets (area-law-decay slope `γ_held`;
> spatial variance `σ²_W33`). The bounded null, with all four probes
> registered before their runs and the disposition selected from a
> pre-stated fallback table, is the durable result on this cell.

This sentence is the thing under review. If it is too strong, the packet
should be revised before any public surface exists.

## Four-Receipt Summary

| Probe | Signature | Target | Primary bin-purity@5 | RAND margin | Verdict | Receipt |
| --- | --- | --- | --- | --- | --- | --- |
| v0 | vocab v1 (bare 8-dim mean+var) | γ_held (LS slope, vocab v1) | 0.31042 | +0.01042 | `YM-P2-NEG-A` | [`receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md) |
| v1 | vocab v4 (APE-smeared 8-dim mean+var, α=0.5, N_sm=10) | γ_held (LS slope, vocab v1) | 0.29375 | −0.00208 | `YM-P2-NEG-A` | [`receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md`](receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md) |
| v2 | vocab v5 (bare 20-dim connected 2-point correlator) | γ_held (LS slope, vocab v1) | 0.30833 | +0.02083 | `YM-P2-NEG-A` | [`receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md`](receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md) |
| v3 | vocab v1 (bare 8-dim mean+var; re-read from v0 CSV, SHA-256 asserted) | σ²_W33 spatial variance (vocab v2) | 0.32917 | +0.02708 | `YM-P2-NEG-A` | [`receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md`](receipts/2026-05-29_SU2_3D_phase2_v3_no_rank_local_structure.md) |

Chance baseline (tertile bins): 1/3 ≈ 0.3333. Promotion gates (all four
probes share these): primary bin-purity@5 `≥ 0.5` AND margin over
`CTRL_RAND` `≥ 0.10`. All four probes fail both gates.

Ensemble health (Phase-1-inherited gates) passed on all four probes:
burn-in `≥ 2000`, τ_int(plaquette) `≤ 16` combined sweeps (observed
≈ 0.5–1.0), thinning interval `32 ≥ 2 · τ_int`, heatbath fallback
fraction `≤ 0.001` (observed `0` over millions of link updates), link
unitarity `≤ 1e-10` (observed `≤ 1.1e-15`), orientation isotropy spread
`≤ 5e-2` (observed `≤ 3.1e-3`).

## Files To Read

Primary:

- [`receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md)
  - capstone bounded-null synthesis; the load-bearing statement under review.
- [`../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
  - registered envelope and locked reviewer questions.
- [`../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md)
  - the one P0 amendment filed during the run; freezes APE smearing
  parameters and adds the `YM-P*-QUAR-E smearing_drift` quarantine.

Secondary (per-probe binding specs and probe-spec rationales):

- [`../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md`](../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md)
  - v0 spec; defines γ_held, per-β tertile bin convention, distance metric,
  controls battery.
- [`specs/2026-05-29_phase2_v1_smearing_probe.md`](specs/2026-05-29_phase2_v1_smearing_probe.md)
  - v1 probe rationale (UV-noise hypothesis) and the T1/T2/T3 design audit.
- [`../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md`](../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v1.md)
  - v1 smeared-signature binding spec.
- [`specs/2026-05-29_phase2_v2_correlator_probe.md`](specs/2026-05-29_phase2_v2_correlator_probe.md)
  - v2 probe rationale (UV-noise falsified; T2 promoted; locked
  displacement slate).
- [`../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md`](../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v2.md)
  - v2 correlator binding spec.
- [`specs/2026-05-29_phase2_v3_target_redesign_probe.md`](specs/2026-05-29_phase2_v3_target_redesign_probe.md)
  - v3 probe rationale (target-side redesign; pre-stated v4 fallback table
  including PAUSE-and-synthesize default).
- [`../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md`](../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md)
  - v3 σ²_W33 binding spec.

Lit-pass / prior-art context (not load-bearing on the bounded null):

- [`../YANG_MILLS_LITPASS_MEMO.md`](../YANG_MILLS_LITPASS_MEMO.md)
  - records what the 2026-05-29 literature pass found and what it
  quarantined.

Runnable / audit support (under `scripts/` at repo root, not under
`docs/`):

- `scripts/yang-mills-phase1-gauge-smoke.mjs` - U(1) 2D Phase 1 runner.
- `scripts/yang-mills-phase1-su2-gauge-smoke.mjs` - SU(2) 2D Phase 1
  runner.
- `scripts/yang-mills-phase1-su2-3d-gauge-smoke.mjs` - SU(2) 3D Phase 1
  runner.
- `scripts/yang-mills-phase2-su2-3d-ensemble.mjs` - Phase 2 v0 per-β
  ensemble generator (12³).
- `scripts/yang-mills-phase2-su2-3d-aggregate.mjs` - v0 aggregation.
- `scripts/yang-mills-phase2-v1-su2-3d-aggregate.mjs` - v1 smeared
  aggregation.
- `scripts/yang-mills-phase2-v2-su2-3d-aggregate.mjs` - v2 correlator
  aggregation.
- `scripts/yang-mills-phase2-v3-su2-3d-aggregate.mjs` - v3 σ²_W33
  aggregation.
- `scripts/lib/yang-mills-u1-2d-core.mjs`,
  `scripts/lib/yang-mills-su2-2d-core.mjs`,
  `scripts/lib/yang-mills-su2-3d-core.mjs`,
  `scripts/lib/yang-mills-su2-3d-smearing.mjs`,
  `scripts/lib/yang-mills-su2-3d-correlator.mjs` - shared modules.

Result directories (under `results/yang-mills/` at repo root, not in
git; include separately if sending a ZIP):

- `results/yang-mills/phase1/U1_2D/...`,
  `results/yang-mills/phase1/SU2_2D/...`,
  `results/yang-mills/phase1/SU2_3D/...` - Phase 1 smoke outputs.
- `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta<β>_ensemble_v0/`
  for β ∈ {2.0, 2.4, 2.8} - the three v0 ensemble dirs reused by v1, v2,
  v3 aggregations.
- `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_relative_locality_v0/`,
  `..._v1/`, `..._v2/`, `..._v3/` - the four aggregation result dirs.

## If The Reviewer Has Only One Comment

Ask them to answer this:

> Is there any standard lattice gauge theory reading of this small-lattice
> small-ensemble setup that we are missing, under which the four NEG-As
> would be either (a) the **expected** outcome no matter what signature
> was used (so the bounded null says little about Sundog's apparatus), or
> (b) the **artifact** outcome of a known finite-volume / β-aliasing /
> N-too-small effect (so the bounded null says little about the substrate)?
> If yes, please name it. If no, the bounded-null framing is the right
> conservative call on this envelope.

## Output We Want From Review

Any of:

- "OK as bounded internal null; keep public claims blocked or cautious."
- "Quarantine the synthesis because X."
- "The four NEG-As are dominated by N=32 noise; re-run with N ≥ Y per β
  before claiming a bounded null about signature classes."
- "The signature/target disjointness is fine, but `γ_held` LS-slope over
  three points has a known noise issue; the bounded null on γ_held is
  about LS-slope noise, not about signature class. The σ²_W33 read
  stands."
- "The σ²_W33 target is too correlated with mean W33 for the v3 probe to
  be the independent test you claim; do W; the v0–v2 reads stand."
- "The probe ladder missed obvious combination C (e.g. Polyakov target,
  HYP smearing, blocked signature) that any lattice gauge theorist would
  have tested before declaring bounded null; flag this in the synthesis."
- "The pre-registration discipline is defensible; the bounded null is in
  family with standard 'small-lattice rank-locality is hard' results;
  publishable as a methodological null."

## Packet Hygiene

Do not send a polished public page first. Send this packet or a PDF
rendering of it. A future `yang-mills.html` generality-gallery card can
be made later only if it carries the same load-bearing bounded-null
statement and a clear "external review pending" or "external review
returned: <verdict>" banner.

The packet does NOT include code review or implementation audit beyond
what is needed to verify the Phase-1-inherited ensemble-health gates
and the per-probe aggregation pipelines. If the reviewer wants
code-level audit (e.g. of `yang-mills-su2-3d-smearing.mjs`), that is a
separate scope; this packet is for the result and its framing.

Public-language boundary inherited from P0 lock §"Public Language
Boundary" remains binding on every output of this review process:

- Allowed: "Sundog is drafting a finite-lattice Yang-Mills certificate
  lane. It asks whether gauge-invariant shadows preserve bounded
  structure beyond controls."
- Forbidden: "Sundog has a Yang-Mills result." "Sundog proves
  confinement." "Sundog found a mass gap." "Sundog is approaching the
  Clay problem directly." "Finite-lattice correlations imply the
  continuum theorem."

A bounded-null receipt is **not** "a Yang-Mills result"; it is a
finite-lattice certificate-program null inside the registered envelope.
The forbidden phrasing remains forbidden even after a positive review.
