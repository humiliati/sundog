# Yang-Mills Informative-Null External Review Packet

> Minimal packet for an external sanity check by a lattice gauge theorist.
> This is not a public page and not a Clay-problem claim. It exists to make
> it easy for a reviewer to say "yes, this informative null is correctly
> characterized," "no, this overstates X," or "you missed standard issue Y."

> **What changed since the first draft (2026-05-31 re-point).** The original
> packet asked whether a *four-probe bounded null* (Wilson-loop targets, v0–v3)
> was honestly framed. That null turned out to be **uninformative**: a follow-up
> audit showed its held-out targets were not powered enough for a null to mean
> anything. The lane then ran an audit-first escalation (v4 → v5 → v6a) and
> produced an **informative powered null**: on a genuinely powered and disjoint
> *confinement-order-parameter* target (the finite-temperature Polyakov loop),
> the small-loop signature still showed no rank-locality. **This packet now asks
> a different, sharper question** — is that powered test sound, and is the null
> expected or surprising on gauge-theory grounds?

**Date:** 2026-05-31
**Status:** draft reviewer packet (re-pointed to the informative-null story)
**Primary synthesis:** [`receipts/2026-05-31_SU2_3D_phase2_informative_null_synthesis.md`](receipts/2026-05-31_SU2_3D_phase2_informative_null_synthesis.md)

## Reviewer Snapshot

Repository map:

- Primary working packet: `docs/yang-mills/`.
- Main ledger: [`../SUNDOG_V_YANG_MILLS.md`](../SUNDOG_V_YANG_MILLS.md).
- Lit-pass: [`../YANG_MILLS_LITPASS_MEMO.md`](../YANG_MILLS_LITPASS_MEMO.md).
- P0 domain lock: [`../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md).
- P0 amendment 1 (APE smearing): [`../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md).
- P0 amendment 2 (Polyakov target class + finite-T slate): [`../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md`](../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md).
- Current conclusion: [`receipts/2026-05-31_SU2_3D_phase2_informative_null_synthesis.md`](receipts/2026-05-31_SU2_3D_phase2_informative_null_synthesis.md).

Why we looked at Yang-Mills:

> Pure SU(N) lattice gauge theory was used as a bounded stress test for whether
> Sundog's "lossy gauge-invariant signature preserves a held-out observable
> label in rank space beyond controls" discipline can earn a finite-lattice
> certificate on a real gauge substrate. It did not, on the registered cell —
> and, crucially, the lane kept escalating until the test that failed was a
> **powered, disjoint, order-parameter** test, so the failure is informative
> rather than empty. The review question is whether that powered finite-T test
> is sound and whether the null is expected given standard lattice gauge theory.

Current status:

- Phase 1 instrumentation closed across U(1) 2D / SU(2) 2D / SU(2) 3D
  (gauge-randomization smokes, all `P1-A smoke_pass`).
- Phase 2 ran four pre-registered relative-locality probes (v0 / v1 / v2 / v3)
  on the SU(2) 3D primary cell against Wilson-loop targets; all four landed
  `YM-P2-NEG-A no_rank_local_structure`.
- An audit-first escalation then tested whether any held-out target was even
  **powered** enough to make a null informative: v4 (Wilson re-summaries) and
  v5 (symmetric Polyakov) both returned `YM-P2-UNDERPOWERED`; a finite-T pilot
  voided once on an unbracketed β peak; **v6a** supplied the first powered and
  disjoint target — the finite-temperature Polyakov loop — and the small-loop
  signature **still** returned `YM-P2-NEG-A`.
- An arc-complete informative-null synthesis is filed; the lane is paused at
  this endpoint (no further automatic probes; reopen requires external
  motivation).
- No Yang-Mills, confinement, mass-gap, continuum, or Clay-problem claim is
  live. The v6a null is a **credible negative** about the small-loop signature
  on this cell, not a statement about the physics of confinement.
- The public surface is blocked until an external sanity check confirms or
  corrects the informative-null framing.

Core mapping:

| Sundog concept | Yang-Mills instantiation | Current read |
| --- | --- | --- |
| Lossy gauge-invariant signature | small Wilson-loop summaries: bare mean/var (vocab v1, 8-dim), APE-smeared mean/var with frozen `(α, N_sm) = (0.5, 10)` (vocab v4, 8-dim), bare connected 2-point correlator at 5 cubic-symmetry displacements (vocab v5, 20-dim). **The v6a headline uses the unchanged bare v1 signature.** | three signature classes registered; all gauge-invariant by construction (`CTRL_GAUGE_RAND` to ≤ 1e-12) |
| Held-out observable label | Wilson-loop targets: `γ_held` area-law slope, `σ²_W33` spatial variance (vocab v1/v2). **Confinement order parameter (vocab v4, P0 amendment 2): the Polyakov loop `P(x_⊥;μ)` summaries `{abs_mean_P, mean_abs_P, χ_P}`, on a finite-temperature `12²×4` cell where `⟨\|P\|⟩` is a genuine order parameter.** | Wilson targets later shown un-powered (v4); the finite-T Polyakov target **powered and disjoint** (v6a) |
| Powered-target audit (the new discipline) | before any rank score, each held-out target must clear power (split-half `ICC ≥ 0.50` ∧ tertile agreement `≥ 0.50`, all three β) and disjointness (`CV-R²(target \| v1 signature) ≤ 0.25`); `γ_held` carried as a must-**fail** self-validation | v4/v5 found no powered+disjoint target (correct stop); v6a admitted all three Polyakov candidates (powered ∧ disjoint) |
| Relative-locality certificate | within-β k-NN bin-purity@5 against per-β tertile labels on the admitted target | v6a primary@5 `0.3042` vs `CTRL_RAND 0.3292` (chance 1/3) — no certificate |
| Leakage controls battery (P0 lock) | seven entries: CTRL_META, CTRL_RAW, CTRL_RAND, CTRL_RAND_STRAT, CTRL_PERM, CTRL_GAUGE_RAND (all scored); CTRL_FINITE_SIZE declared but deferred to Phase 4 | all health gates passed across every probe; one bug caught by `CTRL_GAUGE_RAND` (APE staple orientation, see methodology note below) before any score was interpreted |
| Pre-registration discipline | every probe filed a dated spec before its runner ran; smearing parameters and the finite-T β slate frozen at the amendment that admitted them; each escalation step evidence-triggered (v4 only after v0–v3; v6 only after v5 underpowered) | followed verbatim; the v4/v5 underpowered **stops** are the discipline refusing to score against a target that could not carry a meaningful null |

Ways to falsify or weaken this packet:

- Show that the finite-T `12²×4`, `N_t=4` cell at the deconfinement crossover
  is a poor place to pose this test (e.g. the crossover is too rounded at `12²`
  transverse volume for `abs_mean_P` tertiles to be physically meaningful), so
  the v6a null is about the cell, not the signature.
- Show that a small-loop gauge-invariant signature is **expected** on standard
  gauge-theory grounds to carry no rank-locality of the Polyakov order parameter
  (e.g. a center-symmetry / locality argument), so the v6a null is the textbook
  outcome and says little about Sundog's apparatus — and, if so, say whether it
  is still a clean confirmation or simply uninteresting.
- Show that the powered-target audit (split-half ICC over transverse-site
  parity, the `CV-R² ≤ 0.25` disjointness estimator) is not a sound way to
  certify a target as "powered and disjoint," so "powered" is overstated.
- Show that 32 configurations per β is below the threshold for any k-NN
  rank-locality scoring against tertile labels, regardless of target — making
  even the v6a read an "N too small" finding rather than a signature finding.
- Identify a standard finite-temperature lattice artifact at `12²×4`
  (`N_t=4` coarseness, finite-volume crossover rounding, β near the
  susceptibility peak) that would have made any positive result a known false
  signal in the first place.
- Recommend that the informative-null language be narrowed (e.g. from "the
  small-loop signature does not carry the order parameter's rank-locality" to
  "… on this finite-T cell at this N").

## One-Sentence Ask

Please sanity-check whether our **informative-null** conclusion is correctly
framed: on a powered (split-half ICC `0.965`) and disjoint (leakage CV-R²
`−0.332`) finite-temperature Polyakov-loop (confinement order parameter) target
at the SU(2) 2+1D deconfinement crossover (`12²×4`, `N_t=4`, β slate
`{6.3, 6.55, 6.8}` with the susceptibility peak at `6.55` ≈ literature
`β_c = 6.53661(13)`), the unchanged bare small-loop gauge-invariant signature
did **not** earn a relative-locality certificate (within-β bin-purity@5 `0.3042`
vs `CTRL_RAND 0.3292`), and this powered null is the durable result rather than
an artifact of the finite-T cell, the powered-target audit, N, or the β slate.

## What We Are Not Asking

- Not asking for a review of, or position on, the Clay Yang-Mills existence
  and mass-gap problem.
- Not asking whether this is evidence for or against confinement, a mass gap,
  or continuum behavior. (The Polyakov loop is used here only as a powered,
  disjoint held-out *label* for a rank-locality test, not as a physics probe of
  the deconfinement transition.)
- Not asking for endorsement of Sundog, the broader apparatus, or any public
  presentation.
- Not asking the reviewer to debug the entire repository.
- Not asking for a lit-pass adjudication of recent claimed Clay solutions
  (Jacobsen withdrawn, Glimm 2025, Odusanya 2026 — quarantined in
  [`../YANG_MILLS_LITPASS_MEMO.md`](../YANG_MILLS_LITPASS_MEMO.md) and
  not load-bearing on this packet).

## What We Are Asking

Please check the following calls. Questions 1–3 are the new centerpiece (the
powered finite-T Polyakov test); 4–7 carry over the original audit, recontextualized.

1. **Is the powered finite-T Polyakov test sound?** v6a poses the
   relative-locality test on a `12²×4`, `N_t=4` SU(2) ensemble at the
   deconfinement crossover (β slate `{6.3, 6.55, 6.8}`, susceptibility peak
   `6.55`), using per-config Polyakov summaries `{abs_mean_P, mean_abs_P, χ_P}`
   (temporal-wrap loop, averaged over `12²` transverse sites) as the held-out
   target. Is this a reasonable way to obtain a powered, disjoint order-parameter
   label for a rank-locality test? Is the `N_t=4` crossover cell a defensible
   choice, or is there a better-conditioned finite-T geometry?

2. **Is the null expected or surprising on gauge-theory grounds — and is our
   abelian-boundary explanation sound?** We now bring a candidate answer rather
   than asking cold. From the portfolio's clean abelian substrate (Shadow
   Faraday) we argue the null is **expected**: the same Wilson-loop operator that
   closes Faraday induction as a freebie — because `dF = 0` is linear and the
   holonomy of an exact form vanishes — loses exactly that freebie non-abelianly,
   where the Bianchi identity `DF = dF + A∧F = 0` drags the connection into the
   covariant derivative; and the abelian Aharonov-Bohm boundary shows the *local*
   field tier (`F`) is control-blind to global/topological content while only the
   *global* loop (`∮A`) suffices — the exact analog of a *local* small-loop
   signature failing to carry the *global* (temporal-wrap) Polyakov target. This
   is laid out in the [Faraday → Yang-Mills bridge
   note](2026-05-31_faraday_abelian_bridge_note.md), resting on the earned Shadow
   Faraday Phase 7 / Phase 8 receipts. **Is this bridge sound as gauge theory**, or
   does it over-reach — is there a center-symmetry or locality subtlety it misses,
   or a reason a compact non-abelian signature *should* still carry the order
   parameter that we are not seeing? If the null is genuinely the expected outcome,
   is it nonetheless a clean confirmation that the apparatus reports "no structure"
   correctly (distinct from the v0–v3 nulls, whose targets were merely unpowered)?

3. **Is the powered-target audit methodology sound?** Before scoring, each
   target had to clear power (split-half `ICC ≥ 0.50` ∧ tertile agreement
   `≥ 0.50` in all three β; split = transverse-site parity) and disjointness
   (5-fold OLS `CV-R²(target | v1 signature) ≤ 0.25`), with `γ_held` carried as
   a must-fail self-validation (it failed, as intended). Is this a defensible
   way to certify a held-out target as "powered and disjoint," so that the
   resulting null genuinely implicates the signature? Observed for the admitted
   target `abs_mean_P`: mean ICC `0.965`, mean leakage CV-R² `−0.332`.

4. **β-slate / crossover location.** Is the post-pilot slate `{6.3, 6.55, 6.8}`
   (chosen by a one-time pre-generation susceptibility pilot, peak at `6.55`) a
   reasonable bracket of the SU(2) 2+1D `N_t=4` deconfinement crossover, given
   the infinite-volume `β_c = 6.53661(13)` (Edwards–von Smekal 2009,
   [arXiv:0908.4030](https://arxiv.org/abs/0908.4030); β = 4/g²a, identical to
   our Wilson action)? Is the finite-`12²` peak sitting slightly above `β_c`
   the expected finite-volume behavior, or a sign the pilot mis-located it?

5. **Standard finite-T lattice artifacts.** Are there standard finite-volume,
   `N_t=4`-coarseness, or β-near-crossover artifacts on a `12²×4` SU(2)
   Wilson-action ensemble that would trivialize either a positive read OR the
   informative-null read? For reference, the pilot plaquette runs `≈ 0.83–0.85`
   across `{6.3, 6.55, 6.8}`; is that in family for this geometry? (We did not
   pre-register a published-range gate; we report the value and ask whether it
   is in family.)

6. **Probe-ladder completeness.** The lane tested, in order: `{v1 bare,
   v4 smeared, v5 correlator} × γ_held`, `v1 bare × σ²_W33` (v0–v3); then
   powered-target audits over Wilson re-summaries (v4) and symmetric Polyakov
   (v5); then the powered finite-T Polyakov target (v6a). Is the finite-T
   Polyakov loop the natural powered, disjoint order-parameter target to end on,
   or is there a more standard powered held-out observable (e.g. a different
   order parameter, a topological-charge proxy, a blocked/HYP signature) that a
   lattice gauge theorist would have reached for first? Topological-charge and
   4D targets remain deferred by P0; flag if that deferral hides the obvious
   test.

7. **Pre-registration / anti-p-hunting discipline.** Each escalation step was
   evidence-triggered (v4 only after the v0–v3 nulls; the finite-T v6 build only
   after v5 returned underpowered) and the endpoint disposition (PAUSE at the
   first powered null, no further automatic probes) was pre-stated. Is this a
   defensible safeguard against target-shopping — i.e. is "keep escalating until
   the target is powered, then stop whatever the sign" a sound way to avoid
   manufacturing a false positive? Or are there gauge-theory-specific concerns
   (non-uniqueness of "natural" small-loop signatures, β-slate aliasing,
   crossover-location sensitivity) that this discipline papers over and that we
   should flag explicitly in the synthesis?

A short reply is enough. The most useful possible answers are:

```text
Yes, the informative-null framing is basically right.
No, this overstates X (e.g. "powered" is too strong because Z).
The v6a null is the expected/textbook outcome because A; it confirms the
  apparatus but is not surprising — say so in the synthesis.
The finite-T cell is mis-conditioned; re-run at N_t = B / volume C.
The powered-target audit is unsound because D; the null is not yet informative.
The probe ladder should have ended on target E, not the Polyakov loop.
```

## Methodology Note: Discipline Working As Designed

Two events worth a reviewer's eye, both showing the controls/audit catching
problems before any score was interpreted:

**`CTRL_GAUGE_RAND` bug catch (v1 smearing).** During APE-smearing
implementation, an early draft used the heatbath-staple traversal order (matrix
product sequence optimized for Boltzmann conditional sampling of a single link
given its three-link staple sum) instead of the gauge-equivariant plaquette
traversal required for APE smearing to commute with gauge transformations. The
bug surfaced immediately when `CTRL_GAUGE_RAND` re-ran the smearing pipeline on
a Haar-randomized ensemble and found the smeared signature was **not** invariant
— `YM-P1-NEG-A gauge_leakage` fired before any rank-locality score could be
interpreted. The corrected implementation re-ran and produced the clean v1
receipt numerics (det drift 6.66e-16, post-smearing unitarity 9.42e-16, gauge
residual 1.44e-15).

**Underpowered stops (v4, v5).** v4 and v5 returned `YM-P2-UNDERPOWERED` —
**not** `NEG-A` — because no held-out target cleared the power+disjointness
audit. The lane refused to score a rank-locality test against a target that
could not carry a meaningful null, and routed instead to building the
finite-temperature cell where a powered target could exist. This is the
guardrail that turns the eventual v6a `NEG-A` into an *informative* null rather
than a fourth empty one.

We mention both so a reviewer can audit the discipline working as designed
rather than just the green checkmarks.

## Reviewer Time Budget

Suggested review paths:

- **10 minutes:** read the informative-null synthesis receipt's "The upgrade"
  section (the load-bearing statement) and the arc table.
- **25 minutes:** additionally inspect the v6a receipt's Stage-1 powered-target
  audit table (the ICC/leakage numbers that admit the Polyakov target) and the
  Stage-2 score table (primary vs the six controls).
- **60 minutes:** additionally inspect P0 amendment 2 (Polyakov target class +
  finite-T slate freeze), the v6 binding spec's pilot/anti-scope-creep clauses,
  and the v4/v5 underpowered receipts (the audit that establishes the v0–v3
  targets were not powered).

## Core Claim To Audit

From the synthesis:

> On a **powered** (split-half ICC `0.965`) and **disjoint** (leakage CV-R²
> `−0.332`) held-out target — the Polyakov loop, the SU(2) confinement order
> parameter, measured on a finite-temperature `12²×4` ensemble at the
> deconfinement crossover (β slate `{6.3, 6.55, 6.8}`, susceptibility peak
> `6.55` consistent with literature `β_c = 6.53661(13)`) — the unchanged bare
> small-loop gauge-invariant signature `{W11,W12,W13,W22}` does **not** preserve
> within-β rank-local structure beyond controls (within-β bin-purity@5 `0.3042`
> vs `CTRL_RAND 0.3292`; across-β primary `= CTRL_RAND_STRAT` exactly). This is
> an informative null: a powered, disjoint, physically meaningful test, and the
> small-loop signature carried no rank-locality of the order parameter on this
> cell.

This sentence is the thing under review. If it is too strong — if "powered" is
overstated, or the null is the expected textbook outcome, or the finite-T cell
is mis-conditioned — the packet should be revised before any public surface
exists.

## Receipt Arc Summary

The full escalation, every step evidence-triggered:

| Probe | Signature | Target | Powered? | Disjoint? | Primary@5 | RAND margin | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0 | v1 bare 8-dim | `γ_held` (Wilson) | no¹ | n/a | 0.31042 | +0.01042 | `YM-P2-NEG-A` (uninformative) |
| v1 | v4 smeared 8-dim | `γ_held` (Wilson) | no¹ | n/a | 0.29375 | −0.00208 | `YM-P2-NEG-A` (uninformative) |
| v2 | v5 correlator 20-dim | `γ_held` (Wilson) | no¹ | n/a | 0.30833 | +0.02083 | `YM-P2-NEG-A` (uninformative) |
| v3 | v1 bare 8-dim | `σ²_W33` (Wilson) | no¹ | n/a | 0.32917 | +0.02708 | `YM-P2-NEG-A` (uninformative) |
| v4 | — (target audit) | Wilson re-summaries `{W14,W23}` | no (squeeze²) | — | — | — | `YM-P2-UNDERPOWERED` |
| v5 | — (target audit) | symmetric Polyakov | no (`⟨P⟩≈0`) | yes (all 3) | — | — | `YM-P2-UNDERPOWERED` |
| v6 pilot | — | finite-T β-slate selection | — | — | — | — | `Z beta_peak_unbracketed` (void) |
| **v6a** | **v1 bare 8-dim** | **finite-T Polyakov** (`abs_mean_P`) | **YES (ICC 0.965)** | **YES (CV-R² −0.332)** | **0.30417** | **−0.02500** | **`YM-P2-NEG-A` (informative)** |

¹ The v0–v3 targets' lack of power was established *after the fact* by the v4/v5
audits, which showed no Wilson or symmetric-Polyakov target in the registered
envelope cleared the power+disjointness gates. The v0–v3 nulls are therefore
uninformative about the signature.
² Power-vs-disjointness squeeze: the one near-powered Wilson candidate
(`mean_W14`, β2.0 ICC `0.487`) leaked into the signature (β2.8 CV-R² `0.576`);
the disjoint candidates lacked power.

Chance baseline (tertile bins): 1/3 ≈ 0.3333. Promotion gates (every probe
shares these): within-β primary bin-purity@5 `≥ 0.5` AND margin over `CTRL_RAND`
`≥ 0.10` (plus margin `≥ 0.10` over `CTRL_META`/`CTRL_RAW`, across-β `≥ 0.05`
over `CTRL_RAND_STRAT`, `CTRL_PERM` within 0.05 of 1/3, `CTRL_GAUGE_RAND`
`≤ 1e-12`). The v6a read fails the primary gates while the target is powered and
disjoint — that is the informative content.

v6a Stage-2 detail (within-β, k-primary 5):

| lane / control | purity@5 |
| --- | --- |
| PRIMARY | 0.304167 |
| CTRL_RAND | 0.329167 |
| CTRL_META | 0.30625 |
| CTRL_RAW | 0.31875 |
| CTRL_PERM | 0.31309 |
| CTRL_GAUGE_RAND | 0.304167 (`= PRIMARY`, gauge-invariant) |
| across-β PRIMARY | 0.345833 (`= CTRL_RAND_STRAT`) |

Ensemble health (finite-T v6a, Phase-1-inherited gates) passed on all three β:
burn-in `≥ 2000`, τ_int(plaquette) `≤ 16` combined sweeps (observed `≈ 0.5–1.0`),
thinning interval `≥ 2 · τ_int` (observed `≥ 33×`), heatbath fallback fraction
`0`, link unitarity residual `≤ 7.9e-16`. Polyakov target-side gauge residual
max `4.44e-16` (`CTRL_GAUGE_RAND` invariance on the target).

## Files To Read

Primary:

- [`receipts/2026-05-31_SU2_3D_phase2_informative_null_synthesis.md`](receipts/2026-05-31_SU2_3D_phase2_informative_null_synthesis.md)
  - capstone informative-null synthesis; the load-bearing statement under review.
- [`2026-05-31_faraday_abelian_bridge_note.md`](2026-05-31_faraday_abelian_bridge_note.md)
  - the abelian-boundary explanation behind Q2: why the same Wilson operator that
  closes Faraday induction is bounded-null non-abelianly (rests on the earned
  Shadow Faraday Phase 7 / Phase 8 receipts).
- [`receipts/2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md`](receipts/2026-05-31_SU2_3D_phase2_v6a_finite_t_polyakov_neg_a.md)
  - the powered finite-T Polyakov probe: pilot, Stage-1 powered-target audit,
  Stage-2 scores.
- [`../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
  - registered envelope and locked reviewer questions.
- [`../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md`](../prereg/yang-mills/P0_AMENDMENT_2026-05-31_polyakov.md)
  - admits the Polyakov loop as a held-out target class and the finite-T slate.

Secondary (the escalation that makes the null informative):

- [`receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md`](receipts/2026-05-31_SU2_3D_phase2_v4_underpowered.md)
  - Wilson re-summary powered-target audit; establishes the v0–v3 targets are
  not powered (power-vs-disjointness squeeze).
- [`receipts/2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered.md`](receipts/2026-05-31_SU2_3D_phase2_v5_polyakov_underpowered.md)
  - symmetric-Polyakov audit; disjoint but underpowered in the confined cell.
- [`receipts/2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md`](receipts/2026-05-31_SU2_3D_phase2_v6_pilot_unbracketed.md)
  - the first finite-T pilot void (peak on the grid boundary) — the discipline
  refusing to freeze an unbracketed β slate.
- [`../prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6.md`](../prereg/yang-mills/PHASE2_SU2_3D_finite_t_polyakov_v6.md)
  - v6/v6a binding spec (finite-T cell, pilot, gates, anti-scope-creep).
- [`specs/2026-05-31_phase2_v4_powered_target_probe.md`](specs/2026-05-31_phase2_v4_powered_target_probe.md),
  [`specs/2026-05-31_phase2_v5_polyakov_probe.md`](specs/2026-05-31_phase2_v5_polyakov_probe.md)
  - the powered-target-audit probe rationales and fallback locks.

Original four-probe context (now the uninformative leg of the arc):

- [`receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md)
  - the v0–v3 four-probe synthesis (superseded as the lane conclusion, valid for
  its four probes).
- [`../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md`](../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v0.md)
  - v0 spec; defines `γ_held`, per-β tertile bin convention, distance metric,
  controls battery (inherited verbatim by v6a Stage 2).

Lit-pass / prior-art context (not load-bearing on the null):

- [`../YANG_MILLS_LITPASS_MEMO.md`](../YANG_MILLS_LITPASS_MEMO.md)
  - records what the 2026-05-29 literature pass found and what it quarantined.

Runnable / audit support (under `scripts/` at repo root, not under `docs/`):

- `scripts/yang-mills-phase1-gauge-smoke.mjs`,
  `scripts/yang-mills-phase1-su2-gauge-smoke.mjs`,
  `scripts/yang-mills-phase1-su2-3d-gauge-smoke.mjs` - Phase 1 runners.
- `scripts/yang-mills-phase2-su2-3d-ensemble.mjs` - Phase 2 v0 per-β ensemble
  generator (12³).
- `scripts/yang-mills-phase2-v4-su2-3d-aggregate.mjs`,
  `scripts/yang-mills-phase2-v5-su2-3d-aggregate.mjs` - the v4/v5 powered-target
  audits.
- `scripts/yang-mills-phase2-v6-finite-t-polyakov.mjs` - the v6a finite-T pilot
  + generation + audit runner (one locked invocation).
- `scripts/lib/yang-mills-su2-3d-core.mjs` - shared SU(2) 3D core, generalized
  to asymmetric `(Lx,Ly,Lz)` lattices for the finite-T cell (the cubic path is
  asserted bit-for-bit unchanged).
- `scripts/lib/yang-mills-su2-3d-smearing.mjs`,
  `scripts/lib/yang-mills-su2-3d-correlator.mjs` - v4/v5 signature modules.

Result directories (under `results/yang-mills/` at repo root, not in git;
include separately if sending a ZIP):

- `results/yang-mills/phase1/...` - Phase 1 smoke outputs.
- `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta<β>_ensemble_v0/`
  for β ∈ {2.0, 2.4, 2.8} - the three v0 ensemble dirs reused by v1–v5.
- `results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_finite_t_polyakov_v6a/`
  - the powered finite-T Polyakov run (pilot, ensembles, audit, scores).

## If The Reviewer Has Only One Comment

Ask them to answer this:

> We ran a powered (ICC `0.965`), disjoint (CV-R² `−0.332`), finite-temperature
> Polyakov-loop (confinement order parameter) rank-locality test, and the
> small-loop gauge-invariant signature showed no within-β rank-locality beyond
> controls. Two ways this could still be empty: (a) the finite-T `12²×4`,
> `N_t=4` cell or the powered-target audit is mis-conditioned, so "powered" is
> overstated; or (b) it is a **known** gauge-theory fact that small local loops
> cannot resolve the non-local Polyakov order parameter config-to-config, so the
> null is the textbook outcome. If either holds, please name it. If neither, the
> informative-null framing — "a powered, disjoint, order-parameter test that the
> small-loop signature failed" — is the right conservative call on this envelope.

## Output We Want From Review

Any of:

- "OK as informative bounded null; keep public claims blocked or cautious."
- "Quarantine the synthesis because X."
- "The v6a read is dominated by N=32 noise; re-run with N ≥ Y per β before
  claiming a powered null."
- "'Powered' is overstated: the split-half ICC audit does not certify a usable
  tertile label on this finite-T cell because Z."
- "The null is the expected textbook outcome (small local loops cannot carry the
  non-local Polyakov order parameter); it confirms the apparatus reports 'no
  structure' correctly but is not a surprising result — frame it that way."
- "The finite-T cell is mis-conditioned (N_t too small / volume too small /
  β slate off the crossover); re-run at W."
- "The probe ladder should have ended on order parameter / observable E, not the
  Polyakov loop; that is the natural powered disjoint target."
- "The escalation discipline is defensible; the informative null is in family
  with standard 'small-loop signatures don't carry order-parameter locality'
  expectations; publishable as a methodological null."

## Packet Hygiene

Do not send a polished public page first. Send this packet or a PDF rendering of
it. A future `yang-mills.html` generality-gallery card can be made later only if
it carries the same load-bearing informative-null statement and a clear
"external review pending" or "external review returned: <verdict>" banner.

The packet does NOT include code review or implementation audit beyond what is
needed to verify the Phase-1-inherited ensemble-health gates, the powered-target
audit, and the per-probe aggregation pipelines. If the reviewer wants code-level
audit (e.g. of the asymmetric-lattice generalization in
`yang-mills-su2-3d-core.mjs`), that is a separate scope; this packet is for the
result and its framing.

Public-language boundary inherited from P0 lock §"Public Language Boundary"
remains binding on every output of this review process:

- Allowed: "Sundog is drafting a finite-lattice Yang-Mills certificate lane. It
  asks whether gauge-invariant shadows preserve bounded structure beyond
  controls."
- Forbidden: "Sundog has a Yang-Mills result." "Sundog proves confinement."
  "Sundog found a mass gap." "Sundog is approaching the Clay problem directly."
  "Finite-lattice correlations imply the continuum theorem."

An informative-null receipt is **not** "a Yang-Mills result"; it is a
finite-lattice certificate-program null inside the registered envelope — here, a
*credible* null because the held-out target was powered and disjoint. The
forbidden phrasing remains forbidden even after a positive review.
