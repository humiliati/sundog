# External Review Email Draft (Yang-Mills Bounded Null)

> Send to a lattice gauge theorist. Owner fills `[Name]`, `[reviewer
> specialty signal]`, `[link or attachment to packet]`, and `[Your name]`
> before sending. Three versions below; pick the one that matches your
> relationship with the reviewer.

## Short Version

**Subject:** Quick sanity check request: bounded-null finite-lattice SU(2)
relative-locality test

Hi [Name],

Could I ask for a small sanity check on a finite-lattice Yang-Mills note?
This is not a Clay-problem claim and not a confinement / mass-gap claim.
The point is the opposite: we ran a Sundog-style "lossy gauge-invariant
signature must preserve a held-out observable label beyond controls"
apparatus on pure SU(2) lattice gauge theory and now think the correct
conclusion is a bounded cell-local null.

The claim I want checked is:

> Inside the registered SU(2) 3D, 12³, Wilson-action ensemble at β slate
> {2.0, 2.4, 2.8} with 32 configurations per β (Creutz + Kennedy-Pendleton
> heatbath + Brown-Woch overrelaxation, 1 HB + 4 OR per combined sweep,
> τ_int(plaquette) ≈ 0.5–1.0, thinning interval 32 sweeps), the v1 small-
> Wilson-loop signature class (mean and variance of W11/W12/W13/W22 across
> all positions and three plane orientations) does not preserve within-β
> rank-local structure for either of two held-out targets — area-law decay
> slope γ_held on (W14, W23, W33) or per-config spatial variance σ²_W33 —
> with k-NN bin-purity at k=5 against per-β tertile labels. Three signature
> variants were registered before scoring: bare loop moments, APE-smeared
> loop moments at frozen (α, N_sm)=(0.5, 10), and connected 2-point
> correlator at a frozen displacement slate. All four (signature × target)
> probes landed within ±0.04 of chance baseline 1/3 and failed both
> registered promotion gates (absolute purity ≥ 0.5, margin over random
> ≥ 0.10).

The most useful review would be a short "yes, that framing is conservative"
or "no, this overstates X." In particular, I would love your view on three
questions:

1. **Signature/target disjointness.** Is the v1 signature (small Wilson
   loops W11/W12/W13/W22 mean + variance) disjoint enough from the
   held-out target (larger loops W14/W23/W33 entering either an LS-slope
   γ_held or a per-config spatial variance σ²_W33) that any positive
   would have been the natural read rather than trivial leakage?
2. **N and ensemble health.** Is 32 configurations per β at 12³ large
   enough for k-NN tertile-bin rank-locality scoring to mean anything?
   Are the locked τ_int / thinning thresholds (τ_int ≤ 16 combined sweeps,
   thinning ≥ 2 · τ_int) defensible at the registered β values?
3. **Probe-ladder completeness.** Is there an obvious small-loop
   signature class or held-out summary that a lattice gauge theorist
   would expect to test before declaring a bounded null on this cell?
   The four probes ran (signature × target) combinations bare/smeared/
   correlator × γ_held plus bare × σ²_W33. Pre-stated but unrun: σ²_W14,
   σ²_W23, Polyakov-loop target, smeared × σ²_W33.

Packet: [link or attachment to docs/yang-mills/EXTERNAL_REVIEW_PACKET.md]

No endorsement requested. A one-paragraph reply is genuinely enough; the
most useful possible answer might be a one-liner like "this is dominated
by N=32 noise; re-run with N ≥ Y before claiming anything," or "the σ²_W33
target is too correlated with mean W33, do X."

Thanks,
[Your name]

## Slightly Warmer Version

**Subject:** Small finite-lattice SU(2) sanity check, mostly to prevent
overclaiming

Hi [Name],

I have a small, bounded sanity-check request if you have the bandwidth.
We've been testing a Sundog "lossy gauge-invariant shadow" apparatus on
pure SU(2) lattice gauge theory. The result is not a positive claim. It
is a bounded cell-local null: after four pre-registered relative-locality
probes on the same finite-lattice SU(2) 3D × β slate ensemble, the
conservative conclusion appears to be that the small-loop signature
classes we tested do not earn a within-β rank-locality certificate on
the held-out targets we tested.

The four probes (registered in order, each as a separate pre-run spec)
were:

- v0: bare-loop mean+variance signature vs γ_held LS-slope target →
  bin-purity@5 = 0.310 (chance 0.333), margin over CTRL_RAND = +0.010,
  fails both promotion gates;
- v1: APE-smeared mean+variance at frozen (α, N_sm) = (0.5, 10) vs same
  γ_held target → bin-purity 0.294, **slightly worse** than the bare
  baseline (so UV-noise dominance was falsified as the v0 explanation);
- v2: bare connected 2-point correlator at a frozen displacement slate
  vs same γ_held target → bin-purity 0.308, also fails;
- v3: bare-loop signature (re-read from v0 with SHA-256 assertion)
  against a new held-out target σ²_W33 = per-config spatial variance of
  the largest held-out loop → bin-purity 0.329, also fails.

What I want to know is whether that "four signature × target
combinations, four named nulls on the registered envelope, no
relative-locality certificate" framing is mathematically and statistically
honest as a lattice-gauge result, given finite-lattice / finite-ensemble
standard practice. If it is too strong, or if one of the four is better
called an N-too-small artifact or a target-leakage artifact, I want to
quarantine it before any public surface exists.

The review packet is here: [link or attachment to docs/yang-mills/EXTERNAL_REVIEW_PACKET.md]

The packet is written so a 10-minute skim of the synthesis section and
the four-receipt matrix should be enough to answer the main question.
Methodology context (gauge-invariance smoke tests, ensemble health
gates, pre-registration discipline) is one click deeper. No endorsement
requested; even "this is standard, cite X, and don't call it Y" would be
extremely helpful.

Specifically, I would value your view on three things if you have time:

1. Is the v1 signature (small-loop mean + variance) disjoint enough from
   the held-out target (larger-loop area-law slope, or large-loop spatial
   variance) that this is a genuine independence test rather than
   trivial signature-into-target leakage?
2. Is 32 configurations per β at 12³ enough for any k-NN tertile-bin
   rank-locality scoring to be meaningful, given the τ_int / thinning
   rule we registered?
3. The pre-stated unrun probes (σ²_W14, σ²_W23, Polyakov-loop target,
   smeared × σ²_W33) — should any of those be flagged as "you should
   have done this before declaring bounded null" rather than "optional
   with external scientific motivation"?

Thanks,
[Your name]

## Follow-Up If They Say Yes

Thank you. The most useful path is:

1. Read the load-bearing bounded-null statement, the four-receipt matrix,
   and the PAUSE Disposition in
   `receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`.
2. Check the seven questions in `EXTERNAL_REVIEW_PACKET.md` — the first
   five are direct adaptations of the questions the P0 lock pre-registered
   as the review surface; questions 6 and 7 are bounded-null-specific.
3. Reply with any of:
   - "framing is conservative";
   - "quarantine the synthesis because X";
   - "this is dominated by N=32 noise; re-run with N ≥ Y";
   - "the v3 σ²_W33 target is too correlated with mean W33; do X";
   - "the probe ladder missed obvious combination C; that should be in
     the ladder before declaring bounded null";
   - "cite this standard reference; the bounded null is in family with X."

I am not asking for a full referee report.

## Follow-Up If They Decline

No worries at all, and thank you for considering it. If there is someone
who would be a better fit for a quick lattice gauge theory sanity check —
ideally someone with [reviewer specialty signal, e.g. "experience with
small-lattice SU(2) Wilson-loop / area-law work" or "with rank-locality
or signature-based ML methods on lattice ensembles"] — I would be
grateful for a pointer.

## Owner Fill-In Checklist Before Sending

Before sending either the short or the warmer version, verify:

- [ ] `[Name]` filled with the reviewer's name in salutation and the
      reviewer-specialty signal in the "Decline" follow-up.
- [ ] `[link or attachment to docs/yang-mills/EXTERNAL_REVIEW_PACKET.md]`
      points to a stable URL or has the packet attached (PDF rendering is
      acceptable; the packet was designed for that).
- [ ] `[Your name]` filled in the sign-off.
- [ ] Subject line matches one of the two registered options above (do
      not freelance the subject line — the packet hygiene rule in the
      packet forbids polished public-page-style language).
- [ ] No file at this path includes the forbidden phrases per the public-
      language boundary: "Sundog has a Yang-Mills result," "Sundog proves
      confinement," "Sundog found a mass gap," "approaches the Clay
      problem," "implies the continuum theorem."
- [ ] Optional: include a ZIP of the four Phase 2 result directories
      under `results/yang-mills/phase2/SU2_3D/` if the reviewer asks for
      raw artifacts.
