# Scientific Criteria

This document records how the Sundog repo has been reframed for scientific
inspection.

## Research Object

The current research object is not the broad theorem in full generality. It is
a controlled instance of the theorem:

> Indirect photometric feedback can be sufficient for alignment in a bounded
> mirror-pointing task, even without target-position access.

That object is small enough to test.

The Sundog program also publishes a named class of artifact alongside the
controlled experiment: the **structural-zero receipt** and its siblings
(named quarantine, closed-form separability, bounded operating envelope).
These artifacts come out of pre-registered audit chains — constants and
outcome categories registered in code and spec before any row is interpreted
— so failure modes are visible before the result is. The audit-chain
discipline is itself part of the apparatus that the Scientific Criteria
below should be read against.

## Scientific Criteria Met

**Operationalized task.** Alignment is measured as target detector intensity,
not as an aesthetic or post-hoc judgment.

**Explicit observations.** The photometric agent receives detector readings and
joint state only. It is denied target Cartesian position and oracle geometry.

**Baselines.** The experiment compares against:

- `doa_direct`: target-aware analytic oracle;
- `doa_noisy`: analytic oracle with 5 cm static position noise;
- `random`: lower-bound random action policy.

**Matched seeds.** Every condition sees the same laser placement and initial
joint perturbation per seed.

**Quantitative metrics.**

- terminal target intensity;
- time-to-threshold;
- terminal joint stability;
- bootstrap confidence intervals;
- Mann-Whitney U comparisons on terminal intensity.

**Stress tests.** The repo records sweeps over detector noise, beam width, scan
duration, laser height, and joint limits.

**Negative result included.** The joint-limit sweep documents a failure mode
instead of hiding it.

**Named-artifact classes for theorem-adjacent work.** Several audit-chain
artifacts are now in the public ledger and each names its falsification
surface before result interpretation:

- **Structural-failure boundary map** (`/structural-failure`) — pre-registered
  five-locus falsifier for closed-form traceability; P0/P1 passed, Cut 2
  separability held, Cut 3 still open.
- **Mesa-trap operating envelope** (`/mesa`) — 22-policy in-vitro cliff at
  λ ≈ 0.953 with a 5D mechanistic locus at `net.7` and a Large-tier
  envelope extension.
- **K_facet v0.3h verdict** (`/isotrophy`) — 20 of 21 strict G.2
  single-curve choreographies returned structural-zero receipts at
  m₃ = 1; the 21st (O_617) is held back as a named quarantine for a
  bridge direction outside the valid D₃ representation. Audit chain
  intact; theorem-facing result is not closed.
- **Shadow Faraday Branch A receipt** (`/faraday`) - a pre-registered
  classical-vacuum test where local, gauge-invariant shadow data is sufficient
  to close the Faraday residual in the registered domain. Phase 3 lands the
  algebraic Branch A structural zero and Phase 4 support checks pass 5/5.
  Source-bearing, topological, plasma, QED, and curved-spacetime extensions
  remain outside the closed claim and are routed to the Phase 7 boundary audit.
- **Shadow Method safety-template sketch** (`/safety-method`) - a method essay
  that translates the Faraday receipt into three AI-safety questions: local
  readout, structural zero, and named quarantine. It is public and useful, but
  it is explicitly a sketch/bridge, not a proof of AI safety or a new Faraday
  derivation.

The structural-zero receipt is a named, defensible artifact class — not a
tolerance gate. A row earns one when `c_i = d_i = 0` by construction.

## Criteria Partially Met

**Reproducibility.** Scripts and saved outputs are present. A fully polished
research release should still add dependency pinning, one-command setup, and a
fresh-run checksum or expected-output note.

**Reviewer-ready paper shape.** `docs/PAPER_v1_draft.md` is close to a compact
paper draft, but needs final citation cleanup, author metadata, and figure path
normalization before submission.

**Ablation depth.** Scan duration is tested. Probe frequencies, gradient gain,
scan amplitude, detector count, and target-detector position remain open.

## Criteria Not Yet Met

**Hardware validation.** No physical optics rig is included.

**Theorem generality.** The present result supports one constrained alignment
task. It does not prove the theorem across arbitrary agent classes.

**Independent replication.** The repo includes run artifacts, but no outside
replication result.

**High-DoF scaling.** The scan phase will not scale naively to many control
dimensions.

**Product evidence as science.** EyesOnly, Dungeon Gleaner, and Money Bags are
application evidence. They are not controlled experiments unless each is given
its own task definition, baseline, metric, and reproducible study.

## Falsifiable Expectations

The current hypothesis would be weakened if:

- the photometric controller fails to match oracle terminal accuracy under
  fresh matched seeds at the same operating point;
- a stronger target-unaware baseline reaches equal accuracy with less
  acquisition time;
- small observation noise destroys convergence earlier than the recorded sweep
  suggests;
- the result depends on accidental implementation leakage from oracle state;
- the scan/seek/track loop fails under modest geometry variation not covered
  by the current stress tests.

It would be strengthened by:

- independent reproduction of the reported statistics;
- a hardware mirror-alignment demonstration;
- stronger non-oracle and oracle baselines;
- detector-count, scan-amplitude, and probe-frequency ablations;
- an occlusion result building on `docs/PHASE2_BLOCKS_DESIGN.md`;
- a second controlled task outside geometric optics.

## Paper Posture

The paper should use precise language:

- Say "photometric mirror alignment without target-position access."
- Say "no detected terminal-intensity difference at n=30 in this MuJoCo task."
- Say "the cost is slower acquisition, roughly 16x in the reported core run."
- Say "the theorem motivates the controller and application program."

Avoid saying:

- "proof of general AI alignment";
- "universal compression of physics";
- "oracle-beating controller";
- "scientifically validated across all products";
- "K_facet v0.3h closes the isotropy theorem" (the audit chain is intact;
  the theorem-facing result is not closed);
- "21/21 receipts" (it is 20/21 plus one named quarantine);
- "O_617 was a weak-admission failure" (it is a clean opposite-strict row;
  the defect lives in the bridge representation outside the valid D₃
  representation);
- "Faraday Branch A proves AI safety" (it is a closed classical-vacuum
  electromagnetic receipt);
- "`/safety-method` is a proof" (it is a marked sketch that imports the
  Faraday receipt as a worked example).

The cleanest academic version is modest and therefore harder to dismiss.
