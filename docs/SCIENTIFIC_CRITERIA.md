# Scientific Criteria

This document records how the Sundog repo has been reframed for scientific
inspection.

## Research Object

The current research object is not the broad theorem in full generality. It is
a controlled instance of the theorem:

> Indirect photometric feedback can be sufficient for alignment in a bounded
> mirror-pointing task, even without target-position access.

That object is small enough to test.

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
- "scientifically validated across all products."

The cleanest academic version is modest and therefore harder to dismiss.
