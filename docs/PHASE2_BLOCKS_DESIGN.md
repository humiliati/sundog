# Phase-2 design: pushable blocks shaping the photometric signal

Phase-1 (current rebuild) defends the small claim: an articulated agent
can align a mirrored end-effector to a structured-light source using
only photometric feedback from a fixed detector array, with terminal
accuracy comparable to a target-aware baseline.

Phase-2 extends the perceptual surface. The motivation comes from the
prior leisure-environment artifact (SundogMujoco2.0/) which had pushable
blocks but didn't connect them to any photometric quantity. Here is how
to make blocks load-bearing.

## Goal

Make the agent ACTIVELY shape its photometric field by manipulating
objects in the scene. This is what the original manuscript was reaching
for with "the system listens to itself through its interactions" - except
this version actually requires the agent to interact with something to
get the signal it needs.

## Three options, in order of increasing complexity

### Option A: occluding block on the floor

A static block sits between the mirror and one of the detectors. It
casts a shadow that ATTENUATES the reflected beam if the beam crosses
the block's footprint on the way to the detector.

Implementation:
- Add `block_pos` (xy) to the env state, initialised at random per episode
- In `optics.compute_detector_intensities`, after `floor_hit` returns
  the hit point, check whether the line segment from `mirror_pos` to
  `hit_point` intersects the block's vertical bounding cylinder. If yes,
  multiply that detector's intensity by an attenuation factor (e.g. 0.1).
- Agent gets all 8 detector readings. To peak detector_0, the agent must
  find a mirror orientation where (a) reflected beam lands near
  detector_0, AND (b) the block doesn't occlude the path.

Effect on the claim: the photometric agent now has a richer constraint
to satisfy. Could expose a failure mode where the agent locks onto a
local maximum that doesn't account for block geometry.

Cost: ~50 lines of optics + a block param in the env. Same agents work
unchanged.

### Option B: pushable block as auxiliary action

The agent has a third degree of freedom: a "push direction" that
displaces the block by a small step each tick. Block dynamics are
elementary (overdamped first-order). Optimal strategy: push the block
out of the occlusion path, then align the mirror.

Implementation:
- Extend action vector to (theta_x, theta_y, push_dx, push_dy)
- Block xy updates as `block_xy += alpha * (push_dx, push_dy)` clipped
  to a workspace bound
- Detector intensity attenuation as in Option A

Effect on the claim: now the photometric agent has to learn a
two-stage strategy. The 4D extremum-seeking agent might struggle;
might need a hierarchical controller. This is where the small claim
gets stress-tested.

Cost: ~150 lines. Agents likely need a re-tune on probe frequencies
(adding two more dimensions to the parameter space).

### Option C: block as secondary mirror / reflector

A reflective block whose orientation the agent can steer redirects
the beam BEFORE it hits the primary mirror, or AFTER. The detector
array sees a multi-bounce signal. Block-orientation is a third joint
the agent controls.

This is closest to the manuscript's "harmonic field" framing because
it gives the agent a multi-element optical system to configure.

Implementation: significant. Multi-bounce ray tracing, multi-element
optical scoring. Easily 300-500 lines.

Cost: high. Probably belongs in a follow-up paper rather than Phase-2.

## Recommendation

Option A is the right Phase-2 step. It gives the photometric agent a
richer, more constrained landscape without ballooning the action
space or the optics. The headline claim moves from "alignment under
unconstrained reflection" to "alignment under occlusion" - a measurable
robustness story.

If A works, B is the natural next step. C is a separate paper.

## What 2.0 did with blocks vs. what we should do

2.0 had blocks that the agent could push, but the env's "resonance"
metric only cared about block xy clustering. Blocks were the WHOLE
task, not part of an alignment task. The skeptic's verdict from this
session's audit: the do-nothing agent won 2.0's comparison because
the env was effectively "leave the blocks where they started."

Our Phase-2 keeps alignment as the central task and uses blocks as a
constraint-shaper. Doing nothing yields zero target intensity (bad).
Aligning without accounting for the block yields partial intensity
(better). Aligning around the block yields full intensity (best).
The reward landscape is non-trivial by construction.
