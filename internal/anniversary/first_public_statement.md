# First Public Statement Since Discovery

Internal draft for the Sundog Year 1 anniversary rollout.

Status: draft, 2026-05-16.
Target publication window: 2026-05-19.
Companion rollout packet: [`anni_spam_roadmap.md`](anni_spam_roadmap.md).

## Short Version

One year ago, Sundog began as a strange alignment problem: could a system act
when the target was hidden, using only the trace the world leaked back?

The first version sounded like a theorem. The honest year-one version is
better: Sundog is a traceability harness for indirect-inference alignment.

We build small, inspectable systems where the decisive state is hidden, the
world still leaks structure, and action can be taken from the trace. Then we
measure where that stops working.

The current claim is not that every shadow is enough. It is that some shadows
are structured, some structures can be controlled from, and the boundary can be
made visible.

## Full Statement

A year ago, Sundog began with a practical refusal:

> the system does not get to see the target.

That refusal came from a real alignment problem. A laser, a fastener head, and
an occluded line of sight forced a field question into mathematical form: could
placement be confirmed through the loss and return of a signal, rather than by
direct inspection of the seated target?

The first language around the discovery was wild. It had theorem energy before
it had theorem discipline. "Shadow becomes signal" was the right intuition, but
not yet a careful claim.

The year since has been the work of making the intuition smaller, sharper, and
more inspectable.

Sundog Research Lab now studies systems that act without full sight. The
recurring pattern is simple:

- the decisive state is hidden;
- the world leaks structure through an indirect signal;
- that signal is transformed into a control-relevant signature;
- the system acts from the signature;
- the operating envelope and failure boundary are measured.

That pattern appears in photometric mirror alignment, halo geometry,
three-body dynamics, Balance, Pressure Mines, Mesa-Trap experiments, and a few
application surfaces. Not all of those carry the same evidence weight. Some are
controlled results. Some are operating-envelope studies. Some are prototypes.
Some are speculative proof paths.

That distinction matters.

The defensible year-one claim is not that Sundog is a completed universal
alignment theory. It is that Sundog has become a traceability harness for
indirect-inference alignment: a place where hidden state, indirect signal,
route fidelity, action, baseline, and failure boundary can be separated and
tested.

This matters because partial observability is not an edge case. Real systems
are occluded, delayed, noisy, expensive, sensor-limited, or deliberately
withheld from privileged state. Conventional software often responds by asking
for more state. Sundog asks a narrower question:

> when direct inspection is unavailable, is the trace enough to act?

Sometimes yes. A shadow can be a sensor. A deformation can be telemetry. A
pressure field can be a decision surface. A local field reading can carry more
usable structure than an impossible full model.

Sometimes no. The trace can be too weak, too ambiguous, too easy to spoof, or
insufficient for the next action. A system can converge for the wrong reason. A
probe can decode a hidden variable without proving the agent used it. A
beautiful analogy can hide an ordinary proxy.

That is why the failure boundary is not an embarrassment. It is the credential.

The next Sundog work is aimed directly at the hardest version of the question:
not merely "did the system land," but "did it act through the claimed route,
and how would we know?" The current answer is causal and structural. A traceable
system should move when the indirect signal is counterfactually edited, and it
should fail where the closed-form inverse loses identifiability. If it sails
through a boundary where the route is supposed to break, then it found another
route. That may still be interesting. It is not the claim.

So the anniversary announcement is deliberately modest:

We are not claiming the final form of a theorem. We are opening the first year
of a research apparatus. The repo, docs, workbenches, stress tests, and
boundaries are public because the idea is only useful if it survives contact
with people trying to break it.

The project began with a shadow. It became a question about useful
incompleteness.

Our claim is not that every shadow is enough.

Our claim is that some shadows are structured, some structures can be
controlled from, and the boundary can be measured.

## Evidence Paragraphs To Swap In

Use one or two, depending on channel.

### Photometric Alignment

The original controlled result is photometric mirror alignment without
target-position access. A controller denied Cartesian target coordinates aligns
from sparse photometric feedback and proprioception in a bounded MuJoCo task.
The cost of indirect feedback is slower acquisition, about 16x in the reported
core run; the reported terminal accuracy claim must be kept narrow and paired
with that cost.

### Geometry

The halo-geometry lane turned a visual atlas into an honesty machine. It
separates rendered primitives from anchored inverse routes, names the strict
eligible set, records the 29 degree tangent/circumscribed merge and the 32.2
degree CZA cutoff, and treats coverage failures as results rather than
inconveniences.

### Mesa

The Mesa-Trap front is the strongest warning against overclaiming. In the
tested shadow-field navigation family, signature-trained policies hold across
some selection-pressure pockets and mixed-signal policies collapse at a mapped
Medium threshold. That is not universal immunity. It is a measured operating
envelope with a mechanistic failure boundary.

The newest Large-tier subset should be framed even more carefully: `lambda=0.99`
recovers by terminal-alignment eval (`mean_terminal_alignment ~= 0.885`), but
the eval summary does not yet compute `old_basin_pref`. So the current reading
is basin-reaching by the eval metric, not verified basin-attractor avoidance.
The strategic claim is coherent-signal protection under tested conditions, not
unique signature immunity.

### Applications

Three-Body, Balance, Pressure Mines, EyesOnly, Dungeon Gleaner, and Money Bags
are surfaces, not proofs. Their job is to expose where the pattern travels,
where it degrades, and which baselines must be added next.

## Boundary Paragraph

Use this in any broad public artifact:

> Sundog is not a completed universal alignment theory. The current evidence is
> narrower: photometric mirror alignment without target-position access, bounded
> operating-envelope studies, and a public falsification roadmap. The broader
> theorem language is a research program, not a finished result.

## Title Options

- A Year of Sundog
- Shadow Becomes Signal, One Year Later
- Useful Incompleteness
- A Year of Trying to Falsify Sundog
- Systems That Act Without Full Sight
- From Theorem Energy to Traceability Harness

## Pull Quotes

- "The failure boundary is not an embarrassment. It is the credential."
- "We started with a shadow and ended up with a question about useful
  incompleteness."
- "The question is not whether the system can know everything. It is whether
  the trace is enough to act."
- "A probe can decode a variable without proving the agent used it."
- "The project began as a claim. It survived the year by becoming an
  instrument."

## Publication Checklist

- [x] No universal-theorem language.
- [x] 16x acquisition cost is visible anywhere terminal-accuracy is mentioned.
- [x] Mesa language says "mapped threshold" or "operating envelope," not
      "immunity."
- [x] Geometry language distinguishes rendered from anchored.
- [x] Applications are called surfaces/workbenches, not proof.
- [x] Every broad artifact includes the boundary paragraph or close paraphrase.
