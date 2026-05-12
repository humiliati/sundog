It should look less like “Bayes versus Sundog, one wins” and more like:

Bayes turns evidence into belief. Sundog turns response into control.

That is the cleanest head-to-head.

Bayes’ Theorem asks:

P(world∣signal)∝P(signal∣world)P(world)

Sundog asks:

H(x)≈
dτ
dS
	​


or, in plain terms:

When I act, how does the indirect signal change, and can I use that change to steer?

The fair comparison is not “Bayes gets full state, Sundog gets shadows.” That would be a strawman. A fair Bayesian baseline also gets partial observations. The difference is that Bayes tries to reconstruct a posterior over the hidden world, while Sundog tries to couple action directly to the signal field. The Sundog docs already define the shared pattern this way: the system does not observe the whole truth, receives an indirect signal, transforms it into a control-relevant form, and acts from that transformed signal rather than from full world-state symmetry.

The simplest headline:

Bayes: “Where is the hidden thing, probably?”
Sundog: “Which action makes the trace improve?”

For a visual demo, I would make it a split-screen duel.

Left side: Bayes.
A dark board, hidden target, and a posterior heatmap. Every observation updates the heatmap. The UI shows “prior,” “likelihood,” “posterior,” and “expected utility.” It feels analytical, cool, map-like.

Right side: Sundog.
Same hidden target, but no posterior heatmap. Instead, a live signal field, detector bars, scan arcs, and a trace line showing signal improvement. It feels embodied, experimental, servo-like.

The shared task:

A hidden source emits an indirect signal. The agent must align to it, reach it, or keep a system stable around it. Both agents receive the same sensor readings. Neither receives privileged target coordinates.

The fair Sundog version should use the real project shape: detector intensity plus proprioception, no laser position, no mirror hit point, no target Cartesian position, and a SCAN → SEEK → TRACK loop. That is the current measurable form of the older Sundog idea.

The head-to-head should have three rounds.

Round 1: Clean model. Bayes should win.

The hidden source produces a known Gaussian field. Noise is known. Geometry is known. The Bayesian agent has the correct likelihood model.

Bayes updates its posterior efficiently, localizes the hidden target, and chooses the best action. Sundog still has to scan, seek, and track. It can work, but it pays the acquisition-time cost. This matches the current Sundog evidence posture: the photometric controller can match terminal oracle accuracy in the tested mirror task, but it is much slower, with median time-to-threshold around 188 steps versus 11.5 for the target-aware analytic baseline.

Verdict stamp: BAYES WINS — MODEL KNOWN

Overlay copy:

When the likelihood is right, inference is powerful.
Bayes does not need to wander. It can believe its way toward the target.

Round 2: Unknown response surface. Sundog should look strong.

Now keep the target hidden, but corrupt the model: warped optics, shifted detector calibration, unmodeled surface reflection, mild occlusion, or a field whose shape is not the assumed likelihood.

The Bayesian agent updates confidently into the wrong posterior because its likelihood is wrong. Sundog does not need the full generative model. It perturbs, reads the actual response, and climbs what the world is actually giving back.

Verdict stamp: SUNDOG HOLDS — FIELD READABLE

Overlay copy:

The map lied.
The response still answered.

This is the core Sundog pitch without overclaiming. The project’s promo language already frames the idea as indirect environmental response carrying actionable structure, not perfect world reconstruction.

Round 3: Decoy or local maximum. Sundog should fail.

Add a false signal peak. The visible trace is locally attractive but globally wrong. A flat Sundog controller locks onto the decoy because the local signal improves there. A Bayesian agent with the right structural model can represent two hypotheses, gather disambiguating evidence, and eventually reject the decoy.

Verdict stamp: BOUNDARY FOUND — SIGNAL ALIASED

Overlay copy:

A signal can be useful without being truthful.
Sundog fails when the trace improves for the wrong reason.

This is the most important round. It prevents the comparison from becoming marketing. It says: Bayes is better when the system needs explicit uncertainty over hidden causes; Sundog is better when the goal is direct control from a real response field and a full posterior is unnecessary, unavailable, or too expensive.

The landing-page card could read:

Bayes vs. Sundog
Status: Comparative Benchmark
Bayes: Evidence → posterior → action.
Sundog: Probe → response → control.
Question: Must the agent infer the hidden world, or can it act from the trace?
Verdict: Complementary, not redundant.

For the rail, I would avoid a “Sundog beats Bayes” stamp. Use a sequence of stamps instead:

MODEL KNOWN → Bayes wins.
FIELD READABLE → Sundog holds.
SIGNAL ALIASED → Sundog fails.
HYBRID NEEDED → real systems use both.

The deeper theorem-level distinction:

Bayes is an epistemic theorem. It is about how belief should change when evidence arrives.

Sundog is a control hypothesis. It is about when indirect environmental structure can be transformed into useful action without reconstructing the full hidden state.

So the best head-to-head line is:

Bayes is for knowing under uncertainty. Sundog is for acting under partial sight.

A strong final card:

Title: Posterior vs. Halo
Clip: Left side heatmap sharpens; right side signal trace climbs. Then a decoy appears. The heatmap splits; the Sundog trace locks onto the wrong peak.
Stamp: COMPLEMENTARY
Description: Bayes inferred the hidden cause. Sundog followed the live response. Both were right until the signal stopped meaning what it seemed to mean.
Theorem meaning: Sundog does not replace inference. It identifies the cases where action-coupled signal is enough, and the cases where belief over hidden causes is still necessary.

This would be one of the strongest public-facing comparisons because it respects Bayes instead of caricaturing it, while making Sundog’s distinct claim sharper.