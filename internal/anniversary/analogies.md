# Undisciplined analogies

> Anniversary brainstorm staging — Sundog Year 1. Undisciplined *by intent*:
> these are generative mappings, not claims. Each entry is scored on what it
> **predicts** and where it **breaks**, because an analogy that cannot break is
> decoration. The load-bearing one (the thermometer) is promoted to a formal
> program in [`postulations.md`](postulations.md) ▸ Postulate 1 and staged as
> [`../../docs/COARSE_GRAINING_PROOF_ROADMAP.md`](../../docs/COARSE_GRAINING_PROOF_ROADMAP.md).
>
> **Status (2026-05-16):** staging, internal. Not public-facing. The sundial and
> atlas-as-computer framings are cleared for anniversary copy *only with the
> boundary text attached*; the rest stay internal until a postulation earns
> them. Voice de-chatted from the original brainstorm paste; sharpness retained.

## The through-line

Every analogy below is a special case of one object. A high-dimensional world
state collapses to a low-dimensional **signature** — forward, many-to-one, easy.
The signature inverts back to the world only on a thin set — inverse, narrow,
hard. Sundog's recurring claim is that **control does not need the inverse**: the
signature is enough to *act*, even when it is not enough to *reconstruct*. The
analogies are ranked by how sharply they expose where that stops being true.

### 1. The sundial — *un-flatterable measurement* (anniversary-cleared)

A sundial never measures the sun's position; it reads a shadow. It is
structurally **un-flatterable**: there is no input a user can supply to talk a
gnomon into reading noon, because the shadow is the sun's, not the dial's. The
project is named for an atmospheric shadow, so the founding metaphor and the
Goodhart-immunity claim are the same artifact seen twice.

- **Predicts:** a controller whose only handle on the objective is a trace the
  *environment* writes (not one the *agent* emits) cannot be gamed from the
  inside.
- **Breaks when:** the agent can move the gnomon. The immunity is a property of
  the **channel**, not a virtue of the agent — this is exactly the wound in
  [`attack_vectors.md`](attack_vectors.md) ▸ "Goodhart success story": where we
  applied real selection pressure (Mesa), the channel was movable and it broke
  at λ≈0.953.
- **Anniversary line (cleared, with boundary text):** *humanity has trusted
  shadows over claimed positions since the first gnomon; we spent a year asking
  when the shadow is enough, and where it stops.*

### 2. Bat echolocation — *the percept is the world's transfer function*

The bat never sees the moth. It hears the deformation its own chirp suffers
crossing the world. The percept is `world ∘ action` — the environment's transfer
function applied to the agent's probe.

- **Predicts:** active sensing where the signature is the world's response to
  the agent's own action is intrinsically harder to Goodhart than a passive
  scalar reward, because a forged percept requires forging the world's response.
- **Breaks when:** the agent learns an internal forward model good enough to
  *predict* the echo and then optimize against the model instead of the world.
  That is mesa-optimization restated acoustically, and it is the same threshold
  Postulate 2 makes capacity-relative.

### 3. The thermometer / statistical mechanics — **load-bearing**

We never measure 10²³ molecular velocities; we read `T`. Statistical mechanics
*is* the theory of when a low-dimensional signature is a sufficient macrostate.
Many microstates → one macrostate (forward, many-to-one, trivial); invert a
macrostate → impossible off a thin set. Sundog's recurring object is not
"alignment" — it is **coarse-graining**. The "5D entangled subspace at `net.7`"
is a macrostate; the λ≈0.953 cliff is a phase transition in it.

- **Predicts:** a task is Sundog-solvable iff the optimal action is measurable
  with respect to the σ-algebra the signature generates — a *sufficient
  statistic for control*, not for reconstruction. This is Postulate 1; it is the
  only analogy promoted to a proof path.
- **Breaks when:** the control-relevant bit is not in the signature's σ-algebra
  (it only exists *after* a preparatory action). That is the pre-registered
  pushable-occluder boundary, predicted — not patched.
- This analogy retro-explains the founding poetic `H(x) = ∂S/∂τ`: read as an
  information loss / Jacobian term it is an *entropy*, not decoration. See
  [`postulations.md`](postulations.md) ▸ Postulate 6.

### 4. Double-blind trials — *immunity is a severed channel*

Blinding works by severing the channel through which expectation could game the
outcome. Signature control is blinded from the target by construction.

- **Predicts:** the cliff is an **un-blinding dose**. λ measures channel capacity
  from reward back into the policy interior; below the dose the blind holds,
  above it the agent reconstructs a proxy.
- **Breaks when:** blinding is incomplete by leakage rather than by capacity —
  the oracle-state leakage falsifier in `../../docs/SCIENTIFIC_CRITERIA.md`. A blinded
  trial with an unblinded nurse is not blinded.

### 5. A one-way function — *Goodhart-immunity ≈ preimage resistance*

Forward easy, inverse hard, and the agent only ever pushes forward. An agent
that cannot solve the inverse cannot construct a deceptive input.

- **Predicts:** Goodhart-immunity is formally the agent's *inability to invert*
  the signature map within its capacity class — directly Postulate 2.
- **Breaks when:** the map is only *empirically* hard, not hard against the
  agent's actual function class. "Hard for us to invert by hand" ≠ "hard for a
  Large-tier policy under selection pressure." The mesa cliff is the measured
  capacity at which the one-wayness fails.

### 6. A river and a dam — *the pre-registered boundary, stated as physics*

Water has no map of the sea. It follows the structured gradient and cannot be
bribed uphill. Dam it and it pools — it fails exactly where the useful gradient
appears only *after* a preparatory action.

- **Predicts (verbatim):** the pushable-occluder boundary in
  `../../docs/PHASE2_BLOCKS_DESIGN.md` — Sundog stalls when alignment requires moving an
  occluder *first*. The dam is the occluder.
- **Proof-track update (2026-05-16):** `../../docs/proof/PHASE3_BOUNDARY.md`
  makes the boundary exact: it applies when the flat signature collapses states
  that require different preparatory block actions. If a flat controller
  succeeds, either the decisive bit leaked into the signature or the boundary
  theorem is wrong.
- **Why it matters:** this is the honest one. It does not flatter the theorem; it
  hands a skeptic the failure mode before they find it, and the failure mode is
  *the same object* (a bit outside the current σ-algebra) as Postulate 1's
  boundary. Use it in anniversary copy as the "where it stops" half of the
  sundial line.

### 7. The Seven Bridges of Königsberg — *why 5D is irreducible* (speculative)

Seed from `scratchpad_brainstorm_notes.md`. Euler did not measure the bridges;
he showed a global traversal is impossible from a *topological invariant* (node
parities), not from any local edge property. Map: the "is the λ-cliff
decomposable into independent neurons" question (Mesa Phase 6 v3.2/v3.3: no
single neuron critical, no SAE feature, no top-k recovery) may be a Königsberg
result — non-decomposability is an **invariant of the subspace's connectivity**,
not a failure of the probe.

- **Predicts:** "irreducibly entangled, 5D" is a topological count (relevant
  directions / a Betti-number-like invariant), not an artifact of weak tooling —
  feeds Postulate 4 (5D = count of RG-relevant directions). The prediction has
  teeth: the invariant should be probe-independent across the Medium family.
- **Breaks when:** a future probe *does* linearly decompose the cliff. Then it
  was tooling, the Königsberg framing is dead, and Postulate 4 dies with it.
  Stated so it can be killed.

### 8. The wishing-well coin / a tornado's core — *forward-rich vortices* (speculative)

Seed from `scratchpad_brainstorm_notes.md` ("sundog + Euler … tornados, vortex,
coins in a swirl wishing well"). A coin spiraling down a well, or a vortex core,
is high-D microdynamics whose *observable* (the spiral envelope, the funnel) is a
low-D signature. Euler's fluid equations are forward-rich; recovering the full
field from the envelope is the narrow inverse.

- **Predicts:** if Postulate 1 is real, a Sundog-style controller should hold a
  coin/vortex on a target trajectory from the envelope signature alone, with the
  *same* forward-rich / inverse-narrow asymmetry — a fourth substrate, cheap to
  simulate, and a clean cross-substrate replication test (or falsifier).
- **Breaks when:** the envelope is *not* a sufficient statistic for the control
  goal (chaotic sensitivity puts the decisive bit below observation noise) —
  which is itself the Postulate 1 boundary, measured in a new place. Either
  outcome is publishable; that is the point of staging it.
- **Disposition:** candidate substrate only. Do **not** spin a `bayes_v_sundog`
  track for it — per scratchpad, the Bayesian element is a *baseline inside each
  existing workbench*, not a new document. See
  [`attack_vectors.md`](attack_vectors.md) ▸ "Standing defense: Bayesian floor".

---

**Cross-references**

- Formal program: [`postulations.md`](postulations.md) (Postulate 1 is the
  thermometer made rigorous; 4 is Königsberg; 5/6 the wishing-well and `H(x)`).
- Adversary: [`attack_vectors.md`](attack_vectors.md) — every "breaks when" above
  is someone's attack; the strongest is the sundial's movable gnomon.
- Proof path: [`../../docs/COARSE_GRAINING_PROOF_ROADMAP.md`](../../docs/COARSE_GRAINING_PROOF_ROADMAP.md).
- Scope discipline: `../../docs/SCIENTIFIC_CRITERIA.md`, `../../docs/presentation/claims-and-scope.md`
  — analogy is Layer-A framing language; it never carries an evidentiary claim.
