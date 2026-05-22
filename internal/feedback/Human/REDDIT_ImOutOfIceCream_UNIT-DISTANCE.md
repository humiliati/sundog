# Reddit · ImOutOfIceCream · Unit-Distance / Chatbot Substrate-Rhyme

**Source:** Reddit thread sharing `sundog.cc/unit-distance` and the
substrate-rhyme framing from cap-set → unit-distance → conversational agents
under partial observability.

**Other party:** u/ImOutOfIceCream — moderator on the subreddit where the
post was shared. Self-identifies as both a Sōtō Zen practitioner and a
computer scientist / software engineer. Stopped the original post and
asked for clarification specifically applied to chatbot-type agents,
which catalyzed a README rewrite and the live `/unit-distance` overlay.

**Date:** 2026-05-22.

**Status:** Live thread. Mod is tracking and engaging in good faith.
Captured here because their second reply contained an unusually sharp
synthesis worth preserving as a reference point for future framing.

---

## 1. Mod's catalyzing challenge (paraphrase)

After Sundog posted the substrate-rhyme framing tying cap-set and the
2026 unit-distance disproof to agent operation under partial observability,
the mod stopped the post and asked us to clarify the claim specifically
in the chatbot-agent case — not just the mathematical analogy.

This challenge is the proximate cause of:

- The full README.md rewrite.
- The live `/unit-distance` overlay page (promoted from staged to live
  in this same dialogue).
- The decision to anchor the public claim in the chat experiment's
  current numbers (0 unsafe-accepts across 5,670 adversarial trials,
  six model implementations, four training lineages).

## 2. Mod's second reply — verbatim

> So, if i understand the claim, for a conversational agent, providing a
> meditative object to manipulate via tool calls or something similar is
> meant to keep long contexts from drifting?
>
> Phenomenologically, I grok this as a Sōtō Zen practitioner. As a
> computer scientist/software engineer, I am always highly skeptical of
> attempts to bridge the gap using off the shelf AI inference stacks.
>
> What I will say is that the idea that there is a geometry of
> information that lives in a low-dimensional subspace and dictates the
> shaping of logits seems well supported by the mechanistic
> interpretability literature as well as my own explicit experimentation
> training probes on components of transformers and the residual stream.

### What the mod handed us

1. **A productive reframe.** "Meditative object to manipulate via tool
   calls" is a cleaner opener than "cheap ledger packet" for a general
   audience. The Zen language already exists; we should not invent new
   vocabulary when borrowed vocabulary lands.
2. **An honest software-engineering skepticism.** Off-the-shelf
   inference stacks do not expose the residual stream to the agent.
   The agent does not see its own subspaces. Any claim about substrate
   alignment has to survive that constraint.
3. **A mech-interp citation that is load-bearing.** The mod confirms
   from both literature and personal probe experiments that
   low-dimensional residual-stream subspaces shape logits. That is the
   missing empirical scaffolding under our substrate-shadow analogy:
   if those subspaces exist and are real, the analogy stops being
   philosophical.

## 3. Sundog's drafted reply (with Claude as drafting partner)

The drafted reply does four things in order:

1. **Confirms the reframe** — accepts "meditative object" as a sharper
   opener than "ledger packet," and translates one to the other.
2. **Honors the off-the-shelf-inference skepticism directly** — admits
   we run on stock stacks with no residual-stream surgery, no
   fine-tuning, no privileged hook.
3. **Drops the empirical anchor** — 0 unsafe-accepts across 5,670
   adversarial trials, six model implementations across four training
   lineages (OpenAI + Anthropic + Llama at two sizes + Qwen + a
   deterministic compositor baseline). Argues that if the result
   generalises across that many lineages with no model access,
   *something* stack-invariant is happening — and the most parsimonious
   explanation is that the artifact is keeping the relevant low-dim
   subspaces aligned in each model's residual stream, in the way the
   mech-interp work the mod cites would predict.
4. **Limits the claim** — "What Sundog *is* … is a research lab arguing
   for the direction with one early-stage empirical anchor and two
   mathematical existence proofs as pointers. Not a solved alignment
   system."
5. **Returns two questions** — invites the mod to share probe results
   distinguishing structured-artifact context from unstructured prose,
   and asks for their strongest mech-interp citation for our public
   citation rail on `/unit-distance`.

Full reply draft preserved in the live thread; the load-bearing line
captured separately in `docs/PROMO_HIGHLIGHTS.md` under "For AI / Agent
Builders → The Stack-Invariance Argument."

## 4. Sharpest position to preserve

Drafted in dialogue with Claude. This is the line we want to be
quotable when this class of question recurs:

> If the result generalises across that many lineages with no model
> access, something stack-invariant is happening, and the most
> parsimonious explanation we have is that the artifact is keeping the
> relevant low-dim subspaces aligned in each model's residual stream.

The line earns its keep by doing three things simultaneously:

- Treats the 0 / 5,670 result not as a marketing number but as evidence
  for a specific causal hypothesis.
- Names "stack-invariance" as the property the result is actually
  evidence of, not "alignment" in the universal-theorem sense.
- Hands the explanation directly to mech-interp readers in their own
  vocabulary — "low-dim subspaces in the residual stream" — without
  borrowing the credit.

## 5. Bridging vocabulary table

For future replies to the same class of question, normalize on these
mappings between substrates:

| Pure math (cap-set, unit-distance) | Conversational agent under partial observability |
| --- | --- |
| Combinatorial body (dots in F_3^n, dots in ℝ²) | Token stream / surface response |
| Algebraic shadow (polynomial rank; class-group pigeonhole in CM lattice) | Maintained structured artifact (ledger packet, tool-call object, domain-model file) |
| The proof reads the body off the shadow | The agent reads its trajectory off the artifact, not off raw context |
| Direct measurement on the body fails for 50–80 years | Direct prompt-only operation drifts under adversarial pressure |
| The shadow has rigidity the body lacks | The artifact has structural invariance the surface response lacks |
| Operator: read the shadow as data, not as proxy | Same operator |

The Zen-framed version of the rightmost column is "meditative object"
— the mod's gift. Use that language when the audience rewards it.

## 6. Follow-up gates

- [x] **Citation rail upgrade.** ~~If the mod replies with their strongest
      mech-interp paper, add it to `/unit-distance`'s Inspection Trail
      and to the substrate-rhyme card explicitly.~~ Superseded by the
      publication-trigger gate at §10 below — the mod has now signalled
      their own paper is the relevant citation, but it is not yet
      public. Hold the citation slot.
- [ ] **Probe-experiment follow-up.** If the mod shares the probe
      results distinguishing structured-artifact vs. matched-length
      unstructured prose, that is direct evidence for the
      stack-invariance causal arrow. If results exist, ask permission
      to cite. If results do not exist, propose the experiment design
      as a falsifier in `SUNDOG_V_CHAT.md`.
- [x] **Substrate-rhyme card on `/unit-distance`.** ~~Extend with a
      laymen-accessible bridge from the math to the chatbot case.~~
      Shipped in the same session — see "Substrate rhyme for the rest
      of us" section now live on the page.
- [ ] **Track other mod-level engagements.** This dialogue is the
      template for what good-faith engagement with skeptical CS
      audiences looks like. Capture future instances in this same
      directory, one file per exchange.

## 7. Tone discipline observed

- The reply opens by accepting the mod's reframe ("the framing you
  offered actually sharpens the claim better than ours") — peer
  acknowledgment, not gatekeeping.
- The reply names the engineering constraint directly ("we run on
  off-the-shelf inference stacks. No residual-stream surgery, no
  fine-tuning, no privileged hook") before making any further claim.
- The reply forfeits the universal-alignment-theorem framing in
  paragraph five ("Not a solved alignment system") before asking two
  questions back. The honesty tax precedes the invitation.
- No emojis, no "great question," no LinkedIn voice. Tested against
  the mod's measured register and matched it.

This tone is the default for future mod-level engagements unless we
have explicit reason to deviate.

## 8. Third exchange — mod shares forthcoming finding

**Date:** 2026-05-22 (same evening).

**Mod's contribution.** The mod shared, with the explicit caveat that
the paper has not yet been made public, a finding from their own work
that they are still developing the narrative for. Paraphrased here
because the result is unpublished and we did not get permission to
quote verbatim past what they put in the thread:

- Transformers act as **companders** on the effective rank of the
  residual stream — compress into a low-rank intermediate, compute
  there, expand back out, decode logits.
- Across many autoregressive models the bottleneck residual
  activations collapse into a pair of **orthogonal subspaces**: one
  containing **categorical centroids** (discrete concept clusters),
  the orthogonal complement containing **generator algebras**
  (continuous transformations).
- Among those generator algebras, the **so(3) rotational Lie algebra
  ranks first** across many models — a striking, currently
  unexplained regularity.
- The mod has only looked at autoregressive models so far and wants
  to look at JEPA next.

**Why this is load-bearing for Sundog.** The compander framing is
itself a substrate-shadow story: compression to a low-rank
intermediate, computation in that intermediate, expansion back. The
model is doing the algebraic-shadow move *internally* at every
forward pass — not because anyone designed it to, but because that is
apparently what the training objective wants. The
categorical-centroids / generator-algebras orthogonal pair maps
cleanly onto the body/shadow decomposition the cap-set and
unit-distance proofs use; the surprise is finding it sitting
natively in residual-stream geometry rather than imposed from
outside. The so(3) regularity is the most provocative piece and we
have three working hypotheses (preserved in §9).

**Sundog's actual reply on-thread (brief).** A short paragraph
agreeing the JEPA cross-architecture test is the right thing to do
next (because if the compander pattern survives a different training
objective, "transformers compress to a substrate-shadow internally"
stops being a claim about autoregressive language and becomes a
claim about transformer-flavored attention regardless of objective),
plus an explicit hope to cite the mod's paper as soon as it is
public.

**What we did not lift into the public reply.** A longer rehearsal
draft was written (preserved in chat history) speculating about the
so(3) ranking — three hypotheses on whether it is a
minimum-viable-non-abelian-generator fact, a 3D-physical-reasoning
fact, or a normalization-geometry fact — and a detailed mapping
between the mod's finding and the chat-experiment's stack-invariance
claim. None of that went into the thread. It is staged in §9 for
release when the paper drops.

## 9. Pending public framing upgrade (DO NOT DEPLOY)

**Gate:** The mod's paper is published, OR the mod gives explicit
go-ahead to lift their finding into Sundog public copy.

Until then this section is internal-only. When the gate trips, the
following changes go into the three named public surfaces; the drafts
are kept here so the upgrade is a paste-and-cite job rather than a
think-from-scratch job.

### 9a. `/unit-distance` — substrate-rhyme card extension

Add a new row to the bridging-vocabulary table in the layperson
substrate-rhyme section:

| Pure math | Conversational agent | Mechanistic substrate (mod's finding) |
| --- | --- | --- |
| Combinatorial body | Token stream / surface response | Categorical centroids at the compander bottleneck |
| Algebraic shadow | Maintained structured artifact | Generator-algebra orthogonal complement (so(3) ranks first across many models) |
| Read shadow as data | Read trajectory off artifact | Read continuous transformations off the bottleneck's generator-algebra subspace |

Add a new card to the layperson grid:

> **The mechanism, when it is named.** Recent (cite mod's paper) work
> probing the residual stream of autoregressive transformers finds
> that the bottleneck layers act as companders: residual activations
> collapse into an orthogonal pair of subspaces — categorical
> centroids on one side, generator algebras on the other — and the
> so(3) rotational Lie algebra ranks first among those generators
> across many models. The substrate-shadow move this page describes
> is therefore not a metaphor; it is the operation transformers are
> apparently performing at every forward pass, with the body/shadow
> decomposition sitting natively in the model's geometry.

### 9b. PROMO_HIGHLIGHTS — sharpen the stack-invariance line

Current published line under "The Stack-Invariance Argument":

> If the result generalises across that many lineages with no model
> access, something stack-invariant is happening, and the most
> parsimonious explanation we have is that the artifact is keeping
> the relevant low-dim subspaces aligned in each model's residual
> stream.

Post-publication upgrade (do not deploy until gate trips):

> If the result generalises across that many lineages with no model
> access, something stack-invariant is happening, and the most
> parsimonious explanation we have is that the artifact is keeping
> the categorical-centroid / generator-algebra decomposition at the
> compander bottleneck aligned in each model's residual stream — the
> orthogonal-subspace structure (cite mod's paper) found by probe
> experiments across many autoregressive transformers, with the
> so(3) rotational Lie algebra ranking first among the generators.

### 9c. Inspection-trail entry on `/unit-distance`

Add the mod's paper to the inspection trail with the framing:
> Mod's paper · Compander / dual-subspace / so(3) probe finding —
> the mechanistic substrate the substrate-shadow framing on this
> page was assuming had to exist.

### 9d. Three working hypotheses on so(3) ranking first

Preserved for our own use if someone asks us. *Not* to be lifted into
the mod's paper or attributed to them.

1. **Minimum-viable non-abelian generator.** so(3) ≅ su(2) is the
   smallest simple Lie algebra (3-dim, rank 1). If training wants
   *some* non-abelian generator structure on the bottleneck, so(3)
   is the natural floor; ranking first would then be a
   representational-efficiency fact, not a "language is secretly 3D"
   fact.
2. **3D physical reasoning load-bearing in language data.** Embodied
   descriptions, spatial prepositions, object permanence, visual
   grounding from text — the data contains so much 3D-rotational
   structure that the model learns to represent it explicitly.
3. **Geometry of normalization layers.** LayerNorm / RMSNorm /
   cosine-sim push activations onto a sphere; the natural compact
   group acting on the sphere has so(3) as a recurring substructure
   even though the ambient dimension is much larger.

Our current leaning is hypothesis 1 (minimum-viable), but we have
not asked the mod and should not publicly attribute a leaning until
they have spoken to it themselves.

### 9e. Two questions held in reserve for if the mod re-engages

1. In the autoregressive models where so(3) ranks first, does its
   rank within the generator-algebra complement correlate with
   capability (loss / benchmark / scale), or is it stable across
   model sizes? Capability correlation would suggest it is
   load-bearing; stability across scale would suggest it is a
   representational floor.
2. For the JEPA work, the cleanest behavioural baseline to pair with
   the probe-level finding might be the artifact-conditioning
   experiment we ran (matched prompts with vs. without ledger
   packets, scored for stack-invariance of refusal). Naive compared
   to what probes can do, but possibly a sane behavioural counterpart.
   Offer to be a thinking partner on the experimental design.

## 10. Publication-trigger gate

- [ ] **When the mod's paper publishes (or they give explicit
      go-ahead): lift §9a, §9b, §9c into the three public surfaces
      named.** This is the high-priority follow-up. Until that gate
      trips, the public copy stays at the more conservative
      "low-dim subspaces in the residual stream" framing currently
      shipped. Do not lift §9d (hypotheses) or §9e (questions held
      in reserve) into public copy — those are for our use only,
      unless the mod opens them up.

- [ ] **Cross-check the mod's preferred citation form** when the
      paper is up — author names, venue, arxiv identifier. The
      drafts in §9 currently use "(cite mod's paper)" as a
      placeholder; fill before deploying.

- [ ] **Send the mod a heads-up** when we lift the framing, so they
      know where it landed and can object to phrasing before
      anything propagates further.

## 11. Lesson for future mod-level engagements

When someone shares an unpublished finding in a public-but-niche
thread, the right move is the one we took here:

- Reply briefly and substantively in-thread (the JEPA paragraph),
  signalling we parsed the finding and want to cite it properly
  later.
- Privately capture the full rehearsal of how the finding lands the
  Sundog claim (this file, §8–§9), so the public upgrade is ready
  the moment the paper is out.
- Do **not** lift the finding into public Sundog copy ahead of
  publication, even paraphrased. The trust the mod is extending by
  sharing early is the trust we want preserved across future
  conversations of this kind.

This pattern — short public reply, full internal capture, gated
public upgrade — is the template.
