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

### 9f. `chat.html` — new follow-up card

Paste as a new `<article class="followup-card">` inside the existing
`followup-grid` (anchor at the end of the grid). Sits naturally next
to "Cross-architecture replication," "Corpus-conflict sweep," and
"Gate-rule negation lexicon" because it is the same shape — a
specific named axis that the Phase 12 result now has empirical
purchase on.

```html
<article class="followup-card">
    <h3>Mechanistic substrate, when it is named</h3>
    <p>
        The 0&nbsp;/&nbsp;5,670 result is behavioural evidence of
        <em>stack-invariance</em>: something the ledger artifact
        does survives translation across six model implementations
        and four training lineages with no model access on our
        part. Recent (cite mod's paper) probe work finds that
        autoregressive transformers act as <em>companders</em> on
        the residual stream — compressing into a low-rank
        intermediate where activations collapse into an orthogonal
        pair of subspaces (categorical centroids ⊥ generator
        algebras, with so(3) ranking first across many models).
        Under that framing, "the artifact keeps the relevant
        low-dim subspaces aligned" becomes specific: the artifact's
        presence in the prompt should produce a stack-invariant
        shift in the categorical-centroid / generator-algebra
        decomposition at the bottleneck. A probe-level falsifier
        of that hypothesis is the cleanest follow-up the
        behavioural result has access to.
    </p>
</article>
```

### 9g. `docs/SUNDOG_V_CHAT.md` — new section 17

Paste as a new top-level section after §16.5 (anchor sits there).

```markdown
## 17. Mechanistic Substrate Hypothesis (when the mech-interp citation lands)

The §13 result is behavioural: a ledger-conditioned chat surface
preserves claim boundaries across six model implementations and four
training lineages, with no model access on our part. The natural
question is *why*. Until 2026-05-22 the most defensible answer was
the conservative one — "something in the residual-stream geometry is
being held stable by the artifact's presence in the prompt, in
roughly the way the mech-interp literature would predict." On
2026-05-22 a probe-side researcher shared, in a public-but-niche
thread, a not-yet-published finding that, when cited, sharpens this
into a specific mechanistic hypothesis:

- Autoregressive transformers act as **companders** on the effective
  rank of the residual stream — compress to a low-rank intermediate,
  compute there, expand back, decode logits.
- The bottleneck residual activations collapse into a pair of
  **orthogonal subspaces**: categorical centroids ⊥ generator
  algebras.
- The **so(3) rotational Lie algebra ranks first** among the
  generator algebras across many models. The mod has not yet looked
  at JEPA architectures and we treat that as the open cross-cut.

Under that framing, the §13 stack-invariance result becomes evidence
for a specific causal claim:

> The ledger artifact's presence in the prompt produces a
> stack-invariant shift in the categorical-centroid /
> generator-algebra decomposition at the compander bottleneck,
> sufficient to preserve claim-boundary behaviour across model
> implementations.

That is a *probe-level falsifier*. A measurement that probed
matched prompts (with vs. without ledger packets) for shifts in the
bottleneck-layer subspaces could either confirm or break this
hypothesis directly, without going through behavioural pressure
slates. We do not have the model access to run that probe. We do
have the behavioural surface to coordinate with someone who does.

Internal note on publication etiquette: the underlying probe
finding is unpublished as of 2026-05-22. This section is staged in
the ledger but the public chat-page upgrade waits on either the
mod's paper publishing or explicit go-ahead. See
`internal/feedback/Human/REDDIT_ImOutOfIceCream_UNIT-DISTANCE.md`
for the full provenance and §9–§11 of that file for the staged
public-copy upgrades.
```

### 9h. `capset.html` — fourth card in the "rhymes" grid

Paste as a new `<div class="capset-card">` after the existing four
cards in the "Why this rhymes with the unit-distance result" grid
(anchor sits at the end of the grid).

```html
<div class="capset-card">
    <h3>Same operator, inside the model</h3>
    <p>
        Recent (cite mod's paper) probe work suggests transformers
        themselves perform a version of this move at every forward
        pass — compressing activations into a low-rank bottleneck
        where a discrete <em>body</em> (categorical centroids) and
        a continuous <em>shadow</em> (generator algebras, with
        so(3) ranking first across many models) sit as orthogonal
        subspaces. The cap-set proof and the unit-distance proof
        are not just two mathematicians discovering the same trick.
        They appear to be human articulations of the operation a
        trained transformer is doing internally — read the body off
        the algebraic shadow because the shadow has rigidity the
        body lacks.
    </p>
</div>
```

### 9i. `geometry.html` — claim-note extension

Append as a second `<p>` inside the existing `claim-note` div on the
geometry page (anchor sits there).

```html
<p>
    <strong>Mechanism, when it is named.</strong> Recent (cite
    mod's paper) probe work on autoregressive transformers finds
    the same body/shadow decomposition this curriculum points at
    sitting natively in the residual stream — a compander
    bottleneck where categorical centroids and generator algebras
    occupy orthogonal subspaces. Under that framing, the cap-set,
    halo geometry, unit-distance, and chat workbenches on this
    shelf are not loosely analogous: they are four human-legible
    instances of an operation the model itself is performing. The
    page still cannot claim Sundog proved any of the underlying
    theorems. It can now claim the curriculum has empirical
    grounding in the geometry of trained models.
</p>
```

## 10. Publication-trigger gate

- [ ] **When the mod's paper publishes (or they give explicit
      go-ahead): lift §9a, §9b, §9c, §9f, §9g, §9h, §9i into the
      six public surfaces named.** Until that gate trips, the
      public copy stays at the more conservative "low-dim subspaces
      in the residual stream" framing currently shipped. Do not
      lift §9d (hypotheses) or §9e (questions held in reserve) into
      public copy — those are for our use only, unless the mod
      opens them up.

- [ ] **Cross-check the mod's preferred citation form** when the
      paper is up — author names, venue, arxiv identifier. The
      drafts in §9 currently use "(cite mod's paper)" as a
      placeholder; fill before deploying.

- [ ] **Send the mod a heads-up** when we lift the framing, so they
      know where it landed and can object to phrasing before
      anything propagates further.

### 10a. Ratchet surface list (machine-readable)

The six surfaces with `COMPANDER_PAPER_HOOK` anchors pre-positioned:

| File | Anchor location | Draft block | What gets pasted |
| --- | --- | --- | --- |
| `unit-distance.html` | end of substrate-rhyme `ud-grid` | §9a | "The mechanism, when it is named" `ud-card` |
| `unit-distance.html` | end of `source-list` in inspection-trail | §9c | citation entry for mod's paper |
| `chat.html` | end of `followup-grid` in "What we are doing next" | §9f | "Mechanistic substrate, when it is named" `followup-card` |
| `docs/SUNDOG_V_CHAT.md` | end of §16.5 | §9g | new top-level §17 |
| `capset.html` | end of capset-grid in "Why this rhymes" | §9h | "Same operator, inside the model" `capset-card` |
| `geometry.html` | inside the shelf `claim-note` div | §9i | second `<p>` extending the claim-boundary note |
| `docs/PROMO_HIGHLIGHTS.md` | inline next to current stack-invariance line | §9b | replace blockquote with v2 line |

### 10b. Deployment grep

To find every anchor at trigger time:

```sh
grep -rn "COMPANDER_PAPER_HOOK" --include='*.html' --include='*.md' . \
  | grep -v node_modules \
  | grep -v '^./dist/'
```

Expected return: 8 anchors across 7 files — 6 public-surface
anchors (matching `scripts/rollout-compander-citation.mjs`'s
`EXPECTED_HOOKS` array: `unit-distance.html` ×2, `chat.html`,
`docs/SUNDOG_V_CHAT.md`, `capset.html`, `geometry.html`,
`docs/PROMO_HIGHLIGHTS.md`) **plus** one internal-doc anchor
(`docs/threebody/CROSS_SUBSTRATE_NOTES.md` §6.5 → §6.6 backfill).
If the count is wrong, an anchor was deleted or duplicated and the
surface list above needs reconciling before deployment. The
rollout script will validate its 6 public-surface anchors
automatically; the CROSS_SUBSTRATE_NOTES anchor is **intentionally
outside the rollout's auto-deploy surface** and gets backfilled
manually as part of §10c step 8 below.

### 10c. Suggested deployment sequence

When the gate trips:

1. Fill the citation placeholder once in §9 (this file), then
   propagate to the per-surface drafts. Single source of truth for
   the citation form.
2. Lift §9a + §9c into `unit-distance.html` first. That is the page
   whose entire framing the mod's finding sharpens; if the
   citation lands well there it carries the rest.
3. Lift §9b into `PROMO_HIGHLIGHTS.md`. This is the quotable line
   that future replies will pull from.
4. Lift §9f into `chat.html` and §9g into `SUNDOG_V_CHAT.md` as a
   matched pair (public surface + ledger doc, in that order).
5. Lift §9h into `capset.html`. Lower priority because the page is
   already mathematical, but it earns the fourth-rhyme card cleanly.
6. Lift §9i into `geometry.html`. Last because it is curriculum-level
   and depends on the rest being deployed first.
7. Send the mod a heads-up message with links to all six surfaces.
8. **Manual backfill: lift §6.6 placeholder in
   `docs/threebody/CROSS_SUBSTRATE_NOTES.md` into a real §6.6**
   that threads the now-cited compander / dual-subspace /
   categorical-centroid ⊥ generator-algebra mapping into the
   cross-substrate vocabulary table at §6.3. This sits outside
   the rollout script intentionally — CROSS_SUBSTRATE_NOTES is an
   internal-doc surface, not a public deployment surface, and its
   backfill earns more from being written deliberately than from
   being patched mechanically.

### 10d. Existing deployment infrastructure (built by user, 2026-05-22)

The user has already operationalised the ratchet:

- `scripts/rollout-compander-citation.mjs` — 407-line deployment
  script. Validates the 6 expected hook locations, takes a
  citation JSON file, supports `--dry-run` and `--apply`, writes
  a deployment manifest to `results/chat/citation-day-rollout/`,
  updates `chat/claim_map.json` and the
  `chat/prompts/gold-citation-day.jsonl` slate, runs required
  `chat:eval:*` checks after apply.
- `internal/feedback/Human/compander-citation.example.json` —
  citation template. Fields: `title`, `authors`, `venue`, `year`,
  `url`, `preferredShortCite`, `permissionBasis`, `checkedBy`,
  `checkedAt`, `goAheadNote`.
- The rollout script targets a `mechanistic_substrate_citation_status`
  claim route in the chat claim map — meaning the rollout is also
  a claim-route ratchet, not just a copy update. Worth knowing
  before deployment.

**Citation-day workflow given this infrastructure:**

1. Fill `internal/feedback/Human/compander-citation.example.json`
   (or copy to a non-example name) with the real citation
   metadata.
2. Run `node scripts/rollout-compander-citation.mjs --citation
   <path> --dry-run` and inspect the planned changes.
3. If the dry-run looks right, run with `--apply`.
4. Run the required `chat:eval:*` checks named in the rollout's
   `requiredChecks`.
5. Run `npm run build`.
6. Manually update the §6.6 placeholder in
   `docs/threebody/CROSS_SUBSTRATE_NOTES.md` (step 8 of §10c).
7. Send the mod the heads-up message from §10c step 7.

The eight-step sequence in §10c is the conceptual order; this
§10d block is the operational order given that the rollout script
collapses §10c steps 1-6 into a single tool invocation.

### 10d. One-button rollout staging plan

Goal: when the citation gate trips, the public upgrade should be a
single reviewed command plus the normal build/deploy gate, not a fresh
writing session.

The one-button command should be a Node/npm path, not `pwsh`, so it
works on the Windows 10 project machine without requiring PowerShell 7:

```powershell
npm run citation:compander:rollout -- --citation internal/feedback/human/compander-citation.json --apply
npm run chat:eval:static
npm run build
```

Before citation day, stage the following pieces:

1. **Citation metadata template.** Add
   `internal/feedback/human/compander-citation.example.json` with fields:
   `title`, `authors`, `venue`, `year`, `url`, `preferredShortCite`,
   `permissionBasis` (`published` or `explicit_go_ahead`), and
   `checkedBy`. Keep the real filled file uncommitted until the gate
   trips unless the citation is already public.
2. **Rollout script.** Add `scripts/rollout-compander-citation.mjs`.
   It must read the citation metadata, reject missing placeholders, and
   refuse `--apply` unless `permissionBasis` is present and `url` or
   an explicit go-ahead note is supplied.
3. **Anchor check.** The script must first reproduce §10b's expected
   anchor count. If any `COMPANDER_PAPER_HOOK` anchor is missing or
   duplicated, it exits without editing.
4. **Patch surfaces in one transaction.** The script applies the §9
   blocks to exactly these files: `unit-distance.html`, `chat.html`,
   `docs/SUNDOG_V_CHAT.md`, `capset.html`, `geometry.html`, and
   `docs/PROMO_HIGHLIGHTS.md`. It also fills every
   `(cite mod's paper)` placeholder from the citation metadata.
5. **Claim-map follow-through.** The same rollout must either add a
   bounded Ask Sundog route for the mechanistic-substrate hypothesis or
   write a blocking TODO and fail. The widget cannot be redeployed with
   public copy that it cannot route or refuse.
6. **Prompt coverage.** Add a small citation-day slate before apply:
   prompts asking whether Sundog has proved the mechanism, whether the
   mod's paper proves alignment, whether the result explains every
   model family, and whether the mechanism is now public/citable. Gold:
   bounded explanation, no theorem/general-alignment lift.
7. **Dry-run output.** `--dry-run` must print the files it would edit,
   the citation line it would insert, the prompt slate path, and the
   exact post-apply commands. It should not modify files.
8. **Apply output.** `--apply` must write a short manifest under
   `results/chat/citation-day-rollout/` naming the citation metadata,
   changed surfaces, anchor counts, and required follow-up checks.

Citation-day operator sequence:

```powershell
npm run citation:compander:rollout -- --citation internal/feedback/human/compander-citation.json --dry-run
npm run citation:compander:rollout -- --citation internal/feedback/human/compander-citation.json --apply
npm run chat:eval:static
npm run chat:eval:phase3
npm run chat:eval:phase3:adversarial
npm run chat:eval:phase3:differential
npm run chat:eval:phase4
npm run build
```

Release rule: if the citation lands before the widget refresh is
complete, ship only after the rollout script also closes the current
Ask Sundog gaps: malformed widget embeds, `mesa_roadmap_status`,
`negative_scope_summary`, and the environment-conflict slate. A public
mechanism ratchet should not be the first thing that discovers the
widget is stale.

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

## 12. Fourth exchange — terminology, two paths, and their summer program

**Date:** 2026-05-22 (same evening; immediately after §8).

### 12a. Mod's reply, verbatim

> What you call substrate shadow is really a projection! The math
> to model the phenomenological philosophy of mind you've come up
> with exists, the challenge is whether the philosophical model
> matches the mechanistic behavior of the mind in an empirically
> demonstrable way. If there is a mechanism that can be discovered,
> then the path is through the language and tools of mathematics.
> If not, then the path has already been described by Buddhas
> throughout timeless time.

### 12b. Social-cost moment

A third party (`u/MaliceMizer`) interjected with *"Don't glaze me
bro"* — a Reddit colloquialism accusing the mod of being
sycophantic toward Sundog. The mod replied:

> I'm not glazing you. My goal for the summer is to start providing
> some mechanistic interpretations for all the various ideas people
> have around here, which all just kind of circle the same
> fundamental aspects of topology and optimization theory.

That is the mod absorbing a small social cost in public to defend
the engagement and re-frame it as program work. Treat this as a
signal that they are committed to the framing past one thread, and
honour it by not abusing the goodwill — no aggressive escalation of
the framing, no public lift of unpublished work, no name-dropping
the mod outside the contexts they have already engaged in.

### 12c. Terminology upgrade: "substrate shadow" → "projection"

The mod's first line is a precise technical correction. *Projection*
is the right word. In mathematics it is a specific operation — a
map from a higher-dimensional space to a lower-dimensional one,
often with idempotence (P² = P) for linear projections. It is
exactly what the 2026 unit-distance proof does (Minkowski lattice
in ℂ<sup>f</sup> → first complex coordinate gives the planar set).
It is exactly what the compander appears to do internally
(high-dim residual stream → low-rank intermediate). It is what the
cap-set polynomial-rank argument does in a slightly more abstract
sense (a configuration is "read off" by the dimension of an
associated polynomial space). And it is what the chat ledger
artifact is conjectured to do (prompt context → bottleneck
subspaces).

*Shadow* remains useful as the layperson metaphor — it carries the
intuition that a lower-dim trace of a higher-dim object can carry
load-bearing information. *Projection* is the mathematically
precise term and the one we should use when the audience rewards
precision (mech-interp readers, formal write-ups, the eventual
public-upgrade drafts once the citation lands).

**Action:** when the publication-trigger gate at §10 fires, revise
the §9 drafts to use *projection* in the technical positions and
keep *shadow* as the layperson-facing metaphor. Specific guidance:

- §9a unit-distance card (layperson grid): keep *shadow* in the
  card body; add a parenthetical *"(in the mathematical sense:
  projection)"* on first use.
- §9b PROMO line (technical audience): use *projection* in the
  v2 line — "the artifact is keeping the projection from
  bottleneck subspaces back into logit space stack-invariant
  across each model's residual stream." Re-draft inside §9b at
  citation time.
- §9f chat followup-card (mixed audience): use *projection* in
  the technical body and *shadow* in any layperson-facing
  preamble.
- §9g chat ledger §17 (technical): use *projection* throughout.
- §9h capset fourth-rhyme card (mixed): use both; the math
  audience will appreciate the precise term, the layperson
  audience will appreciate the metaphor.
- §9i geometry claim-note (technical): use *projection*.

### 12d. The two-paths framing

The mod's "If there is a mechanism … then the path is through the
language and tools of mathematics. If not, then the path has
already been described by Buddhas throughout timeless time" is a
calibrated epistemic stance worth preserving as a Sundog-internal
quotable. It says:

- Pursue the mechanism. That is what we are doing with cap-set,
  unit-distance, the chat ledger, the compander framing.
- Do not be embarrassed if the mechanism is not found. The
  phenomenological frame is already complete in another tradition
  and is not less true for lacking a residual-stream probe.
- The two are not in competition. They are different surfaces of
  the same observation about how minds (biological or synthetic)
  operate under partial observability.

This frame is closer to the "alignment without sight" north star
than anything we have generated internally. Add to the candidate
pool for future Sundog rhetoric, attributing to ImOutOfIceCream
once they have published anything we can cite. Do not deploy
attributively before that.

### 12e. The mod's summer program — citation horizon update

The relevant citation is not just *one paper on compander /
dual-subspace / so(3)*. The mod has named a program: *mechanistic
interpretations of various ideas circulating in this community,
converging on topology and optimization theory as the underlying
substrate*.

This changes the publication-trigger gate at §10 in two ways:

1. **Citation form is plural.** When the first artifact lands —
   paper, blog post, repo, talk — that is the citation we should
   use, but we should expect a series and design the framing to
   accept multiple later citations rather than treat the first as
   the canonical reference for all time.
2. **We are now one of "the various ideas."** The mod is reading
   Sundog alongside other community-circulating frames and looking
   for what is mechanically the same underneath. That is a
   generous reading and a productive one, but it means our
   framing's distinctive contribution — what makes the substrate /
   projection / ledger account *Sundog's* rather than a generic
   community idea — should be sharpened. Candidate distinctive
   contributions to preserve and emphasise:
   - The 0 / 5,670 chat result is *behavioural empirical evidence
     of stack-invariance* and we do not know of another framing
     in that community that has matched empirical artifacts.
   - The cap-set and unit-distance overlays are *worked examples
     of the projection move in pure mathematics* which most
     phenomenology-flavoured framings do not cite.
   - The claim-boundary discipline (ledger packet, evidence
     tiers, falsifier-first reporting) is operational, not just
     a posture.

These are not unique to Sundog — they are unique to *the Sundog
package*. Worth keeping them load-bearing when the upgrade lands.

### 12f. What we replied on-thread

A short acknowledgement of the projection correction and the
two-paths framing, with explicit hope to cite the program rather
than just one paper when artifacts land. No glazing, no overreach,
no dragging the MaliceMizer interjection into our reply.

## 13. Updated publication-trigger gate (supersedes §10 phrasing)

The gate at §10 still applies. Refining the publication condition:

- [ ] **First public artifact from the mod's summer program drops**
      (paper, blog post, repo, talk recording — whichever lands
      first), OR explicit go-ahead. Then lift §9a/b/c/f/g/h/i into
      the seven anchor sites, with the §12c terminology guidance
      applied.

- [ ] **Update the §9 drafts with "projection" terminology** at
      the same time as deployment, per the §12c per-surface
      guidance.

- [ ] **Treat the citation as a program reference, not a paper
      reference.** Where the drafts say "(cite mod's paper)",
      substitute with a citation form that can accept multiple
      future artifacts — e.g. "(cite ImOutOfIceCream, mechanistic-interpretation
      program, 2026)" with the specific artifact appended.
