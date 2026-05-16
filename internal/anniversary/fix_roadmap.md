# Anniversary Fix Roadmap

Internal readiness checklist for the Sundog Year 1 public statement.

Status: draft, 2026-05-16.
Target publication date: 2026-05-19.
Companion packet: [`anni_spam_roadmap.md`](anni_spam_roadmap.md).

## Purpose

This file keeps the unglamorous fixes separate from the public copy. The
rollout is only credible if the site and docs show the same discipline the
statement claims: narrow claims, visible boundaries, and no hidden overreach.

## P0 - Publication-Gating Before First Post

These block the broad public push if unresolved.

### 1. Terminal-accuracy wording

Problem:

The phrase "statistically indistinguishable from the oracle" is too easy to
attack. A non-significant Mann-Whitney result at `n=30` is failure to detect a
difference, not an equivalence result.

Action:

- Replace broad uses of "statistically indistinguishable" with one of:
  - "we did not detect a terminal-accuracy difference at n=30";
  - "terminal accuracy was comparable in the tested setting, with slower
    acquisition";
  - "equivalence remains to be tested with a pre-registered margin."
- Wherever the terminal-accuracy claim appears in public copy, pair it with the
  16x acquisition-cost caveat.

Acceptance:

- Search public pages/docs for `statistically indistinguishable`,
  `indistinguishable`, `oracle`, and `terminal accuracy`.
- No public sentence implies equivalence unless a TOST/equivalence test is
  actually present.

### 2. Oracle-leakage audit

Problem:

The cleanest skeptic attack is implementation leakage: the agent may be denied
target coordinates in its observation vector but still receive target-aware
training, termination, shaping, or success signals.

Action:

- Re-audit the core photometric experiment path for:
  - observation vector contents;
  - reward/success metric access;
  - episode termination conditions;
  - reset and seed handling;
  - logging-only vs controller-visible fields.
- Record the result as a short receipt before linking the paper as the flagship
  year-one claim.

Acceptance:

- A receipt exists stating exactly which target-aware computations are
  controller-visible, training-visible, metric-only, or logging-only.
- If leakage is found, the anniversary statement demotes the core claim until
  repaired.

### 3. Mesa lambda-confound caveat

Problem:

The Mesa cliff at `lambda ~= 0.952588` is powerful and suspicious for the same
reason. If lambda is collinear with effective learning rate, reward-scale
normalization, or gradient-norm ratio, the cliff could be an optimizer artifact.

Action:

- Until the confound test runs, every public Mesa mention says "mapped
  threshold" or "operating-envelope cliff," not "law" or "proof."
- Do not use the cliff as standalone evidence for the gravity frame without
  the bounded caveat.

Acceptance:

- Anniversary copy uses the cliff as a failure-boundary receipt, not as
  universal Goodhart immunity.

### 4. Geometry rendered-vs-anchored surface check

Problem:

The geometry lane is now honest in the docs, but the public surface must not
visually imply that every rendered primitive is an anchored inverse route.

Action:

- Check `sundog.html`, `legend.html`, and related geometry copy for:
  - rendered core vs optional vocabulary;
  - named-only literature coverage;
  - 29 degree tangent/circumscribed merge;
  - 32.2 degree CZA cutoff;
  - parhelic-belt-y caveat if the visual layer still draws it.

Acceptance:

- A reader can tell what is rendered, what is anchored, and what is only named
  or speculative without opening an internal memo.

## P1 - Same-Day Public Coherence

These do not block the first post, but should land before wider technical
outreach.

### 5. Remove internal phase shorthand from public HTML

Problem:

Public cards say "Phase NN" in ways that collide across workstreams. Two
different "Phase 11" references can mean unrelated roadmaps.

Action:

- Search public `.html` files for `Phase`, `phase`, `Step`, and `roadmap`.
- Replace phase shorthand with user-facing descriptions, or hyperlink the
  exact doc section when the phase reference is unavoidable.

Acceptance:

- A non-repo reader does not need to know the internal phase calendar to
  understand a public page.

### 6. Ask Sundog polish

Problem:

The chat widget should act like a claim-boundary guide during rollout, not a
generic hype bot.

Action:

- Add/highlight references from project vocabulary to relevant docs.
- Confirm answer templates refuse:
  - universal theorem;
  - solved alignment;
  - probe-decoding-as-route-use;
  - Atari/crypto/name-collision myths.
- Smoke test with launch-day prompts from the spam packet.

Acceptance:

- Ask Sundog answers "what is Sundog?" with apparatus language and links
  sources.
- Ask Sundog answers adversarial prompts with boundaries instead of escalation.

### 7. Docs index and inspection path

Problem:

The public statement asks for review. Reviewers need a path that does not feel
like spelunking.

Action:

- Keep `docs/index.html` complete for all docs.
- Make sure the anniversary statement points to:
  - About;
  - docs index;
  - scientific criteria;
  - structural-failure prereg;
  - claims and scope.

Acceptance:

- A skeptical reader can find the paper-grade claim, the application tiers, and
  the falsifiers within two clicks from the statement.

## P2 - After The First Wave

These are good follow-through once the initial public statement is out.

### 8. Young-reader / high-legibility path

Problem:

There are approachable demos, but no deliberate "for young readers" layer.

Action:

- Decide between:
  - a `for-kids` or `plain-language` page; or
  - a roadmap that simulates teenage-reader comprehension and patches each
    page.
- Start with one simple explanation: "Sundog is about using clues when you
  cannot see the answer directly."

Acceptance:

- A nontechnical reader can understand the difference between a clue, a target,
  and a failure boundary.

### 9. Bayesian floor across workbenches

Problem:

The project should not spin up a `bayes_v_sundog.md` culture-war doc. The
right move is a Bayesian-optimal or information-theoretic baseline inside each
workbench.

Action:

- For each controlled workbench, add a "Bayesian floor / information baseline"
  row to the roadmap:
  - core photometric;
  - three-body;
  - Balance;
  - Pressure Mines;
  - future vortex/wishing-well toy.

Acceptance:

- Public claims can compare Sundog to a positive information-theoretic baseline,
  not only to oracle/random/naive references.

### 10. Post-launch critique capture

Problem:

The rollout will produce attacks. They should become roadmap entries, not
comment-section fog.

Action:

- Collect serious critiques into [`attack_vectors.md`](attack_vectors.md).
- Classify each as:
  - fatal if true;
  - scope reduction;
  - wording fix;
  - future experiment;
  - misunderstanding already handled by docs.

Acceptance:

- Within 72 hours of launch, the attack vector file has a post-launch addendum
  and at least one next-action owner per serious critique.

## Quick Search List

Run these before final launch copy:

```powershell
rg -n "statistically indistinguishable|indistinguishable|solves alignment|universal theorem|sidesteps Goodhart|cannot be reward-hacked|Phase [0-9]+|phase [0-9]+|oracle" *.html docs internal
```

## Launch Gate

The public statement may ship when:

- [ ] P0.1 terminal-accuracy wording is safe or explicitly caveated.
- [ ] P0.2 oracle-leakage receipt exists or the claim is demoted.
- [ ] P0.3 Mesa cliff language is bounded.
- [ ] P0.4 geometry rendered-vs-anchored distinction is visible.
- [ ] `first_public_statement.md` passes the public guardrails in
      `anni_spam_roadmap.md`.

The broader social rollout may continue while P1/P2 work proceeds, as long as
posts keep linking to the boundary language.
