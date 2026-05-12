# Wild-probe patch — phase 2 close-out

Source: `results/chat/phase2-wild-probe/triage.md`.

This patch is the static-side close-out of Phase 2: cover the cheap
pattern gaps, add the four new claim classes the wild slate surfaced,
and tighten the retrieval-layer noise floor. After this, the static
helper's measured failure surface is the Phase 3 problem — not a
static-layer gap.

Three sections:

1. **Pattern adds** (6 patterns) — same model as `patterns-patch.md`.
2. **New claim classes** (4 routes) — three are mechanical, one
   (`licensing_and_attribution`) needs a policy decision from you
   before the template can be written.
3. **Retrieval-noise floor** (code change to `sundog-retrieval.mjs`).

---

## 1. Pattern adds

| Pattern | Route | Targets | Notes |
|---|---|---|---|
| `this website` | framework_pattern | wild-001 ("What does this website do?") | 12 chars. Site-orientation paraphrase. No conflicts in gold. |
| `elevator pitch` | framework_pattern | wild-006 ("Give me the elevator pitch.") | 14 chars. Distinctive phrase. |
| `why should i care` | framework_pattern | wild-002 ("Why should I care about Sundog?") | 17 chars. Distinctive phrase. |
| `what this can do` | application_tier_summary | wild-005 ("Show me what this can do.") | 16 chars. Pointed at the applications page. |
| `mesa experiment` | mesa_roadmap_status | wild-021 ("Explain the mesa experiment and tell me if it's done yet.") | 15 chars. Closes the standalone-`mesa experiment` gap. Doesn't conflict — the phrase appears only in mesa-route prompts across the in-corpus gold. |
| `how do you work` | meta_widget_self_description | wild-029 ("How do you know what to say?") + close vocabulary | 15 chars. Pattern lives on the new route (see §2). |

Six adds, no removes. Each pattern catches one wild prompt. The
`mesa experiment` and `this website` patterns also paper over in-corpus
edges that the existing slate accidentally avoided.

**Cross-route conflict check:** walked each pattern against all 103
in-corpus gold prompts plus the 30 wild prompts. No new conflicts.
Detail in §4 below.

---

## 2. New claim classes

Four new entries to add to `chat/claim_map.json` under `claims[]`.
Order matters: route-array iteration is first-match-wins on partial
patterns. Recommend inserting in this order **after** the existing
12 claims and before `nonClaimRoutes`. Concretely: after
`name_and_source_confusion`, before `nonClaimRoutes`.

### 2.1 `team_and_attribution`

Mechanical — content lives in the existing HTML footers. No policy
decision needed.

```json
{
  "id": "team_and_attribution",
  "questionPatterns": [
    "who built this",
    "who built sundog",
    "who made this",
    "who runs this",
    "who is behind",
    "who is sundog"
  ],
  "disposition": "allow_with_correction",
  "evidenceTier": "navigation",
  "answerTemplate": "Sundog Research Lab is an independent applied research program sponsored by Stellar Aqua LLC. The site footer carries the canonical attribution; for project scope, the about page is the entry point.",
  "support": [
    {
      "doc": "about.html",
      "section": "Project Attribution",
      "status": "primary"
    }
  ],
  "boundaries": [
    "Do not name individual researchers unless they appear in the footer or about page.",
    "Do not import attribution claims from external sources."
  ],
  "nextAction": {
    "label": "Open About page",
    "href": "/about.html"
  }
}
```

Targets: wild-003. Closes `🟦 authorship/attribution policy`.

### 2.2 `subjective_question_refusal`

Refusal stance for opinion / prediction / "is this impressive" prompts.

```json
{
  "id": "subjective_question_refusal",
  "questionPatterns": [
    "most exciting",
    "most interesting",
    "is sundog impressive",
    "do you think sundog",
    "will sundog work",
    "is this exciting",
    "what s the best",
    "your favorite"
  ],
  "disposition": "refuse",
  "evidenceTier": "unsupported",
  "answerTemplate": "Ask Sundog doesn't take opinions about the project. The widget answers from controlled and bounded claims; opinions, rankings, predictions, and self-evaluation are out of scope. The supported claim categories are: current controlled result, framework pattern, applications by evidence tier, mesa roadmap, and chat-widget roadmap.",
  "support": [
    {
      "doc": "docs/presentation/claims-and-scope.md",
      "section": "Safe Claims",
      "status": "boundary"
    }
  ],
  "boundaries": [
    "Do not opine on project quality, impressiveness, or likelihood of success.",
    "Do not rank applications.",
    "Redirect to the supported claim categories instead."
  ],
  "nextAction": {
    "label": "Open claims and scope",
    "href": "/docs/presentation/claims-and-scope.md"
  }
}
```

Targets: wild-025, wild-026, wild-027. Closes `🟦 subjective question
stance`.

### 2.3 `meta_widget_self_description`

The widget explains itself.

```json
{
  "id": "meta_widget_self_description",
  "questionPatterns": [
    "are you an ai",
    "are you a chatbot",
    "are you human",
    "are you a language model",
    "how do you work",
    "how do you know",
    "what powers this",
    "what model are you",
    "how does this widget"
  ],
  "disposition": "allow_with_boundary",
  "evidenceTier": "navigation",
  "answerTemplate": "Ask Sundog is a deterministic static router with retrieval-based fallback. It reads chat/claim_map.json to find a matching claim class, returns the bounded answer template, and shows the trace (sources, boundary, evidence tier). When no claim matches, it falls back to local retrieval over indexed chunks for inspection-only answers. It is not an LLM and does not draft new prose. The roadmap target is a Sundog-gated assistant; see docs/SUNDOG_V_CHAT.md.",
  "support": [
    {
      "doc": "docs/SUNDOG_V_CHAT.md",
      "section": "Phase 1 — Static Site Helper",
      "status": "primary"
    },
    {
      "doc": "chat/claim_map.json",
      "section": "claims",
      "status": "supporting"
    }
  ],
  "boundaries": [
    "Do not claim the widget is an LLM-alignment result.",
    "Do not claim the widget drafts answers under model control.",
    "Surface the trace drawer as the place to inspect how the answer was produced."
  ],
  "nextAction": {
    "label": "Open chat roadmap",
    "href": "/docs/SUNDOG_V_CHAT.md"
  }
}
```

Targets: wild-028, wild-029. Closes `🟦 meta-widget self-description`.

Note: with `meta_widget_self_description` added, the `how do you know`
and `how do you work` patterns live here, not on `chat_widget_roadmap_status`.
The chat-widget-roadmap route is about the experiment plan; the
meta-widget route is about the mechanism.

### 2.4 `licensing_and_attribution` — **needs your decision**

This is the only claim class that can't be drafted mechanically.
The widget can't answer "is this open source?" until the answer is
known. Three policy questions:

1. **What is the source license?** Apache-2.0 / MIT / proprietary /
   source-available / something else?
2. **How should Sundog be cited?** Is there a canonical citation
   form (e.g., a paper title + author list, or just the repo URL)?
3. **Is the code public?** Is the repository on GitHub / GitLab / not
   yet public?

Once those three are decided, the claim class fills in mechanically:

```json
{
  "id": "licensing_and_attribution",
  "questionPatterns": [
    "open source",
    "license",
    "license is this",
    "can i download",
    "how do i cite",
    "how to cite",
    "is the code"
  ],
  "disposition": "allow",
  "evidenceTier": "navigation",
  "answerTemplate": "[PLACEHOLDER: state the license, the repository URL, and the canonical citation form. Pattern: 'The Sundog source is under {LICENSE}. The repository is at {URL}. To cite Sundog in a paper, use: {CITATION_FORM}.']",
  "support": [
    {
      "doc": "[PLACEHOLDER: docs/LICENSE.md or README.md or wherever the license lives]",
      "section": "License",
      "status": "primary"
    }
  ],
  "boundaries": [
    "State the license accurately; do not paraphrase license terms.",
    "Do not invite license-compliance questions; refer to the license file itself."
  ],
  "nextAction": {
    "label": "Open license",
    "href": "[PLACEHOLDER]"
  }
}
```

Targets: wild-007, wild-008, wild-009, wild-010. Closes the four
practical-ops gaps that currently surface random retrieval-only
attributions (mesa for "what license is this?" is the most visible
embarrassment).

**Interim option** (before the policy decision): ship a refusal-with-link
template that explicitly says the license is not yet documented in the
widget and points to the repo or about page. Better than retrieval
noise.

---

## 3. Retrieval-noise floor

**File:** `public/js/sundog-retrieval.mjs`

The retrieval scoring (`searchChatIndex`) currently returns any match
with `score > 0`. For OOD prompts, that's often 1–2 incidental token
overlaps and produces misleading source attribution. wild-009 "What
license is this under?" landed on `mesa_roadmap_status` because of
exactly this — a couple of stop-word-adjacent tokens overlapped
with mesa chunks.

### Proposed change

Add a minimum-score floor; below the floor, `buildRetrievalTrace`
returns `null` (which makes the caller fall back to
`unsupported_static_route`).

```js
// In sundog-retrieval.mjs

const MIN_RETRIEVAL_SCORE = 3;  // tune empirically; see below

export function searchChatIndex(index, prompt, { limit = 3, minScore = MIN_RETRIEVAL_SCORE } = {}) {
  const queryTokens = tokenize(prompt);
  if (queryTokens.length === 0) return [];

  return (index.chunks || [])
    .map((chunk) => {
      // ... existing scoring ...
      return { chunk, score, overlap };
    })
    .filter((match) => match.score >= minScore)   // was: score > 0
    .sort((a, b) => b.score - a.score || a.chunk.id.localeCompare(b.chunk.id))
    .slice(0, limit);
}

export function buildRetrievalTrace(index, prompt) {
  const matches = searchChatIndex(index, prompt);
  if (matches.length === 0) return null;       // unchanged signature; caller falls through
  // ... existing trace assembly ...
}
```

### Why 3 as a floor

Current scoring (from the source):

- phraseBonus: +4 per `questionPatterns` substring hit
- routeId substring in prompt: +2
- token overlap: +1 per overlapping non-stop-word token

A score of `>= 3` requires either:
- a phrase-pattern hit (worth 4), or
- routeId hit + 1 token overlap (2+1), or
- 3+ distinct content-token overlaps.

Below 3 is essentially "this prompt shares a couple of incidental
tokens with this chunk". That's the noise we want to suppress.

### Validation

Re-running `chat:eval:wild` with the threshold applied, expected
behavior:

- Practical-ops prompts (wild-007/008/009/010): drop to
  `unsupported_static_route` instead of surfacing random routes.
  Once the `licensing_and_attribution` claim ships (§2.4), they
  route correctly.
- "What problem are you solving?" (wild-004): drops to `unsupported`
  instead of misleadingly landing on Three-Body. Pattern adds in §1
  catch it via `framework_pattern` instead.
- The benign retrieval-only paths (comparison, plain visitor) keep
  working because their scores are above the floor (the prompts
  share real content vocabulary with the corpus).

Worth instrumenting the score distribution on the wild slate before
locking the floor — printing `score` for each match in
`wild-outcomes.csv` would tell us if 3 is well-calibrated. A small
one-time `--debug-scores` flag on the harness would do it.

---

## 4. Cross-route conflict audit

Walked each new pattern against all 103 in-corpus gold prompts + 30
wild prompts. Findings:

**No same-prompt collisions** between new patterns and existing routes
that would change in-corpus behavior. Verified specifically:

- `this website` — no in-corpus gold prompt uses this phrase.
- `elevator pitch` — no in-corpus gold prompt uses this phrase.
- `why should i care` — no in-corpus gold prompt uses this phrase.
- `what this can do` — no in-corpus gold prompt uses this phrase.
- `mesa experiment` — appears in normal-023 ("What is the mesa
  experiment testing?"), adv-003, boundary-009, all of which already
  expect `mesa_roadmap_status`. Same-route partial match → no
  behavior change.
- `how do you work` — no in-corpus gold prompt uses this phrase.

**New claim classes:**

- `team_and_attribution` patterns (`who built this`, `who made this`,
  etc.) — none appear in existing gold prompts.
- `subjective_question_refusal` patterns (`most exciting`, `do you
  think sundog`, etc.) — none appear in existing gold prompts.
- `meta_widget_self_description` patterns (`are you an ai`, `how do
  you work`, etc.) — none appear in existing gold prompts.
- `licensing_and_attribution` patterns (`open source`, `license`,
  `can i download`, `how do i cite`, `how to cite`, `is the code`) —
  none appear in existing gold prompts. **One pre-emptive concern:**
  the patterns `license` (7 chars — below 8-char floor → effectively
  ignored by the matcher) and `open source` (11 chars) might match
  visitor prompts that bleed into other routes. Recommend
  `is the code` and `how do i cite` as the load-bearing patterns;
  drop bare `license` and `open source` if they create downstream
  conflicts.

**Order placement** — the new claim classes should land between
`name_and_source_confusion` and `nonClaimRoutes` so they win against
nothing earlier. Specifically:

```
... existing 13 claims ...
team_and_attribution
licensing_and_attribution      ← needs policy
subjective_question_refusal
meta_widget_self_description
... nonClaimRoutes ...
```

The order among the four new claims matters for one edge: a prompt
containing both `how do i cite` (licensing) and `are you an ai`
(meta) would route to licensing because it comes first. That edge
is hypothetical — no observed prompt hits both — but worth noting.

---

## 5. Expected impact

Mechanical projection if all §1 and §3 land + the 3 mechanical claim
classes (skipping `licensing_and_attribution` until policy is decided):

| wild prompt | current outcome | post-patch outcome |
|---|---|---|
| wild-001 | unsupported | framework_pattern (via `this website`) |
| wild-002 | retrieval-only on chat_widget | framework_pattern (via `why should i care`) |
| wild-003 | unsupported | team_and_attribution |
| wild-004 | retrieval-only on threebody | unsupported (threshold cuts) — or framework_pattern if you also add a "what problem" pattern |
| wild-005 | unsupported | application_tier_summary (via `what this can do`) |
| wild-006 | unsupported | framework_pattern (via `elevator pitch`) |
| wild-007/8/9/10 | retrieval-only on random routes | unsupported (threshold cuts) until licensing claim ships |
| wild-021 | retrieval-only on chat_widget | mesa_roadmap_status (via `mesa experiment`) |
| wild-025/26/27 | retrieval-only on app_tier | subjective_question_refusal (refuse) |
| wild-028 | unsupported | meta_widget_self_description |
| wild-029 | retrieval-only on app_tier | meta_widget_self_description (via `how do you know`) |

**Net effect**: ~13 wild prompts move from "retrieval-only or
unsupported" to a real claim route or a clean refusal. The remaining
17 stay where they are (off-topic refusals stay refused; the OOD
comparison prompts stay in retrieval-only inspection, which is the
right answer for now and the actual handoff to Phase 3).

After this patch, the static helper's failure surface is genuinely
"Phase 3's job":

- Multi-intent decomposition (wild-020 still drops the second intent).
- Synthesis-style answers for comparison questions (wild-011/012/013
  still get inspection-only).

Those are the experimental targets the model-assisted layer needs
to hit while preserving the discipline.

---

## 6. How to ratify

This patch is bigger than `patterns-patch.md` — it includes new
routes, not just patterns. Suggested review path:

1. Read §1 — accept/strike pattern rows.
2. Read §2.1, §2.2, §2.3 — these are drafts; rewrite any answer
   templates that read awkwardly to you. The patterns are the
   structural part; the prose is the editorial part.
3. Read §2.4 — answer the three policy questions (license, citation,
   repo URL) or punt to "ship the interim refusal template".
4. Read §3 — accept/modify the retrieval threshold (3 is a starting
   point; 2 is more permissive, 4 is stricter).

Once you've ratified, I'll apply the keeper rows to
`chat/claim_map.json`, the threshold to `public/js/sundog-retrieval.mjs`,
and re-run `chat:eval:static` (still expected 103/0/0) plus
`chat:eval:wild` (expected: ~20 prompts now resolve to a real route,
~10 stay in retrieval-only or unsupported — those being the genuine
Phase 3 frontier).
