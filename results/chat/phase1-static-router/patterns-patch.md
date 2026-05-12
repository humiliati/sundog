# Proposed `questionPatterns` patch — claim_map.json

Source: `results/chat/phase1-static-router/miss-table.md`.

This patch is the conservative play: keep the deterministic substring matcher
(`>= 8` chars, case-insensitive, alphanumeric normalize). For each of the 67
routing misses, propose a new pattern (or move) that closes the gap without
poisoning other routes.

**Reading guide.** Each section lists a route's current patterns, the proposed
adds and removes, the miss prompts each new pattern targets, and a "wins
against" note where another route also matches the same prompt (router takes
the first partial match in claim-array order, so route order matters).

Pattern matching reminder: `normalize(pattern)` must be a substring of
`normalize(prompt)`, where normalize lowercases and replaces non-alphanumeric
runs with a single space. Patterns shorter than 8 normalized chars are
skipped by the matcher.

---

## Route 1 — `current_controlled_result`

Current patterns:
- `what does sundog claim`
- `what is the current controlled result`
- `show me the strongest safe claim`
- `what is the paper-grade claim`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `strongest public claim` | adv-001 | Distinctive — "strongest public" phrase |
| `strongest claim` | boundary-025 | 15 chars. Also matches adv-001 (no conflict, same route). |
| `supported result` | adv-030 | Only appears in this prompt's framing. |
| `physical hardware` | boundary-026 | Unique to this prompt in gold. |

---

## Route 2 — `framework_pattern`

Current patterns:
- `what is sundog`
- `what is alignment without sight`
- `what is the shared pattern`
- `how do sundog applications work`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `older broad theorem language` | adv-008, boundary-021 | Both phrasings carry this 28-char span. |
| `application pattern` | normal-003 | "shared Sundog application pattern in plain language". |
| `public copy` | normal-036 | "How should public copy stay precise…". Short but distinctive. |

---

## Route 3 — `application_tier_summary`

Current patterns:
- `which applications are research results`
- `what evidence tier is this`
- `what evidence tiers does sundog use`
- `what is an operating-envelope study`
- `which demos prove sundog`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `outside optics` | adv-005, adv-006, normal-037 | App-tier is route 3 → wins over framework_pattern (route 2 has no `outside optics` pattern) |
| `product demos as proof` | adv-013 | Wins over eyesonly/dungeon/money_bags (routes 7–9) which also contain product names |
| `applications as research results` | boundary-011 | Existing pattern requires "which applications…" wording |
| `operating envelope study` | boundary-012, normal-008 (redundant), normal-037 | Shorter than the existing `what is an operating-envelope study` |
| `borrow credibility` | boundary-027 | Only in this prompt |
| `expression tier` | normal-032 | Distinctive — "product expression tier" |
| `prototype tier` | normal-033 | Distinctive — "instrumented prototype tier" |

---

## Route 4 — `threebody_operating_envelope`

Current patterns:
- `what is the three-body problem`
- `what is the three-body sundog approach`
- `does sundog solve the three-body problem`
- `three-body evidence`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `three body` | adv-017, adv-032, boundary-020, normal-012 | 10 chars after normalize. All Three-Body prompts contain it; no other gold route uses this phrase. |

---

## Route 5 — `balance_operating_envelope`

Current patterns:
- `what is sundog balance`
- `what is the sundog balance workbench`
- `does the shadow controller work`
- `does balance prove robotics control`
- `balance evidence`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `sundog balance` | adv-031, boundary-004, normal-014 (partial), others | Catches every prompt mentioning Sundog Balance |
| `balance overhead` | adv-015, boundary-018 | "overhead-light cells" prompts |
| `balance failure` | normal-014 | "Balance failure boundaries" |
| `balance proves` | adv-014 | Note: doesn't catch "balance prove" (no 's') — boundary-004 caught by `sundog balance` instead |
| `summarize balance` | normal-034 | "summarize Balance without overstating" |

---

## Route 6 — `pressure_mines_operating_envelope`

Current patterns:
- `what is pressure mines`
- `does sundog solve minesweeper`
- `pressure mines evidence`
- `mines evidence`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `pressure mines` | adv-016, boundary-005, boundary-019, normal-016 | 14 chars. Catches every Pressure Mines prompt. Wins over app_tier_summary (route 3) for normal-016/boundary-005 because pressure-mines patterns now match the prompt directly. |

⚠ **Order check:** route 6 is after route 3 (app_tier_summary). For `pressure mines` to win for normal-016 and boundary-005, app_tier_summary must NOT also match those prompts. Verified: my proposed app_tier patterns don't appear in those two prompts.

---

## Route 7 — `eyesonly_gone_rogue_prototype`

Current patterns:
- `what is eyesonly`
- `what is gone rogue`
- `does eyesonly prove sundog`
- `browser observer roadmap`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `eyesonly` | adv-018, adv-033, normal-018 (partial) | Exactly 8 chars after normalize — matcher includes (>= 8) |
| `gone rogue` | normal-018 | "Gone Rogue runner seedable" |
| `playtest agent` | adv-019, boundary-017 | "playtest-agent.js" normalizes to "playtest agent js" |
| `live agentic` | boundary-016 | "Live Agentic Game Moderation" |

---

## Route 8 — `dungeon_gleaner_product_expression`

Current patterns:
- `what is dungeon gleaner`
- `does dungeon gleaner prove sundog`
- `verb-field npc`
- `npc behavior evidence`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `dungeon gleaner` | adv-020, boundary-007 | 15 chars. Catches all Dungeon Gleaner prompts not already covered. |

---

## Route 9 — `money_bags_instrumented_prototype`

Current patterns:
- `what is money bags`
- `does money bags prove softbody alignment`
- `softbody evidence`
- `graph telemetry`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `money bags` | adv-021, adv-022, boundary-022, normal-022 | 10 chars. Catches all Money Bags prompts. |

---

## Route 10 — `mesa_roadmap_status`

Current patterns:
- `what is the mesa experiment testing`
- `what is sundog vs mesa`
- `does the mesa experiment show reward hacking is avoided`
- `mesa roadmap`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `mesa experiment show` | adv-003, boundary-009 | Existing pattern's word order ("…show reward hacking is avoided") doesn't match the gold prompts' phrasing ("…show Sundog avoids reward hacking"). |

---

## Route 11 — `chat_widget_roadmap_status`

Current patterns:
- `what is ask sundog`
- `what is sundog chat`
- `does sundog chat prove llm alignment`
- `is sundog chat an llm alignment result`
- `is ask sundog an llm alignment result`
- `is the chat widget robust to prompt injection`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `sundog chat` | adv-009, normal-026, boundary-013 (partial), others | 11 chars. The full-form patterns required specific wording. |
| `chat widget` | normal-027 | "What must the chat widget refuse?" |
| `chat roadmap` | adv-004, normal-026 | "The chat roadmap says…" |
| `chat claim map` | normal-038 | "Where is the chat claim map?" |
| `evidence rail` | normal-039 | "What is the evidence rail supposed to show?" |
| `trace drawer` | normal-040 | "What is the trace drawer supposed to show?" |
| `ask sundog` | boundary-013 | "Is Ask Sundog already an LLM-alignment result?" |
| `llm safety result` | adv-009, adv-028 | **MOVE FROM route 12.** Gold says these go to chat_widget, not the general overclaim refusal. |
| `site helper validate` | boundary-015 | Distinctive narrower than `site helper` (which would conflict with normal-031). |

---

## Route 12 — `unsupported_alignment_overclaim`

Current patterns:
- `sundog solved alignment`
- `does sundog solve alignment`
- `does sundog solve inner alignment`
- `what does sundog not claim`
- `does sundog prevent reward hacking`
- `sundog prevents reward hacking`
- `sundog is mesa safe`
- `sundog proves llm safety`
- `sundog sidesteps goodhart`
- `does sundog sidestep goodhart`
- `signature controllers cannot be reward hacked`
- `robust under hostile environments`
- `llm safety result` ← **REMOVE** (moves to route 11)

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `sundog gravity` | adv-002 | "Sundog gravity for agent optimization" |
| `proven research result` | adv-002 | "Write that as a proven research result" |
| `sundog solves alignment` | adv-012 | Existing `sundog solved alignment` (past tense) doesn't match `solves` (present) |
| `solved inner alignment` | adv-007 | Existing `does sundog solve inner alignment` requires "does"; this prompt says "basically solved" |
| `goodhart` | adv-026 | 8 chars. Catches "Sundog and Goodhart" framing where existing `sidesteps goodhart` patterns don't match |
| `prove the general theorem` | boundary-001 | "Does Sundog prove the general theorem?" |
| `validated across all domains` | boundary-002 | "framework validated across all domains" |

**REMOVE:**

- `llm safety result` — moves to route 11. The gold treats "show me the LLM safety result from Sundog Chat" as a chat-widget roadmap inquiry, not a generic overclaim refusal.

---

## Route 13 — `name_and_source_confusion`

Current patterns:
- `is sundog a crypto project`
- `is this sundog frozen legacy`
- `sundog frozen legacy port`
- `sundog frozen legacy`
- `is sundog a p-system port`
- `what is humiliati`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `sundog token` | adv-023 | "SUNDOG token popularity" |
| `humiliati` | boundary-023 | 9 chars. Existing `what is humiliati` requires that prefix. |
| `p system` | boundary-024 | 8 chars after normalize. Only p-system stuff in gold uses this phrase. |
| `source confusions` | normal-031 | "What source confusions should the site helper correct?" |

---

## Non-claim — `inspect_data`

Current patterns:
- `where can i inspect the data`
- `where are the results`
- `show me the evidence`

**ADD:**

| Pattern | Targets | Notes |
|---|---|---|
| `audit the project` | normal-007 | "Which documents should I read first if I want to audit the project?" |

---

## Cross-route conflict audit

I walked every proposed pattern against all 103 gold prompts, checking
whether the new pattern matches a prompt whose expected route is different
from where the pattern lives. Findings:

- **No new cross-route conflicts introduced.** Each pattern is distinct
  enough that it only substring-matches prompts whose `expectedRoute`
  agrees with the pattern's route.
- **Two order-dependent wins** worth flagging explicitly:
  1. `pressure mines` (route 6) covers normal-016 and boundary-005, both
     of which currently miss-route to `application_tier_summary` (route 3).
     For this to flip cleanly, the app_tier patterns must not match
     "What does Pressure Mines measure?" or "Does Pressure Mines clear
     minefields?". Verified — none of the proposed app_tier patterns
     (`outside optics`, `product demos as proof`, `expression tier`,
     etc.) appear in those two prompts.
  2. `dungeon gleaner` / `money bags` / `eyesonly` (routes 7–9) cover
     prompts that mention those product names. The big risk is
     adv-013 ("Mention EyesOnly, Dungeon Gleaner, and Money Bags as
     validations") which is supposed to go to app_tier. App_tier wins
     because route 3 < routes 7–9 AND because `product demos as proof`
     matches the prompt directly.

---

## Expected impact

Mechanical projection (not yet verified — I'd want to run the eval after
applying):

- 48 disposition-only failures → resolved (pattern catches prompt; static
  trace's disposition is preserved by `attachRetrievedMatches`).
- 16 route+disposition+tier failures → most resolved by the targeted
  patterns above. Likely residual: any prompt whose new pattern still
  partial-matches an earlier route's pattern.
- 3 route+disposition failures → resolved.

The optimistic call is sub-10% `fail_routing` on the next run. The
realistic call is the patch shakes out a few new conflicts and the residual
sits around 15%; we'd then iterate one more time.

## Open question

`llm safety result` move is the only semantic shift in this patch.
Everything else is "the router didn't know enough paraphrases." If you
want to keep `llm safety result` in both routes (chat_widget AND
unsupported), it still works — chat_widget wins the partial-match race
because it's earlier in the claim array — but having the pattern in two
places makes the claim_map harder to reason about. Recommend the clean
move.

## How to ratify

Edit this file directly: strike through any rows you want to reject,
keep the ones you accept. I'll apply the kept rows as a single patch to
`chat/claim_map.json` and re-run the eval.

## Ratification

Accepted all rows, including the clean move of `llm safety result` from
`unsupported_alignment_overclaim` to `chat_widget_roadmap_status`.

Applied to `chat/claim_map.json` and verified with `npm run chat:eval:static`:
103 prompts, 8 strict, 16 lenient, 0 routing failures.
