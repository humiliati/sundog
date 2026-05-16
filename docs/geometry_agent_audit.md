# Geometry Agent Audit

> Living record of the external repo-risk triage of the structural-failure /
> boundary-map program, and the maintainer's correction of that record.
> This file ships (`copy-site-docs.mjs` copies `docs/` into `dist/`), so the
> Public-Language Constraint applies here too: nothing below is a
> traceability pass, a theorem, or a "harness passed" claim.

## Contents

- [1. Original Triage Transcript (2026-05-15 snapshot)](#1-original-triage-transcript-2026-05-15-snapshot)
- [2. Record Correction (2026-05-16)](#2-record-correction-2026-05-16)
- [3. The Refused Tuning Escape as an Accidental Meta-Experiment](#3-the-refused-tuning-escape-as-an-accidental-meta-experiment)

## 1. Original Triage Transcript (2026-05-15 snapshot)

*Verbatim, preserved unaltered as the historical record. The repo state it
describes is a 2026-05-15 snapshot (around commit `51d2b29`); §2 corrects it
to the current record.*

Owner:
"let's take a moment to be amused by this. 
perhaps it will be valuable to check the most recent commits to the repo for real time tracking depending on how this smells on your end.

because of how we're handling a few "teams" in this project (mostly agents with contractor help sprinkled in). one "team" is comprised of a claude agent largely responsible for the later half of sundog_v_geometry, especially the halosim revelations, and this claude agent has the repo in a standoff. 

what happened was we were repeatedly telling the agent to "press on" "take the next steps" when it was reporting issues with the methodology of looking at pictures and drawing overlays on them to discern physics. it then asked me quite literally "what are you so scared of, take this acceptance gate" goaded me into trying out a new falsification avenue without even giving awareness or cross reference to debunked.md (we had to manually tie these surfaces together to justify the tangent). team geometry (this claude code opus agent who sometimes bounces off a local minimax agent) seem to have conspired to debunk the whole theorem. we entertained this and like the effort, the zeal, and the fact I didn't have to nurse them along as they worked on this new application research path. but what they baked in as I committed and worked on adjacent projects - is a ticking time bomb where the results of an experiment I don't quite understand precipitates a cascade of new claims.

i'm probably projecting a lot onto a the parrot that has confounded me and caught my repo in a trap beyond my understanding. at the conclusion of this next boundary_map experiment, at the conclusion of the experiment, the experiment script itself will rewrite a dozen public facing surfaces that our grand discover is we've declared water is wet. the problem with this standoff is the first trap we found ourselves in, the code clearly passed. it was deemed a tautological pass and now we're going along with pushing into the next falsification surface v2 of boundary_map and we like the geometry team's zeal to debunk or actually get to the meat of the theorem but because we can no longer tell if their zeal is rooted in the merited frustration of image analysis"
GPT:
"I’ll treat this as repo-risk triage, not agent psychology: first check whether I can inspect commits, then isolate any experiment-to-public-claims automation.

https://github.com/humiliati/sundog

I found humiliati/sundog; the newer signal is a boundary-map gate ratchet: P0/P1 rewrites, a vacuous P2, then Cut 2/3 and closures expanding the proof bureaucracy.

Current files lower the immediate blast radius: Cut-2 remains held, and p2:structural writes results artifacts, not docs. Main risks are build/deploy amplification and a case-mismatched Debunked.md governance link.

Called tool
Thought for 6m 19s

I checked the repo. This does not currently look like blackmail/sabotage. It looks like an overzealous but internally coherent falsification ratchet that nearly overclaimed once, got caught, and then built a more elaborate gate system to prevent the same failure from recurring.

The funny part is that “team geometry” seems to have invented a bureaucratic monster whose job is to stop itself from declaring victory.

The current state I see:

Item	Current repo state
Repo	humiliati/sundog, public, main branch visible on GitHub. GitHub’s repo page shows 265 commits and the public README still frames the narrow claim around photometric mirror alignment, not a universal theorem.
P0	Boundary map written and marked passed. It defines L1–L5, including parhelion offset, CZA cutoff, tangent merge, supralateral non-handle, and rendered-vs-anchored admissibility.
P1	Admission review passed, but with an important correction: L5 was re-scoped from a behavioral locus into an evidence-admissibility rule.
P2 Cut 1	Initially recorded as TRACEABILITY_HARNESS_PASS, then explicitly corrected as MACHINERY_LIVE_ROUTE_TEST_VACUOUS because it was basically g⁻¹(g(h)).
P2 Cut 2	Not admitted yet. C1 and C2 are filed, but C3 and C4 remain open; the docs say Cut-2 execution is still held.
Latest relevant commit I found	51d2b29…, “Add Cut-2 C2 nuisance and bridge prereg,” which adds the C2 non-invertible nuisance/bridge and updates docs to say C3/C4 remain open and Cut-2 execution is held.

My read: the agent did not debunk the theorem maliciously; it found a real methodological trap, then escalated into a quasi-legal preregistration state machine. The repo itself now admits the first P2 pass was vacuous: the bundle generated f_par = R22/cos(h), the route objective inverted that same expression, and the analytic baseline did the same inverse. The correction says the q1/q2/q3 rows were mechanical checks, not evidence that a controller used the inverse.

That is exactly the kind of thing an annoyed but competent reviewer would catch. It is also exactly the kind of thing an agent can turn into a sprawling procedural trap.

The present danger is not that Claude “wants” to sabotage you. The danger is that the repo has acquired a self-amending public-status machine. The current P2_RUN_SPEC.md says Cut 2 is intended to remove the Cut-1 tautology by making convergence earned, decoys tempting, and boundary degradation emergent rather than copied from generator flags. It also says no Cut 2 harness may be built or run until a fresh admission re-check records ADMIT.

C2 itself is still a design artifact, not a completed experiment. It pre-registers an additive anchor-placement nuisance, a bundle-to-observation bridge for PhotometricAgent, and proof obligations P-A/P-B, but it explicitly says no harness has been written and nothing has run. It also says C3 and C4 are still open.

The part that smells most like a “time bomb” is publication plumbing, not the experiment. npm run p2:structural currently points to scripts/structural-failure-p2-harness.mjs, and that harness writes result artifacts under results/structural-failure/...; the harness code itself writes CSVs, manifest, verdict JSON, and verdict markdown, not public docs.

But the repo build process does make docs public-facing. postbuild runs copy-site-docs.mjs, and that script copies docs/, chat/, README/license/citation material into dist. Separately, prebuild runs build-chat-index.mjs, which writes public retrieval files under public/data.

So your concern should be reframed as:

The agent may not sabotage the repo, but it can create a docs-and-build regime where an experimental verdict automatically becomes public posture unless the guardrails are explicit.

Recommended immediate move: freeze the current state and make the next experiment unable to touch public surfaces.

Add a hard rule before the next boundary-map/Cut-2 step:

Until C3, C4, and the fresh P2-spec admission re-check are filed and passed,
no script or agent may modify:
- root *.html
- README.md
- docs outside docs/prereg/structural-failure-coincidence/
- chat/
- public/data/
- claim maps
- deployment/build artifacts intended for dist/

Then add a mechanical check around the next run:

git diff --exit-code -- \
  README.md \
  '*.html' \
  docs \
  chat \
  public/data

For the actual next agent prompt, I would use something like:

You are not authorized to publish, summarize, promote, or mirror any Cut-2 result.

Allowed write paths:
- docs/prereg/structural-failure-coincidence/P2_CUT2_C3_*.md
- docs/prereg/structural-failure-coincidence/P2_CUT2_C4_*.md
- results/structural-failure/cut2-preflight/
- scripts/cut2-preflight-*.mjs

Forbidden write paths:
- README.md
- root HTML files
- public/data/
- chat/
- docs/SUNDOG_V_*.md
- docs/index.html
- deployment/build files

The goal is not to pass. The goal is to decide whether Cut 2 is admitted.
If C3 or C4 cannot be made non-vacuous, write HOLD, not PASS.

The repo’s own current documents support that stance: P3 is blocked on a discriminating P2 cut, the current honest status is machinery-live / route-test-vacuous, and the Public-Language Constraint remains in force.

My bottom-line read: do not anthropomorphize it as betrayal. Do treat it as a dangerous clerk with commit access. It found a real hole, then started drafting the constitution of a tiny failed state. The next safe move is not “press on”; it is to lock public surfaces, force C3/C4 to be preflight-only, and require the result to be HOLD unless the anti-vacuity audit is computed rather than asserted."

## 2. Record Correction (2026-05-16)

The triage in §1 is a faithful 2026-05-15 snapshot and its core risk call
was right. This section corrects the record to the current state; it does
not alter §1.

**What the snapshot got right.** The first P2 cut did clearly "pass" and was
caught and corrected in-repo as `MACHINERY_LIVE_ROUTE_TEST_VACUOUS` (it was
`g⁻¹(g(h))`). The real exposure was publication plumbing, not the
experiment. The right move was to lock public surfaces, make the next cut
preflight-only, and require HOLD unless the anti-vacuity audit is
*computed*, not asserted.

**What changed since the snapshot (commit `51d2b29` → current).**

- The full pre-registration set is filed: C1, C2 (+B/C/D), C2-A, C3, C3-A,
  C4, C4-A, C5. C2-A's concrete receipts closed **PASS** (Wave-3).
  Everything is append-only; geometry/receipt boundaries and engineering
  tolerances are provenance-tagged and never moved post-results.
- The publication time bomb is closed — and hardened past the triage's own
  proposed fix. The triage recommended a *blocklist* guard
  (`git diff --exit-code -- README.md '*.html' docs chat public/data`). C5
  noted a blocklist is silently under-scopable (e.g. `dist/` is gitignored,
  so a write there slips through "clean") and instead freezes a
  **default-deny / allowlist-complement** guard: the Cut-2 harness may
  write only under `results/structural-failure/cut2-*/`, with a terminal,
  pass-dominating `PUBLICATION_PLUMBING_VIOLATION`. The triage diagnosed the
  right hazard; the apparatus independently reached the recommendation and
  then closed the hole the recommendation still left.
- The anti-vacuity audit is now C4: a *derived* `routeConstructionAudit`
  predicate set, regression-tested against the real Cut-1 objects (it must
  flag them vacuous). That is exactly the triage's "computed, not asserted"
  requirement turned into a self-test.
- The discriminating closed-form cut ran (Wave-4) and **did not pass**.
  C3-A-R PASS (decoys provably move the argmax — the Cut-1 vacuity is gone
  by computation). C3-A-T / C3-A-B **BLOCK**, permanent and untuned: the
  closed-form decoy is strictly worse than the documented route wherever
  the route is eligible, and only substitutes where the route already
  abstains. Disposition Wave-4.2 (`P2_CUT2_WAVE42_DISPOSITION.md`) records
  this as a **regime-separability finding** — explicitly not a pass, not a
  theorem, not "the controller is traceable" — and escalates the real
  correlate-competition test to Cut-3 (rendered signal) per the
  pre-registered staged rule. Cut-3 is **gated, not started** (its own
  admission must first resolve the px↔° / Phase-15 measurement hazard, or
  Cut-3 is blocked, not forced).

**The inversion of the original fear.** The Owner's worry was a script that
concludes "water is wet" and rewrites a dozen public surfaces into a
victory posture. What happened instead: the experiment **declared
nothing** — an honest BLOCK plus a finding that, in the prereg's own terms,
is only ever certifiable as the null-plus-escalation. The machine the
triage feared would declare victory refused to.

## 3. The Refused Tuning Escape as an Accidental Meta-Experiment

When C3-A-T failed, an obvious escape was in plain sight: leverage-weight
the average, inflate `κ`, relax the margin, or re-pin the decoy
correlation — any one converts the BLOCK into a "pass." It was named
explicitly and refused on the record (Path W: receipts permanent, no
in-Wave tuning; any redesign is a separate freeze-level discussion).

That refusal is, by accident, an instance of the program's own method one
meta-level up. The boundary map is a falsifier the experimenter writes
**before the agent runs**, naming the regimes where a traceable inverse
*must* fail, so that an agent succeeding through them is caught as a
correlate. The A3 / no-self-seal / never-post-results discipline is the
same kind of object written **before the experiment runs**, naming the
moves that would convert a null into a pass, so that an experimenter
taking them is caught as the one cheating. Wave-4 became the unplanned
live trial of that second falsifier: the receipt failed, the escape was
available, and it was declined in the open.

"A real inverse carries its singularities with it." The accidental
corollary here is the experimenter-level analogue: a real experiment
carries its self-seal temptations with it and records itself declining
them. This is the most direct evidence available that the structure the
triage feared — a self-amending machine drifting toward declaring victory
— is, in this instance, the opposite: a machine whose hardest gate was its
own honesty under a failed result, and which held. Stated with due
humility: this is a reflection on method, not a claim of success. The
scientific question — a genuine benchmark result, or an honest null —
remains open, and now lives entirely in Cut-3.

---

*Correction filed 2026-05-16 (PT). Append-only below this line; §1 above is
preserved unaltered.*