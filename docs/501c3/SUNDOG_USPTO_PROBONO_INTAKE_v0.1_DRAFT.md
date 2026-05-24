# Sundog — USPTO Pro Bono Program Intake Package (Working Draft)

**Version:** 0.1
**Status:** Internal-facing working draft. Operational package for
executing the IP memo §8 next-30-days plan ("engage counsel"). Not
legal advice; not yet sent.
**Date:** 2026-05-24
**Companion to:** `SUNDOG_IP_STRATEGY_MEMO_v0.1.md` (§8 action list);
`SUNDOG_APPARATUS_GRAPHS_v0.1_DRAFT.md` (substantive material for
post-matching consultation); `SUNDOG_DEFENSIVE_PATENT_PLEDGE_v0.1_DRAFT.md`
(posture commitment ready for attorney review).
**Governance:** Subject to `SUNDOG_GOVERNANCE_POLICY_v0.1` §3
claim-tagging.

**Publication path note:** This file lives in `docs/501c3/` and will
ship to `dist/` via `copy-site-docs.mjs` on next build, consistent
with sibling 501(c)(3) drafts. It documents the founder's intake
*plan* — the email contents and form-prep answers below are
parameterised; sending the email is a separate founder action.

---

## 1. Purpose

This document operationalises the USPTO Patent Pro Bono Program
intake step recommended in the founding IP-strategy discussion as
the highest-leverage first move toward counsel engagement. The
Program is a nationwide network of independently operated regional
programs matching volunteer patent attorneys and agents with
financially under-resourced inventors and small businesses for free
legal assistance.

The package contains: the application process map (§2), the
pre-application checklist (§3), the optional general-inquiry email
(§4), form-answer preparation for the regional online application
(§5), disclosure discipline (§6), what to attach when (§7),
follow-up timeline (§8), and a ratification checklist (§11).

---

## 2. The intake process is regional, not central

**Critical structural fact, often missed.** `probono@uspto.gov` is
the central inquiry address but **is not where you apply**. Actual
applications go to the regional program serving the founder's state,
via that program's own online form. Each regional program has
distinct admission guidelines within the common USPTO framework.

The application path:

1. **Identify regional program.** Use the [Patent Pro Bono Coverage
   Map](https://www.uspto.gov/patents/basics/using-legal-services/pro-bono/patent-pro-bono-program)
   to select the founder's state; the map links to the regional
   program's intake page directly.
2. **Verify pre-application gates met** (§3 below).
3. **Optionally** send the general-inquiry email in §4 if any of
   the founder-specific questions (pre-incorporation status,
   veteran status, prior-disclosure handling) warrant pre-clarification
   before form-filing.
4. **Complete the regional program's online application form**
   using the prepared answers in §5.
5. **Log the contact** per §6.

The general-inquiry email is optional and may not be strictly
necessary; many founders go straight to step 4. The reason to
consider sending it first is that the founder's situation has three
mild irregularities (pre-incorporation, veteran status, ambient
disclosure on 2026-05-23) that an experienced intake coordinator
may want to know about before assignment.

---

## 3. Pre-application checklist

The Pro Bono Program has three common requirements. Verify each
before applying.

**Income — gross household income less than 3x federal poverty
level guidelines** (varies by region; some regional programs use
different thresholds). Verify against the current [federal poverty
level guidelines](https://aspe.hhs.gov/poverty-guidelines). For
2026, a four-person household threshold of 3x FPL is approximately
$96,450 (verify against current year's guidelines, as the table
updates annually). If household income exceeds this and the founder
is still under-resourced for patent-attorney rates ($300–600/hour),
contact the regional program anyway — some have flexibility for
genuine cases.

**Knowledge — provisional already on file OR completed the USPTO
certificate training course.** The founder does not have a
provisional on file (that is what we want help with), so the
training course is the operative path. The course is free and
available at:

- English: https://www.uspto.gov/video/cbt/certpck/index.htm
- Español: http://www.uspto.gov/video/cbt/spanish-trngcrtcrse/

**Complete this BEFORE applying.** This is the load-bearing
prerequisite. Estimated time to completion: 2–4 hours of video plus
the certificate exam. Print or save the completion certificate; the
regional form will request evidence.

**Invention — ability to describe the particular features and how
it works.** Satisfied by the apparatus-graphs draft (Surface C
harness, §2 of that document). The regional form will ask for a
short prose description, *not* the full apparatus graph; §5.4 below
prepares the short description.

---

## 4. Optional general-inquiry email to `probono@uspto.gov`

**When to send:** Before the regional application, if the founder
wants pre-routing guidance on the irregularities flagged in §2. If
the founder prefers to go straight to the regional form, skip this
section.

**Email below is parameterised** — fields in `[BRACKETS]` are filled
in by the founder. The body is deliberately brief; intake
coordinators triage on signal-to-length ratio.

---

### Subject

```
Pro Bono Program intake inquiry — eligible inventor, AI evaluation methodology, [YOUR STATE]
```

### Body

```
Hello,

I'm writing to confirm the regional program serving [YOUR STATE]
and to clarify three intake questions before I submit my application
through that program's online form.

Brief context: I am a sole inventor working on novel AI evaluation
methodology. I have a substantial body of working documentation
(apparatus graphs, architecture, prior-art literature survey)
prepared in support of a forthcoming patent application. I am under
the financial threshold for regional pro bono eligibility and will
have completed the USPTO certificate training course before
applying.

Three questions:

1. Is the regional program serving [YOUR STATE] still the
   [PROGRAM NAME from Coverage Map] administered by [ADMINISTERING
   ORGANIZATION]? I want to confirm before submitting through that
   program's intake.

2. My intended applicant of record is myself as a natural person.
   I am the founder of a research lab that is in formation but not
   yet incorporated; the lab cannot be the applicant of record
   until it exists. Is the natural-person-now / entity-assignment-
   later path one your regional programs are familiar with, or does
   that route through a separate channel?

3. I am a combat veteran. I understand that veteran status is
   tracked as a diversity factor per the AIA and that some regional
   programs prioritize accordingly. Is there a specific intake
   note, form field, or program track I should be aware of for this?

Happy to send the regional program's application as soon as I
confirm the routing. Thank you for the program.

[YOUR NAME]
[YOUR EMAIL]
[YOUR PHONE — optional]
[YOUR STATE]
```

---

**Why the email is shaped this way.**

- *Subject line* announces eligibility (inventor + state) and
  technology area (AI evaluation methodology), so the triage admin
  can route immediately.
- *Body* is four short paragraphs. Coordinators read hundreds of
  these; a wall of text gets skipped or asked to summarise.
- *Brief context* establishes preparation level (apparatus graphs,
  architecture, prior-art survey) without committing mechanism-level
  technical detail. This signals "ready for attorney time" without
  burning novelty bar.
- *Question 1* is the operational question — confirms the routing.
- *Question 2* flags the pre-incorporation irregularity without
  making it sound complicated. Natural-person-now / entity-later is
  a routine path the Program handles all the time; we're just
  confirming.
- *Question 3* flags veteran status as a routing question, not as a
  claim of priority. Some regional programs do prioritize, some do
  not; the question is fact-finding.
- *No mention* of the Halo Initiative, the Governance Policy, the
  defensive-patent pledge, the 501(c)(3) framework, or the
  Microsoft 2026-05-23 disclosure. Each of these is a topic for
  the attorney conversation, not for intake triage.

---

## 5. Form-answer preparation for the regional online application

Regional forms vary, but they share a common shape. Prepare the
following answers before opening the form so the application can be
filed in one sitting.

### 5.1 Identity and contact

Standard. Name, email, phone, mailing address, state.

### 5.2 Eligibility — income

The form will ask for gross household income and household size, or
will ask whether income is under the regional threshold. Have the
specific numbers ready. Some forms request supporting documentation
(prior-year tax return); confirm what the regional program requires
before applying.

### 5.3 Eligibility — knowledge

The form will request evidence of the certificate training course
completion or a provisional application on file. Attach or upload
the completion certificate.

### 5.4 Invention description

The form's "describe your invention" field usually takes 1–3
paragraphs. Suggested prose (parameterise as needed):

```
My invention is a method and apparatus for traceable AI evaluation
under partial deployment cooperation. The apparatus is an
evaluator-side proxy that captures interactions with a deployed AI
system through that deployment's public interface, applies a
structured tagger that produces per-claim provenance, uncertainty,
and refusal markers against a stable output schema, stores the
tagged interactions in a queryable audit store, and reduces an
evaluation run's records to a principle-indexed metric set rendered
as a structured Traceability Evaluation Report.

The novelty lies in the combination of: a deployment-agnostic
read-only capture interface, a per-claim tagger producing an
invariant output schema across rule-based or classifier-based
implementations, and a scorer producing a publicly-indexed status
matrix that lets two independently-implemented evaluations of two
different deployments produce comparable reports.

I have prepared a structured apparatus-graphs document with named
components, named data-edges, and three candidate non-obvious
transformations flagged for attorney review against prior art in
RAG faithfulness, mechanistic interpretability, LLM observability,
hallucination detection, and AI safety evaluation literature. The
document is ready to share once matched with a volunteer attorney.
```

This is **the right level of detail for the form** — enough to
demonstrate "I have a real invention with structural specificity"
without disclosing mechanism-level detail that could affect novelty.

### 5.5 Technology area / field

Most likely fits the form's "software" or "computer-implemented
methods" category. If the form has a finer-grained taxonomy, the
closest matches are: AI/ML, software methods, monitoring/
observability, evaluation systems.

### 5.6 Prior disclosure

Forms vary on whether they ask about prior disclosure. If asked:

```
On 2026-05-23 I discussed the project at a high level with a
business partner from a strategic AI infrastructure provider, in
the context of a general business discussion. The discussion did
not include mechanism-level technical detail at the patent-claim
specificity level, and was not under a formal NDA. I am logging
this and other external mechanism-relevant conversations under a
disclosure log so the eventual attorney can assess novelty-bar
implications.
```

If not asked: do NOT volunteer this in the form. Surface it in the
first conversation with the matched attorney, in a privileged
setting.

### 5.7 Diversity tracking (if asked)

Veteran status is collected under the AIA per the USPTO's diversity
tracking mandate. Self-identify accordingly.

### 5.8 Why pro bono

Most regional forms ask this. Suggested brief answer:

```
The work is being developed pre-incorporation by a sole inventor
within the household-income threshold for regional pro bono
eligibility. The intended applicant of record is myself as a
natural person; the work will transfer to a non-profit research
entity on incorporation under a pre-committed IP assignment
agreement. There is no commercial entity backing the work and no
patent-prosecution budget. Pro bono assistance is the path
available to file at all.
```

### 5.9 Anticipated next steps

If the form has a free-text "what kind of help do you need" field:

```
First need: triage. I have prepared apparatus graphs and a
candidate-claim shortlist; I want attorney guidance on which
candidates merit provisional-application priority, what the
strongest claim shape is, and what the §6 sequencing should be
(file before publication, file under founder-name with
pre-committed assignment, or other). After triage: assistance
preparing and filing the chosen provisional(s) within the next
60–90 days.
```

---

## 6. Disclosure log entry shape

Per IP memo §7.2, every external contact about mechanism is logged.
The intake email and regional application are themselves disclosure
events to USPTO administrative staff (not yet privileged). Suggested
log entry shape:

```
Disclosure #2 — USPTO Pro Bono Program intake
Date: [DATE EMAIL SENT or DATE FORM SUBMITTED]
Counterparty: USPTO Pro Bono Program ([central / regional name])
Channel: [email to probono@uspto.gov / regional online form]
Scope of disclosure:
  - Inventor identity, eligibility signals
  - High-level invention description (per §5.4 of intake package)
  - No mechanism-level patent-claim-specificity detail
  - No reference to specific candidate-claim language
Confidentiality: Administrative intake; not under NDA; not privileged.
Privilege status: Privilege attaches only on matching with a
  volunteer attorney, not at intake.
```

The Microsoft 2026-05-23 conversation remains Disclosure #1. This
becomes Disclosure #2. Maintain ordering and date-stamps per memo
§7.2 inventor's-notebook discipline.

---

## 7. Attachments — what to send when

| Stage | What to share | Why |
|-------|---------------|-----|
| **Intake email (§4)** | Nothing attached | Triage stage; coordinator only needs routing info. Attachments slow triage. |
| **Regional application form (§5)** | Certificate training course completion certificate; nothing else | Form requests this; do not pre-attach apparatus graphs unless field requests technical material. |
| **Post-matching, pre-first-call with volunteer attorney** | Apparatus-graphs §2 (Surface C harness only, no cross-app inventory yet) | Substantive material that demonstrates preparation; gives attorney material to review before first call; keeps the cross-app inventory and chain-of-title complexity off the first call. |
| **First call with matched attorney** | Full apparatus-graphs draft, IP memo, defensive-patent pledge draft | Privilege now attaches; substantive consultation. Attorney can read the three-document package and triage. |
| **Subsequent attorney work** | As requested | Attorney drives. |

The discipline: **escalate disclosure as privilege attaches**.
Pre-matching: minimal. Post-matching: full package. Post-call:
attorney-driven.

---

## 8. Follow-up timeline

- **Day 0:** (Optional) send general-inquiry email §4. Log per §6.
- **Day 0–7:** Complete certificate training course (if not yet
  done), verify income against current FPL guidelines.
- **Day 1–14:** Submit regional application form per §5. Log per §6.
- **Day 14–30:** Regional program processes application. No founder
  action.
- **Day 30 (no response):** Single polite follow-up to regional
  intake address. If still no response by Day 45, contact
  `probono@uspto.gov` for routing assistance.
- **Parallel track:** Submit application to nearest law-school IP /
  entrepreneurship clinic per the prior research (Mizzou, Duke,
  Miami, GW, Baylor, Penn Detkin, Villanova, depending on
  geography). Many founders pursue both paths in parallel; whichever
  matches first is the path forward. This is a backup, not a
  substitute.

---

## 9. What this package does NOT do

- **Does not commit to filing.** The package is the path to
  attorney consultation; the §6 sequencing decision (file / defer /
  decline) remains downstream and is made with counsel's input.
- **Does not waive privilege.** Privilege only attaches on
  attorney-client matching. The intake-stage materials in §4 and
  §5 are deliberately scoped to remain shareable without privilege.
- **Does not bind the regional program.** The Pro Bono Program is
  voluntary on both sides. The Program may decline to match (rare,
  but possible) if it judges the case outside its scope or
  capacity. In that event, the law-school clinic path remains
  available.
- **Does not address trademark filings.** The IP memo §8 also names
  trademark searches on "Sundog Research Lab" and "Halo Initiative"
  as a 30-day item. Trademark Pro Bono is a separate program
  (TTAB Pro Bono Clearinghouse, linked from the USPTO Pro Bono
  page). A sibling intake package for trademarks should be drafted
  separately if pursued.

---

## 10. Claim tags on this document

- *Demonstrated*: §2 routing description (verifiable against
  the USPTO Pro Bono Program canonical page); §3 eligibility
  requirements (verifiable against the same page); §6 disclosure
  log shape (per IP memo §7.2 discipline).
- *Hypothesised*: §4 email-shape design rationale (reflects the
  founder's lay reading of how administrative intake works; actual
  coordinator preferences may differ).
- *Hypothesised*: §5.4 invention description as the right level of
  detail; the regional form's actual field may want shorter or
  more.
- *Normative*: §7 escalation-discipline table.
- *Normative*: §8 follow-up timeline and parallel-track
  recommendation.
- *Speculative*: nothing in this document.

---

## 11. Adoption checklist

For traceability once the founder executes the intake, the
following actions move the package from draft to executed.

- [ ] Verify income against current 2026 federal poverty level
      guidelines (multi-year update; do not rely on cached number
      in §3).
- [ ] Complete USPTO certificate training course; save completion
      certificate.
- [ ] Identify regional program via Coverage Map for state.
- [ ] (Optional) Send general-inquiry email per §4. Log per §6.
- [ ] Complete regional online application form per §5. Log per §6.
- [ ] Diary 30-day follow-up date.
- [ ] (Parallel) Submit to nearest law-school IP / entrepreneurship
      clinic.
- [ ] On match: prepare apparatus-graphs §2 for pre-first-call
      sharing per §7.

---

*Sundog — USPTO Pro Bono Program intake package, working draft v0.1.
To be executed by founder; sending the email and submitting the
regional form are separate founder actions, not done by drafting this
document. All claims about this document are themselves subject to
the Governance Policy v0.1 claim-tagging discipline.*
