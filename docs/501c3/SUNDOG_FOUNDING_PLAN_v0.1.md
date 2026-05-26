# Sundog Research Lab — Founding Plan

**Version:** 0.1 (working document, internal-facing)
**Publisher:** Founding members of Sundog Research Lab
**Companion to:** *Sundog Research Lab Governance and Claims-Review
Policy v0.1*, *The Halo Initiative v0.2*,
*Halo Traceability Harness Whitepaper Outline v0.2*
**Date:** 2026-05-21
**Status:** Internal — subject to claim-tagging per the Governance
and Claims-Review Policy.

---

## 0. Purpose

The *Sundog Research Lab Governance and Claims-Review Policy v0.1*
describes a regime that assumes specific institutional structures
exist: an incorporated nonprofit, a board with an independent
majority, a research director, a rotating claims reviewer, an
annual conflict-of-interest declaration, a public claims log, and a
written agreement framework for evaluation engagements.

None of these structures exist today. The lab is currently a
half-dozen volunteers around a working software project. The
governance policy, taken literally, is not yet enforceable; it
describes a regime the founders are committed to building.

This Founding Plan bridges the two. It describes the current state,
the target state, the gap between them, the interim rules that
apply while the gap is being closed, the milestones at which
specific policy clauses become operative, and the things the lab
explicitly will not do until the structures supporting them are
real. The plan is the answer to the obvious challenge that the
meta-principle is aspirational rather than enforceable.

The plan is itself subject to the claim-tagging discipline. All
substantive claims carry tags in §11.

---

## 1. Current state

The following is true on the date of this draft.

- The lab exists as an informal working group of approximately
  six contributors. Tagged *Demonstrated*: the working group meets
  and produces artifacts; the artifact trail in this repository
  is the evidence.
- The lab has no legal entity. No incorporation, no fiscal
  sponsorship, no bank account in a lab name, no EIN, no
  registered trademark, no formal IP holdings under a lab name.
- The lab has no board, no research director, no formal claims
  reviewer, no conflict-of-interest declarations, no published
  claims log, and no written contributor agreements.
- The lab has working artifacts in the Sundog repository: the
  halo-rendering toolset (HaloSim integration, the HS-0 mechanism),
  the geometry workbench, the sundog.html public page and the
  Cloudflare Pages deploy, and the Mujoco / mesa / mines /
  three-body strands documented under `docs/`.
- The lab has draft published documents: the Halo Initiative
  (v0.1, v0.2), the Governance and Claims-Review Policy (v0.1),
  the harness whitepaper outline (v0.1, v0.2).
- The lab has published no Traceability Evaluation Reports, no
  research papers under its name, and no evaluations of any
  third-party deployment.

## 2. Target state

The target state assumed by the Governance and Claims-Review Policy
is, in summary:

- An incorporated US 501(c)(3) public-interest entity named
  *Sundog Research Lab*, or such other name as the founders adopt
  if the trademark and reputational review of "Sundog" requires a
  change.
- A board of directors with an **independent majority** as
  defined in policy §10.
- A **research director** with authority over high-stakes
  publications per policy §4.
- A **rotating claims reviewer** with a rotation cadence of no
  longer than 12 months per policy §4.
- A **public claims log** per policy §5.
- **Annual conflict-of-interest declarations** per policy §8 and
  **30-day disclosure of material changes** thereafter.
- **Written agreements** for every evaluation engagement per
  policy §7.
- Contributor agreements (CLAs) on file for every contributor to
  the harness code, the website, the published documents, and
  the Sundog project artifacts assigned to the lab.

The path to that state, in this plan, is **fiscal sponsorship
first, then direct 501(c)(3)**, with Sundog projects becoming
assets of the lab at incorporation.

## 3. Gap analysis

| Target-state structure                | Current status | Closure mechanism            |
|---------------------------------------|----------------|------------------------------|
| Legal entity                          | None           | Fiscal sponsor → direct 501(c)(3) |
| Board with independent majority       | None           | Recruit at incorporation     |
| Research director                     | None           | Founding-member appointment, ratified by interim governance |
| Rotating claims reviewer              | None           | Founding-member rotation, defined in §5 |
| Public claims log                     | None           | Repository file from publication-1 onward |
| Conflict-of-interest declarations     | None           | Founding-member declaration filed before publication-1 |
| Evaluation-engagement agreement       | None           | Template drafted before any paid engagement |
| Contributor agreements (CLAs)         | None           | All current contributors sign before incorporation; new contributors sign on first contribution |
| IP assignment of Sundog artifacts     | Individually held | Assignment agreements signed at incorporation |
| Trademark and name clearance          | None           | "Halo" and "Sundog" reviewed before v1.0 of the Initiative |

Each closure mechanism is mapped to a milestone in §6.

## 4. Founding personnel and roles

The lab has approximately six current contributors. The plan does
not enumerate them by name in this version; founders will fill in
§4 from a private founders' file before this plan is circulated
outside the working group. Each row below lists the role, its
description, and the policy clause it implements when in effect.

| Role                          | Description                                                                 | Policy clause |
|-------------------------------|-----------------------------------------------------------------------------|---------------|
| Interim research director     | Authority over high-stakes publications during the interim period.          | §4            |
| Interim claims reviewer       | Reviews tags on publications; rotates among founding members.               | §4            |
| Project lead, Sundog projects | Continuation of halo-rendering and geometry-workbench work.                 | (lab project) |
| Project lead, Initiative      | Stewardship of the Halo Initiative document and the public comment.    | (lab project) |
| Project lead, harness         | Stewardship of the harness whitepaper and the v0.1 implementation.          | (lab project) |
| Incorporation lead            | Drives fiscal-sponsor selection and the eventual 501(c)(3) filing.          | (this plan)   |
| Secretary                     | Maintains the claims log, the founders' file, the IP register.              | §5, §9, §10   |

A single founding member may hold multiple interim roles, but no
founding member may hold both interim research director and
interim claims reviewer for the same publication.

## 5. Interim governance

Until the target state is reached, the lab operates under the
following interim rules. Each rule has an explicit sunset condition.

- **Decision-making.** Substantive decisions are taken by **rough
  consensus** of the founding members, with at least three
  members participating. A founding member with a material
  conflict on a decision recuses.
  *Sunset:* superseded by the board at incorporation.
- **Publication.** No publication carries the *Sundog Research
  Lab* attribution until the publication has been tagged by its
  author, reviewed by a non-author founding member acting as
  interim claims reviewer, and signed off by the interim research
  director.
  *Sunset:* superseded by the policy §4 procedure when the
  research director role is formalised and the claims-reviewer
  rotation is operational.
- **Tagging.** All claim tags are recorded in a `claims-log.md`
  file in the repository from publication-1 onward, regardless of
  whether the lab has incorporated yet. The discipline applies
  before the institution exists.
  *Sunset:* never. The log persists.
- **Conflict disclosure.** Every founding member files a
  conflict-of-interest declaration with the working group before
  any publication carrying lab attribution. Declarations are
  re-filed at incorporation and annually thereafter.
  *Sunset:* the interim declarations are superseded by the
  post-incorporation annual cycle.
- **Evaluation engagements.** No paid evaluation engagement of a
  third-party deployment is undertaken during the interim period.
  *Sunset:* see §7 pause-points; this rule sunsets only when the
  board, research director, evaluation-engagement template, and
  COI process are all in place.
- **Donations and funding.** During the interim period, the lab
  does not solicit donations under its own name. Grant
  applications may be submitted through a fiscal sponsor once
  one is selected.
  *Sunset:* superseded at fiscal-sponsorship adoption (month 3
  target) and again at direct 501(c)(3) determination (month 12+
  target).
- **Spokesperson rule.** Only the founding members, speaking in
  rough-consensus-affirmed capacity, may make public statements
  in the name of the lab. Individual members are free to
  describe their personal work but may not bind the lab.
  *Sunset:* superseded by the board's communications policy at
  incorporation.

## 6. Transition timeline

Months are counted from publication of this plan as month 0. Dates
are illustrative; the founders may compress or extend as
circumstances allow. All milestones tagged *Hypothesised* unless
otherwise noted.

| Month  | Milestone                                                                              |
|--------|----------------------------------------------------------------------------------------|
| 0      | This plan published internally. *Demonstrated* on circulation to founders.              |
| 0–1    | Founders' file completed with role assignments. CLAs signed by all current contributors.|
| 1      | First conflict-of-interest declarations filed. Claims log opened.                       |
| 1–3    | Fiscal-sponsor selection. Candidate sponsors evaluated; agreement signed.               |
| 3      | Fiscal sponsorship in place. First grant applications drafted.                          |
| 3–6    | Recruit two independent board members in preparation for direct incorporation.          |
| 6      | Articles of incorporation and bylaws drafted. First Traceability Evaluation Report exercise begins under interim governance — see §7 for the publication pause-point. |
| 6–9    | 501(c)(3) application submitted. Trademark and name clearance completed for "Sundog" and "Halo".|
| 9      | Independent board majority in place. Interim research director ratified or replaced by board.       |
| 12     | Target: IRS 501(c)(3) determination. Sundog project IP formally assigned to the lab.    |
| 12–18  | First Traceability Evaluation Report published if pause-points in §7 are met. Initiative v1.0 path opens.|

The plan is reviewed quarterly. Slips are recorded; the plan is
not silently rebased onto new dates.

## 7. Pause-points

The following commitments hold during the interim period and are
binding regardless of external pressure (funder interest, press
attention, opportunistic engagements). Each pause-point names the
policy clause it implements.

- **No Traceability Evaluation Reports of third-party deployments
  under lab attribution** until *both* an independent board
  majority and a formalised research director exist. (Policy
  §4 and §10.) An exercise Report may be drafted internally
  before this, but it is not published under lab attribution and
  is tagged *Experimental* if it is shared.
- **No paid evaluation engagements** until the
  evaluation-engagement agreement template (§7 of policy) is
  drafted and the COI process is operational.
- **No fundraising claims about "the lab"** until incorporation
  or fiscal-sponsorship is in place. Fundraising under a fiscal
  sponsor identifies the sponsor.
- **No language of certification, accreditation, conformance, or
  approval** in any public document. The Initiative §5 forbids
  this and the lab adheres to it from publication-1.
- **No public claim tagged *Demonstrated*** about the harness's
  capabilities until the v0.1 §6 walkthrough has been completed
  and at least one independent party has replicated it under the
  §11 reproducibility statement. All claims before then are
  *Hypothesised* or *Experimental*.
- **No commitments to evaluate a specific deployment** until the
  evaluation surface is real and the COI process is operational.
  The lab may publicly state which deployments it intends to
  evaluate, tagged *Hypothesised*.
- **No silent retraction or revision** of any publication ever,
  even during the interim period. The corrections-and-retractions
  discipline in policy §9 is binding from publication-1.

## 8. Sundog projects as lab assets — transition

The existing Sundog project artifacts (halo-rendering toolset,
geometry workbench, sundog.html, the HaloSim integration, the
Mujoco / mesa / mines / three-body strands) become assets of the
lab at incorporation. The transition steps:

- **Now.** Identify the contributors to each artifact. Record
  current copyright holders.
- **Month 0–3.** All current contributors sign a Contributor
  Licence Agreement covering past and future contributions to the
  Sundog repository. This is the legal foundation for later
  assignment.
- **Month 6.** A draft assignment agreement is prepared, naming
  the (not-yet-incorporated) Sundog Research Lab as recipient.
- **Month 12 (or upon incorporation).** Contributors sign the
  assignment agreement. The lab takes ownership.
- **At incorporation.** The lab publishes an inventory of
  assigned artifacts and their licences. Where artifacts depend
  on third-party code, the lab confirms licence compatibility.

Until assignment is signed, the lab uses Sundog project artifacts
under the contributors' implicit consent for the project's
historical purpose (halo rendering and geometry workbench). It
does not claim institutional ownership in publications.

## 9. IP and contributor agreements

- **CLA.** A Contributor Licence Agreement is required for every
  contributor to the harness code, the website, the published
  documents, and the Sundog project artifacts being transitioned.
  The CLA grants the lab the right to relicense if needed and
  preserves contributor copyright. Initial template:
  [CLA template v0.1](./SUNDOG_CONTRIBUTOR_LICENCE_AGREEMENT_TEMPLATE_v0.1.md),
  not operative until counsel review, founder approval, and selection
  of the actual legal counterparty.
- **Document licensing.** The Halo Initiative, the Governance
  and Claims-Review Policy, and the harness whitepaper remain
  rights-reserved unless a later signed governance decision selects
  a specific public, partner, archive, or publication license. Any
  open-content release is optional, explicit, and document-scoped.
- **Code licensing.** Harness code and Sundog project code remain
  rights-reserved unless a later signed governance decision selects
  a specific open-source, partner, commercial, nonprofit, or
  successor-entity license. Public repository access is not itself a
  reuse grant.
- **Trademark.** "Sundog Research Lab" and "Halo" usage is
  reviewed for trademark conflict before v1.0 of the Initiative
  per the §10 condition.

## 10. Funding theory

The lab's funding theory in priority order:

- **Public-interest grants.** AI safety and public-interest
  research foundations. Preferred because grant terms are
  written and reviewable, and conflict-of-interest implications
  are limited.
- **Individual donations** through the fiscal sponsor or, later,
  the lab directly. Useful for unrestricted operating funds.
- **Paid evaluation services (Traceability Evaluation Reports).**
  Last in priority and gated behind a complete COI process per
  policy §8. The lab will not accept payment for evaluation work
  until the §7 pause-point on paid engagements is met.

The lab does not accept funding that is contingent on specific
findings, and discloses every funder in every relevant publication
per policy §8.

## 11. Versioning and claim tagging

This is v0.1 of the Founding Plan. The plan is reviewed quarterly
and superseded as milestones are met. A v1.0 will be issued when
the lab has incorporated, has an independent board majority, and
has published its first Traceability Evaluation Report — at which
point the plan's bridging function is complete and most of its
content collapses into the standing policy.

The plan's substantive claims carry the following tags:

- *Demonstrated*: the current state described in §1; the
  existence of the working group and its artifacts.
- *Normative*: the choice of fiscal-sponsorship-first over
  direct-incorporation-first; the choice of Sundog-as-asset over
  Sundog-as-separate; the priority order of the funding theory.
  These are values judgments.
- *Hypothesised*: every milestone in §6; any future outbound license
  choices in §9; the assumption that "Sundog" and "Halo" survive name
  clearance.
- *Speculative*: that v1.0 of this plan is reached within 18
  months.

The interim governance in §5 and the pause-points in §7 are
*operative* rules — they bind the lab now. A rule is not a claim
and is not tagged; it is either followed or revised, with revision
recorded as an amendment per policy §9.

---

*Sundog Research Lab — interim governance period. Public comment
not yet open; circulated to founders.*
