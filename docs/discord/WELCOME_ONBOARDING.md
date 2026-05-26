# Sundog Research Lab — Discord Welcome & Onboarding Pack

**Version:** 0.1
**Status:** Internal draft. Paste-ready Discord copy. Not yet posted.
**Date:** 2026-05-22

---

## Purpose

This file holds copy-paste content for the Sundog Research Lab Discord.
The Discord is **public from day one** and serves two audiences at once:

1. **Community side** — `#welcome`, `#general`, `#questions`,
   `#brainstorm`, `#paper-club`. Casual, hang-out, grass-roots ML
   energy. The bar to participation is "be a decent person and stay
   on-topic-ish."
2. **Lab side** — `#sundog-project`, `#engineering`, `#curriculum`,
   `#scheduling`, `#announcements`, `#red-team`, `#blue-team`,
   `#secretary`. The same claim-tagging discipline the lab uses in its
   `docs/501c3/` writing is operative here.

Both sides share one server. The split lives in the channels, the
pins, and the norms — not in member tiers.

This pack covers only the **welcome / onboarding flow**. Channel
topic strings, seed posts per channel, and role/permission structure
are out of scope and noted at the end as follow-ups.

---

## 1. Server-level fields

These are configured in Server Settings, not posted as messages.

### 1.1 Server name

```
Sundog Research Lab
```

### 1.2 Server description (about, ~120 chars)

Use this in the server's "About" field so the description that
appears in the search/invite preview is complete instead of truncated.

```
Independent applied research lab — alignment, observable AI honesty,
and grass-roots ML. Sponsored by Stellar Aqua LLC.
```

(112 chars. The "..." in the current screenshot suggests the field is
currently set to a longer string and is being truncated; replacing
with this shorter version fixes the truncation.)

### 1.3 Optional longer "About" / community-description body

For places that allow a longer description (e.g. Discovery, Community
features, server profile page):

```
Sundog Research Lab is an independent applied research program
sponsored by Stellar Aqua LLC. We work on the Sundog Alignment
Theorem (controllers that align from indirect, observable structure)
and the Conscium Initiative (a voluntary framework for evaluating
the surface honesty of AI systems).

This server is two things at once: a casual place to hang out and
talk machine learning, and the working channel for active Sundog
research and lab activities. Public from day one. Pre-incorporation;
the lab is working through a 501(c)(3) / fiscal-sponsor track in
parallel.
```

---

## 2. `#welcome` pinned message set

These are the messages to **post in #welcome and pin**, in order.
Each one is a single Discord message under the 2000-char limit. Post
them as the same user (the server owner) so the pins read as one
voice.

---

### 2.1 PIN — "What this place is"

```
**Welcome to Sundog Research Lab.**

We're an independent applied research lab sponsored by Stellar Aqua
LLC. Two things happen here:

**A casual ML hangout.** Talk papers, share builds, ask dumb
questions, brainstorm half-formed ideas. `#general`, `#questions`,
`#brainstorm`, `#paper-club` — that's the community half. No
gatekeeping; grass-roots energy welcome.

**Active Sundog research.** `#sundog-project`, `#engineering`,
`#curriculum`, `#scheduling`, `#red-team`, `#blue-team`, `#secretary`
— this is where the lab's working channels live. The bar is a little
higher here, mostly around how claims are stated (see the norms pin
below).

Both halves live on the same server. You don't pick a side. Lurk in
the channels that interest you; speak when you have something to
say.

If you're brand new, the next four pins explain how things work
here, what to read, where to start, and how to get involved.
```

(~970 chars.)

---

### 2.2 PIN — "Community norms"

```
**Norms — the short version.**

1. **Be a decent person.** No harassment, no bigotry, no doxxing, no
   targeting individuals. Disagree about ideas, not about people.

2. **Stay roughly on-topic for the channel.** Off-topic chatter goes
   in `#general`.

3. **Crediting work matters.** If you reference someone's paper,
   tool, or post, link it. If you build on someone's idea here, name
   them.

4. **Be honest about uncertainty.** This one is the lab's signature
   norm and the one most likely to feel unusual at first.

The lab publishes its writing with **claim tags** — every claim is
labelled *Demonstrated*, *Normative*, *Hypothesised*, or
*Speculative*. The same discipline operates in our research
channels (`#sundog-project`, `#red-team`, `#blue-team`,
`#secretary`). Plain-English version: when you make a strong claim
in those channels, indicate whether it's something you can show,
something you think people *ought* to do, a hypothesis you're
working on, or speculation. You don't have to use the literal tags;
"I think" / "I can show" / "I'm guessing" is fine. We're just
allergic to confident-sounding statements that turn out to be
guesses.

Community channels (`#general`, `#questions`, `#brainstorm`,
`#paper-club`) are looser. Speculate freely there.

5. **Moderation.** Server staff can timeout, remove, or ban for
   violations of 1–3. We try to talk to people before doing anything
   heavier than a nudge.

Full Governance and Claims-Review Policy:
github.com/humiliati/sundog → `docs/501c3/SUNDOG_GOVERNANCE_POLICY_v0.1.md`
(still a v0.1 internal-facing draft; it tells you where we're
headed).
```

(~1530 chars.)

---

### 2.3 PIN — "Channel tour"

```
**Channel tour.**

**Community side — casual, low bar:**
- `#general` — anything ML, anything lab-adjacent, anything off-topic-but-friendly.
- `#questions` — ask anything; "is this a dumb question" answer is no.
- `#brainstorm` — half-formed ideas welcome. Speculation explicitly OK here.
- `#paper-club` — a paper a week (or whenever someone proposes one). Threads per paper.

**Lab side — working channels, claim-tag norms apply:**
- `#sundog-project` — the Sundog Alignment Theorem work itself: mirror-alignment, operating-envelope studies, the paper draft.
- `#engineering` — repo, builds, MuJoCo, HaloSim, website, infra.
- `#curriculum` — onboarding new contributors, learning-path discussions, scaffolded exercises.
- `#scheduling` — meeting times, work blocks, coordination.
- `#announcements` — read-mostly; lab-wide updates land here.
- `#red-team` — adversarial review of claims, papers, methods. Strong arguments welcome.
- `#blue-team` — defense and replication of claims. Where the receipts get assembled.
- `#secretary` — minutes, decisions, claim-tag fixes, corrections log.

**Events** and **Server Boosts** are server features in the left
column; not channels.

If you're not sure where something belongs, post in `#general` and
someone will redirect.
```

(~1430 chars.)

---

### 2.4 PIN — "Reading list / where to dig in"

```
**If you want to dig in, here's where to start.**

Everything is in the public repo: **github.com/humiliati/sundog**

**Curious newcomer / "what is this lab?"**
→ `README.md` (top of the repo) for the research program
→ `docs/501c3/CONSCIUM_INITIATIVE_v0.2.md` for the normative
  framework the lab develops and publishes (currently in public
  comment).

**ML person who wants to read a paper:**
→ `docs/PAPER_v1_draft.md` — current paper-shaped writeup of the
  photometric mirror-alignment result.
→ `docs/RESEARCHER_GUIDE.md` — shortest path through the repo for a
  reviewer.

**Prospective contributor:**
→ `docs/501c3/SUNDOG_FOUNDING_PLAN_v0.1.md` §§8–9 for the
  IP-transition logic.
→ `docs/501c3/SUNDOG_CONTRIBUTOR_LICENCE_AGREEMENT_TEMPLATE_v0.1.md`
  for the draft CLA shape (not yet operative).

**Funder / fiscal-sponsor / journalist:**
→ `docs/501c3/INITIATIVE_INDEX.md` — the audience-specific
  reading-order map.

**A note on naming.** "Conscium" overlaps with an existing UK AI
company (Conscium Ltd., founded 2024). The name is provisional in
our public-comment draft and will be resolved at v1.0; see the name
notice at the top of `CONSCIUM_INITIATIVE_v0.2.md`.

The lab is **pre-incorporation**. The 501(c)(3) / fiscal-sponsor
track is underway; founding documents are in `docs/501c3/` with
status legends so you can see what's binding, what's draft, and
what's still hypothesised.
```

(~1500 chars.)

---

### 2.5 PIN — "Getting involved"

```
**Getting involved.**

The honest answer right now: the easiest way to participate is to
**show up in the channels that interest you and start talking.**
We're small. There's no application form.

A few concrete on-ramps:

**Read and comment.** The Conscium Initiative v0.2 is in public
comment. Read it, push back, file issues on GitHub. The lab takes
this seriously — public comment isn't decorative.

**Paper club.** Propose a paper in `#paper-club`. If two people
want to read it, we read it.

**Replicate something.** The mirror-alignment result, the
operating-envelope sweeps, anything in `docs/`. Bring your replication
attempts (and your failures) to `#blue-team`.

**Break something.** `#red-team` is for finding holes in our claims.
Strong adversarial reads are valued, not punished.

**Contribute code or docs.** The repo is available for inspection, but rights
are reserved by Stellar Aqua LLC unless a file clearly states otherwise. Open a
PR only when the contribution can be handled under the current contributor/IP
intake rules.
For anything substantive, expect a future CLA (template is in
`docs/501c3/`); we'll let you know when it goes live.

**Just hang out.** Honestly fine too. Lurking is welcome.

**A reminder:** the lab is pre-incorporation and not currently
taking paid evaluation engagements or formal partnerships. The
Founding Plan §7 has the pause-points that govern when we will and
won't.

Questions about any of this go in `#questions` (or DM a server
moderator).
```

(~1400 chars.)

---

## 3. Optional welcome-bot DM (if you wire one up)

If you install a welcome bot (MEE6, Carl-bot, etc.) and want it to
DM new arrivals, this is a 3-sentence opener that respects the
disciplined side without being cold.

```
Welcome to Sundog Research Lab. This server is half casual ML
hangout, half active research channels for the Sundog program — head
to #welcome for the pinned tour. If you're not sure where to start,
post in #general or #questions; we'll get you oriented.
```

(~310 chars.)

---

## 4. Posting order (do this once)

1. Set the **Server description** (§1.2) and **About body** (§1.3)
   in Server Settings.
2. Open `#welcome`. Post §2.1, §2.2, §2.3, §2.4, §2.5 in order, as
   the server owner.
3. Right-click each → Pin Message → confirm. Discord will reverse
   the order in the pins view, which is fine — the "What this place
   is" pin reads naturally as the most recent.
4. Delete the "User pinned a message" system notifications in
   `#welcome` so the channel reads clean.
5. (Optional) Install a welcome bot and paste §3 as the DM template.

---

## 5. Follow-ups (deliberately out of scope for this draft)

These were flagged by the focus-area question as **not now**. Listing
them so we don't lose track:

- **Channel topic strings** — the one-line topic that shows under the
  channel name. One per channel. ~30 minutes of work; needs a pass.
- **Seed posts per channel** — first message in each working channel
  so it doesn't read empty. `#paper-club` kickoff, `#curriculum`
  learning-path index, `#red-team` posting template, `#secretary`
  minutes template.
- **Role + permission structure** — founders / contributors /
  community / observer; who can post in `#announcements`; how the
  `#red-team`/`#blue-team` workflow gates work in practice; whether
  `#secretary` is post-anywhere read-only or contributor-write.
- **Code of conduct — long form.** §2.2 is the short version.
  A longer CoC document (linked from the pin) would mirror the
  Governance Policy's structure: claim-tag discipline, corrections
  process, escalation.
- **Server icon and banner.** Currently the icon in the screenshot
  is a generic photo. A small-format Sundog mark would land harder.

Each of these is a separate ~30–90 minute pass.

---

## 6. Claim tags on this pack

- *Demonstrated*: that the current Discord has the channel set shown
  in the 2026-05-22 screenshot (welcome, general, sundog-project,
  red-team, blue-team, secretary, announcements, questions,
  curriculum, engineering, scheduling, brainstorm, paper-club); that
  the repo and `docs/501c3/` contain the files referenced.
- *Normative*: the casual-front / disciplined-back split, the
  community norms in §2.2, the channel-tour assignments in §2.3, and
  the posting order in §4. These are recommendations, not facts
  about the world.
- *Hypothesised*: that public-from-day-one with these norms is the
  right onboarding shape for the lab's current pre-incorporation
  phase. Will need a review after the first ~25 non-founder members
  arrive.
- *Speculative*: nothing in this pack.

---

*Sundog Research Lab — Discord onboarding pack. Internal draft.*
