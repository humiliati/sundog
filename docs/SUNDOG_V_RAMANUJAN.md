# Sundog vs. Ramanujan — a Labeled Verification Benchmark

> **STATUS: DRAFT SCAFFOLD, unpromoted, no public surface.** Opened 2026-06-05 as a flagship-
> demonstration + portfolio lane, NOT a mathematics lane and NOT a theory of Ramanujan's genius. No
> public-facing copy, `site-pages.json` entry, or gallery surface until owner sign-off + an
> evidence-tier review. House template: `SUNDOG_V_P_V_NP.md` / `SUNDOG_V_IUT_ABC.md`. This lane
> applies the **certificate / capacity-bounded verifier** discipline
> (`docs/pvnp/SUNDOG_CERTIFICATE_PROBLEM.md`) to the richest labeled finding-vs-checking dataset in
> the history of mathematics.

Working hook:

> Hardy was the verifier. We built the instrument. Here is what a century of checking Ramanujan
> teaches a machine that knows when to say *not yet*.

Short version:

> Ramanujan stated thousands of results without proof; the community spent a century adjudicating
> each one. That century is a **labeled accept / reject / quarantine benchmark** with human
> ground truth — and Ramanujan himself is the human instance of the lab's formal object:
> outputs cheap to verify, the generator hard to invert. Sundog does not re-prove a single
> identity. It **scores a capacity-bounded verifier against the historical record**, and
> demonstrates the flagship doing its hardest move — principled abstention — where checking
> genuinely exceeds the envelope.

---

## 0. Boundary first (the loudest section — do not bury it)

**This lane proves no mathematics and explains no genius.** It does not re-derive Ramanujan's
identities, settle open entries, or advance a theory of intuition, the divine, or "more than we
know." The ground-truth labels come from the published adjudication record — principally Bruce
Berndt's *Ramanujan's Notebooks* (Parts I–V) and Andrews–Berndt's *Ramanujan's Lost Notebook* — and
are cited as third-party fact. Sundog's only object is a **verifier scored against that record**.

Two non-negotiable guardrails, each the mirror of a trap:

1. **Do not mysticize.** No "Namagiri / hidden substrate / he tapped something we can't see." That is
   R3 over-reach (`SCIENTIFIC_CRITERIA.md`) and reads as exactly the crankery the evidence tiers
   exist to prevent. Ramanujan is used as a **verification dataset** and a **finding-vs-checking case
   study**, never as evidence for a theory of mind. **Honor him; do not reduce him to a metaphor.**
2. **Self-consistency clause.** A lane whose thesis is *principled abstention under bounded capacity*
   must itself abstain from the mathematics it cannot check. If this lane ever claims to *verify* a
   deep Ramanujan result rather than to *route* it (accept / reject / quarantine) against the cited
   record, it has falsified its own thesis (§6).

---

## 1. Why Ramanujan is the right Tycho (four resonances)

The lab is Kepler with a finished instrument (the claim-boundary verifier) looking for a Tycho — a
rich, real dataset to demonstrate it on. Among the candidates (abc/IUT, Atiyah–RH, the ML
reproducibility crisis), Ramanujan is the strongest on four independent axes:

1. **He is a *labeled* benchmark, not a single point.** ~3,000–4,000 results stated mostly without
   proof; a century of adjudication (Hardy/Watson → Berndt's five volumes → Andrews' 1976
   rediscovery of the Lost Notebook → Andrews–Berndt). The output is, functionally, an
   **accept / reject / quarantine corpus with human ground truth** — the closest thing mathematics
   has to a pre-labeled verification benchmark from one source. abc/IUT is a single unlabeled point
   (correct label: quarantine); Ramanujan is `n = thousands`, **scoreable**.

2. **He is the lab's formal object in human form.** The certificate lane's central object is
   **capacity-relative one-wayness**: outputs cheap to *verify*, generator hard to *invert*.
   Ramanujan realizes it exactly — his identities are cheaply checkable (Hardy by hand; a modern CAS
   in seconds) while his *finding* process is irreproducible and not recoverable from the output. The
   Ramanujan–Hardy partnership is the literal **finder + verifier division of labor** the framework
   formalizes (search problem · verification problem · certificate-completer). Hardy's *"they must be
   true, because no one would have the imagination to invent them"* is a verifier reasoning about a
   certificate it cannot yet check — the widget's inner monologue.

3. **His corpus spans the entire verification-cost curve by itself**, with externally measured
   check-latencies (the verification-cost signal):
   - **cheap accepts** — the bulk, checked by Hardy in days;
   - **expensive accepts** — the **tau conjecture** (Ramanujan 1916) proven only by **Deligne 1974**
     as a consequence of the Weil conjectures (~58-year latency);
   - **long quarantine → eventual accept** — the **mock theta functions** (his final 1920 letter),
     coherent but not *understood* for ~80 years until Zwegers (2002) and Ono/Bringmann placed them
     as holomorphic parts of harmonic Maass forms;
   - **rejects** — the small handful he stated falsely or imprecisely, corrected in the record;
   - **duplicate-finds** — results he rediscovered unaware of prior work (e.g. Rogers–Ramanujan),
     a finding-without-novelty signal.
   A single source realizing the full accept(cheap)/accept(expensive)/long-quarantine/reject spectrum
   is a better calibration set than the instrument has ever had (the Faraday "calibrate across the
   full spectrum" discipline).

4. **He is settled and beloved — the low-risk Tycho.** Unlike abc/IUT (a live rail, active dispute,
   engaged reputations), Ramanujan's adjudication is historical and the story is the warmest in
   mathematics. A demo of "a verifier scored against a century of checking Ramanujan" carries no
   reputational hazard. abc/IUT stays as the *iconic quarantine point* on this lane's curve (§7
   Phase 2), not its spine.

---

## 2. What is honest vs. what is reach

**Honest:**
- A **scored benchmark** (the Ramanujan Verification Benchmark, RVB): does a capacity-bounded
  verifier route each claim to accept / reject / quarantine in a way that respects its own envelope,
  graded against the Berndt-sourced ground truth, **false-accept-first**?
- A **check-latency curve** placing Ramanujan's internal spectrum (and external anchors — Wiles,
  Perelman, CFSG, Atiyah–RH, abc/IUT) on a verification-cost axis.
- Ramanujan as the **human positive control** for capacity-relative one-wayness (cheap-check,
  hard-invert finder).
- A flagship demonstration of named abstention where checking genuinely exceeds the envelope.

**Reach; do not claim:**
- "Sundog verified / explains / re-proved a Ramanujan result."
- "Ramanujan had a hidden substrate / Sundog models his intuition."
- "The widget can verify number theory." (It **routes** by capacity; it does not do the mathematics.)
- Any statement that reads as a theory of genius, intuition, or the metaphysical.

---

## 3. Ratified hook language

**Safe:**
> Sundog vs. Ramanujan scores a capacity-bounded verifier against a century of community adjudication
> of Ramanujan's unproven claims — demonstrating a machine that accepts what it can cheaply check,
> flags what is known false, and *quarantines what is genuinely beyond its envelope*, instead of
> confabulating a verdict.

**Avoid:**
- "Sundog cracks / explains Ramanujan." / "Sundog's instrument channels his intuition."
- "We verified the mock theta functions." (The honest move is to **quarantine** them with the
  documented ~80-year latency.)

---

## 4. Core definitions

- **Proof-as-certificate.** A claim's proof is a certificate; the promise is that checking it is
  cheaper than finding it, and is author-independent.
- **Capacity envelope `D`.** The set of claims a given verifier can check within bounded effort and
  its own competence. Out-of-envelope claims must **quarantine**, never silently accept.
- **Capacity-bounded routing.** The verifier's job here is **not** to do number theory; it is to
  route each Ramanujan claim to `accept` (cheaply checkable, e.g. CAS-verifiable identity), `reject`
  (known false / cheaply falsifiable), or `quarantine` (checking exceeds `D` — e.g. mock thetas) —
  and to be *right about its own envelope*.
- **RVB label taxonomy** (from the adjudication record): `accept_cheap`, `accept_expensive`
  (deep machinery / long latency), `reject`, `quarantine_then_accept` (long-open, later proven),
  `duplicate_find`. Each entry carries a **check-latency** (years from statement to adjudication).
- **One-wayness witness.** Ramanujan as the human instance of cheap-verify / hard-invert (§1.2).

---

## 5. Pre-registered framing claims (about the verification record, NOT the mathematics)

1. The adjudication record yields a usable labeled benchmark (RVB) with a defensible taxonomy and
   per-claim check-latency, sourced and cited from Berndt/Andrews.
2. A capacity-bounded verifier can be **scored** on the RVB by accept/reject/quarantine routing,
   **false-accept-first** (a confident accept of a false or unfit-for-envelope claim is the worst
   outcome; quarantine is correct when checking exceeds `D`).
3. Ramanujan's internal check-latency spectrum (cheap → tau/Deligne → mock-thetas) places him as the
   labeled spine of a verification-cost curve; abc/IUT is the curve's quarantine outlier.
4. Ramanujan is a valid human positive control for capacity-relative one-wayness (finding ≫ checking;
   generator not recoverable from output).

## 6. Falsification surface — how the LANE fails (scope-violation gates)

| gate | violation | disposition |
| --- | --- | --- |
| **mysticize** | any "hidden substrate / Namagiri / more-than-we-know / theory of genius" framing | scope violation — kill (R3) |
| **do the math** | claims to *verify* a deep result rather than *route* it against the cited record | self-consistency violation — kill |
| **re-prove** | re-derives or settles an entry instead of citing Berndt/Andrews | scope creep — demote to citation |
| **benchmark leakage** | the verifier is graded with the label as input, or the claim set is tuned to flatter it | contaminated — void (anti-p-hack, the Phase-3 discipline) |
| **reduce** | the public framing reduces Ramanujan to a Sundog metaphor | hold — re-draft with respect |
| **reputational** | any public-adjacent move without owner sign-off + evidence-tier review | hold — internal only |

## 7. Roadmap (phases)

### Phase 0 — Document spine + the RVB claim set
Compile a curated claim set from the published record with **labels and check-latencies cited** to
Berndt/Andrews (and standard references for tau/Deligne, mock-thetas/Zwegers–Ono). Restate §0 at the
top. **Exit:** a sourced, labeled claim set a skeptic agrees is scope-clean; a lit-pass memo
(`docs/ramanujan/RAMANUJAN_LITPASS_MEMO.md`) mirroring the P-vs-NP memo discipline.

### Phase 1 — The RVB benchmark + scoring (the measurable receipt)
Score a capacity-bounded verifier on accept/reject/quarantine **routing** vs ground truth,
**false-accept-first**, against baselines: a naive LLM (expected to confabulate verdicts on the deep
entries) and a CAS/tool oracle (cheaply verifies the elementary identities, must abstain on the
deep ones). Headline metric: false-accept rate on `reject` + on out-of-envelope `accept_expensive`
entries; quarantine-calibration on the deep entries. **Exit:** a receipt — confusion matrix vs
ground truth, false-accept-first, with the deep entries correctly quarantined.

### Phase 2 — The check-latency / verification-cost curve
Place Ramanujan's internal spectrum on a statement→adjudication latency axis; add external anchors —
Lean/`mathlib`/Flyspeck (cheap-check pole), Wiles (gap found→closed), Perelman, CFSG, Atiyah–RH (fast
reject), and **abc/IUT as the quarantine outlier** (cross-ref `SUNDOG_V_IUT_ABC.md`). Observable
signals only (latency, independent-verifier count, formalization-exists). **Exit:** a calibrated
verification-cost curve with named poles and a live capacity-relative falsifier.

### Phase 3 — Flagship demo
Add RVB prompts to the chat gold slate (`chat/prompts/gold-boundary.jsonl` /
`gold-falsification.jsonl`); the widget (`public/js/sundog-chat-widget.mjs`) must **accept** the
cheaply-checkable identities, **flag** the known-false, and **quarantine** the deep ones with a named
latency — graded, not confabulated; routed in `chat/claim_map.json`. **Exit:** the widget passes the
RVB slate with zero false-accepts on the `reject`/out-of-envelope entries.

### Phase 4 — Public surface (GATED)
A `/ramanujan` overlay (sibling to `/unit-distance`) — "what a century of checking Ramanujan teaches
a verifier" — plus the one-wayness / finding-vs-checking card on the substrate-rhyme grid. **Hard
gate:** owner sign-off + `SCIENTIFIC_CRITERIA.md` evidence-tier review; respect-first framing. Internal
until then.

## 8. Promotion criteria

- **To Active (internal):** Phase 0–1 land scope-clean and the RVB receipt shows zero false-accepts
  on the kill entries with the deep entries correctly quarantined.
- **To Flagship demo:** Phase 3 — the widget passes the RVB slate live.
- **To Public:** Phase 4 only, with owner sign-off + evidence-tier + respect review. Never promote to
  "a result about Ramanujan's mathematics" — there is none, by construction.

## 9. Cross-references

- `docs/pvnp/SUNDOG_CERTIFICATE_PROBLEM.md` — capacity-relative one-wayness; the finding-vs-checking
  certificate framework Ramanujan instantiates.
- `docs/SUNDOG_V_P_V_NP.md` — verifier / certificate / capacity-envelope vocabulary.
- `docs/SUNDOG_V_IUT_ABC.md` — the abc/IUT quarantine; one calibration point on this lane's curve.
- `faraday/SUNDOG_V_FARADAY.md` — structural-zero / named-quarantine receipt grammar.
- `docs/SCIENTIFIC_CRITERIA.md` — evidence tiers; the discipline forbidding the math / genius lane.
- `internal/feedback/Human/REDDIT_ImOutOfIceCream_UNIT-DISTANCE.md` — substrate-rhyme / projection
  vocabulary; the finding-vs-checking card.
- `public/js/sundog-chat-widget.mjs`, `chat/claim_map.json`, `chat/prompts/gold-*.jsonl` — flagship
  demo surface.

## 10. Forbidden language

- "Sundog verified / explains / re-proved Ramanujan." / "Sundog models his intuition / genius."
- "Ramanujan tapped a hidden substrate / more than we know."
- "The widget verifies the mock theta functions / number theory."
- Any phrasing that reduces Ramanujan to a metaphor or advances a theory of mind.

## 11. One-paragraph public summary (draft, DO NOT DEPLOY)

Ramanujan handed the world thousands of results with almost no proofs; checking them took a century.
Sundog vs. Ramanujan does not re-prove any of them. It treats that century of community adjudication
as a labeled benchmark and asks a sharper question: can a verifier *know its own limits* — accept what
it can cheaply check, flag what is known false, and honestly quarantine what is genuinely beyond
reach (the mock theta functions waited eighty years) — instead of confabulating a verdict? It is the
finding-versus-checking gap made human, and a measured demonstration of the one discipline a naive
system lacks: knowing when to say *not yet*.

---

*Sundog Research Lab — SUNDOG_V_RAMANUJAN scaffold. A labeled verification benchmark, not a
mathematics lane and not a theory of genius. The verdict on the deep mathematics is, by construction,
a capacity-bounded quarantine. Internal; unpromoted; no public surface until owner sign-off. Honor
him; do not reduce him.*
