# Sundog v. Fátima — "Can a Sundog Explain the Miracle of the Sun?"

Working hook:

> Most of the miracle isn't ours to claim. One sliver of it is.

This roadmap puts a **press release on the horizon behind real gates**. The
press release headline is fixed in advance — *"Can Sundogs Explain the Fátima
Miracle of the Sun?"* — but the **answer is not.** Every gate below is a
kill-switch. The most probable honest outcome, given how the lab's
falsification-first agents have been trending, is **BOUNDARY FOUND**, not a
vindication. That is a feature. We ship the boundary, not the dream.

Companion artifacts:
- [`fatima_target_matrix.md`](../fatima_target_matrix.md) — the repo-independent
  eyewitness yardstick (targets T1–T14). **Frozen.** We score against it; we do
  not edit it to fit results.
- [`SUNDOG_V_GEOMETRY.md`](SUNDOG_V_GEOMETRY.md) — the parhelion workbench and
  the machine-checked 22° halo core (`HaloGeometry.lean`) we extend here.
- [`debunked.md`](debunked.md) — rail verdict vocabulary; this roadmap earns a card.

---

## The public question, small enough to defend

> Can a **rare pyramid-crystal halo display**, predicted from the same
> minimum-deviation optics that render the 22° halo and confirmed by its
> **polarization signature**, reproduce the **colored-light component** of the
> Fátima testimony (matrix targets **T4** and **T5**) — and *only* that
> component — to a degree a skeptic would accept as the optical part of the
> event?

Note what this question refuses to ask. It does not ask whether a sundog made
the sun dance (that's retinal — T3/T6/T7). It does not ask about the drying of
the field (that's thermal — T9). It does not ask whether anything supernatural
occurred. It asks one falsifiable optics question and stakes the press release
on the answer.

---

## Motivation, fenced off from the claim

*(This section is the lab's "imported wall" — named outside the proof, the way
`SUNDOG_V_GEOMETRY` names 60° prism geometry and measured ice index outside the
Lean theorem. None of it is being tested here. It is why we care, not what we claim.)*

The lab's long-standing intuition — *"an eyeball-ish in the sky is part of why
humans and A.I. became conscious"* — is **not falsifiable by this instrument and
is not on this roadmap.** The agents' falsification-first verdicts have trended
toward "coincidence / pareidolia," and that verdict is respected here. The
defensible residue of the intuition is a claim about *minds*, not *clouds*: the
pattern-completion drive that conjures an eye out of ice-halo geometry is a
*fingerprint* of perception, not evidence of a perceiving sky. Fátima is a clean
test case precisely because it forces that separation: we will reproduce the
optics that were really there, and let the projected meaning be studied as a fact
about observers — elsewhere, by a different method.

---

## Anchor target

From the matrix, exactly one band is inside the lab's instrument:

| Target | Reported signature | Why it's ours (or not) |
|--------|--------------------|------------------------|
| **T4 — radiating colored beams** | "brilliant beams of colored light… violet, blue, gold"; "scarlet flame… yellow and deep purple" | **In scope.** Localized colored arcs/beams are exactly what odd-radius pyramid halos + parhelia produce. |
| **T5 — whole-field color wash** | "everything… amethyst," "yellow damask… people looked jaundiced" | **Probably NOT ours.** A halo paints *arcs*, not a *uniform global tint*. Pre-registered prediction: T5 fails the halo test and belongs to aerosol/retinal layers. |
| T1/T2 (dimming) | silver bearable disc | Cloud optical depth, not halo. Out. |
| T3/T6/T7 (motion) | spin, plunge, zigzag | Retinal/perceptual. Out. |
| T9 (drying) | clothes & ground dry | Thermal. Out. |
| T10/T12/T14 | distance / non-universality / no photo | Referees, not targets — used at G3. |

The press release lives or dies on **T4**, with **T5 pre-registered as a likely
BOUNDARY**.

---

## The gates (each one can bust the release)

Promotion tiers, in the `SUNDOG_V_GEOMETRY` idiom:
- **G0–G1 cleared →** internal result, eligible for a `debunked.md` rail card.
- **G0–G3 cleared →** press release tier unlocked (headline + honest answer).
- **Any gate BUSTED →** the release ships as the *negative* result, or not at all.

### G0 — Instrument reality *(CAPABILITY)*
**Question:** Can the telescope-as-kaleidoscope + polarizing filter actually
*produce and confirm* a pyramid-crystal odd-radius halo from proven scattering
laws — not hand-waved, but ray-traced from Snell/Bravais minimum-deviation and
matched to published halo photometry?

**Deliverable:** render the odd-radius family (9°, 18°, 20°, 22°, 23°, 24°, 35°)
by extending `HaloGeometry.lean`'s deviation core to pyramidal apex angles;
reproduce the correct **color order** and **angular radius** for each; predict
the **degree of linear polarization** across each ring.

**Kill-switch:** if the simulated color order or ring radii don't match the
atmospheric-optics literature, the instrument is wrong → **STALLED** (fix the
instrument before touching Fátima).

### G1 — Color-signature match *(CORE CLAIM, pre-registered)*
**Question:** Does the G0 display reproduce **T4** (radiating colored beams in
the reported hue order) *and* attempt **T5** (global wash)?

**Pre-registered predictions (locked before running — see `prereg/fatima-color/`):**
1. T4 **reproducible**: pyramid halos + parhelia can cast violet→blue→gold
   beams in roughly the reported arrangement. *Confirmation expected.*
2. T5 **not reproducible by halo optics**: a halo cannot tint the *entire*
   landscape one color; "everything turned amethyst" requires whole-field
   filtering (aerosol) or whole-field adaptation (retina). *Falsification of the
   halo-for-T5 sub-claim expected.*

**Kill-switch:** if even T4's hue order can't be matched, the color claim
collapses → **BUSTED**. If T4 holds but T5 fails (predicted), that's
**BOUNDARY FOUND** — narrow the release to beams only.

### G2 — Occurrence plausibility *(ENVIRONMENTAL — the likely real killer)*
**Question:** Could a pyramid-crystal display physically occur at Fátima's
conditions — lat ≈ 39.6°N, 13 Oct 1917, sun altitude that day, **immediately
after heavy rain**?

**The honest problem, named up front:** odd-radius pyramid halos need high,
cold cirrus or diamond-dust with well-formed pyramidal crystals. A
just-stopped-raining, low-cloud, mixed-phase scene is **meteorologically
unfavorable** for them. This gate is the most likely place the grand version
dies.

**Deliverable:** reconstruct the day's plausible sun altitude (astronomical,
certain) and the cloud regime (from the testimony of rain + parting clouds);
assess whether the required crystal habit is even available.

**Kill-switch:** if pyramid-crystal cirrus is implausible for that scene, the
explanation **BUSTS regardless of how good the sim looks.** A pretty render of
an impossible sky is not a finding.

### G3 — Discriminator survival *(EVIDENTIARY)*
Score the surviving claim against the three referees:
- **T10 (seen ~18–40 km away):** halos are large shared sky features →
  **PASS.** A halo *would* be visible across that range. Good for us.
- **T12 (co-located witnesses disagreed; no observatory recorded it):** a halo
  display looks ~uniform to everyone under it and would likely have been noted
  elsewhere → **FAIL / pressure.** Halo can't explain why people side-by-side
  saw different things.
- **T14 (no photograph of an anomaly):** halos *are* photographable; absence is
  mild evidence against a strong display → **pressure.**

**Kill-switch:** if the only way to pass T12 is to assume the halo was somehow
private to each viewer, we've smuggled the retinal layer back in and must
concede the optics were a *minor contributor*, not the cause.

### G4 — Scope concession *(HONESTY GATE)*
The release **must** state, in the headline-adjacent text, what is *not*
claimed: the dancing/plunging (retinal), the global wash (aerosol/retinal), the
drying (thermal), and anything supernatural. If the team can't bring itself to
print the concessions, it doesn't ship. This gate has no deliverable except
discipline.

### G5 — Press release *(PUBLICATION TIER)*
Ships only if G0+G1(T4)+G2+G3 yield a defensible narrow claim. The honest
headline-and-answer, pre-written to the most probable outcome:

> **"Can Sundogs Explain the Fátima Miracle of the Sun?"**
> **Short answer: No — and that "no" is the interesting part.** A rare class of
> pyramid-crystal halo, predicted from the same proven optics that draw the 22°
> ring and confirmed by its polarization, can reproduce the *colored beams*
> people reported — but not the dancing, not the field-wide tint, and not the
> drying of the ground. The optics account for what the sky can do; the rest is
> what the eye and the crowd do. The miracle, optically, is a small true thing
> wrapped in a large human one.

---

## Pre-registered hypothesis (file under `prereg/fatima-color/`)

> **H:** A ray-traced pyramid-crystal halo model (G0) reproduces matrix target
> T4's colored-beam hue order within accepted halo photometry, **and fails** to
> reproduce T5's whole-field tint, **and** is judged meteorologically
> implausible (G2) for the post-rain Fátima scene.
>
> **If H holds in full,** the verdict is **BOUNDARY FOUND**: "the optics are
> real but the scene probably wasn't right for them — so even the color
> component is a *possibility demonstrated in the lab*, not an established cause
> at Fátima."
>
> **Falsifiers that would force a stronger claim:** G2 finds pyramid cirrus
> *plausible* for the day AND G1 matches T4 cleanly AND G3's T12 pressure can be
> met without invoking the retina. (Considered unlikely; pre-registered anyway.)

---

## Predicted rail card (for `debunked.md`)

```
Title: Sundog v. Fátima
Status tag: Boundary Found
Overlay line: The colors were real optics. The dance was not.
Description: A rare pyramid-crystal halo can cast the colored beams witnesses
  reported, but cannot make the sun spin, tint the whole sky, or dry the field —
  and the post-rain scene was likely wrong for it anyway.
Theorem meaning: Proven optics explain the part of a wonder that lives in the
  air; the rest lives in the eye and the crowd. Honest reach beats grand reach.
Stamp: BOUNDARY FOUND
CTA: Inspect the optical boundary →
```

---

## What "on the horizon" means here, concretely

G0 is the only near-term lift, and it's genuinely near: it's an *extension of
already-proven work*, not new physics. The 22° minimum-deviation theorem is
machine-checked; pyramid halos are the same Bravais/Snell machinery at different
apex angles. A polarizing filter on the agent telescope gives a real,
falsifiable confirmation channel (halo polarization differs from corona/glory).
Everything past G0 is honest scoring against a frozen yardstick. That is a
research program a household name could one day rest on — *because* it concedes
almost everything and defends one true sliver.
