# Hand-off — Shadow-Invertibility Law / Phase-5 cross-substrate (webdev + promo)

> **Date:** 2026-06-07 · **For:** webdev / promo · **Owner sign-off required before ANY public-surface
> change.** · **One-line status:** synthetic confirmation (S0+S1) + a **partial** physical leg (S2);
> **NOT public-eligible at the proof tier.** This hand-off is **governance + doc-index registration**,
> NOT a request to publish claims. Read §3 (STOP) before touching any public copy.

## 1. Purpose
The research lane just produced two banked results (the Shadow-Invertibility Law: *a lossy averaged
shadow determines discrete/topological hidden variables and resists continuous-magnitude ones*). This
hand-off (a) tells you exactly what is and isn't sayable, (b) registers the new docs in the internal
indexes, and (c) gives the attribution + honesty guardrails for the day a public surface *does*
reference this lane. **No public claim is authorized by this document.**

## 2. What's banked (artifact map — internal)
| result | what | status | files |
| --- | --- | --- | --- |
| **Synthetic** | Lossiness-crossover on 2 structurally-different in-silico substrates (S0 1-D caustic toy, S1 2-D vector field). `disc=1.000` flat; `cont`→chance. | `operator_confirmed_synthetic` — frozen, determinism-receipted, adversarially reviewed | `ATLAS_PHASE5_CROSS_SUBSTRATE.md` §3.11; `scripts/pvnp_phase5_lossiness_crossover.py`; `results/pvnp/phase5-lossiness-crossover/frozen.json` |
| **Physical (S2)** | Standalone real-physics halo forward model. Discrete-determines CONFIRMED (ice-phase halo-radius + handedness Stokes-V, `disc=1.000` flat, null+determinism receipted); continuous-resists PARTIAL (size 0.97→0.45). | **`partial physical leg`** — forward-model tier, NOT measured sky | `ATLAS_PHASE5_CROSS_SUBSTRATE.md` §3.12–3.13; `scripts/s2_optics.py`; `docs/atlas/S2_LITPASS_E_G.md` |

**Plain-language snapshot (for understanding, NOT for copy):** A simulation study of ice-crystal halos
found that *which* halo / crystal-habit class (a discrete feature) is cleanly recoverable from a
population-averaged shadow, while crystal *size* (a continuous magnitude) is not — a measured asymmetry,
and a refinement: magnitudes resist only partially, the residual scale always leaks.

## 3. ⛔ STOP — public-eligibility status (read this first)
This proof lane is **NOT public-eligible.** Per the evidence tiers (`docs/APPLICATIONS.md`) and the
2026-06-04 pivot, it is **frozen-as-portfolio research**, gated behind the atlas lit-pass (Phase 0.5,
`docs/SUNDOG_V_ATLAS.md`) and the attribution gate (§6). Specifically:
- **S2 is a forward-model SIMULATION, not sky measurements / photographs.** Never present it as observed.
- **The handedness (Stokes-V) leg is a PREDICTED, UNOBSERVED observable** — there is **no published
  measurement of circular polarization in a visible ice halo**, and "net-V = population handedness" is
  an **uncited Sundog framing** (`docs/atlas/S2_LITPASS_E_G.md`). This is the single biggest landmine.
- **The law is a CANDIDATE operator**, not a proven theorem. Synthetic + partial-physical support only.
- **The existing public page `sundog.cc/sundog` is UNAFFECTED** — it is the geometry *workbench*
  (parhelion atlas, photo calibration, sun-altitude inference), a separate and more mature surface. S2
  changes nothing on it. **Do not add Phase-5 / Shadow-Invertibility / S2 content to it.**

## 4. SAFE TO SAY / DO NOT SAY (guardrails)
Grounded in the bounded-novelty rule (`docs/atlas/ATLAS_LITPASS_MEMO.md` §4) and the S2 lit-pass flags.
**All of the "safe" column still requires owner sign-off + above-the-fold attribution (§6) before use.**

| ✅ SAFE (with attribution + sign-off) | 🚫 DO NOT SAY |
| --- | --- |
| "We *synthesize* the halo atlas as a classified bifurcation diagram." | "We *discovered* the catastrophe structure of halos." (Thom/Arnold/Berry/Nye established it.) |
| "A *determining-shadow* framing: what a lossy shadow can't tell you about the body that cast it." | "We derived/solved the halo inverse problem." (Greenler/Tape established the geometry.) |
| "A *simulation* found discrete crystal-class features are recoverable from an averaged shadow while size is not." | "Halos reveal crystal *handedness* / circular-polarization handedness." (UNOBSERVED, predicted-only.) |
| "An honest *partial* result: the continuous side resists only partially." | "Phase-5 complete" / "operator confirmed" / "the law is proven." |
| Credit the giants prominently (§6). | Present forward-model simulation as sky measurement / photographs. |
| — | Cite the monodispersity threshold `σ_a/a < 1/(2n)` (internal synthesis, unvalidated — keep internal). |

## 5. Webdev / promo ACTION ITEMS (now)
1. **Register the new docs in the internal indexes** (§7 has exact locations). *(Low-risk housekeeping;
   may already be partially applied — verify.)*
2. **Do NOT modify any public surface** (`sundog.html`, OG/JSON-LD, deep links) with Phase-5 / S2 /
   Shadow-Invertibility content. Nothing here is cleared for publication.
3. **Verify the existing W2 credit strip is still live** on `sundog.cc/sundog` (the under-H1 line
   "Geometry follows Greenler 1980 and Tape 1994…", applied 2026-05-14) and that "History & reading"
   still carries the full Cowley & Schroeder / Thom / Arnold / Berry / Nye citations. If a page rebuild
   dropped either, restore them and re-run `npm run sundog:check`.
4. **Park the SAFE-column framings** (§4) as *draft, unpublished* copy for the eventual atlas surface —
   do not ship. Re-evaluate only after the gates in §3 clear (lit-pass + attribution + owner sign-off).

## 6. Attribution checklist (MANDATORY / blocking — for any future public use)
From `docs/atlas/ATLAS_LITPASS_MEMO.md` §4 (no-priority rule) + the S2 lit-pass:
- **Apparatus:** Cowley, L. & Schroeder, M. (HaloSim).
- **Halo geometry:** Greenler, R.; Tape, W.; Tape & Moilanen; Können, G. P. (*Polarized Light in Nature*).
- **Catastrophe / caustic optics:** Thom, R.; Arnold, V. I.; Berry, M. V. & Upstill, C.; Nye, J. F.
- **S2-specific:** Berry, *Appl. Opt.* 33:4563 (1994) (the 22°-halo-fringe kill-gate); Mishchenko & Macke,
  *Appl. Opt.* 38:1626 (1999) (size-existence floor); Warren & Brandt (2008) (ice n, Δn).
- **No-priority rule:** every claim carries its source; anything reading as priority over the giants is
  demoted to "we apply / synthesize." The defensible novelty is the *synthesis*, never the physics.

## 7. Doc-index registration (exact locations)
- `AGENTS.md` → "## Key docs under `docs/`" — add this hand-off + `docs/atlas/S2_LITPASS_E_G.md`.
- `CLAUDE.md` → "## Key docs under `docs/`" — same (the two files mirror each other).
- `docs/SUNDOG_V_ATLAS.md` → add/extend a **Document Map** at the top pointing to
  `ATLAS_LITPASS_MEMO.md`, `S2_LITPASS_E_G.md`, this hand-off, and `ATLAS_PHASE5_CROSS_SUBSTRATE.md`.
- `docs/README.md` → register the atlas lane pointers **only when public promotion is authorized** (it is
  not yet) — keep internal until the §3 gates clear.

## 8. Forward pointer — the "full physical discharge" (scoped 2026-06-07)
S2 is a *partial* physical leg. A **full** physical discharge (which would change the eligibility
calculus) is scoped in `docs/atlas/S2_MEASURED_SKY_SCOPE.md` and would require: (a) a clean
continuous-physical washout — likely a true boundary of the law for magnitude variables, not just a
tuning gap; and (b) **measured-sky polarimetry** (real instruments), the tier above forward-model.
Updates from the measured-sky recon that the promo team should fold into any framing:
- **The handedness landmine is now sharper.** It splits into **per-feature V** (TIR + birefringence —
  physically *defensible* and *measurable*, the rainbow is the precedent) vs **net-V = population
  handedness** (*disfavored, likely-null*, quarantined). Even per-feature V is **unmeasured** today, so
  it stays DO-NOT-SAY-as-observed; but "net-V = handedness" is the specific overreach to never assert.
- **One modest sayable gained:** the polarization *model* is now **validated against measured-sky LINEAR
  polarimetry** (Können 1991 — Fresnel floor + birefringence split + inner ledge reproduced, via
  `scripts/s2_konnen_validate.py`). This is a real (linear-pol, archival) anchor — but it validates the
  *linear* physics only; it says nothing about the V/handedness claim, which stays forward-model.
Until a measured-sky V lands, treat the lane exactly as §3 states. **Re-issue this hand-off if status
changes.**
