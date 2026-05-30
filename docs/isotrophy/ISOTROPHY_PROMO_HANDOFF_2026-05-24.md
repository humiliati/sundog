# Isotrophy Promo Handoff (2026-05-24)

Status: **SUPERSEDED FOR CLAIM COPY as of 2026-05-30.** This handoff remains as
historical provenance for the v0.3-v0.9 pause. Use
`docs/isotrophy/kfacet/kfacet_isotrophy_program_pause.md` and
`docs/isotrophy/kfacet/kfacet_v11_m3_conditional_vf_rank_form.md` for current
copy: the v0.10/v0.11 frontier now says velocity-fraction stratifies stability
within fixed `m3` strata (`AUC_cond = 0.6783`, exact `p = 2.046e-7`) while
failing as a mass-marginal held-out predictor (`AUC = 0.4125`).

Original 2026-05-24 status: ready for promo-team execution. The isotrophy
program paused at end-of-v0.9 on 2026-05-24; three public-facing pages need
updating to reflect the seven-chapter arc, the load-bearing substantive positive
at v0.7, and the program-pause framing.

Audience: promo team writers, public-copy reviewers. Anyone who
will edit `index.html`, `threebody.html`, or `isotrophy.html` to
reflect the current state.

Primary internal source documents (read before editing public copy):

- `docs/isotrophy/kfacet/kfacet_isotrophy_program_pause.md` -- the
  authoritative program-pause document with seven-chapter summary,
  load-bearing findings, methodology surface, and eight reopening
  avenues for lab initiates.
- `docs/isotrophy/kfacet/kfacet_v07_writeup.md` -- the v0.7
  qualified-positive chapter close (the load-bearing positive
  lives here).
- `docs/isotrophy/kfacet/kfacet_v09_writeup.md` -- the v0.9
  structural-negative chapter close (the most-recent chapter; the
  monotone-increasing meta-finding lives here).
- `docs/isotrophy/sundog_v_isotrophy.md` -- chapter-by-chapter dated log
  (cross-reference for every claim below).

If any line below conflicts with the internal docs, the internal
docs win.

## The Headline

> The isotrophy program (Sundog vs Li-Liao supplementary-B
> piano-trios) paused 2026-05-24 at end-of-v0.9 after seven
> sequential pre-registered chapters with seven distinct
> chapter-close types. The chain produced one load-bearing
> substantive positive (v0.7a' chi^2 = 16.43, branch-independent,
> on 250 of 273 catalog rows), one monotone-increasing
> meta-finding (v0.9a), and a comprehensive methodology surface
> (ten locked disciplines). Bandwidth has redirected to
> three-body Phase 15+ (survival pocket).

That headline is publishable as written. It's accurate at the
load-bearing level and respects the chain's discipline.

## The Seven-Chapter Arc (Public-Friendly Version)

```text
v0.3   domain-of-applicability:
       Gamma_i is D_3-equivariant; the strict G.2 piano-trio
       catalog yields 20 structural zeros + 1 named quarantine
       (O_617). Daughter catalog (supp-B) is Z_2-or-smaller --
       outside the original mechanism's domain.

v0.4   structural-negative on Z_2 shadow:
       The natural Z_2-equivariant mechanism candidates fail.

v0.5   projection-limit:
       The catalog branch hash on (m_3, z_0) stratifies in-sample
       at p ~ 1e-7 but does NOT generalize across held-out mass
       bins.

v0.6   conditional-independence:
       The continuous catalog-coordinate (E, |L|) shadow passes
       the in-sample audit but is essentially a re-projection of
       the v0.5 branch hash; no independent stability content.

v0.7   integration-attrition + qualified-positive on restricted
       domain (THE LOAD-BEARING POSITIVE):
       The Floquet velocity-fraction direction shadow stratifies
       stability on 250 of 273 catalog rows at chi^2 = 16.43,
       p ~ 9.3e-4, branch-independent. 23 rows could not be
       evaluated at the locked variational precision; the
       restricted-domain positive is the chain's first
       non-branch-aligned substantive finding.

v0.8   structural-negative on unsigned direction-purity:
       The hypothesized symmetric U-shape mechanism fails;
       the v0.7 signal lives in signed direction, not unsigned
       distance from mixed.

v0.9   structural-negative on U-shape + monotone-increasing
       meta-finding:
       Under physical zones (not data-driven quartiles), the
       actual physical pattern is monotone-increasing in
       velocity-fraction (positional 11% S < mixed 34% S <
       velocity-heavy 44% S). chi^2 = 7.42 does not clear the
       registered p = 0.01 floor, but the trend is real and
       physically interpretable.
```

The program paused -- it did not retire. Eight concrete reopening
avenues are documented internally; the first (a Jonckheere-Terpstra
trend test on v0.9a data) is low-effort and likely to land a clean
follow-on positive.

## The Load-Bearing Positive (With All Its Caveats)

The v0.7a' result is the publishable substantive finding. **Every
public claim about it must include the caveats**:

```text
WHAT IT IS:
  The Floquet velocity-fraction (vf) of the largest-real-part
  eigenvector of the monodromy operator, projected as a geometric
  direction in phase space after center-of-mass reduction and
  mass-weighted normalization, stratifies stability on the
  Li-Liao supplementary-B piano-trio catalog.

  Numerically: chi^2 = 16.43 against chi-squared(3) at p = 0.01
  (critical 11.34); p ~ 9.3e-4; quartile contingency 49% / 18% /
  29% / 43% S-fraction across Q1 (positional-leaning) through Q4
  (velocity-leaning).

WHAT IT IS NOT:
  - NOT a catalog-wide result. The variational integrator at
    rtol = atol = 1e-12 attrited 23 of 273 rows (8.4%) at the
    long-period high-m_3 regime; the result holds on the 250
    analyzable rows only.
  - NOT a held-out predictor. v0.7b (held-out) was not licensed
    under the catalog-wide attrition verdict; the in-sample
    chi-squared has not been promoted to a generalization claim.
  - NOT a U-shape mechanism. v0.9a confirmed the apparent U-shape
    was a quartile-boundary artifact; the actual physical pattern
    is monotone-increasing in vf.
  - NOT a circularity-free mechanism interpretation. The
    eigenvalue ordering (largest real part) is used to disambiguate
    which eigenvector to extract; the v0.7 form lock explicitly
    addresses this with a non-circularity sentence (Floquet
    eigenvectors as geometric directions only, never as growth
    rates).

LANGUAGE GUARDRAILS:
  - SAY: "stratifies stability on the analyzable subset"
  - SAY: "branch-independent (alignment 0.70)"
  - SAY: "first non-branch-aligned positive in the chain"
  - SAY: "the monotone-increasing pattern (velocity-heavy =
          more stable) matches the broader physical thesis"
  - DO NOT SAY: "predicts stability"
  - DO NOT SAY: "the catalog confirms"
  - DO NOT SAY: "the sundog theorem proves"
  - DO NOT SAY: "U-shape mechanism" (v0.9 retired this)
```

The chain has been disciplined; the public copy should be too.

## Per-Page Update Priorities

### `index.html` -- LOW PRIORITY (mostly fine)

**Current state**: The isotrophy pillar (lines ~1964-2032) reads
"Isotrophy K_facet v0.3h" and quotes the v0.3h verdict directly
("20 of 21 strict G.2 single-curve choreographies returned
structural-zero receipts; O_617 was quarantined"). This is
accurate at the v0.3h level and honestly framed.

**Update suggestion (optional, not urgent)**:
Add one breadcrumb sentence after the v0.3h headline:
> "The v0.3h verdict is the first chapter of the seven-chapter
> isotrophy program; the program paused at v0.9 on 2026-05-24
> with a substantive positive at v0.7 (see the `/isotrophy`
> page for the chain summary)."

If you'd rather leave it alone, that's defensible -- the v0.3h
pillar is a coherent standalone claim and the chain summary lives
on the dedicated isotrophy page anyway.

### `threebody.html` -- MEDIUM PRIORITY

**Current state**: The "Catalog sidecar — 21 + 4 choreographies"
section (lines ~1441-1647) cites the v0.3h verdict and the
sigma_3 detector reconciliation. The section is accurate but
treats the isotrophy work as ENDING at v0.3h.

**Specific edit targets**:

1. **Sources section (lines ~1629-1645)**: add a Cross-reference
   line for the seven-chapter pause:
   > "The K_facet audit chain extended through v0.9 after v0.3h;
   > the program paused 2026-05-24 with a substantive positive
   > at v0.7 (Floquet velocity-fraction direction shadow,
   > chi^2 = 16.43 on 250 of 273 catalog rows, branch-independent).
   > See `/isotrophy` for the full chapter summary."

2. **Catalog sidecar lede (lines ~1443-1447)**: expand the framing
   from "K_facet v0.3h verdict" to "K_facet v0.3-v0.9 audit chain"
   with the v0.3h verdict as the leading line and a single sentence
   noting the program paused with a substantive positive at v0.7.

3. **Optional: brief callout box** (similar pattern to the
   existing iso-crossref) pointing readers to the v0.7
   positive and the methodology surface.

**Tone**: this page is for technical readers; you can use the
chi-squared values directly. Don't include the caveats in detail
here -- link to `/isotrophy` for the full version.

### `isotrophy.html` -- HIGH PRIORITY

**Current state**: The page is narrowly scoped to v0.3h. The
hero, lede, status ribbon, and load-bearing statement all reflect
the v0.3h verdict. None of v0.4-v0.9 is mentioned.

**Specific edit targets**:

1. **Hero / lede (lines ~425-435)**: keep the v0.3h verdict as
   the leading claim, but add a paragraph immediately after
   situating it:
   > "v0.3h is the first chapter of a seven-chapter pre-registered
   > program (v0.3 through v0.9), now paused as of 2026-05-24.
   > The chain produced one load-bearing substantive positive at
   > v0.7 (Floquet velocity-fraction direction shadow,
   > chi^2 = 16.43 on 250 of 273 catalog rows, branch-independent)
   > and a comprehensive methodology surface (ten locked
   > disciplines)."

2. **Status ribbon (line ~435ish)**: expand from
   `v0.3h | 2026-05-22 | 20 structural-zero receipts | ...`
   to add a second ribbon below:
   `v0.7 (load-bearing positive) | v0.9 close | program paused 2026-05-24`

3. **Post-verdict section (after line ~475)**: insert a NEW
   "Program chapter arc" section between the v0.3h load-bearing
   statement and the "what's next" footer. Use the seven-chapter
   summary above (in this handoff) verbatim or paraphrased.
   Include all SEVEN chapter-close types as distinct entries.

4. **Add a "Load-bearing positive" callout**: a dedicated section
   for the v0.7a' result, with the caveats locked in copy. Use
   the exact language guardrails above.

5. **Add a "Reopening avenues" footer**: a brief note that the
   program paused with eight concrete reopening avenues for lab
   initiates, with a link to the internal pause document
   (`docs/isotrophy/kfacet/kfacet_isotrophy_program_pause.md`).
   Don't list all eight -- one or two examples (e.g., "a
   Jonckheere-Terpstra trend test on v0.9a is the low-effort
   first follow-on") is enough.

**Tone**: this is the destination page for anyone curious about
isotrophy. Be technical but explain the projection-language
vocabulary briefly. The page should make the load-bearing positive
visible without overstating it.

## Cross-References for Verification

Every public claim in the proposed updates above traces back to
one of:

```text
docs/isotrophy/kfacet/kfacet_v03h_writeup.md     v0.3h verdict
docs/isotrophy/kfacet/kfacet_v04_writeup.md      v0.4 close
docs/isotrophy/kfacet/kfacet_v05_writeup.md      v0.5 close
docs/isotrophy/kfacet/kfacet_v06_writeup.md      v0.6 close
docs/isotrophy/kfacet/kfacet_v07_writeup.md      v0.7 LOAD-BEARING POSITIVE
docs/isotrophy/kfacet/kfacet_v08_writeup.md      v0.8 close
docs/isotrophy/kfacet/kfacet_v09_writeup.md      v0.9 close
docs/isotrophy/kfacet/kfacet_isotrophy_program_pause.md
                                                program pause + reopening avenues
docs/isotrophy/sundog_v_isotrophy.md                      dated chapter-by-chapter log

results/isotrophy/k-facet-v07a-velocity-fraction-audit/manifest.json
                                                load-bearing chi^2 = 16.43 receipt
results/isotrophy/k-facet-v07a-prime-restricted-scope/manifest.json
                                                restricted-scope PASS receipt
results/isotrophy/k-facet-v09a-signed-vf-three-zone/manifest.json
                                                v0.9a fails + monotone meta-finding
```

If you want a number, formula, or claim that isn't in one of these
files, run it past me (or a research-side reviewer) before
publishing. The chain has been disciplined; the public surface
should mirror that discipline.

## What Promo Should NOT Do

- Do NOT remove the v0.3h pillar from `index.html`. It is still
  load-bearing for the "audit chain catches itself working" pitch.
- Do NOT promote the v0.7a' result to a catalog-wide claim. The
  attrition is permanent on the receipt; bypassing it in the copy
  would invalidate the chain's discipline.
- Do NOT use the word "predict" for any v0.7 result. The held-out
  predictor (v0.7b) was not licensed. The result is an in-sample
  stratification finding, not a generalization claim.
- Do NOT describe the program as "retired." It is **paused** with
  explicit reopening avenues; the language matters.
- Do NOT remove the O_617 quarantine mention from any page. It is
  the methodological touchstone that proves the audit chain
  catches itself working.

## Suggested Sequence of Public-Copy Work

1. **First**: update `isotrophy.html` (high priority; the
   destination page for the chain narrative). Land the seven-
   chapter summary + load-bearing positive callout + reopening
   avenues footer.
2. **Second**: update `threebody.html` cross-references (medium
   priority). Link to the new isotrophy.html sections.
3. **Third**: optionally add a one-sentence breadcrumb to
   `index.html` (low priority).
4. **Fourth**: run a public-copy review pass for the language
   guardrails (no "predicts", no "catalog-wide", no "U-shape
   mechanism", no "retired", etc.).

Estimated time: 2-4 hours for an experienced public-copy editor;
6-8 hours if also writing a new chapter-arc explainer.

## Open Questions for Promo Team

- Does isotrophy.html want a visual element for the seven-chapter
  arc (timeline diagram, contingency-table reproduction, vf
  distribution histogram)? The v0.7a' contingency makes a clean
  bar chart; the v0.9a monotone-increasing pattern makes a clean
  3-bar chart. Both could land within the existing page aesthetic.
- Should the load-bearing positive get its own dedicated subpage
  (e.g., `/isotrophy/v0.7-positive`) for SEO purposes? Probably
  not necessary; the section on `/isotrophy` is sufficient.
- Threebody.html's catalog-sidecar gallery is interactive; should
  v0.7-positive context be integrated into the gallery's modal
  copy, or kept in the sources/footer only? Recommend
  sources/footer for now to avoid scope creep on the gallery
  feature.

## Sign-Off Checklist for Each Page Update

```text
[ ] All chi-squared values cited match an internal source doc.
[ ] All caveats from "Load-Bearing Positive" section are present
    where the v0.7a' result is mentioned.
[ ] No "predicts", "catalog-wide", "U-shape mechanism", or
    "retired" language.
[ ] Links to internal pause document are functional.
[ ] O_617 quarantine narrative preserved.
[ ] Tone matches the chain's discipline (precise, hedged where
    appropriate, no overclaim).
[ ] A research-side reviewer (the principal, or me) has signed
    off on the final copy before deploy.
```

---

Handoff complete. Public surfaces will reflect the load-bearing
positive without overstating it. The chain's discipline carries
through to the public copy by design.
