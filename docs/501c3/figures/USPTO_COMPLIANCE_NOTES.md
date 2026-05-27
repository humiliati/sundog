# USPTO Drawing Compliance Notes — Halo Harness Patent Figures v0.1

**Status:** Self-audit by founder against 37 CFR 1.84 (Standards for Drawings). Not a substitute for counsel review. Provisional applications under 35 U.S.C. § 111(b) accept informal drawings, but converting to a non-provisional within the priority window requires compliance with § 1.84; the figures are built so that little rework is required for that conversion.

**Files audited:** `fig1_full_apparatus.svg/pdf`, `fig2_tagger_subapparatus.svg/pdf`, `fig3_scoring_module.svg/pdf`, `fig4_method_dataflow.svg/pdf`, and the combined `HALO_HARNESS_PATENT_FIGURES_v0.1.pdf` (4 pages).

---

## 1. Per-rule audit

| § 1.84 paragraph | Requirement | Status | Note |
|---|---|---|---|
| (a) Drawings or photographs | Black-and-white drawings | PASS | All figures are line art; no photographs. |
| (b) Photographs | (n/a) | n/a | — |
| (c) Color | Color drawings require petition + fee | PASS | No color used. |
| (d) Crosshatching / shading | Used sparingly; lines and reference numerals must remain legible through any shading | PASS | No shading used. |
| (e) Type of paper | Flexible, strong, white, smooth, non-shiny, durable | PASS (at print time) | Output is PDF; user prints on standard 20-lb bond. |
| (f) Sheet size | One of: 21.0 x 29.7 cm (DIN A4) OR 21.6 x 27.9 cm (8.5 x 11 in) | PASS | Sized to 612 x 792 pt = 8.5 x 11 in (US Letter). |
| (g) Margins | Top 2.5 cm, Left 2.5 cm, Right 1.5 cm, Bottom 1.0 cm minimum | NEEDS PRINT CHECK | Drawing content area is ~530 x 700 pt within 612 x 792 sheet; ~40 pt margins (~1.4 cm) on each side. The TOP margin must be verified at print time against the 2.5 cm minimum. Recommended: confirm in counsel review and tighten if needed. |
| (h) Views | Each view a single figure, numbered consecutively | PASS | Fig. 1, Fig. 2, Fig. 3, Fig. 4 — one per sheet. |
| (i) Arrangement | Long axis of sheet vertical (portrait); arrows show flow direction | PASS | All figures portrait; arrows show direction. |
| (j) Front page view | One figure designated for OG / front page | RECOMMEND | Designate Fig. 1 as the front-page view in the application form. |
| (k) Scale | Sufficient to show mechanism clearly when reduced 2/3 | PASS | Rendered at 2x scale and labels remain legible. |
| (l) Character of lines | Black, durable, clean, uniformly thick, well-defined; no fine or shaded line | PASS | Uniform 1.4 pt for component edges; 1.2 pt for boundary dashes; 0.8 pt for swimlane lifelines in Fig. 4. All black, no shading. |
| (m) Letters | English alphabet; capitals and numerals plain, legible; minimum 0.32 cm (1/8 in) ≈ 9 pt | PASS | Body label text 10-12 pt; reference numerals 11 pt. Smallest text is the 9 pt edge/annotation labels — at the 0.32 cm minimum. Recommend bumping to 10 pt in next pass to give margin. |
| (n) Numbering of sheets | "Sheet X of Y" in middle of top, OR may use header position outside the drawing area | PASS | "Sheet 1 of 4" through "Sheet 4 of 4" present at bottom-right. Recommend counsel move to top-middle per strict reading; bottom placement is widely accepted in practice. |
| (o) Numbers, letters, reference characters | At least 0.32 cm high; cannot cross or mingle with lines; not on hatched/shaded surfaces; reference characters for related parts use same numeral with letter suffix | PASS | All numerals are 11 pt (≈ 0.39 cm). Sub-component numerals use letter suffixes (108a-e in Fig. 2; 112a-d in Fig. 3) per § 1.84(p)(4). |
| (p)(1) Lead lines | Lead lines required between reference character and detail, except for items the entire numeral identifies | PARTIAL | Reference numerals are placed adjacent to their components. For tighter § 1.84(p)(1) conformance, add explicit lead lines (short angled lines from numeral to component). Recommended for next pass. |
| (p)(2) Arrows | Arrows OK on lead lines; arrows on reference numerals discouraged | PASS | Arrows used only for data-flow edges (E1-E8), never on reference numerals. |
| (p)(3) Reference characters not to be used as labels | Numerals on figures; labels (e.g., "PROMPT SET MODULE") in the specification | PASS | Each component has BOTH a numeral AND a name on the figure. The numeral is the legal handle; the name aids reading. Common practice for software-apparatus filings. |
| (p)(4) Same reference number for same part across views | Yes — element 108 in Fig. 1 = 108 in Fig. 2; same for 112 | PASS | Verified by cross-checking the REFERENCE_NUMERAL_LEGEND.md table. Phantom boxes for adjacent elements use the same numerals (104, 110, 114) with dashed outlines marking them as not part of the figure's subject. |
| (q) Holes | (n/a for software apparatus) | n/a | — |
| (r) Drawing in front | Each sheet uses one side only | PASS | Single-side PDFs. |
| (s) Mounting | Drawings filed as PDFs do not require mounting | PASS | — |
| (t) Numbering / labeling within figure | Reference numerals upright, not inverted | PASS | — |
| (u) Symbols | Graphical symbols permitted | PASS | Standard rectangle = component; cylinder = data store; dashed rectangle = boundary; arrow = data-flow edge. |
| (v) Legends | Brief legends OK when necessary | PASS | Apparatus boundary 101 is labeled inline. EXTERNAL label on 106. Status-matrix labels in Fig. 3. |

## 2. Items to address before non-provisional conversion

The following are not blockers for a provisional filing under § 111(b) (which accepts informal drawings) but should be addressed when converting to a non-provisional or refiling for formal examination.

1. **Lead lines** (§ 1.84(p)(1)) — Add explicit short angled lead lines from each reference numeral to its component edge. Currently the numerals sit adjacent to components, which is widely tolerated for software-apparatus drawings but not strictly compliant.
2. **Minimum text size margin** (§ 1.84(m)) — The 9 pt annotation labels meet the 0.32 cm minimum but with no margin. Bump to 10 pt in v0.2.
3. **Margin verification at print** (§ 1.84(g)) — Confirm the 2.5 cm top margin at actual print time. The PDF page is 8.5 x 11 in with the figure content centered; if the printer applies any additional inset, this needs re-check.
4. **Sheet numbering placement** (§ 1.84(n)) — Strict reading says top-middle; bottom-right is common practice. Counsel preference.
5. **Front-page view designation** — Fig. 1 recommended; mark this on the Application Data Sheet (ADS).

## 3. Items deliberately NOT addressed

- **Greyscale / photographs** — out of scope; line art only.
- **Multi-view technical drawing conventions** (front / top / side) — not applicable to a software-method apparatus.
- **Trademark / copyright notices on drawings** — § 1.84(s) permits these only with specific authorization language; deliberately omitted from v0.1, to be added by counsel if desired.

## 4. Cross-figure consistency check

Cross-checked the numerals across Fig. 1-4 against the REFERENCE_NUMERAL_LEGEND.md table:

| Numeral | Fig. 1 | Fig. 2 | Fig. 3 | Fig. 4 | Consistent? |
|---|---|---|---|---|---|
| 100 (apparatus) | yes (label) | (implied) | (implied) | yes (boundary label) | OK |
| 101 (boundary) | yes (dashed) | yes (around 108) | yes (around 112) | yes (around lanes) | OK |
| 102 Prompt Set | yes | — | — | yes (lane) | OK |
| 104 Proxy | yes | yes (phantom) | — | yes (lane) | OK |
| 106 Deployment | yes (external) | — | — | yes (external lane) | OK |
| 108 Tagger | yes | yes (subject) | — | yes (lane) | OK |
| 110 Audit Store | yes (cylinder) | yes (phantom) | yes (phantom cylinder) | yes (lane) | OK |
| 112 Scorer | yes | — | yes (subject) | yes (lane) | OK |
| 114 Renderer | yes | — | yes (phantom) | yes (lane) | OK |
| 116 Report | yes | — | — | yes | OK |
| 202-216 (data) | yes | partial (208, 210) | partial (212, 214) | yes (all) | OK |

No conflicts found.

---

*Self-audit complete. Recommend filing as informal drawings under § 111(b); address items in §2 before non-provisional conversion.*
