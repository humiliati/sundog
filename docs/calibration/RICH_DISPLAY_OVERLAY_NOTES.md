# Rich-Display Overlay Notes

Phase 10 of `docs/SUNDOG_V_GEOMETRY.md` uses these notes to keep the
optional vocabulary layers separate from the calibrated parhelion core.
Image 1 is the label key; images 2, 7, and 13 are the first tuning set.

## Artifacts

| image | overlay | role |
| --- | --- | --- |
| `1.Photometeor-jeff-mod_marked_red.jpg` | source annotation | Label key for CZA, supralateral, suncave Parry, upper tangent, Parry supralateral candidate, infralateral, parhelic circle, lower tangent, parhelia, 22° halo, and 46° halo. |
| `2.Photometeor-jeff_mod_red.jpg` | `overlays/p2.rich-vocabulary.overlay.png` | Clean modern version of the annotated display; strongest rich-vocabulary tuning source. |
| `7.625544777_10241089047341957_4435817776770420050_n.jpg` | `overlays/p7.rich-vocabulary.overlay.png` | High-sun / partial-visibility stress case. |
| `13.480859565_17934474635991868_323320248088719839_n.jpg` | `overlays/p13.rich-vocabulary.overlay.png` | Square social-crop / logo-crop tuning source. |

## Anchor Summary

| image | sun px | R22 px | parhelion offset | inferred h | note |
| --- | ---: | ---: | ---: | ---: | --- |
| p2 | `(567, 496)` | 182 | 192 | 18.6° | Reuses the clean multi-photo calibration anchor; dagger residuals are -1 px / -1 px. |
| p7 | `(1033, 946)` | 200 | 393 | 59.4° | Reuses the high-sun anchor; CZA should be treated as not applicable at this altitude even when the vocabulary overlay draws it. |
| p13 | `(557, 372)` | 210 | 220 | 17.3° | Rough hand anchor for social-crop morphology; dagger residuals are +6 px / +8 px against rough parhelion marks. |

All generated overlays now apply `--parhelic-y-offset-r22 = -0.05`, raising
the parhelic belt and dagger markers by 5% of the observed 22° halo radius.
This fixes a shared vertical belt residual without changing the 22°/46° halo
registration.

## Vocabulary Classification

| primitive | p2 | p7 | p13 | promote? |
| --- | --- | --- | --- | --- |
| CZA | visible | not applicable / high sun | cropped or not visible | Core label yes, but hide by sun-altitude rule when applicable. |
| Supralateral arc | visible | candidate only | not visible / cropped | Keep as optional vocabulary; p2 carries the strongest evidence. |
| Upper tangent arc | visible | visible / broad | candidate | Promote as stable logo/animation shape language. |
| Lower tangent arc | visible near lower 22° contact | not clear | not clear | Annotation only until another clean low-sun source supports it. |
| Suncave Parry arc | visible/candidate using image 1 key | weak candidate | weak candidate | Candidate label only; useful for education, not logo default. |
| Parry supralateral arc | candidate in image 1 | not visible | not visible | Do not promote beyond optional annotation. |
| Infralateral arcs | visible on p2 periphery | not visible / cropped | not visible / cropped | Good vocabulary label; not a core logo layer unless the design brief wants a rich-display variant. |

## Design Consequence

For the Phase 11 characterized logo toolkit, the safe shape language is:

- core sun;
- 22° halo / iris;
- left and right parhelion glints;
- parhelic-circle sweep;
- upper tangent / eyelid arc;
- optional CZA for large or animated marks.

Hold these back as annotation-only layers unless a later pass finds stronger
support across multiple clean references:

- Parry supralateral;
- suncave Parry;
- lower tangent;
- infralateral arcs.

The logo can borrow the *idea* of rich halo vocabulary, but the default mark
should remain readable at small sizes and trace back to stable atlas
primitives.
