# Wikipedia Outreach Plan

This document outlines our strategy for contributing to and enriching the Wikipedia article on Parhelion (Sundog) phenomena.

## Context

The Sundog Project has developed what we believe to be the **first interactive mathematical visualization** that demonstrates parhelion arc formation using real-time physics calculations. This represents a significant educational contribution to public understanding of atmospheric optical phenomena.

## Current Wikipedia Article Status

Article: [Sun dog](https://en.wikipedia.org/wiki/Sun_dog) · Talk: [Talk:Sun_dog](https://en.wikipedia.org/wiki/Talk:Sun_dog) · Last verified revision: `1339499942` (2026-02-20).

**Live table of contents** (the only sections that exist — anchor names are exact):

1. `Formation_and_characteristics` — physics: refraction through plate-shaped hexagonal ice crystals (cirrus, cirrostratus, diamond dust), 22° minimum deflection, color order red→orange→blue, link to parhelic circle and circumzenithal arc, sun-altitude dependence, other-planet variants. Cites Atmospheric Optics (atoptics.org.uk) and Georgia State HyperPhysics.
2. `Terminology`
3. `Etymology`
4. `History` (with subsections `Antiquity`, `Wars_of_the_Roses`, `Early_modern_era`, `Late_modern_era`)
5. `See_also` — Anthelion, Circumhorizontal arc, Corona (optical phenomenon), Crown flash, Liljequist parhelion, Moon dog
6. `References`
7. `Further_reading` — single entry: Minnaert, *Light and Color in the Outdoors* (Springer, 1993)
8. `External_links` — 7 entries (atoptics.org.uk, starrynightphotos, two YouTube videos, IOL news photo, NASA SDO Sundog Mystery, APOD jumping-sundog)

**What the live article already covers** (don't claim we're filling these holes):
- The 22° minimum-deflection physics
- Plate-shaped hexagonal ice crystal mechanism
- Color ordering and parhelic circle / circumzenithal arc relationships
- Sun-altitude effect on sun-dog separation from the halo
- Sun dogs on other planets (Mars CO₂-ice, Jupiter ammonia, etc.)

**What is genuinely missing from the article:**
- An *interactive* (rather than textual) walkthrough of arc assembly
- A single-frame schematic that places parhelia, 22° halo, and tangent arcs in one labeled diagram (the four article images are photographs, not schematics)
- Equation-to-arc correspondence shown visually
- Open-source educational implementation

## Our Contribution

### What We Can Offer

1. **Interactive Mathematical Visualization**
   - Real-time canvas-based animation showing parhelion formation
   - Sequential arc drawing with corresponding equations
   - Demonstrates:
     - Left and right parhelion arcs (22° refraction)
     - Upper and lower tangent arcs
     - Complete halo system formation
     - Mathematical equations for each component

2. **Educational Value**
   - Breaks down complex optical physics into understandable phases
   - Shows the relationship between ice crystal geometry and observed arcs
   - Demonstrates why parhelia appear at specific angles
   - Makes abstract concepts visually concrete

3. **Open Source Implementation**
   - Code is publicly available at https://github.com/humiliati/sundog
   - Can be embedded or adapted for educational use
   - Self-contained HTML/JavaScript requiring no external dependencies

## Proposed Wikipedia Enhancements

### Option 1 (preferred): External link in the existing `External_links` section

This is the lowest-friction insertion. Match the wikitext style of the seven existing entries — bullet, single-bracket external URL, optional trailing description. Insert near the atoptics.org.uk and starrynightphotos entries (which are the closest-kin resources). Proposed entry:

```wikitext
* [https://sundog.cc Sundog — Interactive parhelion formation visualization], step-by-step canvas demonstration of refraction (n = 1.31), 22° minimum-deviation halo, parhelia, and tangent arcs with the governing equations displayed alongside each arc.
```

Note: existing External-links entries use `*[https://… Title], description` format with a single space after `*`. Match that exactly. Do **not** use the markdown `[Title](URL)` form.

### Option 2 (de-prioritized): "Educational resources" section does not exist

The live article has no `Educational_resources` (or similar) section. Creating one is a substantially heavier lift than Option 1 — Wikipedia editors typically resist new top-level sections that look like external-resource directories, and treat them as WP:ELNO violations. Keep this option only as a fallback if Option 1 is rejected for being insufficiently encyclopedic, and pitch it on the Talk page first rather than landing it directly. If we do pursue it, the closest existing precedent on related articles is a short prose paragraph inside `Formation_and_characteristics`, not a new section.

### Option 3: Wikimedia Commons + image into `Formation_and_characteristics`

The article's `Formation_and_characteristics` section currently carries four photographs (Sun City West AZ, Salem MA, Hesse, Saskatoon) but **no labeled schematic**. A clean SVG schematic that labels parhelia, 22° halo, upper/lower tangent arcs, parhelic circle, and the deflection angle would fill a real gap.

Upload target: [Wikimedia Commons category `Parhelion`](https://commons.wikimedia.org/wiki/Category:Parhelion) — the article's existing `{{Commons|Parhelion|Sun dog}}` template already points there.

Proposed thumbnail wikitext (matches existing image syntax in the section):

```wikitext
[[File:Parhelion_schematic_with_deflection_equation.svg|thumb|upright=1.2|Schematic of a sun dog: parhelia (P) sit on the 22° halo at the same elevation as the Sun, with deflection D(α) = 40° − 2·[arcsin(n·sin α) − α] for ice (n = 1.31).]]
```

License: CC-BY-SA 4.0 or CC0. Source SVG should be derivable from the same equations the hero uses, so we can publish provenance ("rendered from the open-source Sundog hero canvas — github.com/humiliati/sundog").

## Insertion Targets — Live Article Anchors

| Asset | Section | Anchor URL | Existing neighbors to match in tone |
|---|---|---|---|
| External link to sundog.cc | External links | https://en.wikipedia.org/wiki/Sun_dog#External_links | atoptics.org.uk parhelia page; NASA SDO Sundog Mystery |
| Schematic SVG | Formation and characteristics | https://en.wikipedia.org/wiki/Sun_dog#Formation_and_characteristics | Existing four photo thumbnails (Sun City West, Salem, Hesse, Saskatoon) |
| Talk-page proposal | Talk:Sun dog | https://en.wikipedia.org/wiki/Talk:Sun_dog | Post under a new `== Proposal: …  ==` heading |
| Commons upload | Category:Parhelion | https://commons.wikimedia.org/wiki/Category:Parhelion | The article's `{{Commons\|Parhelion\|Sun dog}}` already links here |

The article also has `See_also`, `Further_reading`, and `References`, but none are appropriate insertion targets for our resource: `See_also` is reserved for inter-Wikipedia article links (we have no Wikipedia article), `Further_reading` is for printed/cited works (Minnaert 1993 is the lone entry), and `References` requires a published, citable source — eligible only after a paper lands.

## Implementation Strategy

### Phase 1: Community Engagement (Week 1-2)

1. **Create Wikipedia Account**
   - Register account specifically for scientific contributions
   - Establish edit history with minor, non-controversial edits
   - Build reputation score

2. **Talk Page Proposal**
   - Post proposal on the Sun dog article's Talk page
   - Explain the educational value
   - Request feedback from regular editors
   - Address any concerns about self-promotion vs. educational value

3. **Gather Community Input**
   - Listen to feedback from experienced Wikipedia editors
   - Adjust approach based on community guidelines
   - Ensure compliance with Wikipedia's external links policy

### Phase 2: Content Preparation (Week 2-3)

1. **Neutral Point of View (NPOV)**
   - Frame contribution as educational resource, not project promotion
   - Focus on the mathematical visualization itself, not the Sundog Project
   - Avoid promotional language
   - Ensure factual accuracy

2. **Reliable Sources**
   - Cite atmospheric optics textbooks for the physics
   - Reference peer-reviewed papers on halo phenomena
   - Connect our equations to established scientific literature
   - Document that our visualization implements known physics

3. **Create Supporting Materials**
   - Screenshot sequence of arc formation
   - Static diagram showing equation-to-arc mapping
   - Technical documentation of the implementation
   - Educational guide explaining the physics

### Phase 3: Contribution (Week 3-4)

1. **Initial Edit**
   - Start with modest addition to external links
   - Provide clear edit summary explaining educational value
   - Reference Talk page discussion

2. **Monitor Response**
   - Watch for reverts or challenges
   - Respond professionally to any concerns
   - Be prepared to defend educational merit without being defensive

3. **Iterate Based on Feedback**
   - Adjust language if needed
   - Provide additional sources if requested
   - Consider alternative placements if external links are rejected

### Phase 4: Media Contribution (Ongoing)

1. **Wikimedia Commons Upload**
   - Create high-quality static diagrams
   - Generate animated GIF sequences
   - Ensure proper licensing (CC-BY-SA or CC0)
   - Provide detailed descriptions

2. **Integration with Article**
   - Propose specific diagram placements in article
   - Ensure diagrams complement existing text
   - Follow Wikipedia image guidelines

## Guidelines and Best Practices

### What to Avoid

- **Self-promotion**: Don't mention the Sundog Project's broader research agenda
- **Original research**: Frame as visualization of established physics, not new discoveries
- **Conflict of interest**: Be transparent about our connection to the resource
- **External links spam**: Only propose if genuinely valuable to readers

### What to Emphasize

- **Educational value**: Focus on helping people understand parhelion formation
- **Accessibility**: Makes complex physics more understandable
- **Accuracy**: Based on established atmospheric optics equations
- **Open source**: Freely available for educational use

### Wikipedia Policies to Follow

1. **WP:ELNO** (External Links - What not to include)
   - Ensure our link doesn't violate external links guidelines
   - Must be relevant and educational
   - Not promotional

2. **WP:COI** (Conflict of Interest)
   - Disclose our connection on Talk page
   - Let community decide on inclusion
   - Accept rejection gracefully

3. **WP:NPOV** (Neutral Point of View)
   - Write from neutral perspective
   - Don't claim "world's first" in Wikipedia text
   - Let the resource speak for itself

4. **WP:V** (Verifiability)
   - Cite sources for the physics we're visualizing
   - Don't make unsourced claims

## Messaging

### Talk Page Proposal Template

```
== Proposal: External link to interactive parhelion visualization ==

I'd like to propose adding a single entry to the External links section.

The resource ([https://sundog.cc sundog.cc]) is an open-source canvas-based
visualization that animates the assembly of a sun-dog halo system step by step:
left and right parhelia, upper and lower tangent arcs, the 22° halo, and the
46° outer halo, with the governing equation shown alongside each step. The
physics is the same as already described in the Formation and characteristics
section — refraction through plate-shaped hexagonal ice crystals at n = 1.31,
minimum deflection D(α) = 40° − 2·[arcsin(n·sin α) − α] — but rendered
interactively rather than as text.

In tone and scope it's closest to the atoptics.org.uk pages already linked
from External links.

'''Disclosure''': I contributed to developing this visualization. The code
is open source (MIT) at https://github.com/humiliati/sundog and the page
itself contains no advertising or paywall.

'''Proposed addition''' (External_links section, matching existing entry style):

* [https://sundog.cc Sundog — Interactive parhelion formation visualization],
  step-by-step canvas demonstration of refraction, the 22° halo, parhelia,
  and tangent arcs with the governing equations displayed alongside each arc.

Happy to also upload a labeled SVG schematic to Commons:Category:Parhelion
if editors think Formation and characteristics would benefit from a diagram
to complement the existing four photographs.

~~~~
```

### Edit Summary Template

```
Adding interactive educational visualization demonstrating parhelion formation with
real-time physics calculations. See Talk page for discussion.
```

## Success Metrics

1. **Contribution Accepted**
   - Link or content remains in article for 30+ days
   - No revert wars or serious opposition
   - Positive feedback from editors

2. **Community Value**
   - Cited by other educational resources
   - Used in classroom settings
   - Referenced in related articles

3. **Educational Impact**
   - Increased understanding of parhelion physics
   - Reduced confusion about halo phenomena
   - More accessible entry point for students

## Fallback Options

If Wikipedia contribution is rejected or controversial:

1. **Academic Outreach**
   - Offer to atmospheric optics professors
   - Submit to educational resource directories
   - Share with science museums

2. **Independent Education Platform**
   - Standalone educational site
   - Integration with Khan Academy or similar
   - Partnership with science communication organizations

3. **Scientific Communication**
   - Blog post explaining the physics
   - Video walkthrough for YouTube
   - Social media educational threads

## Timeline

- **Week 1**: Create account, build edit history, review policies
- **Week 2**: Draft Talk page proposal, gather feedback
- **Week 3**: Prepare materials, refine approach based on feedback
- **Week 4**: Make initial contribution, monitor response
- **Ongoing**: Maintain, update, respond to community input

## Resources Needed

1. High-quality screenshots of visualization
2. Static diagrams for Wikimedia Commons
3. Documentation of the physics and equations
4. Citation list for atmospheric optics sources
5. Edit monitoring tools to track article changes

## Notes

- Wikipedia editors are volunteers with high standards
- Patience and respect for community norms is essential
- Educational value must be genuinely high, not just promotional
- Be prepared for rejection and have alternate plans
- Success is measured by reader benefit, not our visibility

## Contact Points

- Wikipedia Username: [To be created]
- Talk Page: https://en.wikipedia.org/wiki/Talk:Sun_dog
- Wikimedia Commons: [For media uploads]

## Related Documentation

- [Message House](message-house.md) - Keep promotional claims separate from Wikipedia content
- [Claims and Scope](claims-and-scope.md) - Remember NPOV when writing for Wikipedia
- [Asset Tracker](asset-tracker.md) - Track Wikipedia-related deliverables

---

**Key Principle**: We contribute to Wikipedia because we can genuinely help readers understand parhelion physics better, not because we want to promote the Sundog Project. The visualization stands on its educational merit.
