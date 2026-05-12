# Landing Page Outline

This document defines the structure, content, and architecture for the Sundog Project landing page.

## Five Core Functions

The landing page must:

1. **Explain what Sundog is** — in accessible language
2. **Show it working** — concrete demonstrations
3. **Make it legible** — establish academic credibility
4. **Establish seriousness** — evidence-first posture
5. **Route people deeper** — clear navigation to documentation, demos, code

## Page Architecture

### Hero Section

**Purpose:** Immediate orientation and emotional hook

**Elements:**
- **Title:** "Sundog" or "The Sundog Project"
- **Tagline:** "Alignment without sight" or "A framework for indirect measurement and agent reasoning under partial observability"
- **Background Visual:** Animated sundog halo / indirect signal visualization / theorem motif
- **Value Proposition (1 sentence):**
  "The Sundog Project turns indirect signals into usable control — building software that doesn't need perfect information to behave intelligently."

**Call-to-Action Buttons:**
- "Explore the Theorem" → Section 1 or `/docs/theorem/overview`
- "View Applications" → Section 3 or `/docs/applications/`
- "Watch Demos" → Section 3 or YouTube channel

**Design Notes:**
- Clean, severe, slightly technical
- Animated background should be subtle, not distracting
- Typography: monospace or technical serif
- Color palette: cool tones, high contrast

---

### Section 1 — What It Is

**Purpose:** Accessible explanation of the core concept

**Heading:** "Indirect Measurement, Direct Results"

**Content (2-3 paragraphs):**

The clearest example is one you already know. The visible arcs of a sundog — the bright rings and daggers around the sun — are not the thing. They are the upper slices of complete geometric circles anchored to where the sun is and what the ice crystals are doing. You can't see the sun's altitude as a number; you read it from the geometry. You can't see the ice crystals at all; you read them from which arcs appear. The sky is doing indirect measurement at planetary scale, and the math is exact: the daggers always sit at `R / cos(h)` from the sun, where R is the 22° halo radius and h is the sun's altitude. Move the sun; the daggers move. Nobody can move the daggers any other way.

The Sundog Project is a framework for turning the same pattern — indirect signals from environmental geometry — into actionable software control. Where conventional approaches demand complete world state, Sundog asks whether the partial signal already contains enough structure to act.

In the core controlled experiment, a controller aligns a reflected beam without target coordinates, using only sparse photometric feedback. In product systems, the same pattern informs procedural agents acting under occluded state, verb-field NPC behavior, softbody motion made interpretable through graph signatures, and a parametric atmospheric-optics workbench that renders the parhelion display from its physical parameters and can run backwards to recover the sun's altitude from a photograph alone.

**Visual:**
- The parametric parhelion render (the geometry workbench hero), with the sun-altitude slider visible and the daggers tracking it
- Diagram: full implied circles vs visible upper-arc signatures
- Animated comparison showing target-aware vs photometric control (the formal experiment, kept as a secondary anchor)

---

### Section 2 — Why It Matters

**Purpose:** Establish practical value across audiences

**Heading:** "Why Indirect Signals Matter"

**Content (Bullets with short explanations):**

- **Works where direct inspection is hard**
  Occlusion, expense, or design constraints often make full state access impossible. Sundog operates from partial information.

- **Useful for agents with incomplete knowledge**
  Agents that know less can feel more alive. Sundog enables coherent behavior from compressed state.

- **Interpretable proxy signals**
  Instead of raw simulation noise, Sundog transforms physical traces into legible metrics: alignment, torsion, deformation, recovery.

- **Practical value in AI, games, simulation, and tooling**
  Demonstrated across procedural roguelikes, physical simulation, and softbody terrain systems.

**Visual:**
- Four icon cards representing each benefit
- Optional: short looping animation for each

---

### Section 3 — Applications

**Purpose:** Proof layer — show it working in real systems

**Heading:** "Working Systems"

**Content:**

Application cards linking to detailed pages. Each card includes:
- Application name
- One-sentence description
- Screenshot or animated preview
- "Learn More" link

**Application Cards:**

**1. Photometric Mirror Alignment**
"A controller aligns a reflected beam without target coordinates, matching oracle baseline accuracy in controlled experiments."
→ Link to `/docs/theorem/` or repo README

**2. EyesOnly / Gone Rogue**
"Procedural roguelike agents act from compressed perception using stop-conditioned action batches."
→ Link to `/docs/applications/eyes-only`

**3. Dungeon Gleaner**
"Verb-field NPC behavior: unmet needs diffuse across dungeon nodes, producing idle orbits without scripted planners."
→ Link to `/docs/applications/gamejam2026`

**4. Money Bags**
"Softbody terrain system with graph-based telemetry: torsion, deformation, symmetry, and recovery made legible."
→ Link to `/docs/applications/money-bags`

**Layout:**
- Responsive grid (2x2 on desktop, stack on mobile)
- Cards should be visually distinct but consistent
- Hover effects to indicate interactivity

---

### Section 4 — Before / After

**Purpose:** Concrete visual proof of difference

**Heading:** "Classic AI vs Sundog AI"

**Content:**

Side-by-side comparison showing:
- **Same environment**
- **Same task**
- **Different control logic**
- **Observable difference in behavior**

**Example Comparisons:**
1. **Photometric alignment:** Oracle vs indirect controller
2. **Agent behavior:** Omniscient agent vs compressed-perception agent
3. **Simulation cost:** Full physics vs Sundog approximation
4. **Interpretability:** Raw telemetry vs graph-enriched metrics

**Format for Each:**
- Video clips or animated GIFs
- Behavior graphs (state traces, performance over time)
- Short caption explaining the difference
- Quantitative metrics where available

**Visual:**
- Split-screen or side-by-side layout
- Toggle or slider to switch between conditions
- Graphs below each comparison

---

### Section 5 — Visual Proof

**Purpose:** Establish scientific credibility with charts and data

**Heading:** "Evidence and Metrics"

**Content:**

Gallery of key evidence visuals:

1. **Terminal target intensity comparison**
   Bar chart: photometric vs oracle baselines
   Caption: "Photometric controller reaches comparable terminal accuracy (p=0.264)"

2. **Time-to-convergence**
   Line graph: convergence curves
   Caption: "The cost of indirect feedback is time, not accuracy"

3. **Stress test results**
   Curve showing failure boundary
   Caption: "Known failure at tight joint limits"

4. **Application metrics**
   Charts from EyesOnly, Dungeon Gleaner, Money Bags
   Caption: "Utility demonstrated across domains"

**Layout:**
- Grid or carousel of key graphs
- Click to expand/view full resolution
- Link to `/results/` or detailed analysis pages

---

### Section 6 — Research Posture

**Purpose:** Establish seriousness and invite collaboration

**Heading:** "Ongoing Research"

**Content:**

The Sundog Project is an independent applied research initiative. The defensible scientific claim is narrow: photometric mirror alignment without target-position access in a controlled MuJoCo experiment. The broader applications demonstrate practical utility across procedural systems, simulation, and agent design.

We are continuing to formalize the mathematics, strengthen experimental evidence, and explore new application domains.

**Sub-sections:**

**What we're sharing:**
- Open repository with reproducible experiments
- Comprehensive documentation
- Application examples and integration guides
- Stress test results and failure boundaries

**What we're inviting:**
- Independent replication
- Collaboration on formalization
- Application-specific studies
- Benchmarks and comparisons
- Critical review

**Links:**
- GitHub Repository
- Documentation Index
- Research Roadmap
- Scientific Criteria

---

### Section 7 — Call to Action

**Purpose:** Route visitors to engagement points

**Heading:** "Explore Sundog"

**Content:**

Large, clear call-to-action buttons:

**Primary Actions:**
- **"View the Repository"** → GitHub
- **"Read the Documentation"** → `/docs/`
- **"Watch Demos"** → YouTube channel

**Secondary Actions:**
- **"Follow Updates"** → Mailing list / RSS / Social
- **"Contact / Collaborate"** → Contact page or email
- **"View Roadmap"** → Research roadmap

**Footer:**
- Project links (GitHub, YouTube, etc.)
- Documentation sections
- Related projects
- License information
- Contact

---

## Optional Sections

### Value Proposition Tabs

Interactive tabbed interface showing value for different audiences:

**Tabs:**
1. For Developers
2. For Researchers
3. For Game Designers
4. For Simulation Engineers
5. For Curious Visitors

**Each Tab Contains:**
- Headline
- 3 bullet points
- 1 graph or visual
- 1 "Learn More" button

This adds interactivity and personalizes the landing experience.

---

## Page Flow

**Visitor Journey:**

1. **Hero** → Immediate orientation: "This is Sundog, it does X"
2. **What It Is** → Understanding: "Here's how it works conceptually"
3. **Why It Matters** → Motivation: "Here's why you should care"
4. **Applications** → Proof: "Here are real systems using it"
5. **Before/After** → Demonstration: "Here's the concrete difference"
6. **Visual Proof** → Credibility: "Here are the numbers"
7. **Research Posture** → Legitimacy: "Here's our scientific stance"
8. **Call to Action** → Conversion: "Here's how to engage"

**Total Length:** Single long-scroll page, approximately 5-7 screen heights

---

## Design Language

### Typography
- Headlines: Bold, clean sans-serif or technical serif
- Body: Readable sans-serif, 16-18px
- Code/Data: Monospace for metrics and references

### Color Palette
- **Primary:** Deep blue or cool gray (technical, serious)
- **Accent:** Warm gold or sundog yellow (sundog motif)
- **Background:** White or very light gray
- **Data visualization:** High-contrast, colorblind-safe

### Visual Motifs
- Sundog halo / indirect reflection
- Graph/node structures
- Light rays and shadows
- Clean geometric forms
- Subtle animations (not distracting)

### Tone
- Measured
- Clear
- Slightly severe
- Quietly ambitious
- Evidence-first

---

## Technical Considerations

### Performance
- Optimize all images and videos
- Lazy-load below-the-fold content
- Keep animations lightweight
- Fast initial load critical

### Accessibility
- Semantic HTML
- ARIA labels where needed
- Keyboard navigation
- Screen reader compatibility
- High contrast mode support

### Responsive Design
- Mobile-first approach
- Breakpoints: mobile (< 768px), tablet (768-1024px), desktop (> 1024px)
- Stack cards on mobile
- Simplify animations on mobile

### Analytics
- Track scroll depth
- Monitor CTA click-through rates
- Track video engagement
- A/B test messaging variants

---

## Copy Tone Examples

### Good Examples
- "The Sundog Project turns indirect signals into usable control."
- "Demonstrated across procedural systems, simulation, and agent design."
- "We are continuing to formalize the mathematics and strengthen experimental evidence."

### Avoid
- "Revolutionary breakthrough in AI!"
- "Solves the alignment problem!"
- "The future of all software!"

---

## Assets Needed

**Graphics:**
- [ ] Hero background animation (sundog halo loop)
- [ ] Indirect measurement diagram
- [ ] Application screenshots (4)
- [ ] Before/after comparison videos (2-4)
- [ ] Key evidence graphs (4-6)
- [ ] Icon set for benefits section

**Copy:**
- [ ] Final hero tagline
- [ ] Section 1 body text (polished)
- [ ] Application descriptions (final)
- [ ] Before/after captions
- [ ] Research posture text (reviewed)

**Technical:**
- [ ] Page layout/wireframe
- [ ] Responsive breakpoints
- [ ] Animation specifications
- [ ] Color palette finalized
- [ ] Typography stack selected

---

## Next Steps

1. **Wireframe the layout** — sketch or low-fidelity mockup
2. **Draft all copy sections** — complete text for each section
3. **Identify existing assets** — what graphs/images already exist?
4. **Commission missing assets** — logo, animations, screenshots
5. **Build static prototype** — HTML/CSS mockup
6. **Review and iterate** — test with target audiences
7. **Implement production version** — static site or framework
8. **Launch and monitor** — track engagement and iterate
