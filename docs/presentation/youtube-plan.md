# YouTube Channel Plan

This document outlines the strategy, content pillars, and initial video sequence for the Sundog Project YouTube channel.

## Channel Philosophy

**You do not need polished "influencer" energy.**

**You need a channel that acts like a lab notebook with production value.**

The tone should be:
- Serious but accessible
- Exciting but not grandiose
- Evidence-first but emotionally engaging
- Technical but not exclusionary

## Channel Identity

**Channel Name:** The Sundog Project

**Tagline:** "Indirect measurement, direct results" or "Alignment without sight"

**Channel Description:**
"The Sundog Project is a framework for turning indirect signals into usable software control. This channel documents our experiments, applications, and research progress — showing what happens when systems learn to act from incomplete information."

**Visual Identity:**
- Consistent thumbnail template with Sundog branding
- Clean, technical aesthetic
- Animated sundog halo/logo intro bumper (3-5 seconds)
- End card with links to repo, docs, related videos

---

## Content Pillars

The channel should organize around four core content types:

### Pillar A — What Is Sundog?

**Purpose:** Explainers and conceptual introductions

**Target Audience:** New visitors, curious developers, researchers evaluating the project

**Video Topics:**
1. What is the Sundog Project?
2. What problem is it solving?
3. Why indirect measurement matters
4. Sundog theorem for game developers
5. Sundog theorem for AI people
6. Sundog theorem for physics/simulation-minded people
7. The math intuition (accessible version)
8. Direct vs indirect perception

**Tone:** Educational, accessible, patient

**Format:**
- 5-12 minute explainers
- Mix of talking head, diagrams, animations
- Concrete examples throughout
- Clear takeaways

---

### Pillar B — Application Videos

**Purpose:** Concrete demonstrations of Sundog in working systems

**Target Audience:** Developers, game designers, technical practitioners

**Video Topics:**
1. Photometric mirror alignment demo (core experiment)
2. EyesOnly / Gone Rogue demo (procedural agents)
3. Dungeon Gleaner demo (pressure washing)
4. Dungeon Gleaner demo (glass/window reflection)
5. Money Bags demo (softbody telemetry)
6. Behavior comparisons: Classic AI vs Sundog AI
7. Softbody rig analysis walkthrough
8. Light reflection cost comparison
9. Before/after: agent behavior under occlusion
10. Turn envelope demonstration

**Tone:** Demonstrative, evidence-focused, comparative

**Format:**
- 3-8 minute demos
- Screen capture with voiceover
- Side-by-side comparisons
- Graphs and metrics overlaid
- Code snippets where relevant

---

### Pillar C — Research Notes

**Purpose:** Transparency, credibility, and methodological honesty

**Target Audience:** Researchers, skeptics, collaborators, academic evaluators

**Video Topics:**
1. What we got wrong last year
2. Reframing the claim
3. What counts as evidence
4. How we think about alignment
5. Why this isn't just marketing language
6. The gap between core experiment and applications
7. Known failure boundaries
8. What's defensible vs what's aspirational
9. How we measure indirect signals
10. Stress test interpretation

**Tone:** Honest, reflective, measured, credible

**Format:**
- 6-15 minute discussions
- Can be more informal (lab notebook style)
- Show uncertainties and limitations
- Build trust through transparency

---

### Pillar D — Devlog / Build Log

**Purpose:** Continuity, engagement, documentation of progress

**Target Audience:** Followers, collaborators, contributors

**Video Topics:**
1. Repo walkthroughs
2. Feature changes and updates
3. New graphs and metrics
4. New experiments in progress
5. Application updates (EyesOnly, Dungeon Gleaner, Money Bags)
6. Benchmark results
7. Integration examples
8. Community contributions
9. Roadmap updates
10. Q&A sessions

**Tone:** Informal, conversational, progress-focused

**Format:**
- 4-10 minute updates
- Can be lower production value
- Frequent, consistent uploads
- Encourages ongoing engagement

---

## First Video Sequence

Launch with these five videos to establish the channel and cover essential ground:

### Video 1 — "What Is the Sundog Project?"

**Purpose:** Introduce the framework and project

**Length:** 8-10 minutes

**Outline:**
1. **Hook (30s):** "Most systems do not reveal their truth directly. You infer them from signatures — shadows, feedback, distortions, response curves. The Sundog Project is our attempt to formalize and apply that idea."

2. **Problem (1m):** Direct state access is expensive, unavailable, or unrealistic in many systems. Agents need to act from incomplete information. Simulations need to compress physics without losing coherence.

3. **What Sundog Contributes (2m):** A framework for turning indirect signals into actionable control. Show H(x) intuition without requiring proof. Examples: photometric feedback, occluded game state, deformation traces.

4. **Core Example (3m):** Walk through the photometric mirror alignment experiment. Show the setup, the constraint (no target coordinates), the result (matches baseline accuracy).

5. **Broader Applications (1.5m):** Quick previews of EyesOnly, Dungeon Gleaner, Money Bags. This isn't just a lab trick.

6. **What We Are / Are Not Claiming (1m):** "We're not claiming a universal theory. We're presenting an evolving framework with measurable results in specific domains."

7. **Call to Action (30s):** "Explore the repo. Watch the demos. Read the docs. This channel will document the research as it develops."

**Visuals:**
- Animated sundog halo
- Diagram: direct vs indirect signal
- Screen capture of MuJoCo experiment
- Quick cuts of application footage
- Text overlays for key claims

**Deliverables Needed:**
- Script (full transcript)
- Voiceover recording
- Visual assets (diagrams, animations, footage)
- Thumbnail
- Description with links

---

### Video 2 — "Why We Reframed the Claim"

**Purpose:** Honesty, credibility, maturity

**Length:** 6-8 minutes

**Outline:**
1. **Hook (30s):** "Last year we introduced Sundog as a discovery. Some people were excited. Others dismissed it. Here's what we learned."

2. **What We Did Last Year (1m):** Presented a bold theorem-shaped idea. High ambition, early evidence. Some of the language overshot what we could defend.

3. **The Feedback (1m):** Provoked curiosity but also skepticism. The gap between the claim and the evidence was too large. We needed to show utility, not just vision.

4. **What We Did This Year (2m):** Focused on application, refinement, and legibility. Built working systems. Collected baselines and metrics. Narrowed the scientific claim while keeping the broader research program alive.

5. **The New Structure (1.5m):** Layer A (bold idea), Layer B (legible claim), Layer C (proof). Anchor ambition in evidence.

6. **What Changed (1m):** The core insight is the same. The framing is clearer. The evidence is stronger. The claim is defensible.

7. **Call to Action (30s):** "This is what honest research looks like. We're sharing the process, not just the wins."

**Tone:** Reflective, honest, mature

**Visuals:**
- Split screen: old framing vs new framing
- Text excerpts from original theorem vs current paper claim
- Graphs showing experimental results
- Screenshots of applications

**Deliverables Needed:**
- Script
- Voiceover
- Comparison visuals
- Thumbnail
- Description

---

### Video 3 — "Classic Game AI vs Sundog AI"

**Purpose:** Hook developers with concrete proof

**Length:** 5-7 minutes

**Outline:**
1. **Hook (20s):** "Same game. Same enemy. Same environment. Different logic. Watch what happens."

2. **Setup (1m):** Introduce the comparison. Classic AI: omniscient, knows full state. Sundog AI: compressed perception, acts from partial information.

3. **Demo 1: EyesOnly / Gone Rogue (2m):** Show agent operating in procedural roguelike. Highlight turn envelope, stop conditions, compressed state. Compare to debug-state agent.

4. **Demo 2: Dungeon Gleaner (1.5m):** Show pressure-washing behavior. Highlight hose state, spray projection. Compare to simpler hitscan.

5. **Why It Matters (1m):** Agents feel more alive when they don't cheat. Partial information becomes a design feature. Performance can be better because the signal is compressed.

6. **The Trade (30s):** Sometimes slower. Sometimes misses information. But the behavior is coherent and the cost is lower.

7. **Call to Action (30s):** "Explore the applications. See the code. Try it in your projects."

**Visuals:**
- Side-by-side game footage
- Overlay graphs (state, performance, behavior metrics)
- Code snippets (optional)
- Before/after stats

**Deliverables Needed:**
- Game capture footage (both conditions)
- Voiceover script
- Overlay graphics
- Thumbnail
- Description

---

### Video 4 — "Indirect Measurement in Practice"

**Purpose:** Explain the math intuition without requiring proof

**Length:** 7-10 minutes

**Outline:**
1. **Hook (30s):** "You can't see the target. But you can see what the world does back to you. That's enough."

2. **The Core Idea (1.5m):** Most systems don't give you direct access. You have to infer from projections, feedback, distortions. This is indirect measurement.

3. **The Photometric Example (2m):** Walk through the mirror alignment setup. No target coordinates. Only light intensity and joint angles. How does it work? Scan, seek, extremum tracking.

4. **The Graph Example (2m):** Show Money Bags softbody rig. Can't directly measure "good terrain handling." But you can measure alignment, torsion, deformation, recovery. Graph metrics make physics interpretable.

5. **The Pattern (1.5m):** Deny full state. Observe indirect signal. Transform into control-relevant signature. Act from signature. Measure failure boundary honestly.

6. **Why It Generalizes (1m):** Same pattern across optics, agents, physics, graphs. The signal type changes. The approach is the same.

7. **Call to Action (30s):** "Read the formalization. Replicate the experiments. Apply the pattern."

**Visuals:**
- Animated diagrams
- MuJoCo experiment footage
- Money Bags telemetry graphs
- Unified pattern diagram
- Text overlays for key concepts

**Deliverables Needed:**
- Script
- Voiceover
- Custom diagrams/animations
- Experiment footage
- Thumbnail
- Description

---

### Video 5 — "A Tour of the Repo"

**Purpose:** Onboarding for curious researchers and developers

**Length:** 8-12 minutes

**Outline:**
1. **Hook (30s):** "The Sundog repo is open. Here's how to navigate it."

2. **Repo Structure (1.5m):** Walk through folders: agents, experiments, results, docs, runners, applications. Explain what each contains.

3. **The Core Experiment (2m):** Point to experiment scripts, analysis code, result summaries. Show how to run the baseline comparison.

4. **The Documentation (1.5m):** Highlight key docs: researcher guide, scientific criteria, applications map, promo highlights. Who should read what.

5. **The Applications (2m):** Point to integration examples: Gone Rogue runner, playtest bundles, ADRs. Show how Sundog connects to real systems.

6. **Reproducing Results (1.5m):** Step-by-step: clone, install, run experiments, inspect outputs. Show expected artifacts.

7. **Contributing / Exploring (1m):** How to engage. Replication, benchmarks, new applications, formalization, critique.

8. **Call to Action (30s):** "Clone it. Run it. Break it. Tell us what you find."

**Visuals:**
- Screen recording of repo navigation
- Terminal commands
- File structure diagrams
- Example outputs (graphs, summaries)
- Code snippets

**Deliverables Needed:**
- Script
- Screen recording
- Voiceover
- Thumbnail
- Description with repo link

---

## Video Script Structure Template

Every script should follow this shape:

1. **Hook** (20-40s) — Grab attention, establish stakes
2. **Problem** (1-2m) — What question are we addressing?
3. **What Sundog Contributes** (2-3m) — Our approach
4. **Example** (2-4m) — Concrete demonstration
5. **What We Are / Are Not Claiming** (30s-1m) — Boundary language
6. **Call to Action** (20-40s) — How to engage further

**Example Hook Templates:**
- "Most systems do not reveal their truth directly..."
- "Same [X]. Same [Y]. Different logic. Watch what happens."
- "You can't see the target. But you can see what the world does back to you."
- "Last year we introduced Sundog as a discovery. Here's what we learned."
- "The Sundog repo is open. Here's how to navigate it."

---

## Production Guidelines

### Recording
- **Voiceover:** Clear, measured pace. Not too fast. Emphasize key terms.
- **Music:** Subtle, technical, atmospheric. Not distracting.
- **Sound effects:** Minimal. Use for transitions or emphasis only.

### Visuals
- **Screen recordings:** 1080p minimum, 60fps for smooth motion
- **Diagrams:** High contrast, clean lines, minimal text
- **Animations:** Purposeful, not decorative. Illustrate concepts.
- **Text overlays:** Large, readable fonts. Short phrases only.

### Editing
- **Pacing:** Steady. Allow concepts to land. Cut dead air but don't rush.
- **Transitions:** Simple cuts or fades. Avoid flashy effects.
- **Captions:** Always include. Accessibility and silent viewing.

### Thumbnails
- **Consistent template:** Sundog branding element + title text
- **High contrast:** Readable at small size
- **Descriptive:** Viewer should know what the video is about
- **Not clickbait:** Honest representation of content

### Descriptions
- **First 2 lines:** Hook and value proposition (visible before "show more")
- **Timestamps:** For videos over 5 minutes
- **Links:** Repo, docs, related videos, applications
- **Credits:** Attribution for footage, tools, collaborators

---

## Upload Schedule

**Phase 1 — Launch (Weeks 1-2):**
- Upload all 5 initial videos
- Space them 2-3 days apart
- Create playlists: "Getting Started", "Applications", "Research Notes"

**Phase 2 — Regular Content (Ongoing):**
- Target: 1-2 videos per month
- Alternate between pillars
- Prioritize application demos and devlog updates

**Phase 3 — Expansion (Later):**
- Guest collaborators
- Replication reports
- Case studies
- Interviews

---

## Playlists to Create

1. **Getting Started with Sundog**
   - Video 1: What Is the Sundog Project?
   - Video 4: Indirect Measurement in Practice
   - Video 5: A Tour of the Repo

2. **Applications and Demos**
   - Video 3: Classic Game AI vs Sundog AI
   - All application-specific demos
   - Before/after comparisons

3. **Research Notes**
   - Video 2: Why We Reframed the Claim
   - Research transparency videos
   - Stress test interpretations
   - Methodology discussions

4. **Devlog**
   - All build log and update videos
   - Repo walkthroughs
   - Feature announcements

---

## Community Engagement

### Comments
- Respond thoughtfully to questions
- Acknowledge critiques
- Provide links to relevant docs
- Invite replication and contributions

### Community Tab
- Share repo updates
- Poll viewers on topics of interest
- Highlight external replications or applications
- Announce new videos

### End Cards
- Link to related videos
- Subscribe button
- Link to repo/docs
- Next suggested video

---

## Metrics to Track

- **Watch time** — Are people staying engaged?
- **Click-through rate** — Are thumbnails and titles effective?
- **Audience retention** — Where do people drop off?
- **Traffic sources** — Where are viewers coming from?
- **Subscriber growth** — Are we building a community?
- **Link clicks** — Are people visiting the repo/docs?

---

## Assets Needed for Launch

**Video 1:**
- [ ] Full script
- [ ] Voiceover recording
- [ ] Sundog halo animation
- [ ] Direct vs indirect diagram
- [ ] MuJoCo experiment footage
- [ ] Application preview clips
- [ ] Thumbnail

**Video 2:**
- [ ] Full script
- [ ] Voiceover recording
- [ ] Comparison visuals (old vs new framing)
- [ ] Experimental results graphs
- [ ] Thumbnail

**Video 3:**
- [ ] Full script
- [ ] Voiceover recording
- [ ] EyesOnly gameplay capture (both AI types)
- [ ] Dungeon Gleaner capture
- [ ] Overlay graphics (stats, metrics)
- [ ] Thumbnail

**Video 4:**
- [ ] Full script
- [ ] Voiceover recording
- [ ] Custom animated diagrams
- [ ] MuJoCo footage
- [ ] Money Bags telemetry graphs
- [ ] Pattern diagram
- [ ] Thumbnail

**Video 5:**
- [ ] Full script
- [ ] Voiceover recording
- [ ] Repo navigation screen recording
- [ ] Terminal command demos
- [ ] Example outputs
- [ ] Thumbnail

**Channel Assets:**
- [ ] Channel banner
- [ ] Profile picture (Sundog logo)
- [ ] Intro bumper animation (3-5s)
- [ ] Outro end card template
- [ ] Thumbnail template
- [ ] Playlist cover images

---

## Next Steps

1. **Write full scripts** for Videos 1-5
2. **Record voiceovers** or arrange for voice talent
3. **Gather/create visual assets** (footage, diagrams, animations)
4. **Edit videos** using consistent template
5. **Design thumbnails** using template
6. **Write descriptions** with links and timestamps
7. **Upload and schedule** Videos 1-5
8. **Create playlists** and organize content
9. **Promote initial launch** through repo, social, communities
10. **Monitor and iterate** based on engagement

---

## Long-Term Vision

The channel should become:
- **The primary video resource** for understanding Sundog
- **A living documentation layer** complementing written docs
- **A credibility signal** showing active research and transparency
- **A community hub** for collaborators and adopters
- **An engagement funnel** driving traffic to repo and applications
