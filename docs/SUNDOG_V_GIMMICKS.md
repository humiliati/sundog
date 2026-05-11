# Sundog Gimmicks Ledger

Working hook:

> Not better game AI. Different cognition under occlusion.

Sundog Gimmicks is the candidate-workbench ledger for game-native Sundog
applications that are still being considered, compared, and pressure-tested
before any single one is promoted into a full roadmap document like
`SUNDOG_V_BALANCE.md`.

This document exists to keep the brainstorming disciplined. The repo already has
two public workbench families with bounded claims:

- `SUNDOG_V_BALANCE.md`: hidden body, visible shadow;
- `SUNDOG_V_THREEBODY.md`: denied full state, indirect instability signatures.

The next game-facing workbenches should not drift into "better game AI,"
"beats AlphaGo," or "searches harder." The intended shape is different:

1. the decisive state is hidden;
2. the world leaks structure through traces, shadows, wakes, gradients,
   pressures, delays, distortions, or partial reactions;
3. the agent acts from those traces rather than full privileged access;
4. the usable region and failure boundary are named explicitly.

The goal of this document is to compare candidate games by that standard before
committing to implementation.

## Claim Boundary

This document does **not** claim that any listed game already expresses Sundog
well enough to be a public workbench. It is a staging ledger for evaluating
which gimmicks actually support the theoremic move and which ones only sound
good in pitch form.

A good candidate for promotion should satisfy most of the following:

- familiar enough that a first-time visitor understands the hidden-state problem
  quickly;
- browser-native and fast to simulate;
- naturally partial-observation rather than artificially obscured for drama;
- supports clear baselines;
- supports a bounded operating-envelope story;
- does not reduce immediately to "just search deeper" or "just do Bayes harder";
- can be phrased as a smaller, attackable claim rather than a broad boast.

## Anti-AlphaGo Framing

The comparison target is rhetorical, not competitive.

Do **not** say:

> Sundog beats AlphaGo.

Say instead:

> AlphaGo is a monument to perfect-information mastery. Sundog is aimed at a
> different class of problem: when the decisive state is hidden, but the world
> still leaks usable structure through shadows, traces, wakes, delays, and
> distortions.

Short version:

> AlphaGo saw the board. Sundog asks what remains possible when the board is
> denied.

This distinction matters because the public temptation will be to read any
game-facing workbench as "AI for winning games." That is not the point. The
point is to show bounded action from incomplete traces.

## Evaluation Criteria

Each candidate below is scored informally against the same questions:

- **Hidden-state fit:** is the decisive thing genuinely hidden?
- **Indirect-signal fit:** is there a natural shadow/trace/proxy channel?
- **Public legibility:** will people understand the setup quickly?
- **Scientific cleanliness:** is the task easy to bound and benchmark?
- **Implementation burden:** can a browser workbench and seeded harness be built
  without disproportionate engineering?
- **Misread risk:** how likely is the public to collapse it into gambling,
  brute-force search, or ordinary puzzle logic?

## Shortlist Recommendation

Current first-pass order of merit:

1. **Pressure Mines** — strongest theorem pedagogy.
2. **Shadow Fleet** — strongest public hidden-state metaphor.
3. **Occluded Code** — strongest clean scientific experiment.
4. **Shadow Shoe** — strongest casino-adjacent statistical workbench.
5. **Scent Snake** — simplest teaching demo.
6. **Occluded Peg Field** — strongest visual spectacle, better as a second-wave
   demo than the first rigorous one.

The first promotion candidate is currently **Pressure Mines**. This ledger
exists so the other options remain reasoned about rather than forgotten or
reintroduced sloppily.

---

## Candidate 1 - Shadow Fleet *(Battleship-like)*

Working hook:

> The ship is hidden. The water still carries its wake.

### Why it is strong

Shadow Fleet may be the strongest public anti-AlphaGo metaphor. The decisive
geometry is hidden by design. Partial observation is native, familiar, and easy
to explain. Seeded trials are straightforward. Baselines are legible. The
shadow metaphor can be made literal through sonar pings, wake fields, pressure
rings, or distorted local disturbance maps.

This is one of the clearest ways to express the intended contrast:

> Go rewards seeing the whole board. Shadow Fleet rewards acting when the board
> refuses to appear.

### Why it is weaker

Classic Battleship already invites probabilistic heatmapping. If the workbench
collapses into "pick the highest posterior cell," then the Sundog expression
has not yet been earned. The indirect signal needs to be more than ordinary
hit/miss plus combinatorics.

### Sundog variant

After each shot, the agent does not receive a clean `hit` / `miss` only. It
also receives a bounded, lossy wake or sonar field generated by nearby ship
geometry. The ships remain hidden; the surrounding water carries a distorted
trace of their presence.

### Sundog expression

- **Hidden target:** ship locations and local ship geometry.
- **Indirect signal:** wake field, sonar intensity, pressure ring, or adjacency
  disturbance rather than full board state.
- **Transformation:** probe / localize / track through repeated noisy field
  reads.
- **Actionable output:** fire, scan, reposition probe, or abstain.
- **Failure boundary:** overlapping wakes, long delay, weak field strength, or
  adversarial clustering make geometry unreadable.

### Current recommendation

Very strong second candidate after Pressure Mines. Better public metaphor than
Cards; slightly less clean than Mines for first operating-envelope work.

---

## Candidate 2 - Occluded Code *(Mastermind-like)*

Working hook:

> The code is hidden. The guesses cast alignment shadows.

### Why it is strong

Occluded Code is probably the cleanest non-physics Sundog experiment. Hidden
target, repeated probes, indirect feedback, clear baselines, tiny browser
implementation, and very legible evaluation. It is academically tidy and easy
to pre-register.

### Why it is weaker

Unmodified Mastermind risks reading as logic deduction rather than dynamic
control. Without noise, drift, or a bounded action economy, the task may feel
too symbolic and too close to exact inference.

### Sundog variant

Instead of exact black/white peg feedback, each guess produces a distorted
alignment or resonance score: enough to suggest partial structure, not enough
to reconstruct the code directly. The controller acts from that noisy alignment
surface rather than precise peg counts.

### Sundog expression

- **Hidden target:** secret code sequence.
- **Indirect signal:** noisy alignment or resonance score.
- **Transformation:** SCAN through candidate probes, SEEK promising subspaces,
  TRACK improving alignments, REACQUIRE after ambiguity.
- **Actionable output:** guess, probe, commit, or reset.
- **Failure boundary:** score aliasing, heavy noise, code drift, or low probe
  budget makes the signal insufficient.

### Current recommendation

Strongest "clean experiment" candidate. Slightly weaker as a public flagship
because it can feel abstract without careful presentation.

---

## Candidate 3 - Pressure Mines *(Minesweeper-like)*

Working hook:

> The mine is hidden. The field still bends around it.

### Why it is strong

Pressure Mines is the strongest bridge between common observation and uncommon
inference. Minesweeper already teaches the viewer the right lesson: danger is
hidden, but local structure leaks through the board. A fuzzy pressure-field
variant turns that familiar lesson into a direct Sundog workbench.

It is browser-native, easy to explain, easy to baseline, and naturally suited
to a failure-boundary story.

### Why it is weaker

Standard Minesweeper is already solved by exact local logic plus probability.
If the modified field ends up being a trivial re-encoding of exact counts, the
workbench has not earned its claim.

### Sundog variant

Tiles emit a fuzzy pressure value influenced by nearby mines, terrain, blur,
delay, or noise rather than exact adjacency counts. The controller reveals,
flags, scans, or retreats under confidence gating.

### Sundog expression

- **Hidden target:** mine occupancy and downstream local hazard.
- **Indirect signal:** fuzzy pressure, gradient, scan returns.
- **Transformation:** field-reading with confidence gating.
- **Actionable output:** reveal, flag, scan, abstain.
- **Failure boundary:** blur, overlap, density, delay, and dropout collapse
  local distinction.

### Current recommendation

Current best first promotion candidate. See planned full roadmap document:
`sundog_v_minesweeper.md`.

---

## Candidate 4 - Shadow Shoe *(Blackjack-like / cards)*

Working hook:

> The hand is hidden. The shoe still casts a shadow.

### Why it is strong

Cards are familiar, browser-friendly, and well suited to seeded repeated
trials. Hidden state, visible public sequence, and bounded action under
uncertainty all map naturally to the Sundog pattern. A shoe-shadow or discard
field could support a clean harness with strong statistical comparisons.

### Why it is weaker

This candidate carries casino and advantage-play baggage. Public readers may
reduce it to card counting, gambling optimization, or "AI for blackjack," which
is not the repo's desired frame. It also risks becoming more about expected
value arithmetic than about interesting indirect observability.

### Sundog variant

A blackjack-like game denies direct access to decisive latent structure such as
hidden deck composition, dealer information, or a round-level hidden modifier.
The controller sees only bounded discard summaries, noisy shoe indicators, or
other lossy public traces and chooses hit / stand / double / bet / abstain
actions under confidence constraints.

### Sundog expression

- **Hidden target:** round-level action value under hidden shoe state.
- **Indirect signal:** discard-derived shoe summary, public exposure imbalance,
  noisy shoe-shadow meter.
- **Transformation:** SCAN/SEEK/TRACK policy over indirect advantage estimates.
- **Actionable output:** hit, stand, double, press, hedge, or abstain.
- **Failure boundary:** shuffle frequency, signal delay, small sample depth,
  volatility, or rule ambiguity collapse the edge.

### Current recommendation

Keep as a serious later-phase workbench. Strong statistical harness candidate,
but not ideal as the first public game expression.

---

## Candidate 5 - Occluded Identity *(Guess Who?-like)*

Working hook:

> It does not identify the face. It reads the silhouette left by answers.

### Why it is strong

The hidden target is intuitive and the SCAN/SEEK/TRACK pattern maps naturally
onto question selection and belief collapse. It is familiar and easy to explain
at a high level.

### Why it is weaker

The content burden is higher than the mechanics suggest. Without a distinctive
indirect-answer mechanic, it risks collapsing into ordinary twenty-questions.

### Sundog variant

The controller cannot ask direct feature questions. Instead it receives partial,
axis-wise disturbances in a latent feature space: the answer perturbs the
"glasses / hair / age / color / shape" manifold without directly stating the
feature value.

### Current recommendation

Interesting but content-heavy. Not first-wave.

---

## Candidate 6 - Shadow Mansion *(Clue-like)*

Working hook:

> The board does not tell who did it. It leaks the shape of the lie.

### Why it is strong

Clue has genuine hidden state, public revelations, information asymmetry, and a
natural indirect-inference shape. It can support multiplayer or one-player
deduction loops and would make the anti-perfect-information framing legible.

### Why it is weaker

Scope. A clean workbench is harder to build quickly than Mines or Fleet. The
deduction loop requires the controller to see public movements, rumor fragments,
and partial denials rather than explicit card knowledge, then choose where to
inspect or when to accuse.

### Current recommendation

Strong long-form idea; too large for first promotion.

---

## Candidate 7 - Shadow Table *(Poker-lite)*

Working hook:

> The cards are hidden. The table still leans.

### Why it is strong

Poker is culturally familiar and genuinely about hidden state, public traces,
betting pressure, and incomplete information.

### Why it is weaker

It has a heavy AI literature already, so the comparison set becomes noisy and
the public misread risk is high. It can quickly become "we made poker AI,"
which is precisely the wrong framing for this repo.

### Sundog variant

Strip the game down to a single-opponent toy table with small deck, fixed
betting rounds, and interpretable public traces. The controller reads betting
pressure and exposed cards, not true hand state.

### Current recommendation

Useful cautionary candidate; not recommended for first-wave work.

---

## Candidate 8 - Occluded Four *(Connect Four with fog)*

Working hook:

> The pieces are buried. The columns still carry pressure.

### Why it is strong

Connect Four is familiar, visual, and easy to render. It offers a deliberate
contrast to Go-like perfect-information play.

### Why it is weaker

The hidden-state modification can feel artificial. People may ask why the board
is fogged in the first place.

### Sundog variant

Only recent layers, top surfaces, or per-column pressure summaries are visible;
older structure fades into hidden mass. The controller acts from column
pressure, not full position reconstruction.

### Current recommendation

Maybe useful as a rhetorical contrast piece, but not a first-priority workbench.

---

## Candidate 9 - Trace Cards *(Memory / Concentration)*

Working hook:

> The card is face down, but remembers.

### Why it is strong

Very simple, visual, browser-native, and non-gambling. Easy to teach and fast
to simulate.

### Why it is weaker

In its vanilla form, it mostly tests explicit recall. Without a stronger
residue mechanic, it risks being too slight for the theorem.

### Sundog variant

Cards leave incomplete traces when previously revealed: color drift, temporal
residue, adjacency echo, or category shadow. The controller acts from those
residues rather than exact memory.

### Current recommendation

Good teaching demo, weaker flagship.

---

## Candidate 10 - Blind Flipper *(Pinball-like)*

Working hook:

> The ball disappears. The cabinet still speaks.

### Why it is strong

Very strong demo appeal. It is kinetic, visual, familiar, and naturally about
timed intervention under partial observation.

### Why it is weaker

Engineering burden is higher. If the occlusion is not central, the task
degrades into ordinary physics control.

### Sundog variant

The ball passes behind occluding panels. The controller sees only impacts,
brief glimpses, cabinet vibration, or sound-like event traces and times flipper
actions accordingly.

### Current recommendation

Excellent later-phase spectacle; too costly for first theorem workbench.

---

## Candidate 11 - Occluded Peg Field *(Plinko / Pachinko-like)*

Working hook:

> The basin is hidden. The first impacts still speak.

### Why it is strong

Spectacular visuals, stochastic structure, and a natural place for shadow
trails and basin inference.

### Why it is weaker

Agency is weaker unless the player can nudge, choose lane, cash out, or alter
subsequent drops. It is better as a second demo than a first scientific proof
surface.

### Sundog variant

Hidden peg perturbations create stable but unseen bias fields. The controller
receives early bounce signatures and decides whether to intervene, abort, or
change the next drop.

### Current recommendation

Strong second-wave spectacle candidate.

---

## Candidate 12 - Shadow Queue *(Tetris with hidden bag)*

Working hook:

> It does not know the next piece. It keeps the board alive under queue
> pressure.

### Why it is strong

Familiar, fast, action-heavy, and easy to explain as uncertainty about the
future rather than the present.

### Why it is weaker

Hiding the queue can feel like an arbitrary difficulty modifier unless the
shadow signal is interesting enough to justify the change.

### Sundog variant

Instead of exact next-piece preview, the controller sees a distorted pressure
vector over possible future pieces and builds a board robust to that uncertain
future.

### Current recommendation

Interesting bridge to game-systems language; not first-wave.

---

## Candidate 13 - Scent Snake *(Snake-like)*

Working hook:

> The food is hidden. The air still carries its scent.

### Why it is strong

Simple, immediate, browser-native, and a very clean teaching demo. SCAN /
SEEK / TRACK maps naturally onto navigation under weak gradients.

### Why it is weaker

It may feel too small or too toy-like unless the scent field and hazard design
are carefully tuned.

### Sundog variant

Food is invisible until close. The controller receives scent gradients, stale
trails, and misleading wind-like drift, then navigates by disturbance.

### Current recommendation

Very good teaching/demo candidate; less strong than Mines or Fleet as a flagship
research workbench.

---

## Candidate 14 - Blind Labyrinth *(Hot/Cold maze)*

Working hook:

> The goal is hidden. The maze still answers at a distance.

### Why it is strong

This may be the purest theorem demo. Indirect signal is central, failure
boundary is easy to show, and the hidden-state logic is immediate.

### Why it is weaker

It is less culturally sticky than Battleship, Minesweeper, or Cards.

### Sundog variant

Walls, hazards, or the target remain hidden. The controller sees only thermal,
echoic, pressure, or proximity gradients and chooses move / scan / wait /
backtrack.

### Current recommendation

Strong pure-form candidate; weaker public hook.

---

## Candidate 15 - Biased Wheel *(Roulette-like)*

Working hook:

> The wheel is biased. The bias never appears directly.

### Why it is strong

Hidden bias plus indirect observation is a real observability story, and the
statistical harness would be straightforward.

### Why it is weaker

Very high gambling misread risk. This would invite the wrong discourse and
skepticism immediately.

### Sundog variant

Use a toy wheel with hidden bias fields. The controller reads spin decay,
bounce timing, and outcome history to decide whether to act or abstain.

### Current recommendation

Only worth pursuing with aggressive synthetic-task disclaimers. Not recommended
for early public work.

---

## Stronger / Weaker Comparison Summary

### Pressure Mines

Stronger because it naturally says:

- danger is hidden;
- nearby structure leaks through the board;
- useful action is possible without full visibility;
- the failure boundary is easy to understand.

### Shadow Fleet

Stronger because hidden geometry is the whole game. Better metaphor than Cards,
and a stronger anti-AlphaGo surface.

### Occluded Code

Strongest scientific cleanliness. Best for a small, defendable experiment.

### Shadow Shoe

Strong repeated-trial harness, but public readers may reduce it to gambling or
card counting.

### Occluded Peg Field / Blind Flipper

Best visuals, but weaker first-proof surfaces because they require more custom
mechanics or more engineering to keep the claim honest.

## Promotion Guidance

Promote a gimmick from this ledger into its own `SUNDOG_V_*.md` roadmap only if
it satisfies all of the following:

- a bounded claim;
- a browser workbench;
- a shared browser/headless core;
- clear baseline set;
- explicit hidden / indirect / transformation / output block;
- observability-boundary sweeps;
- and a public statement of where the gimmick should **not** be used.

## Current Recommendation

Proceed in this order:

1. `sundog_v_minesweeper.md`
2. `SUNDOG_V_FLEET.md`
3. `SUNDOG_V_CARDS.md`
4. `SUNDOG_V_PLINKO.md` or `SUNDOG_V_FLIPPER.md`

That sequence keeps the first workbench scientifically legible, the second
publicly resonant, the third statistically rich, and the fourth visually
promotional.

## Broadcast-Aligned Summary

For public communication, the gimmick family can be summarized this way:

> Sundog is not aimed at perfect-information mastery. The game-native
> workbenches ask a different question: when the decisive state is hidden, can
> useful action still be taken from the traces it leaves behind? Pressure Mines
> asks this through a hidden minefield and a fuzzy pressure field. Shadow Fleet
> asks it through hidden ship geometry and wake-like disturbance. Shadow Shoe
> asks it through cards and shoe-state traces. The point is not that shadow
> signals always suffice. The point is to map when they do, and where they
> fail.
