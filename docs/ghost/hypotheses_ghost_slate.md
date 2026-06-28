# Ghost Hypotheses Slate (cross-lane), 2026-06-28

> Conjecture hooks generated after the Ghost lane closed (Phases 1-5). They point
> the Ghost instrument at other Sundog lanes. **Conjecture-cooking, NOT claims** -
> same fencing as `SUNDOG_V_GHOST.md`. Status: OPEN (unvetted).
>
> Routing: this shippable slate holds the two theorem-shaped, low-overclaim-risk
> hooks (H1, H2). Three higher-risk / strategy-adjacent siblings (H3 generality-
> strikeout reframe, H4 alignment route-fidelity, H5 P-vs-NP "bait") live in the
> gitignored `internal/slates/GHOST_HYP_INTERNAL_2026-06-28.md` and are NOT here.

## The instrument

Ghost closed with a reusable tool, not just a result:

- **recognizability radius** = Mosse's *constant of recognizability* (Durand &
  Leroy, arXiv:1610.05577) - the finite radius at which a local patch determines
  its place in the global hierarchy;
- a sharp **bounded <-> undecidable phase boundary** (substitution /
  hierarchical = finite recognizability radius; SFT / Wang extension =
  undecidable, unbounded);
- secondary metrics: **local-derivability radius / MLD**, **hull cohomology**.

The sharpest fresh hypotheses measure another lane's "outside debt" with this
instrument.

## H1 - Ghost is congruent to Shadow-Invertibility

**Claim under test.** Shadow-Invertibility's banked result (lossy shadow
DETERMINES discrete/topological hidden structure, RESISTS continuous hidden
variables) and Ghost's bounded/unbounded boundary are the **same dichotomy**, and
their separators correspond.

Proposed correspondence:

| Shadow-Invertibility | Ghost / tiling |
| --- | --- |
| discrete / topological hidden structure DETERMINED by the shadow | substitution hierarchy RECOGNIZABLE at a finite radius (Mosse-Solomyak) |
| continuous hidden variables RESIST a lossy shadow | cut-and-project window phase NOT recoverable from any finite patch |
| `charFun` / Cauchy separator (PROVEN in Lean) | diffraction spectrum / pure-point dynamics (Baake-Grimm) |
| trained nonlinear encoder DEFEATS continuous-resist (banked) | a non-local / global map can recover the window phase that local rules cannot |

The last row is the sharp axis: the real variable is the **power of the recovery
operator**. Local / bounded (linear shadow; finite-radius rule) -> discrete
recoverable, continuous resists. Unbounded / nonlinear (trained encoder; non-local
map) -> continuous also recovered. So the dichotomy is not "discrete vs
continuous" per se but "what the recovery operator can reach."

**Inroad.** If the correspondence holds, the two lanes' formal cores are halves of
one statement: the Lean `charFun` separator and the recognizability machinery
cross-license, and the "continuous resists" result inherits a tiling-side witness
(cut-and-project) and vice versa.

**Load-bearing question (harden this first).** Does the `charFun`/Cauchy separator
actually correspond to the diffraction / dynamical-spectrum analysis of the
tiling, or is that only a Fourier-flavored rhyme? Everything rides on this bridge.

**Privileged-truth anchors.** Cut-and-project sets have pure-point diffraction
(Baake-Grimm, known); substitution recognizability <-> nonperiodicity
(Mosse-Solomyak, Ghost Q2). Both halves already have citable referents.

**Kill if:** the "continuous resists" in Shadow-Invertibility is a different
continuity than the cut-and-project window (e.g. it is purely about a trained
nonlinear encoder on i.i.d. data with no spatial / hierarchical structure), so the
separators are unrelated; OR the charFun separator has no diffraction-spectrum
counterpart once both are written precisely. Then it is a rhyme, not a functor.

Links: [[SUNDOG_V_GHOST.md]] Q2; Shadow-Invertibility lane (charFun Lean core).

### H1 HARDENED 2026-06-28 - SPLIT (H1a survives, H1b killed)

First hardening pass read the actual proven Lean statement
(`sundogcert/Sundogcert/ShadowDecayGeneral.lean`, axiom-clean) instead of the
memory summary, and cross-checked the tiling side against cited diffraction
theory. Result: the conjecture splits. The charFun separator IS the
diffraction-spectrum dichotomy (survives), but it is NOT the recognizability
radius (killed with a receipt). The real bridge lands on Ghost's spectral /
global axis, not its headline local one.

**H1a (SURVIVES) - charFun separator = the diffraction-spectrum dichotomy.**

- Lean core: averaging the continuous fringe factors as
  `int cos(2pi(c+lam x)t) dmu = Re[ exp(2pi i c t) * charFun mu(2pi lam t) ]`
  (`shadow_decay_charFun`). `charFun mu` = Fourier transform of the population =
  the **diffraction amplitude**.
- RESIST = `||charFun mu|| -> 0` (Riemann-Lebesgue; absolutely-continuous
  populations) = **continuous diffraction, no Bragg** (`resistance_general`;
  Cauchy separator `cauchy_is_separator`). DETERMINE = charFun does not decay =
  **pure-point / Bragg** (the "lattice charFun = cos does not wash" case).
- Tiling side is bedrock mathematical diffraction theory (Baake-Grimm,
  *Aperiodic Order* Vol. 1): Riemann-Lebesgue is exactly "an absolutely-
  continuous measure contributes no Bragg peaks."
- Quasicrystal sharpening: the DETERMINE class is LARGER than periodic -
  **cut-and-project sets (Penrose) have pure-point diffraction despite being
  aperiodic** (Shechtman 1984, already in the lit-pass spine). So
  Shadow-Invertibility's "a discrete label survives" = "ordered-but-aperiodic
  still diffracts to points," meeting Ghost's "ordered != periodic" exactly at
  the quasicrystal.
- Baake-Lenz (arXiv:math/0302061, verified): pure-point diffraction <=>
  pure-point **dynamical** spectrum. So the charFun axis is precisely Ghost Q2's
  hull / dynamical-spectrum global-invariant slot - not the recognizability slot.
- Verdict: a real cross-lane unification; the Lean charFun separator and tiling
  diffraction theory are the same statement, both sides proven/cited.

**H1b (KILLED, clean) - charFun separator is NOT the recognizability radius.**

The naive H1 read "Ghost" as its headline metric (recognizability). False:
recognizability holds for ALL primitive aperiodic substitutions (Mosse),
independent of diffraction type. Privileged-truth receipt (measured radii from
`npm run ghost:metric:test` + cited diffraction):

| substrate | recognizability radius L (measured) | diffraction (charFun axis) |
| --- | --- | --- |
| Fibonacci | 1 (recognizable) | pure-point -> DETERMINE |
| period-doubling | 1 (recognizable) | pure-point -> DETERMINE |
| Thue-Morse | 2 (recognizable) | singular continuous -> RESIST |

Thue-Morse is recognizable (finite L) yet sits on the RESIST side, so
recognizability is independent of the charFun/diffraction axis. The receipt is
machine-checkable (radii from the Phase 4 suite; Thue-Morse singular-continuous
per Baake-Gahler, "The singular continuous diffraction measure of the Thue-Morse
chain"; Fibonacci / period-doubling pure-point per Baake-Grimm).

### H1a frontier RESOLVED 2026-06-28 - confirmed (refined), lands on known theory

The open hook (tiling analog of Shadow-Invertibility's "trained nonlinear encoder
DEFEATS continuous-resist"): can a **non-local** reader recover structure the
diffraction declares absent? Pre-registered probe `scripts/ghost-nonlocal-probe.mjs`
on a common length N=987. Receipt:

| substrate | Bragg B = max (1/N^2)\|sum w e^{-2pi i f n}\|^2 | non-local desub-validity (best phase) |
| --- | --- | --- |
| periodic `abab` | 1.000 (detector validated) | - |
| Fibonacci | 0.352 (pure-point) | recognizable, L=1 |
| period-doubling | 0.446 (pure-point) | 1.000 |
| **Thue-Morse** | **0.058** (no Bragg; singular continuous) | **1.000** |
| random | 0.009 (absolutely continuous) | 0.489 |

**Load-bearing result (CONFIRMED):** the non-local deterministic reader recovers
Thue-Morse perfectly (desub 1.000) and rejects randomness (0.489), while TM carries
**no Bragg component** (B=0.058 sits in the singular-continuous tier, an order of
magnitude below the pure-point tier 0.35-1.0). So a non-local reader recovers
deterministic order that the **pure-point / Bragg channel of diffraction declares
absent**. That is the frontier claim, with a receipt.

**Honest refinement (two pre-registered Bragg thresholds FAILED, informatively):**
the probe's other two checks assumed singular-continuous TM would read like random.
It does not - TM (0.058) is ~7x random (0.009). **Diffraction is lossy, not
blind:** it relegates TM's order to a faint *continuous* smear rather than erasing
it. The thresholds were left as they fired (not re-tuned to force a pass); the
mechanism is corrected here.

**The clean van Enter-Miekisz instance the probe surfaced:** period-doubling is a
*factor* of Thue-Morse and is pure-point (B=0.446, strong Bragg), yet TM's own
diffraction is a faint SC smear (0.058) - even though both are equally
deterministic and the non-local reader scores both 1.000. The order diffraction of
TM under-reports is exactly the order visible in its factor / to the non-local
reader. (Van Enter & Miekisz 1992: order at a "molecular" scale recovered by
factor structures.)

**Landing (the honest default holds).** This resolves into KNOWN theory, NOT a new
invariant: diffraction = averaged **two-point** correlations and is homometry-
limited; the **dynamical spectrum / higher-order correlations / factor structures**
see more (Baake-Grimm, "Mathematical diffraction of aperiodic structures",
arXiv:1205.3633; van Enter-Miekisz 1992). The Shadow-Invertibility "nonlinear
encoder defeats continuous-resist" is the same phenomenon: a nonlinear / non-local
statistic beats the linear power spectrum.

Status: **H1a SURVIVES** (cited); **H1b KILLED** (receipt); **H1a frontier
RESOLVED** - confirmed against the pure-point channel, refined for full diffraction
(lossy not blind), lands on homometry / dynamical-spectrum theory.

## H2 - Recognizability is legibility (Ghost -> Least Reader-Action)

**Claim under test.** "Reader-action" (the work a cold reader does to recover a
claim, the Least Reader-Action lane's functional) IS a recognizability radius on
an argument's block structure: the local window of text needed to recover which
lemma-block ("supertile") a sentence belongs to and its role in it.

**Inroad.** Gives an active lane a citable backbone (the Mosse constant) instead
of a folk metric, plus a falsifiable probe: model a proof / document as a sequence
with a declared block decomposition (sections / lemmas = level-1 supertiles),
measure the minimal window that determines block membership + role, and call that
the reader-recognizability radius. Compare before / after an Euler-pass rewrite -
a good rewrite should shrink it.

**Privileged-truth anchor.** The lane already has a worked example (Snell / Fermat)
and the `B/Phi/T/A/I/F/R` chart; reader-action should track the recognizability
radius on the argument graph, and the lane's own pre-registered falsifier (cold
readers still cannot name import / falsifier / receipt after the pass) is the
ground truth to correlate against.

**Kill if:** real prose has no stable block decomposition, or the radius is
ill-defined / unbounded on actual argument graphs - then "legibility =
recognizability" stays metaphor and the lane keeps its folk metric.

Links: [[project_sundog_least_reader_action]]; [[SUNDOG_V_GHOST.md]] Q2.

### H2 HARDENED 2026-06-28 - SURVIVES (refined), with an independent receipt

First hardening pass read the actual Least Reader-Action artifacts
(`SUNDOG_V_LEAST_ACTION.md`; `docs/least_action/PHASE1_SUNDOGCERT_READER_ACTION_AUDIT.md`)
rather than the memory summary. Outcome: H2 splits like H1 - a strong surviving
core and a killed over-strong identity - and the surviving core is grounded by an
audit the lane ran *before* this hypothesis existed.

**H2a (SURVIVES) - the lane's recoverability scale IS a reader-recognizability
radius.** The Phase 1 audit scores each `B/Phi/T/A/I/F/R` coordinate on a
recoverability scale: check = recoverable by a cold reader; warn = recoverable but
costs a reread / context-hop; cross = not recoverable without help. That scale is
a coarse **reader-recognizability radius**: check = bounded-small; warn =
bounded-with-a-hop (look further / elsewhere); cross = unbounded (role not
recoverable from the document without going non-local). The lane's pre-registered
falsifier ("two cold readers cannot name import / falsifier / receipt without
help") is exactly "the reader-recognizability radius for I / F / R is not finite at
the front door." An Euler pass = shrink every coordinate's reader-radius to
bounded-at-the-front-door. Backbone: Mosse's *constant of recognizability*
(Durand-Leroy). This also answers the lane's open question ("does `J` need real
scoring?"): the **locality** terms of `J` admit a real per-coordinate score - the
reader-recognizability radius - even while the rest of `J` stays qualitative.

**H2b (KILLED) - reader-action `J` is NOT the recognizability radius.** `J` is a
sum of six costs (missing-inference, notation-hop, cross-reference, hidden-import,
status-ambiguity, replay-friction). The recognizability radius captures only the
**locality** terms (notation-hop, cross-reference, missing-inference: how far you
must look to recover a coordinate's role). The **coverage** terms (hidden-import,
status-ambiguity, replay-friction) are a separate axis: whether a role *exists / is
correct / is runnable at all*, not how far you look. So recognizability radius is
**one coordinate of reader-action**, not the whole functional - same shape as H1b.

**Receipt (independent, privileged-truth).** Applying the reader-recognizability
lens reproduces the Phase 1 audit's own per-coordinate verdict:

| coordinate | audit verdict | reader-recognizability radius reading |
| --- | --- | --- |
| `I` imports | check-check (strongest; stated up front) | bounded-small |
| `R` receipt | check-check (referee-free, self-enforcing) | bounded-small |
| `B` body | warn ("the word 'body' never appears"; one notation-hop) | bounded + one hop |
| `F` falsifier | cross->warn (weakest; lives in the sibling ledger, not the README) | unbounded at the front door (recoverable only non-locally) |

And the audit's **headline catch** - a stale worked-example count
(status-ambiguity) - is a **coverage** failure (the info was present at small radius
but WRONG), which the lens correctly classifies as OFF the radius axis, matching
the audit's own "status-ambiguity cost" label. The lens agrees with a human audit
run before the hypothesis - that is the check.

**Honest scope.** H2's receipt is **analytic** (lens reproduces an independent
audit), not machine-computed like H1's. A computable version - model a document as
role-tagged sentences and measure the window needed to recover each tag on a
structured vs scrambled version, or via an LLM-reader proxy - is a possible
follow-up if a number is wanted; it is not built here.

**Landing (not a new invariant).** Recognizability radius is a borrowed instrument
(Mosse), not a new legibility theorem: it gives the *locality coordinate* of
reader-action a citable backbone and a real score, while reader-action as a whole
stays a Sundog construct with a separate coverage axis.

Status: **H2a SURVIVES** (grounds an active lane's metric + answers its open
J-scoring question; receipt reproduces the Phase 1 audit); **H2b KILLED**
(recognizability radius = the locality coordinate of reader-action, coverage is
orthogonal).

## Next

Both shippable hooks are now hardened, and they hardened to the **same shape**: an
`a` half that survives + a `b` half killed as an over-strong identity, with
"recognizability is ONE coordinate, not the whole thing" as the recurring verdict.

- **H1** (2026-06-28): H1a SURVIVES (charFun = diffraction, both sides
  proven/cited), H1b KILLED (recognizability != diffraction, receipt), H1a
  frontier RESOLVED (non-local reader recovers what the pure-point channel
  declares absent; diffraction lossy-not-blind; lands on homometry /
  van Enter-Miekisz).
- **H2** (2026-06-28): H2a SURVIVES (the lane's recoverability scale + falsifier
  ARE a reader-recognizability radius; answers its open "does J need scoring?"
  question; receipt reproduces the Phase 1 audit), H2b KILLED (J != radius; radius
  = the locality coordinate, coverage is orthogonal).

Highest-utility follow-through (actionable, in an ACTIVE lane): **DONE 2026-06-28
(owner license).** H2a landed in `SUNDOG_V_LEAST_ACTION.md`: new section "Scoring
the Locality Coordinate: the Reader-Recognizability Radius" (the locality/coverage
split of `J`, the per-coordinate radius score, the Mosse backbone with the
"a document is not a substitution" fence); the lane's open "does `J` need real
scoring?" question marked partially-resolved; Safe Language updated (Use +=
"reader-recognizability radius", Avoid += "legibility is recognizability" /
"a document is a substitution system"). This revived a lane that had been left
stale and nearly quietly closed.

H1a write-up: **DONE 2026-06-28** - promoted to its own cross-substrate home at
[`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md) section 10
("Shadow-Invertibility charFun = the diffraction spectrum of aperiodic order"),
the portfolio's typed-generalization catch-can. Both shippable hooks are now
hardened AND landed (H2a in `SUNDOG_V_LEAST_ACTION.md`, H1a in
`CROSS_SUBSTRATE_NOTES.md` section 10). Remaining: internal siblings (H3-H5)
parked unless one earns attention.
