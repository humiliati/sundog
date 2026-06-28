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

**Named frontier (the one genuinely open hook).** Shadow-Invertibility also banked
"a trained nonlinear encoder DEFEATS continuous-resist." Tiling analog: can a
**non-local** reader recover structure the diffraction (charFun) declares absent
(singular-continuous / Bragg-poor)? Baake-Lenz fixes the linear/spectral picture;
the open question is whether a nonlinear / non-local map sees more - the tiling
counterpart of the trained-encoder result. That is the next probe if H1a is
pushed further.

Status: **H1a SURVIVES** (recorded, cited); **H1b KILLED** (receipt); frontier
OPEN (non-local / trained-encoder analog).

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

## Next

**H1 hardened 2026-06-28** (above): the charFun <-> diffraction bridge is REAL
(H1a survives, both sides proven/cited), and the naive charFun<->recognizability
bridge is killed with a machine-checkable receipt (H1b). Open choices:

- promote **H1a** to its own pre-registered entry / short note (it is a genuine
  cross-lane unification on the spectral axis) and/or probe the named frontier
  (non-local / trained-encoder analog of "defeats continuous-resist");
- harden **H2** next (recognizability = legibility), the other theorem-shaped hook;
- leave the internal siblings (H3-H5) parked unless one earns attention.
