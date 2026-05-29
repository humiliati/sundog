# Yang-Mills P0 - Domain and Receipt Lock

Status: **design lock, filed 2026-05-29**. This is not an executed
pre-registration and admits no claim by itself. No runner code is admitted
until this document is referenced from a Phase 1 runner manifest that fills
every Admission Requirement.

Roadmap: [`../../SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
Lit-pass: [`../../YANG_MILLS_LITPASS_MEMO.md`](../../YANG_MILLS_LITPASS_MEMO.md)
Holding pen: [`README.md`](README.md)
Sibling pattern: [`../riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../riemann/P0_DOMAIN_AND_RECEIPT_LOCK.md)

Purpose:

> Freeze the shape of `sundog_v_yang_mills` receipts before the first
> ensemble is generated, so the lane cannot drift from "finite-lattice
> gauge-invariant certificate test" into "Clay-adjacent narrative" after
> seeing output.

## Scope

This P0 lock covers the first finite-lattice certificate program only:

- Phase 1 gauge-invariance smoke;
- Phase 2 relative-locality certificate on held-out gauge-invariant
  observable labels;
- Phase 3 observable-certificate gate (verifier-style, P-vs-NP-style
  source-binding discipline);
- Phase 4 finite-size / coupling generality, opened only after Phase 2 or
  Phase 3 earns a non-vacuous result;
- Phase 5 external-review packet, blocked until a Phase 2 or Phase 3
  receipt exists.

Out of scope at this lock:

- 4D lattice cells (explicitly deferred per roadmap §1; admission requires
  a fresh P0 amendment after a Phase 2 or Phase 3 SU(2) 3D result);
- improved actions, anisotropic actions, fermions, Higgs sectors;
- smearing or blocking inside the signature vocabulary (fixed loop sets
  only — see "Primary Signature Vocabulary" below);
- topological-charge proxies, Polyakov-loop proxies (deferred until a
  later P0 amendment binds the finite-temperature setup);
- continuum-limit reasoning of any kind.

## Initial Domain Envelope

### Cell Ladder (locked)

The Phase 1/2/3 program runs three cells in sequence. 4D is explicitly
not in the ladder.

| Order | Cell label | Group | Dim | Role |
| --- | --- | --- | --- | --- |
| 1 | `U1_2D` | U(1) | 2 | Abelian instrumentation / leakage smoke only. Not evidence for non-Abelian Yang-Mills. |
| 2 | `SU2_2D` | SU(2) | 2 | First non-Abelian harness cell. 2D YM has a known Wilson-loop structure — useful as a forward oracle for the leakage controls, never as a Clay proxy. |
| 3 | `SU2_3D` | SU(2) | 3 | First plausible non-trivial certificate cell. The primary Phase 2 read happens here. |

### Lattice Slate (locked)

Per-cell lattice sizes, frozen at this lock; revisable only by a dated
P0 amendment, never inside a phase manifest after first run.

| Cell | Sizes | Boundary |
| --- | --- | --- |
| `U1_2D` | 16×16, 32×32 | periodic in both directions |
| `SU2_2D` | 16×16, 24×24 | periodic in both directions |
| `SU2_3D` | 8×8×8, 12×12×12 | periodic in all directions |

The smaller lattice in each pair is the registered Phase 4 finite-size
split partner; promoting a Phase 2/3 result requires the split.

### Action (locked)

Standard Wilson plaquette action for every cell. No improved or
anisotropic action is admitted at this lock.

```text
S = β · sum_{plaquettes P} [ 1 - (1/N) · Re Tr U_P ]   for SU(N)
S = β · sum_{plaquettes P} [ 1 -        Re U_P ]       for U(1)
```

`N = 2` for SU(2). `N = 1` collapses the prefactor for U(1).

### Coupling Slate (locked shape, β values locked here)

A 3-point β slate per cell, frozen now:

| Cell | β slate |
| --- | --- |
| `U1_2D` | { 1.0, 1.5, 2.0 } |
| `SU2_2D` | { 1.5, 2.0, 2.5 } |
| `SU2_3D` | { 2.0, 2.4, 2.8 } |

These β values may be revised exactly once, inside the Phase 1 runner
manifest, before any ensemble is generated, if a pilot reveals that all
three points sit on the same side of the Wilson-loop area-law signal.
After ensemble generation, no β revision is admitted — that would
collapse the pre-registration.

### Ensemble Generator (locked)

In-repo from scratch. No external lattice-gauge library is admitted at
this lock. The algorithm spine is:

- SU(2) cells: Creutz pseudo-heatbath
  ([Creutz 1980](https://doi.org/10.1103/PhysRevD.21.2308)) with the
  Kennedy-Pendleton improvement
  ([Kennedy-Pendleton 1985](https://www.osti.gov/etdeweb/biblio/6427299)),
  alternated with Brown-Woch overrelaxation
  ([Brown-Woch 1987](https://www.osti.gov/biblio/6510289)).
- U(1) cells: staple-based Metropolis with single-link proposal; used
  only as an Abelian instrumentation control.

Each generated configuration must record:

- code commit hash;
- random seed;
- per-sweep acceptance rate (Metropolis) or update count (heatbath);
- wall-clock and sweep count up to first measurement.

### Burn-in, Thinning, Autocorrelation (locked rules)

Frozen rules, applied per cell × β:

- **Burn-in:** minimum 2000 combined sweeps before any measurement. Each
  cell × β must record the burn-in actually used and prove it is ≥ 2000.
- **Thinning:** thinning interval frozen in the Phase 1 runner manifest
  based on a pilot integrated autocorrelation time τ_int(plaquette);
  registered interval must satisfy `thinning ≥ 2 · τ_int(plaquette)`. The
  pilot must be a dated artifact under
  `results/yang-mills/phase1/<cell>/<receipt-id>/autocorr_pilot/`.
- **Autocorrelation reporting:** every receipt reports τ_int(plaquette)
  computed on the post-burn-in measurement series. If post-hoc τ_int
  exceeds half the registered thinning interval, the run is a
  `YM-P1-NEG-X autocorrelation_underflow` quarantine.

### Primary Signature Vocabulary (locked)

Gauge invariant by construction. Fixed loop set only — no smearing or
blocking is admitted at P0.

Per-configuration signature vector consists of the plaquette and three
small Wilson-loop classes:

| Loop class | Loop shape(s) | Per-configuration summary |
| --- | --- | --- |
| `W11` | 1×1 plaquette | mean and variance of `(1/N) Re Tr U_loop` over all positions and (for d=3) plane orientations |
| `W12` | 1×2 (= 2×1) | mean and variance |
| `W13` | 1×3 (= 3×1) | mean and variance |
| `W22` | 2×2 | mean and variance |

For SU(2) cells the signature uses `(1/2) Re Tr U_loop`. For U(1) cells
the signature uses `Re U_loop`. Position and orientation averaging is
**part of the signature** and cannot be undone by a downstream stage.

Forbidden as primary signature components, restated for the lock:

- raw link variables;
- gauge-fixed potentials (admitted only as diagnostic controls, see
  Leakage Controls Battery);
- metadata fields (β, lattice size, cell label, seed);
- any held-out target loop (see below) copied directly or transformed
  invertibly into the signature.

### Held-Out Observable Label (locked)

The held-out target for Phase 2's relative-locality certificate is a
**larger Wilson-loop area-law proxy class** — loops strictly outside
the signature vocabulary.

| Held-out loop class | Loop shape(s) | Used to derive |
| --- | --- | --- |
| `W14` | 1×4 (= 4×1) | |
| `W23` | 2×3 (= 3×2) | per-cell exponential-decay rate γ_held fit on (`W11`, `W14`, `W23`, `W33`) and area-law residual ε_held |
| `W33` | 3×3 | |

The label is the quantile bin of γ_held within the cell × β slate
(low-decay / mid / high-decay). The bin edges are fixed at the tertiles
of the per-cell γ_held distribution measured on the ensemble after
thinning, and **must be frozen before any nearest-neighbor scoring
happens**. Bin-edge freezing is a manifest field.

Why this is not target leakage:

- The signature contains only loops up to area 4 (`W22`).
- The target contains only loops with area ≥ 4 (`W14`, `W23`) or area 9
  (`W33`).
- The pair (`W22`, `W14`) shares no sub-loop of the signature with the
  target. Area-law correlation between them is exactly what is being
  tested, not assumed.

If a Phase 1 smoke shows `W14` or `W33` is recoverable from the
signature to within receipt tolerance without the area-law mechanism
under test (i.e. trivially, via a one-loop linear identity), the cell
is voided as `YM-P3-NEG-A certificate_spoof / target_leakage`.

## Leakage Controls Battery

Every Phase 2 certificate run must include all of the following controls
on the same neighbor graph and same held-out labels. Receipts must
report each control's rank-locality score alongside the primary.

| Control id | Description | Role |
| --- | --- | --- |
| `CTRL_META` | nearest neighbors in metadata-only space (β, lattice size) | metadata-shortcut detector (§7 #2) |
| `CTRL_RAW` | nearest neighbors in raw link / axial-gauge-fixed representation | gauge-variant diagnostic; primary must NOT need this lane to carry signal (§7 #1) |
| `CTRL_RAND` | uniform-random neighbors | floor |
| `CTRL_RAND_STRAT` | random neighbors within same β bin | stronger floor; isolates β-class trivial recovery |
| `CTRL_PERM` | held-out label permuted over the frozen neighbor graph | confirms the scoring is sensitive to label structure, not graph |
| `CTRL_GAUGE_RAND` | primary signature recomputed on configurations after a random gauge transformation | invariance witness — primary must match within numerical tolerance; raw-link control must NOT |
| `CTRL_FINITE_SIZE` | nearest-neighbor matches restricted to the held-out lattice-size partner | finite-size artifact detector (§7 #4) |

A Phase 2 run with any of the following is **void**, not interpretable:

- `CTRL_META` rank-locality ≥ primary's;
- `CTRL_GAUGE_RAND` shows the primary signature is not invariant within
  numerical tolerance;
- `CTRL_PERM` shows non-zero rank-locality at the same significance as
  the primary (indicates graph contamination).

## Admission Requirements

No Phase 1, 2, 3, or 4 run is admitted unless the runner manifest
states:

- cell label (one of `U1_2D`, `SU2_2D`, `SU2_3D`);
- lattice size (one of the two registered sizes for the cell);
- β value (one of the three registered slate points);
- boundary condition (must be `periodic`);
- action (must be `Wilson` for this lock);
- generator algorithm and update mix ratio;
- random seed;
- burn-in sweep count (≥ 2000);
- pilot τ_int(plaquette) source and value;
- registered thinning interval (must satisfy `≥ 2 · τ_int`);
- signature vocabulary version (must be `v1` for this lock);
- held-out target vocabulary version (must be `v1`);
- γ_held bin-edge values, frozen before scoring;
- control set used (must include all seven entries in the Leakage
  Controls Battery);
- output directory;
- code commit hash;
- exact command line (gates = the exact unchanged command, per
  spec-self-consistency discipline);
- compute cap declaration (each invocation ≤ 10 minutes wall time on the
  repo reference machine; over-cap runs must be re-staged at a smaller
  lattice and re-locked).

If any field is missing, the output is exploratory and cannot be cited
as a receipt.

## Outcome Branches

Phase-level branches. A run lands in exactly one branch.

| Branch | Trigger | Disposition |
| --- | --- | --- |
| A — bounded positive | primary signature beats every control on held-out γ_held label, and `CTRL_GAUGE_RAND` confirms invariance | bounded relative-locality receipt at cell × β |
| B — metadata-only | `CTRL_META` matches or beats primary | named null `YM-P2-NEG-B metadata_only` |
| C — no rank-local | primary fails to beat `CTRL_RAND` or `CTRL_RAND_STRAT` | named null `YM-P2-NEG-A no_rank_local_structure` |
| D — gauge leakage | `CTRL_GAUGE_RAND` breaks signature invariance | quarantine `YM-P1-NEG-A gauge_leakage`; lane voided until algorithm corrected |
| E — target leakage | held-out loop class proves derivable from signature without area-law mechanism | quarantine `YM-P3-NEG-A certificate_spoof` |
| F — finite-size void | signal disappears or reverses under `CTRL_FINITE_SIZE` partner | quarantine `YM-P4-DEFERRED_FINITE_SIZE` |
| G — coupling triviality | result tracks β bin only, not γ_held inside β bin | named null `YM-P2-NEG-C coupling_triviality` (§7 #5) |
| H — scope leak | interpretation requires changing cell, lattice slate, β slate, signature vocabulary, or held-out target after run | domain-leak quarantine, file new dated probe spec |

Phase 1 has its own subset (A / D / and a special `A-with-suspicious-RAW`
verdict where `CTRL_RAW` is anomalously invariant, which quarantines as
implicit-gauge-fixing).

## Anti-Scope-Creep Rule

If a Phase 1, 2, or 3 run lands in a null or quarantine branch, the
next action is a new dated probe spec under
`docs/yang-mills/specs/YYYY-MM-DD_<cell>_<phase>_<short-label>.md` with
a new falsifier — never a silent domain expansion.

Examples of forbidden silent expansion:

- adding `W15` to the signature after `W22` failed to discriminate;
- shifting the SU(2) 3D β slate after seeing γ_held bin distributions;
- promoting 16×16 SU(2) 2D to 32×32 inside the same receipt because the
  smaller lattice was noisy;
- introducing smearing inside the same receipt after fixed-loop signal
  was weak;
- jumping to 4D without an explicit P0 amendment after a 3D positive.

## Receipt Storage

Raw outputs:

- `results/yang-mills/phase1/<cell>/<receipt-id>/` — gauge-invariance
  smoke artifacts (autocorr pilot, gauge-randomization residuals);
- `results/yang-mills/phase2/<cell>/<receipt-id>/` — relative-locality
  scoring artifacts;
- `results/yang-mills/phase3/<cell>/<receipt-id>/` — observable
  certificate artifacts;
- `results/yang-mills/phase4/<cell>/<receipt-id>/` — finite-size /
  coupling generality artifacts.

Reviewed receipts:

- `docs/yang-mills/receipts/YYYY-MM-DD_<cell>_<phase>_<short-verdict>.md`

The reviewed receipt must cite the raw result directory and fill the
template at [`../../yang-mills/RECEIPT_TEMPLATE.md`](../../yang-mills/RECEIPT_TEMPLATE.md).

## Public Language Boundary

Allowed surface, exactly per roadmap §12:

> Sundog is drafting a finite-lattice Yang-Mills certificate lane. It
> asks whether gauge-invariant shadows preserve bounded structure beyond
> controls.

Forbidden surface, restated and binding on every receipt:

- "Sundog has a Yang-Mills result."
- "Sundog is approaching the Clay problem directly."
- "Sundog found a mass gap."
- "Sundog proves confinement."
- "Finite-lattice correlations imply the continuum theorem."

Competitor framing — L-CNN, finite-N bootstrap, Wilson-loop neural,
equivariant diffusion — must be treated as live baseline language, not
as a "raw ML" foil (per lit-pass §D).

## External Reviewer Category (locked)

Phase 5 reviewer-packet target category: **lattice gauge theorist**.

The reviewer packet must be written assuming familiarity with Wilson
action, plaquette / Wilson-loop / area-law vocabulary, heatbath /
overrelaxation, integrated autocorrelation time, and finite-volume
artifacts — but not assuming familiarity with the Sundog receipt /
relative-locality framing, which the packet must orient.

Specific reviewer questions (locked here for §5 packet drafting):

1. Is the held-out target γ_held independent enough from the signature
   vocabulary that an area-law mechanism is the natural explanation for
   any positive result?
2. Are the seven leakage controls sufficient, or is there a known
   gauge-theory failure mode they miss on a finite lattice this small?
3. Is the τ_int / thinning rule strict enough at the registered β
   values?
4. Is the SU(2) 3D β slate { 2.0, 2.4, 2.8 } a reasonable choice for
   spanning confinement-to-perturbative regimes at 8³ and 12³?
5. Are there standard lattice-gauge artifacts at 8³ that would
   trivialize a positive read?

Reviewer is asked **not** to assess: continuum-limit implications, mass
gap implications, confinement proof implications.

## Open Decisions Closed By This Lock

| Roadmap §11 item | Resolution |
| --- | --- |
| 1. SU(2) only, or abelian toy first? | `U1_2D` instrumentation cell + `SU2_2D` harness cell + `SU2_3D` primary cell (staged ladder). |
| 2. Which library/generator? | In-repo Creutz heatbath + Kennedy-Pendleton + Brown-Woch overrelaxation for SU(2); staple-based Metropolis for U(1). No external library at this lock. |
| 3. Which observable label is least likely to be a metadata shortcut? | Larger-Wilson-loop area-law proxy class (γ_held tertile bin), with signature vocabulary strictly disjoint from target vocabulary. |
| 4. Phase 2 fixed loop sets only, or smearing/blocking? | Fixed loop set only at P0. Smearing/blocking deferred to a later P0 amendment after the fixed-loop signal either passes or fails. |
| 5. Reviewer category? | Lattice gauge theorist. |

## Current State

- 2026-05-29: roadmap [`SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
  and lit-pass [`YANG_MILLS_LITPASS_MEMO.md`](../../YANG_MILLS_LITPASS_MEMO.md)
  filed. Prereg holding pen [`README.md`](README.md) opened.
- 2026-05-29: this P0 lock filed; cell ladder, lattice slate, action,
  β slate, generator algorithms, burn-in / thinning rules, signature
  vocabulary v1, held-out target vocabulary v1, seven-entry leakage
  controls battery, manifest admission requirements, outcome-branch
  table, anti-scope-creep rule, public-language boundary, and reviewer
  category all frozen. Five §11 roadmap decisions resolved.
- 2026-05-29: no runner code admitted yet. Next artifact:
  `docs/yang-mills/RECEIPT_TEMPLATE.md`, then a Phase 1 runner manifest
  that fills the Admission Requirements above.
