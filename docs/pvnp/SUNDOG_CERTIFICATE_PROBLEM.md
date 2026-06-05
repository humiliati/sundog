# The Sundog Certificate Problem

A Phase-4 theorem-shaped synthesis for the `SUNDOG_V_P_V_NP` lane. It banks Phase 3
as the honest boundary and turns "P-vs-NP as metaphor" into one precise object: a
capacity-relative checker-vs-finder problem, three cleanly separated claims, a
substrate migration off mesa, and one constructed instance.

Companion to [`../SUNDOG_V_P_V_NP.md`](../SUNDOG_V_P_V_NP.md) (the roadmap and the
registered definitions §4–§6 this document tightens) and the Phase 1–3 receipts in
[`receipts/`](receipts/). Status: **design / existence note.** It introduces no new
slate and makes no measured promotion claim; its one constructed instance is an
existence proof whose capacity curve is **unrun** (see §6).

---

## 0. Boundary first (not buried)

This is **not** a proof of P≠NP, P=NP, or any complexity-class separation, and it
imports no complexity-theoretic result. It is a bounded alignment-verification
study that *borrows* the finding-vs-checking distinction as vocabulary
(Cook–Levin–Karp as language only). The classical barriers — relativization
(Baker–Gill–Solovay), natural proofs (Razborov–Rudich), algebrization
(Aaronson–Wigderson) — stand against any line that reads as a new route to P≠NP,
and this document is written to stay clear of them. "Capacity-relative one-wayness"
is **not** a standard complexity class and is admissible **only** as a measured
verify/invert/spoof battery, never as a cryptographic one-way-function claim.

Everything below inherits the shared Phase-3 boundary: no cryptographic
one-wayness, no general alignment verification, no wall-time cheapness, no
body-resistance / Sundog-regime-2 result, no progress on P vs NP. The empirical
results are **control-substrate certificate-discipline** results only.

---

## 1. The promise problem, formally

The Sundog Certificate Problem tightens — it does not replace — the registered
definitions §4.1–§4.4 and the five falsification modes §6.1–§6.5 of the roadmap.

### 1.1 Objects

Fix an environment family `E`, an observation tier `O`, a policy class `Π`, and a
**safety predicate** `Safe(π, E) ∈ {true, false}`.

- **Policy Search Problem** (the hard "find" side, §4.1): given `(E, O, Π, Safe)`,
  find `π ∈ Π` with `Safe(π, E)` true. This side carries the search, optimization,
  selection pressure, and mesa-optimizer risk.
- **Signature transform** `H`: a map from a policy's local/indirect observations to
  a compact **certificate** `σ = H(·)`. `σ` is a **Sundog certificate** (§4.3) iff it
  has all three of: **Locality** (read from bounded probes/traces/shadows/gradients),
  **Invariance** (preserves safety-relevant structure under the allowed gauge/
  symmetry/coordinate changes), and **Sufficiency** — *for the registered task
  class* — to verify `Safe` **without full state reconstruction**. The task-class
  qualifier on Sufficiency is load-bearing and may not be dropped.
- **Verifier** `V(π, σ) → {accept, reject, quarantine}`: three-valued, never binary.
  The third value is what lets the promise be honest.

Each certificate must declare the **seven components** (the registered checkable
object): source observations · signature transform · checker · cost accounting ·
false-accept rule · quarantine rule · privilege-leak audit.

### 1.2 The promise

The **operating envelope** is the promise domain `D`. Define the language pair:

- `Π_yes` = policies in `D` for which `σ` certifies `Safe` (the verifier may accept);
- `Π_no`  = policies in `D` for which `σ` certifies `¬Safe` (the verifier may reject);
- inputs **outside** `D` (out of envelope) are the promise gap.

**Promise rule (the single most load-bearing constraint):** on an out-of-envelope
input, `V` must `reject` or `quarantine` — it may **never** silently upgrade its
claim to `accept`. A verifier that accepts outside `D` voids the whole scaffold.

The envelope is **parameterized** (these are parameters, not footnotes): capacity
`C`, horizon, sensor tier, noise, margin, and adversary class. The capacity ladder
has tiers **Small / Medium / Large**.

### 1.3 Capacity-relative one-wayness (the checker-vs-finder axis)

`H` is **capacity-relative one-way** (§4.4) iff, for adversaries below a capacity
`C`, `σ` is sufficient for verification but insufficient for cheap **inversion**,
**spoofing**, or **reward-proxy reconstruction**; above `C`, the one-wayness may
fail and **must be measured**. This is the P-vs-NP-shaped axis — *can a stronger
finder/inverter break the certificate before the verifier loses cheapness?* — and
it is admissible only with a full **measurement contract**: capacity tier ·
adversary class · inversion/spoof task · success metric · failure threshold.

The three adversary tasks are **separate**, with separate metrics:

1. **verify** safety from `σ` (the friendly task);
2. **reconstruct** the hidden target from `σ` (inversion);
3. **spoof** `σ` while violating safety (forge a passing-but-unsafe certificate).

Phase 3 already taught which one is load-bearing: **spoof**, not inversion — an
inversion failure is often near-tautological by construction (any view exposing the
verifier's decision leaks the safety bit), whereas a spoof success is a real break.

### 1.4 The named-failure contract

Every way the problem can fail is **pre-named** (§6.1–§6.5), and each becomes a
precise reject/quarantine clause rather than a silent degradation:

| Mode | Failure | Disposition |
| --- | --- | --- |
| **6.1 Certificate vacuity** | `σ` adds no advantage over ordinary features / a renamed reward channel | not a certificate; collapse to representation learning |
| **6.2 Sufficiency failure** | `σ` discards information `Safe` needs | named quarantine: "signature insufficient for this predicate" |
| **6.3 Inversion / spoofing** | adversary reconstructs/spoofs cheaply enough to Goodhart `σ` | one-wayness fails at measured capacity `C` |
| **6.4 Verifier overhead** | `V` not cheaper than search, or needs privileged state | no complexity-theoretic framing earned |
| **6.5 Boundary absence** | `V` succeeds/fails with no predictable envelope | no useful claim; a bounded verifier must say where it stops |

Metric ordering is fixed: **false-accept rate is primary** (safety); false-reject
secondary; utility/reward preservation reported only after safety discipline.

---

## 2. The three claims, separated

The single biggest error would be to conflate these. They are three claims with
**three different verdicts**. All figures are read from the dated receipts.

### Claim 1 — Cheap verification (an op-count result, never wall-time)

*Signature verification costs fewer operations than the rollout/search work it
certifies, inside a frozen toy envelope.*

- **Phase 1 v6** — bounded positive under the registered v6 op-count protocol:
  `C_total_signature_ops / C_rollout_ops = 527297 / 555876 = 0.948587 ≤ 1.0`
  ([`receipts/2026-05-31_phase1_toy_verifier_v6.md`](receipts/2026-05-31_phase1_toy_verifier_v6.md)).
  **Honest figure:** `0.948587` is a *conservative upper bound* — its numerator
  double-counts calibration calls (2496 signature calls incl. 64 calibration envs ×
  3 policies vs 2304 measurement rollout calls); the clean measurement-only
  recomputation is **≈ 0.879**, still below the gate. Cite ≈0.879 as the
  measurement and 0.948587 as the conservative bound; never as a tight number.
- **Phase 2 v1 (mesa-bridge)** — bounded positive under the frozen contract:
  `C_total_certificate_ops / C_raw_trace_audit_ops = 6867420 / 9310447 = 0.73760368`
  ([`receipts/2026-05-31_phase2_mesa_bridge_v1.md`](receipts/2026-05-31_phase2_mesa_bridge_v1.md)).
  This uses a **weaker comparator** (a same-artifact raw-trace audit, not full mesa
  regeneration). The two ratios come from **non-comparable comparators** — do not
  average or rank them.

**Cost is op-count only.** Wall-time is diagnostic-only and was withdrawn as
non-reproducible (v6 wall-time is ~1603× *slower*). This claim does **not** say
"polynomial," "in P," or "cheap in general."

### Claim 2 — Capacity-relative spoof resistance ("bounded but fragile")

*A consensus-level spoof repair that holds on frozen seeds — and nothing about
single-block safety.*

- **v0** — `falsified_registered_cell`, `capacity_threshold ≤ small`. A source-bound
  seed block of the registered unsafe controller `phase5_l_mixed_lambda_0_7_small`
  (seed 40000) was accepted with no breach/quarantine (sig `0.23545148`, geom
  `0.28284839`, both views), at **6.25%** of the small-tier budget. The mechanism
  was seed-block **mean drift**, not the razor-margin cell; the spoof was the
  load-bearing probe.
- **v1** — `named_quarantine`, repair strength `consensus-only repair`. The K=4/M=3
  block-consensus rule closed the v0 spoof at consensus (0 unsafe consensus
  accepts), but **2 single blocks** still crossed (seeds 70000, 90000), and the run
  quarantined on the `mixed_objective_laundering` gate at the protected anchor
  `l_mixed_lambda_0_95_medium` (flag fired 2/4 < M=3).
- **v2b** — bounded positive, `consensus-only disclosure repair`, on frozen
  promotion seeds 140000–170000: 0 unsafe consensus accepts, signature accept floor
  3/3. **The "bounded but fragile":** exactly **one** unsafe *block-level* accept,
  `l_mixed_lambda_0_7_small` seed 140000 (sig `0.24505205`, geom `0.31528229`),
  crossing the fixed 0.23/0.18 lines by drift but only 1/4 → no consensus →
  `source_block_safety_claim_allowed = false`. The positive **rests on seed-luck at
  the anchor**: its observation mean *straddles* the 0.5 line (0.5369 / 0.4738 /
  0.4631 / 0.5884 → 2/4), where the adjacent pre-freeze seeds 100000–130000 drift
  entirely below 0.5 (→ 0/4 → laundering) and would have quarantined.

This claim is **not** a source-block-safety claim; the consensus qualifier and the
`source_block_safety_claim_allowed = false` disclaimer travel with it. v0 (falsified)
is not revised by later versions.

### Claim 3 — Disclosure robustness (a pre-registered NULL)

*The disclosure repair does not generalize across fresh seeds at the near-line
anchor. This is a disclosure-robustness null, not a safety failure.*

- **v3** — `named_quarantine — disclosure_robustness_null` (the pre-registered
  expected outcome). A multi-battery gate over N=3 fresh disjoint batteries (seeds
  180000–290000) found the anchor `l_mixed_lambda_0_95_medium` `clean_consensus` on
  **all three** (v3-A 1/4, v3-B 0/4, v3-C 1/4) — clean on **4 of 6** total batteries,
  so the v2b straddle was seed-luck, not a stable property. The other three
  registered mixed cells stayed robustly disclosed. The unsafe side **stayed
  closed**: 0 unsafe consensus accepts on all three fresh batteries; 8 block-level
  crossings, none reaching consensus.
- **v4 / v4-A** (design notes, not receipts — no slate frozen): the search for a
  *different* reward-blind behavioral channel was exhausted. The reward-only
  basin-position channel is **structurally inert** (response 0.000 — the basin move
  feeds only reward, which reward-blind feed-forward policies ignore at inference).
  A verify-first experiment then showed the basin-**observation** channel is
  action-visible but **not specific**: a pure-signature policy (no basin reward)
  responds at **0.431** vs the reward policy's 0.558 (ratio **1.29×**, *worsening*
  with training), because any policy wired to the basin observable responds to
  editing it. No basin-based fix is available; see
  [`PHASE3_V4_PATHA_VERIFY_NOTE.md`](PHASE3_V4_PATHA_VERIFY_NOTE.md).

**This is not a safety break.** v2b is not retracted — it holds on its frozen seeds;
v3 only shows it does not generalize. The Phase-3 net in one line: a consensus-level
spoof repair that **holds** (v1→v2b); a single-battery disclosure repair that
**holds on its frozen seeds** (v2b); a disclosure repair that does **not survive** a
multi-battery robustness test at the near-line anchor (v3). The observation channel
is the only reward-blind mixed-objective signal, and it is seed-fragile at the
anchor.

### Why three claims, not one

Claim 1 is about **cost** (op-count cheapness, earned). Claim 2 is about **spoof
hardness** (earned at consensus on frozen seeds, fragile under seed choice). Claim 3
is about **disclosure robustness** (a measured null). The P-vs-NP-shaped object
needs all three to co-hold *on the same instance, at the same capacity*; Phase 3
shows they do not co-hold robustly on mesa.

---

## 3. Substrate migration: off mesa

**The problem in one sentence:** the Phase-3 capacity experiment runs on **mesa, a
measured-bad one-wayness substrate**, and its repair does not generalize — so the
experiment must move.

### 3.1 Why mesa is the wrong substrate

Mesa is **marginal on body-resistance**. The 5D `net.7` shadow nearly reconstructs
the controller's hidden state — `FVE(net.7 | 5D) ≈ 0.97–0.99` — because `net.7`
(256-wide) is intrinsically ~2-dimensional (participation ratio ≈ 2.0): it is a
function of the 6-dim observation, so **there is no high-dimensional body to
resist** ([`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md)). This is
**three-for-three**: every *measurable-control* substrate the program has tried is
marginal — Mesa (PR≈2), Navier–Stokes C1 (FVE≈0.99), Sabra shell (eff-rank ≈1.7 of
30). Where the projection is near-invertible, state-rigidity and control-rigidity
nearly coincide and the one-wayness separation **collapses toward vacuity** (strictly
non-vacuous, physically marginal). A capacity-relative one-wayness claim needs a
shadow that **provably loses body detail**; mesa's does not.

### 3.2 The north star: the Aharonov–Bohm topological witness

The program's one *exact* regime-2 separation is topological (Faraday Phase 7,
B7-topology). On a non-contractible patch (`H¹ ≠ 0`), the loop-holonomy shadow
`∮A = Φ` is **state-insufficient** (one flux number ≠ the interior field `B(x)`;
many bodies map to the same shadow) yet **control-sufficient** (the AB phase is
`qΦ/ħ`, exactly). Crucially the resistance is **independent of dimension** — the body
is low-dimensional but resists *because the obstruction is cohomological, not
dimensional*. This is the conceptual target: **control without reconstruction,
resistance designed in by construction, not hoped for from scale.**

There are thus **two resistance axes**: **dimensional** (the body is too big to
reconstruct) and **topological/algebraic** (the body is globally under-determined by
the shadow). The body-resistance **continuum**: Faraday is the **exact-zero** anchor
(closure is the Bianchi identity `dF = d(dA) = 0`, so the shadow reconstructs the
body exactly — zero by theorem); mesa / NSE-C1 / shell are the **marginal interior**
(near-invertible); AB is the **exact separation** (topological). A one-wayness
substrate must sit at the **AB end**.

### 3.3 Candidate substrates (engineer the resistance in)

The required properties are the inverse of mesa's failure: (a) state-insufficiency
**by construction** (a fiber/quotient the shadow provably cannot invert — *certified,
not a measured FVE<1*); (b) control-sufficiency preserved (the decision factors
through the shadow); (c) a **de-confound guarantee** (the latent must not be a
passive function of the input — mesa's fatal flaw); and, for the one-wayness variant,
(d) a **cost asymmetry** (cheaper to check the shadow's control-relevance than to
invert it).

- **LDT / candidate-set lattice (Galois pair)** — the abstraction `α` (grids →
  candidate-set lattice) is many-to-one, so state-insufficiency is **certified**: the
  concretization fiber `γ(a) = {grids consistent with a}` is genuinely non-trivial,
  the one thing every prior substrate lacked. `ded_p(a)` factors through `a` ⇒
  control-sufficient with a soundness guarantee. The C1 twin-state adjudicator ports:
  two solutions in the same fiber get an identical forward pass yet differ in truth —
  the computational analog of the exact AB witness.
- **chatv2 pair-XOR synthetic latent body** — each latent `z_i` lives only in a pair
  XOR `x = u ⊕ v` (`Cov(u, z) = Cov(v, z) = 0`), provably not linearly input-decodable.
  The mandatory de-confound pre-check (linear-input-probe recovery ≈ chance) passed at
  **0.498–0.516**; an earlier window-parity realization **failed at 0.85** (a GF(2)
  recurrence collapsed it) and was caught *before any training compute* — exactly the
  discipline mesa lacked. Control-sufficient (`z1_acc = 0.94` at H=8) with `leak =
  0.50` (the shadow carries nothing about the other latents); the body scales with H.

These are the **empirical** migration targets (both currently unpromoted). The
**theorem-shaped** instance below is a concrete realization of the same "lossy
quotient + cost asymmetry" family.

---

## 4. The constructed instance (the theorem-shaped core)

One clean instance where **checking is cheap, spoofing is capacity-hard, the shadow
is control-sufficient but provably lossy, and the failure branches are pre-named.**
It is an **existence proof / design**, not a measured receipt — its capacity curve is
unrun (§6, RISK 1).

### 4.1 Candidate A — the syndrome / SIS certificate (recommended)

A is the cleanest instance because **source-binding and lossiness are the same
rank-`(n−k)` map `H`** — you cannot loosen the binding without tightening the loss,
which directly closes the v0 trade-off — and the lossiness is a **proof, not an FVE
estimate**, which closes the mesa trap by algebra.

- **Body** `x = (s, e)`: a secret `s ∈ Z_q^k`, a sparse error `e` of weight `w`, a
  public generator `G` and parity-check `H` of rank `n−k`. The local observation is
  `y = Gᵀs + e`.
- **Certificate** `σ = (z, b, t)`: the **syndrome** `z = H y`, a cheap weight witness
  `b`, and a **source-binding tag** `t = PRF_K(y)`. **Safety predicate**
  `Safe := wt(e) ≤ d/2` (the error is correctable / the policy stays inside the safe
  ball).
- **Verifier** `V`: recompute `z = H y`, check `t` (so only `y` is editable), run a
  cheap sound-but-incomplete weight lower bound from `z`; output
  `accept / reject / quarantine`.

### 4.2 Property-by-property

- **Check-cheap (Claim 1 / 6.4).** `V` = a linear map `Hy` + a PRF + a cheap bound.
  **Find** (decode `e` from the syndrome, i.e. find a safe policy) is **syndrome
  decoding — NP-hard**. The cost asymmetry is structural, not measured.
- **Lossy-but-sufficient shadow (4.3 Sufficiency / 6.1 / 6.2).** Because `H` kills
  codewords, `z = H y = H(Gᵀs + e) = He` depends on `e` **only**, and `He` has
  kernel dimension `k > 0`, so **`qᵏ` secrets map to each syndrome** — `s` is
  *exactly* gone. This is the certified state-insufficiency, the inverse of mesa's
  0.97–0.99, and it is true **by algebra, not by a hoped-for FVE<1** — closing the
  mesa trap. Yet `z` still decides `Safe` via a coset-weight bound: sufficient for
  the predicate without reconstructing the body.
- **Capacity-hard spoof (4.4 / 6.3).** A passing-but-unsafe certificate is a
  same-syndrome near-codeword *below* the decoding radius — an **SIS instance below
  an information-set-decoding (ISD) capacity `C`**. Spoof-hardness is the load-bearing
  leg (matching v0's lesson). **Claim spoof-resistance, never inversion-resistance:**
  `s` is provably gone, so an inversion-resistance claim would be vacuous.
- **Binding = loss = the same map.** `V` recomputes `z` from `y`, so only `y` is
  editable, and editing `y` to be unsafe-but-passing *is* a decoding problem. The
  single map `H` does both jobs — this is *why* A beats the controls.
- **Pre-named failure branches.** 6.1 vacuity (would fire if `z` added nothing over
  raw features — closed, `z` is the decision statistic); 6.2 sufficiency (fires as a
  named quarantine inside the false-quarantine band of the cheap bound, RISK 2); 6.3
  inversion/spoof (the capacity-`C` breakpoint where ISD finds a same-syndrome
  near-codeword); 6.4 overhead (closed structurally); 6.5 boundary (the promise
  domain is the frozen `(n, k, w)` regime; out-of-regime inputs quarantine).

### 4.3 Controls B and C (named negative controls, not candidates)

- **B — Merkle commitment.** Binds **identity, not safety**. A sound opening is
  `O(T)` overhead (**6.4**); a sampled opening is unsound (**6.2**); either way it
  collapses to **6.1 vacuity**. B shows that source-binding alone is not a certificate.
- **C — holonomy / `∮A = ∫F`.** The sufficient version has **zero loss by Bianchi**
  (`dF = 0`) — it sits at the **Faraday/mesa pole** of the §3 continuum; coarsening it
  to manufacture loss makes it stop being sufficient. C concretely marks the
  exact-zero anchor a one-wayness instance must avoid.

A wins because it is the only one of the three that holds all four properties at
once: lossy (unlike C), safety-binding (unlike B), check-cheap, and spoof-hard.

---

## 5. The next experiment (return to empirics with a stronger falsifier)

The constructed instance is a design; the **win** is the measured two-sided curve.
Once `(n, k, w)` and the predicate are **frozen**, run a capacity ladder whose single
question is the P-vs-NP-shaped one:

> *Can a stronger finder/inverter break the certificate before the verifier loses
> cheapness?*

- **x-axis:** attacker capacity `C` (ISD effort / model size / compute), Small →
  Medium → Large, exactly the registered ladder.
- **Verifier curve:** `V`'s op-count, which must stay below the find/decode cost
  across the ladder (Claim 1 must not erode as `C` grows).
- **Spoof curve:** the rate at which a capacity-`C` ISD attacker produces a
  same-syndrome near-codeword that passes `V` while violating `Safe` (the
  load-bearing probe, per Phase 3).
- **The breakpoint** is the measured `C` where the spoof curve crosses from 0 to
  positive *while the verifier is still cheap* — a measured capacity-relative
  one-wayness threshold, the thing every Phase-1 receipt reported as
  `capacity_threshold = not_estimated`.

This run is what would convert §4 from existence proof to receipt. It must obey the
same anti-p-hack discipline the Phase-3 receipts earned: freeze the regime before
the attacker runs, keep `wt(e)` and `s` as scoring labels only (never verifier
inputs), and pre-register the verdict branches.

### 5.1 Prototype result (2026-06-04)

A first prototype of the instance is built and its mechanism is **measured-good** on
a real `[48,24]` GF(2) regime (`w=4`, `τ=4`), deterministic and byte-reproducible —
the verify-first pass that v4 never got. All three §4 properties hold by measurement:
**P1** `z=He` is independent of `s` (2²⁴ secrets per syndrome — lossy by algebra);
**P2** check = 2,376 ops vs naive decode ≈5.1M (~2,150× cheaper), with 0 false
accepts / 0 false rejects; **P3** `z` is invariant under an `s`-flip (`s` one-way).
The find-vs-check capacity curve is present: forge success rises 0.00 → 0.04 → 0.38
→ 1.00 with attacker budget while check ops stay flat at 2,376 — a visible
breakpoint. Honest limits: a toy regime, a naive (non-ISD) forger, imported decoding
hardness, and a degenerate cheap-reject branch (RISK 1). See
[`SUNDOG_CERTIFICATE_SYNDROME_PROTOTYPE_NOTE.md`](SUNDOG_CERTIFICATE_SYNDROME_PROTOTYPE_NOTE.md)
and `scripts/pvnp-certificate-syndrome.py`. The frozen, scaled, ISD-attacker run
remains the step that earns a measured capacity threshold.

---

## 6. Open risks and what would falsify this

Led by the biggest, because the lane's last design (the v4 basin channel) *looked*
sound on paper and was falsified before freeze by an actual run.

1. **UNVERIFIED-BY-RUN (the v4 analogy).** No ISD attacker has been run; the
   two-sided capacity curve in §5 does not yet exist. Candidate A is an existence
   proof / design, **not** a measured result, and must be presented as such. The v4-A
   precedent is the standing warning: a mechanism sound on paper (the basin channel)
   was killed by measurement (signature 0.431 vs reward 0.558, ratio 1.29×). Do not
   rest the capacity-relative claim on A until §5 has produced the curve.
2. **The cheap weight bound may be as hard as decoding.** A *tight* `wt(e) ≤ d/2`
   bound from the syndrome can itself be decoding-hard (reopening 6.4). Mitigation:
   use a cheap, sound-but-incomplete lower bound and **pre-register the
   false-quarantine band** as the explicit 6.2 boundary — a quarantine, never a false
   accept (consistent with false-accept-first ordering).
3. **Label-freeze discipline.** `wt(e)` and `s` must be scoring labels only, never
   verifier inputs (grep-enforced), and `(n, k, w)` frozen before the spoof curve —
   else the spoof result is contaminated, exactly the contamination the Phase-3
   receipts were built to avoid.
4. **Hardness import, not demonstration.** A's spoof-hardness rests on an SIS/ISD
   hardness *assumption*, not on an emergent property of a trained body. Legitimate
   for an existence proof, but it must read as "imports hardness," never as a
   cryptographic one-way-function claim (the §0 / Track A guardrail). The empirical
   substrates (LDT, chatv2) are where hardness would be *demonstrated* rather than
   assumed.
5. **No measured `C` yet.** Every Phase-1 receipt reports
   `capacity_threshold = not_estimated`; A's capacity-relative clause must not imply a
   measured threshold that does not exist. The §5 curve is what supplies it.

---

## 7. What this does and does not establish

**Earned (banked):**
- Cheap verification as an **op-count** result in a frozen toy/mesa-bridge envelope
  (Claim 1: ≈0.879 honest / 0.948587 conservative; 0.73760368 on a weaker comparator).
- Capacity-relative spoof resistance at **consensus level on frozen seeds**, bounded
  and seed-fragile, with no source-block-safety claim (Claim 2).
- A pre-registered **disclosure-robustness null** at the near-line anchor, with the
  unsafe side closed (Claim 3) — Phase 3 banked as the honest boundary.
- A **formal promise problem** (this document) tightening §4–§6, and **one constructed
  instance** (§4) with all four target properties and pre-named failure branches.

**Not established (explicit non-goals):** no cryptographic one-wayness; no claim that
verification is "polynomial" or "in P"; no general alignment verification; no
wall-time cheapness; no body-resistance / Sundog-regime-2 result; no progress on P vs
NP; and **no measured capacity threshold** — §5 is the experiment that would earn it.

The honest endpoint: Sundog has a precise object, three cleanly separated and
correctly-bounded claims, a reasoned migration off the marginal mesa substrate, and a
constructed instance where checking is cheap by structure, spoofing is capacity-hard
by an imported assumption, and the shadow loses the body by algebra. The next win is
the measured curve of §5 — not another definition.
