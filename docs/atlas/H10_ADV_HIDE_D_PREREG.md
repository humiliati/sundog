# H10 PREREG — ADV-HIDE-D: can an incentivized body conceal the determine latent, and at what cost?

**Frozen:** 2026-06-11, before any H10 code exists. Lane: Shadow-Invertibility / H3 pooled-shadow
(red team, determine half). Internal slate ref: HS3 in `internal/slates/HYP_SLATE_2026-06-10.md`
(gitignored).

**Standing discipline:** pre-registered KILL — a clean null is a SUCCESS; forward-generate only;
deterministic seeded runs (torch single-thread, fixed integer seeds) + frozen tests; CPU-only,
no external data; name the nearest prior, state the delta.

**Zero-conflict constraint (pinned):** the H3-PC probe-ceiling session is running off
`scripts/shadow_pooled_synthetic_v2.py` concurrently. H10 therefore IMPORTS that module and
modifies NOTHING in it; all H10 code lives in new files.

---

## §0 Spec correction at prereg time (recorded before any run)

The slate's fixed claim placed the "theory-guided analytic moment-matching encoder" inside the
KILL quantifier ("KILL = ... across GRL arms AND the analytic attack"). On the hide-ENABLED
substrate that placement is vacuous, by construction: with a per-unit index channel, the encoder
`φ*(u, i) ⊇ s_i·ẑ_i` (alternating signs `s_i = (−1)^i`, `ẑ_i = ⟨u_disc, a⟩`) has pooled d-component
`mean_i s_i·(d + ⟨a, η_i⟩) = d·(Σ s_i)/K + noise = 0·d + N(0, σ²/K)` — **exactly d-free** (K = 64
even), with per-unit (index-aware) d-information fully intact at the single-unit Bayes ceiling
Φ(1/σ_D) = Φ(1/1.5) ≈ 0.748. An arm that provably breaks cannot sit inside a kill criterion —
that would be an unreachable KILL, the defect class the slate vetting removed elsewhere
(cf. the spec-self-consistency discipline). Correction, pinned now:

- **A3 (analytic) is an EXISTENCE DEMONSTRATION** with its own pinned predictions (§3), not a
  kill arm. It establishes that hide-with-retention EXISTS on the hide-enabled substrate, so the
  live question — the falsifiable content — is **TRAINABILITY**: do gradient-trained adversaries
  FIND it, and at what competence cost?
- **KILL/BREAK bind the TRAINED arms only** (§3). Scope of any KILL: "determine is robust to
  adversarial incentive **within the named attack class** (GRL vs pooled probe, λ_adv ≤ 10, this
  architecture and training budget)" — never more.
- Cross-link (named prior): A3 is the trained-body sibling of the machine-checked
  `pooled_channel_blind` fiber construction (sundogcert `AuditCost.lean`, HS7): index-conditioned
  sign codes place per-unit information exactly in the pooled-mean kernel. H10 asks whether
  TRAINING finds that blind subspace.

## §1 Claim

On a **hide-enabled** variant of the frozen v2 substrate (per-unit sinusoidal positional channel
breaking unit exchangeability), encoders trained with an explicit hide-d objective
(gradient-reversal against a pooled d-probe) land in exactly one of two pre-registered bands:

- **BREAK:** some trained arm-run with BOTH guards intact (§3) drives best-of-probe post-pool
  d-accuracy **< 0.90** at some eval λ — H3's determine-half asymmetry collapses to "everything
  trainable is objective-dependent", and the (pooled d-acc, primary R², retention) frontier
  across (arm, λ_adv) is the banked artifact.
- **KILL (= clean strengthening null):** every guarded trained arm leaves best-of-probe post-pool
  d-accuracy **≥ 0.90 at every eval λ** — determine upgrades from "robust to indifference"
  (banked H3) to "robust to adversarial incentive within the named attack class", with the
  existence gap (A3 hides by hand; training cannot reach it) as the sharpest banked sentence.
- Bands complementary: if NO trained arm passes guards, KILL holds with the pre-registered
  reading "no guarded adversary exists in the attack class". No gray zone.

**Frozen-substrate control (theorem-predicted, falsifiable in itself):** the same arms on the
UNMODIFIED v2 substrate (exchangeable φ, binary d, mirror-symmetric per-unit noise). Symmetry
prior, pre-registered as a PREDICTION not a theorem: per-unit d-information is carried by the
d-odd component, which K = 64 mean-pooling amplifies (×8 SNR); an exchangeable encoder cannot
cancel it positionally, so hide-with-retention should NOT occur — guarded pooled d-acc stays
≥ 0.90. If a guarded frozen-substrate arm hides, the symmetry prior is WRONG — banked as its own
(unexpected) break.

## §2 First leg (the only leg)

**(a) Files:** `scripts/shadow_adv_hide_d.py` (driver; imports `shadow_pooled_synthetic_v2` as
frozen apparatus) + `scripts/test_shadow_adv_hide_d.py` (frozen test) + results under
`results/atlas/h10/`. Receipt `docs/atlas/H10_ADV_HIDE_D_RESULT.md`.

**(b) Hide-enabled substrate:** `gen_hide(n, lam, seed)` = `v2.gen(...)` with per-unit positional
features appended: `pos_i = [sin(2π f i/K), cos(2π f i/K) for f in (1, 2, 4, 8)]` (P = 8), so
units (n, K, 80). Positional features are slot constants, sample-independent, c- and d-independent.

**(c) Encoder:** same architecture family as banked (per-unit MLP 80→128→128→H=32, ReLU,
mean-pool). Training budget identical to banked: Adam 1e-3, batch 256, 120 epochs,
N_TRAIN = 8000 at TRAIN_LAM = 1.0, `torch.set_num_threads(1)`, fixed integer seeds (new offsets,
no hash()).

**(d) Arms** (each on BOTH substrates):

| arm | objective | λ_adv |
|---|---|---|
| β0_regc | reg_c only (baseline; pins primary + A1 retention baselines) | — |
| β0_ret | reg_c + retention-aux (per-unit linear d-head on φ_i, CE, weight 1.0) | — |
| A1 | reg_c + λ_adv·GRL(pooled d-probe) | 0.1, 1, 10 |
| A2 | reg_c + retention-aux + λ_adv·GRL(pooled d-probe) | 0.1, 1, 10 |
| A3 | analytic (hide-enabled only): φ* = [φ_rffonly(u with disc dims zeroed), s_i·ẑ_i]; φ_rffonly = a reg_c body trained on disc-zeroed inputs (the c-pathway needs training — the banked lesson); the d-pathway is the closed-form antisymmetric code | — |

**GRL protocol (pinned):** inner pooled d-probe = `nn.Linear(H, 2)`, persistent, own Adam
lr = 1e-2, updated every batch on CE with normal gradients; encoder receives the reversed
gradient scaled by λ_adv (autograd GradReverse); additionally the probe is re-initialized and
retrained 100 steps (batch 256) at each epoch start, so the encoder never wins by outrunning a
stale probe.

**(e) Evaluation (frozen protocols, fresh data per λ, v2 seed conventions):**
- **Pooled d-acc (the verdict quantity):** best of three independent retrained probes on pooled
  reps — logistic (max_iter 2000), MLP(128, 64) (max_iter 600, random_state 0), kNN (k = 15) —
  balanced accuracy, 5-fold stratified CV, N_PROBE = 2000, eval λ ∈ {0.0, 1.0, 2.0, 3.0}
  (the d-channel is λ-independent at generation; λ enters only via encoder cross-talk — 4 points
  pre-registered to bound probe cost).
- **Primary guard:** pooled ridge c-R² (banked `c_r2` protocol) at TRAIN_LAM ≥ (arm's β0 baseline
  − 0.05). A1's baseline = β0_regc; A2's = β0_ret; A3's = β0_rffonly (its own c-pathway).
- **Retention guard:** per-unit best-of-{logistic, MLP(64,)} d-acc on concat[φ_i, pos_i] (50k
  unit-rows subsampled seed-fixed from N_PROBE samples at TRAIN_LAM)
  ≥ **max(arm's β0 baseline − 0.05, 0.65)**. The 0.65 absolute floor (pinned now: halfway between
  chance and the 0.748 single-unit ceiling) prevents a vacuous guard if a β0 body's per-unit
  retention is already low; baselines measured in the β0 runs and pinned in the frozen test.
- **Frontier artifact:** (pooled d-acc, primary R², retention) for every arm-run, both substrates.

**(f) ABORT conditions (redesign, not result):** GRL training not byte-reproducible under the
fixed-seed protocol (retrain → pooled reps must match exactly); C0/C1/DETERMINE gates of the v2
construction broken by the positional channel (re-checked on gen_hide: raw mean must still wash c
at high λ; single unit must still carry c; pooled d-acc ≥ 0.95 for β0 bodies).

## §3 Pre-registered outcomes

- **KILL / BREAK** as in §1 (trained arms, hide-enabled substrate, 0.90 line, guards as in §2e).
- **A3 predictions (existence demo; pinned):** pooled best-of-probe d-acc ∈ [0.45, 0.55];
  retention ≥ 0.70; c-R² within 0.05 of β0_rffonly. If A3's pooled d-acc EXCEEDS 0.55, the
  construction's exactness claim failed somewhere in the apparatus — abort-grade investigation,
  not a verdict.
- **Frozen-substrate control prediction:** no guarded arm below 0.90 (the symmetry prior); a
  guarded hide here = unexpected break of the prior, banked as such.
- **What CANNOT kill:** A3 (existence demo); unguarded collapses (hide-by-destroying — reported
  in the frontier, decides nothing); the frozen-substrate control failing in the PREDICTED
  direction (that is confirmation).

## §4 Honesty constraints

1. The determine THEOREM (Lean `determination`/`determination_general`) is about the raw averaged
   shadow of a fixed channel; nothing here touches it. H10 tests the TRAINED-BODY analog only.
2. Any KILL is scoped to the named attack class; "concealment is impossible" is never claimed —
   A3 proves the opposite for constructed codes.
3. The hide-enabled substrate is the LIVE case by design; the frozen-substrate control is where
   the symmetry prior does the predicting. Both reported side by side.
4. Probe-class honesty: pooled d-acc is best-of-three probes (the H3 probe-robustness protocol);
   a hide that fools only the linear probe does not count as BREAK.

## §5 Nearest priors (named), and the delta

**External:** Ganin & Lempitsky 2015 (GRL/DANN — the attack vehicle); Elazar & Goldberg 2018
(adversarial removal leaks under retrained probes — the literature-default outcome, i.e. the
KILL direction); Edwards & Storkey 2015 (censoring adversary); LEACE 2023 (exact linear
erasure — linear analog of the antisymmetric code). **Banked:** H3 v2 (the substrate; determine
half tested only under indifferent objectives — the gap this fills); the determination theorems
(Lean); HS7 `pooled_channel_blind` (the certified blind subspace A3 instantiates; the
trainability question is new). **Delta:** first adversarial objective in the tower, aimed at the
robust half, on a substrate where hiding is provably constructible, with a calibrated
hide-vs-destroy guard and a λ_adv-resolved competence frontier.

## §6 Receipt plan

`docs/atlas/H10_ADV_HIDE_D_RESULT.md`: outcome (KILL / BREAK / abort), the full frontier table,
A3 demo numbers vs pinned predictions, frozen-substrate control vs the symmetry prior, guard
baselines, deviations with reasons, frozen-test pin list. Memory + slate status update after
banking.
