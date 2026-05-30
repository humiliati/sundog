# Chat-v2 Phase 0 — The Minimum Falsifiable Cell (pre-registration)

> 2026-05-29, **draft / not yet run**. The smallest pre-registered experiment
> that could **falsify** the lane claim: *an LLM substrate has a sharp
> body-resistance regime-2.* Phase 0 answers one gate only — **does a
> generatively-trained body resist a narrow decision's shadow, and at what
> complexity?** — before any ledger design, agentic loop, or conversation-scale
> machinery is built. Marginal here (a fourth time) → the whole lane re-scopes;
> sharp → Phase 1 earns its budget.
>
> **Cell chosen (2026-05-29):** D2 = a **synthetic dialable-complexity task**
> (signed off), which couples D1 → a **small transformer trained from scratch**
> (see §5). The synthetic route is the *more informative* minimum: a complexity
> knob lets us watch the verdict move from marginal to sharp inside one
> controlled cell, rather than getting a single yes/no on a fixed task.

## 0. The mechanism this cell is built around

The three substrates measured marginal (NSE C1, Mesa, Sabra shell) share one
property that the cross-substrate notes did not initially foreground: **they are
all control-trained.** Mesa's policy was optimized to *act*; the NSE/shell cells
are dynamics observed under a *control objective*. A system trained only to
produce a decision has gradient pressure to **compress its state down to what
the decision needs** — so its body ends up barely larger than its shadow, which
is exactly the marginal signature (`FVE(body|shadow) ≈ 1`, `eff_dim` small).

An LLM is **generatively trained** (next-token prediction). Its state must carry
*everything predictive of the continuation*, which is vastly more than any one
downstream decision consumes. That is the structural reason an LLM body should
**resist** a narrow decision's shadow — and it is a manipulable hypothesis, not
a given. **Phase 0 builds a substrate where we control training objective,
state dimensionality, and decision width independently, and watch whether
body-resistance appears exactly when the theory says it should.**

## 1. The decisive quantity (a three-number regime-2 fingerprint)

Ported verbatim from the C1/Mesa measure so the result is directly comparable
to the three marginals. With **body** = the residual-stream activations at one
layer and **decision** = a narrow readout the model was *not* solely trained on:

| symbol | meaning | estimator |
| --- | --- | --- |
| `eff_dim(body)` | participation ratio of the body over the task distribution | PCA eigenvalue PR (Mesa pattern) |
| `k_control` | smallest subspace dim that is **control-sufficient** — reads the decision at ≈ full-state accuracy | accuracy-vs-`k` saturation sweep |
| `FVE(body \| shadow)` | how much of the body the **control-shadow** reconstructs, **nonlinearly** | held-out HGB regressor R², perm-controlled (C1 tool) |

where `shadow` = the `k_control`-dimensional control-sufficient subspace.

**Pre-registered verdict** (per swept complexity `H`, see §2):

- **SHARP (the lane's bet wins this gate)** iff there exists a complexity `H*`
  in the swept range where *all three* hold: `eff_dim(body)` **large** (≫ the
  marginals' ~2–18) **AND** `k_control ≪ eff_dim(body)` (`< 1/5`) **AND**
  `FVE(body | shadow)` **low** (`< ~0.6` under the nonlinear estimator — the
  body is *not* slaved to the shadow, unlike C1's 0.99) — *and* this holds
  because the factors are genuinely represented (control-sufficiency real, perm
  control clean), not by artifact.
- **MARGINAL (fourth-in-a-row)** iff across *all* `H`: `FVE ≈ 1` (body slaved to
  the shadow, the C1 pattern) **OR** `k_control ≈ eff_dim` (no compact shadow)
  **OR** `eff_dim` never grows with `H` (the model compresses regardless).

**Headline output = `H*`:** the smallest complexity at which the split turns
sharp (or "none in range"). `H*` is itself the informative result — it locates
*where on the complexity axis* body-resistance begins, which directly predicts
whether a real (very-high-complexity) LLM should be sharp.

## 2. The minimum cell — a synthetic dialable-complexity task

**Generating process (`H`-factor channels).** Each sequence is governed by `H`
**independent latent factors** `z = (z_1, …, z_H)`, `z_i ∈ {0,1}` drawn i.i.d.
per sequence. The token stream interleaves `H` channels round-robin; channel `i`
emits symbols from a distribution biased by `z_i` (e.g. `z_i` flips which of a
channel-specific symbol pair is more likely, or sets a channel-specific Markov
bias). **Predicting channel-`i` tokens well requires inferring `z_i`** from that
channel's earlier symbols — so a model that predicts the whole stream must carry
estimates of *all* `H` latents. The factors are independent by construction, so
the body's *content* is genuinely `H`-dimensional — **the question is whether
the trained model represents it that way or collapses it.**

**Substrate (D1, coupled to D2).** A **small transformer trained from scratch on
next-token prediction** over these sequences (`d_model` ~128–256, 2–4 layers,
few heads — large enough that the residual stream *can* be high-dimensional,
small enough to train in minutes on CPU). Generative training is the load-bearing
choice: it is what lets the body carry more than the decision needs (§0).

**Decision / control objective.** Read out **latent factor `z_1`** from the
residual stream at the final position (a held-out probe target). This is the
narrow decision whose shadow we test. `z_1` is genuinely in the state — the
model needed it to predict channel-1 tokens — so reading it is fair, not
leakage.

**Complexity knob.** `H` ∈ {1, 2, 4, 8, 16, 32} (capped by model capacity). `H`
is the single dial swept; everything else is held fixed across the sweep (spec
self-consistency: the gate is the *same* command at each `H`).

**Expected fingerprint vs `H`** (what each outcome would mean):

- `eff_dim(body)(H)` should **grow** with `H` (more independent factors → higher-
  dim state). If it saturates low → the model slaves/compresses → marginal (a
  real finding, F1).
- `k_control(z_1)` should stay **small and ~constant** (reading one factor needs
  a small subspace, independent of how many other factors exist).
- `FVE(body | z_1-shadow)(H)` should **drop** as `H` grows (`z_1` reconstructs a
  ~`1/H` fraction of an `H`-factor state; the independent `z_2..z_H` resist). If
  it stays ≈1 → the model entangled independent latents into a low-dim code →
  marginal (F1).

## 3. Measures (with the verify-before-file guards baked in)

Computed at **each `H`** (the sweep *is* the experiment):

1. **`eff_dim(body)`** — PCA on the centered body matrix; participation ratio
   `(Σλ)²/Σλ²`. Report raw and per-feature-standardized (the C1 norm lesson).
2. **`k_control`** — sweep `k = 1,2,4,8,16,…`: project the body onto its top-`k`
   PCs, fit a `z_1` classifier, plot held-out accuracy vs `k`; `k_control` =
   smallest `k` reaching ≈ full-body accuracy. **Sanity:** also read `z_2..z_H`
   (they must be recoverable too — confirms the factors are really there, not a
   `z_1`-only artifact).
3. **`FVE(body | shadow)`** — the ported C1 estimator: held-out
   `HistGradientBoostingRegressor` predicting each body coordinate from the
   `k_control`-dim `z_1`-shadow, **block train/test split**, **permutation
   control** (shuffle the shadow → FVE must collapse to ~0). Energy-weighted
   over body coordinates.

**Optional within-cell mechanism control (high value, cheap — flagged for §5):**
train a *second* model on the **same sequences** with a **control-only**
objective (predict `z_1` directly, no next-token loss). Theory predicts it is
**marginal** (`FVE≈1`, `eff_dim` small) where the generative model is sharp —
directly demonstrating that *training objective*, not the data, drives
body-resistance. This is the cleanest single experiment that would explain the
whole three-for-three. Recommended as a Phase-0 add-on; can defer to Phase 1.

## 4. Pre-registered failure modes (named before the run)

- **F1 — Marginal (the expected null given three-for-three).** Across all `H`,
  `FVE(body|z_1) ≈ 1` or `eff_dim` never grows — the model entangles independent
  latents into a low-dim code despite generative training. *Disposition:* the
  body-resistance target may need conversation-scale / time-varying state, or a
  larger model, or it may not appear in this substrate family. Re-scope, don't
  re-skin.
- **F2 — Control-insufficient.** `z_1` is not compactly readable (`k_control`
  large or probe accuracy low) — the model smears `z_1` across the state. High-
  dim body but no compact shadow → not the regime-2 split.
- **F3 — Trivial / leakage.** Apparent sharpness because `z_1` sits as one raw
  coordinate and the task is trivial. Guarded by: projection-only shadow (never
  the label), the `z_2..z_H`-recoverability check, and a `k_control = 1` flag.
- **F4 — Substrate-too-simple / under-trained.** At max `H`, `eff_dim` stays
  small because the model is too small or under-trained (next-token loss not
  converged). *Disposition:* scale model / train longer, **re-register**, not a
  result.

## 5. Design decisions — status

- **D1 — substrate. RESOLVED (coupled to D2):** small transformer trained from
  scratch on next-token prediction. (A pretrained GPT-2 is wrong for a synthetic
  process — it would confound task richness with NL-pretraining.)
- **D2 — task. SIGNED OFF:** synthetic `H`-factor channel process, decision =
  read `z_1`, knob = `H`.
- **D3 — body layer. RESOLVED:** the toy has 2–4 layers — compute the
  fingerprint at *every* layer (cheap) and report the layer with max `eff_dim`
  as the body, plus the depth profile.

**Remaining sub-knobs to confirm before the verdict-bearing run** (defaults in
brackets — these fix the cell per spec self-consistency):

- model size [`d_model=192`, `4 layers`, `4 heads`]; train budget [until
  next-token loss plateaus, cap ~few k steps]; sequence length [≈128] / vocab
  [small, e.g. 2 symbols/channel] / channels-per-`H` mapping [`H` channels];
  `H` sweep [`{1,2,4,8,16,32}`]; **and whether to include the control-only
  contrast model (§3) in Phase 0 [recommended yes].**

## 6. Cost / build / reuse

- **Compute:** train ~6 small transformers (one per `H`) + the fingerprint
  analysis at each. CPU: ~30–60 min total (background-able); GPU: minutes. The
  optional control-only contrast doubles the training count, still cheap. This
  is the "more build" cost the synthetic choice was accepted for.
- **Reuse (do not rebuild):** the `FVE` / effective-rank / perm-control
  estimator is the C1 tool (`scripts/pde_c1_kolmogorov_cell.py`, the
  `state-recon` / `mz-budget` adjudicators); the activation-extraction +
  PCA-subspace pattern is the Mesa tool (`training/mesa/policy.py`, the
  `cliff-pca` basis builder). New: a small from-scratch transformer + the
  synthetic data generator.
- **New code:** one script, `scripts/chatv2_phase0_bodyresist.py` — generate
  data, train the toy LM(s), extract body + `z`, compute the three-number
  fingerprint across `H`, emit `H*` + verdict + manifest. (Not written until the
  §5 sub-knobs are confirmed.)

## 7. What Phase 0 explicitly does NOT do

- It does **not** find or design the conversational **ledger** shadow — Phase 0
  uses a data-driven control-subspace, the cheapest possible shadow. The ledger
  is Phase 1+, earned only if Phase 0 is sharp.
- It does **not** use a real LLM, a multi-turn conversation, or an agentic loop;
  nothing the chat *product* charters (`SUNDOG_V_CHAT*.md`) govern.
- It does **not** claim a result about real LLMs. It returns `H*` + one of:
  *sharp* (proceed to Phase 1, and `H*` predicts the real-LLM expectation), or
  one of F1–F4 (re-scope / re-pick, per the named disposition). **No public
  surface; no promotion.**

## 8. Cross-references

- [`LANE_CHARTER.md`](LANE_CHARTER.md) — the lane mission + the measured mandate.
- [`../threebody/CROSS_SUBSTRATE_NOTES.md`](../threebody/CROSS_SUBSTRATE_NOTES.md) — the two-axis body-resistance framing + the three control-trained marginals this gate extends (and explains, via §0).
- [`../proof/PDE_C2_SHELL_DIMENSIONALITY_PROBE.md`](../proof/PDE_C2_SHELL_DIMENSIONALITY_PROBE.md) — the third marginal + the Path-A-vs-B fork this lane resolves toward B.
- [`../proof/PDE_C1_PROPOSITION.md`](../proof/PDE_C1_PROPOSITION.md) — the C1 measure + the "marginal in every physical norm" ceiling this lane tries to beat.
