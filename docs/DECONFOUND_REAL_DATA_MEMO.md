# The De-confound Wall on Real Data — Problem + Attack Paths

> 2026-06-04. Redirect from the JEPA toy (lit pass found it validation-tier) to the
> **actual blocker**: how to get an *input-undecodable* probing target on a *real*
> substrate, so "the representation **computes** the functional" is non-trivial rather
> than "the functional was already in the input." This is the gate for every real-substrate
> determining-shadow-set / body-shadow read (JEPA, LLMs, the lot). Focused read done; this
> memo frames the requirement, the prior-art attack paths, and a recommended Phase-0.

> **2026-06-04 correction after model-free functional screen:** Othello-GPT-class
> computed-state tasks still open the **de-confound** door, but they do **not** automatically
> provide a linear closure functional. The named Othello slate (legal moves, material
> parity, frontier, mobility) failed the `blocked_by_ill_posed_functional` preflight: none
> produced a non-vacuous linear `k_func << k_state` bracket. Receipt:
> `results/deconfound/othello-functional-screen-2026-06-04.txt`. Othello remains useful for
> de-confound/regime-2; the closure read now routes primarily through semi-synthetic
> injection unless a new canonical linear Othello functional is found before spec lock.

## 1. The requirement (what the toy gave for free)

The instrument needs a functional `u` that is:
- **meaningful** (not a designed artifact, ideally),
- **input-undecodable** — a linear probe on the *raw input* reads it ≈ chance, AND
- **representation-decodable** — the trained body computes it.

Only then is "the body carries `u`" a real claim, not a restatement of "the input contained
`u`." The pair-XOR toy got this by *construction* (parity is input-undecodable). On real
data the input is *given*, and most natural functionals (topic, object class, sentiment)
are sitting right in it → de-confound fails. That's the wall.

## 2. The finding — it is NOT a void. Three attacks, each with prior art.

**A. Computed-state tasks — the natural de-confounded substrate (the breakthrough).**
A model trained to predict a *sequence* often must **compute a hidden state that is not
linearly in the input**. The canonical case is **Othello-GPT** (Li et al.; Nanda's linear
world-representation): input = move sequence, and the model builds a **linear
representation of the board** — which is *computed* from the moves (tracking flips), so it
is **input-undecodable but representation-decodable.** That is *exactly* our de-confound,
satisfied by **task structure rather than construction**, on a *real trained transformer*.
And the board squares are **coupled** by the game rules → the de-confound door is real.
The model-free functional screen corrected the overreach: this does **not** by itself make
the linear closure bracket well-posed. This remains a bridge for de-confound and possible
regime-2 reads; closure needs a separately registered linear functional.

**B. Semi-synthetic injection — the controllable bridge.** Inject a *constructed*
computed functional (e.g. a hidden parity over real features) into real data, à la
**SYNLABEL** (pre-specified ground-truth function → controllable labels). Keeps the toy's
airtight de-confound, upgrades the *substrate* to real distribution + real architecture.
Cleaner than the toy, weaker than A (the functional is still ours).

**C. Causal / amnesic probing — the "computes vs decodes" intervention.** **Amnesic
probing** (Ravfogel et al.) and its successors (**LEACE**, Mean Projection, 2025) remove a
concept from the representation (INLP/LEACE) and test whether *behavior* changes —
sidestepping input-decodability by asking whether the model **uses** the functional, not
whether it's present. A *different* de-confound axis (causal, not informational); useful as
a corroborating second read, not a replacement.

## 3. Assessment

| attack | substrate | de-confound | "more than we know" | cost |
| --- | --- | --- | --- | --- |
| **A. computed-state (Othello-GPT)** | real trained transformer | clean, by task structure | **yes — model-built state** | small model; 1080-feasible |
| B. semi-synthetic injection | real data, real arch | clean, by construction | partial (functional still ours) | medium |
| C. amnesic/causal (LEACE) | any | causal, different axis | n/a (corroborating) | cheap add-on |

**Post-screen assessment:** A dominates for **de-confound** and possible regime-2. B now
dominates for the **closure** read, because it can construct a linear computed functional
while still moving onto a real distribution / architecture.

## 4. Recommended Phase-0 (for the sub-lane, if opened)

**Split the path.** Use Othello-GPT or a small reproduction for the computed-state
de-confound / regime-2 question; use semi-synthetic injection for the closure question
unless the lit pass identifies a nontrivial linear Othello functional. Concretely:
- **De-confound pre-check (ports directly):** confirm the board state is ≤-chance from a
  linear probe on the *input* move-tokens, but well-decodable from the residual stream.
  (Expected to hold — that *is* the Othello-GPT result.)
- **Closure read:** not the named Othello slate. Legal moves, material parity, frontier,
  and mobility are ill-posed for the linear determining-set instrument. Either identify a
  new canonical board functional that is linearly readable from the full board, or pivot
  closure Phase 0 to semi-synthetic injection.
- **Body/shadow read:** does the model's legal-move output (shadow) fail to reconstruct the
  board (body) → state-insufficient/resisting, on a real substrate?
- **Controls:** the `u_null` independent-target control + selection-corrected null carry
  over unchanged; amnesic/LEACE (C) as the corroborating causal read.

## 5. Honest tier

This is a genuine **R2-step-down-payment**: a real trained transformer, a model-computed
state, a free clean de-confound — a real rung above the designed toy. **But** it is a
game-state model, not a frontier LLM; "more than we know" *begins* here (the model's
world-model) and stays bounded. **External review before any promotion; R3 vocabulary
("world model," "understanding") stays gated** even though the literature uses "world
model" for Othello-GPT — we measure de-confound / regime-2 observability structure, and
closure only if a linear functional survives preflight. We do not certify cognition.

## 6. What this unlocks

- The **JEPA lane** gets a sharper wall: computed-state substrates can pass de-confound,
  but a linear closure functional is still a separate requirement.
- The **determining-shadow-set instrument** gets its first real substrate, with the
  body/shadow regime-2 read on a model-built world-state.
- It is the first place the program can honestly ask whether the shadow tells us something
  about an agent **we did not put there**.

## Re-posed decision

The redirect paid off, but with a boundary: **the de-confound wall has a door, and the
linear-closure wall remains.** Recommendation: keep `SUNDOG_V_DECONFOUND` open with two
branches: Othello-GPT for de-confound/regime-2, and semi-synthetic injection for the closure
read unless the lit pass identifies a new canonical linear Othello functional. The kill-gate
worked by refuting the attractive Othello closure slate before any model run.

**Attack-B closure executed:** `docs/deconfound/PHASE0B_ATTACK_B_CLOSURE_RESULTS.md`
records `attack_b_closure_confirmed` on the passed digits 0-pre cell. The
functional-keeper (predict `u = XOR(b_0,b_1,b_2)`) kept `u`
(`k_func=2`) and discarded outside state (`k_state=none`) in 5/5 seeds; the
state-keeper (reconstruct `b`) kept outside state (`k_state=4`) and did not
linearly expose `u` (`k_func=none`) in 5/5 seeds. Median paired
`keeper_gap=7`; `u_null=none` everywhere; selection-corrected `p=0.001`.
Ceiling remains R1.5 because the functional is constructed, not model-discovered.

---

*Sources:* Othello-GPT linear world rep [Nanda](https://www.neelnanda.io/mechanistic-interpretability/othello)
· Sparse probing [arXiv:2305.01610](https://arxiv.org/pdf/2305.01610)
· Amnesic probing / LEACE [arXiv:2506.11673](https://arxiv.org/abs/2506.11673)
· SYNLABEL controllable ground truth [Springer/arXiv:2309.04318](https://arxiv.org/pdf/2309.04318)
· "Fidelity Isn't Accuracy" [arXiv:2506.12176](https://arxiv.org/html/2506.12176)

*Sundog Research Lab — de-confound-wall attack memo, internal. Gates a sub-lane; no run.*
