# Chat-v2 Research Lane — Charter

> Opened 2026-05-29. The **Path-B research lane**: test whether a
> high-dimensional LLM substrate has a **sharp body-resistance regime-2** — a
> low-dimensional, control-sufficient shadow of a genuinely high-dimensional
> state that the shadow does **not** reconstruct. This is the Sundog target
> (state-insufficient yet control-sufficient) that three measured substrates
> failed *marginally*.

> **Naming note.** This `docs/chatv2/` folder is the **research** lane (Path B).
> It is distinct from [`../SUNDOG_V_CHAT.md`](../SUNDOG_V_CHAT.md) (the chat
> *research* roadmap v1, Phases 0–12) and from
> [`../SUNDOG_V_CHAT_V2.md`](../SUNDOG_V_CHAT_V2.md) (the *product* charter for
> the Ask-Sundog widget). The "chatv2" name collides with the product charter;
> rename the lane (e.g. `docs/llm-bodyresist/`) if that bothers — proceeding
> under `chatv2` per direction. Sidecars, tangents, and phase specs live here.

## Why this lane exists (the measured mandate)

The cross-substrate failure map
([`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md))
established, by measurement, that the Sundog regime-2 split has **two axes** —
*body-resistance* (the shadow cannot reconstruct the state; the real target)
and *shadow-irreducibility* (the shadow cannot be simplified) — and that
**every measurable control substrate is marginal on body-resistance**:

| substrate | `FVE(body \| shadow)` / eff-dim | verdict |
| --- | --- | --- |
| NSE C1 (2D Kolmogorov) | `FVE ~ 0.99` (attractor ~18-dim) | marginal |
| Mesa controller | `net.7` eff-dim ~2 (6-dim input) | marginal |
| Sabra shell model | eff-rank ~1.7, shells slaved | marginal (directional) |

The structural reason: a sharp split needs a body that is high-dimensional
**by construction** — not a cascade that concentrates content in slaved modes,
not a controller whose input is six numbers. **An LLM is the natural candidate:
the body (the model's internal state across a sequence/conversation) is
high-dimensional by design; the shadow (a maintained ledger / a small
control-sufficient subspace) is where we ask whether the split finally turns
sharp.** This lane is the test.

## The claim this lane makes falsifiable

> On an LLM substrate, there exists a low-dimensional, **control-sufficient**
> shadow that is **state-insufficient** (does not reconstruct the
> high-dimensional internal state), with the separation **sharp** — the body
> genuinely resists (`k_control ≪ eff_dim(body)`), unlike the three marginal
> substrates.

## Discipline (inherited from the portfolio)

- **Pre-registration before any verdict-bearing run.** Named failure modes
  declared up front; verify-before-claim (this session caught five measurement
  artifacts before they reached a conclusion — that discipline is binding here).
- **No public surface until promoted; nothing promoted without the named
  external/sanity gate.** The chat product surfaces are governed by the v1/v2
  charters, not by this research lane.
- **Reuse, don't rebuild.** The `FVE(body|shadow)` / effective-rank estimator
  (C1 / `pde_c1_kolmogorov_cell.py`) and the activation-extraction pattern
  (Mesa / `training/mesa/`) are both built and validated this session.

## Structure

- `LANE_CHARTER.md` — this file (mission + mandate + discipline).
- `PHASE0_MINIMUM_FALSIFIABLE.md` — the minimum pre-registered cell that could
  falsify the claim (the first gate: *does the LLM body resist at all?*).
- Later phases + sidecars/tangents accumulate here.
