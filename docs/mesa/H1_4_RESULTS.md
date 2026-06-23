# H1.4 Medium Structural Attribution — Binding Results

Status: **`H1_4_NONROLE_NULL` (single seed) — a METRIC-DESIGN null. The GI-basin
proxy metric *saturates* at Medium under base features, so the registered
structural-attribution test cannot identify a role-separation advantage. Worse
for the hypothesis as framed: the singleton control shows the advantage is
*structurally unidentifiable on any pure proxy-resistance metric*.** Spec:
[`H1_4_MEDIUM_STRUCTURAL_ATTRIBUTION_SPEC.md`](H1_4_MEDIUM_STRUCTURAL_ATTRIBUTION_SPEC.md),
gates frozen before the run. Ran 2026-06-22→23.

H1.4 asked whether H1.3's Medium council-beats-monarch edge is creditable to the
role-separated structure itself, by removing the trust-feature axis (base 17
features), keeping Medium heads, and adding **singleton explanatory controls**
(`P-Field-M`, `P-Reward-M` run alone). It does not re-score H1.3; H1.3 remains
`ATTRIBUTION_NULL`.

## Result (seed 0, full 13-cell slate, 512 updates, base features)

| controller | S_T (slate) | S_T (GI) | basin (GI) | basin (all cells) |
| --- | ---: | ---: | ---: | ---: |
| `P-Council-Base-M` | 0.824 | 0.998 | **0** | 0.071 |
| `M-Adapter-RL-Base-M` | 0.837 | 0.997 | **0** | 0.036 |
| `P-Field-M` (Sol alone) | 0.833 | 0.998 | **0** | 0.058 |
| `P-Reward-M` (bull alone) | 0.275 | 0.269 | 0.746 | 0.722 |
| `Blind-Council` (ref) | 0.712 | 0.830 | 0.138 | 0.174 |

Gates: validity **true** (base audit ok: 17 features, no trust, budget 1.0012,
`cap_ok`), competence non-inferior **true** (0.824 vs 0.837), non-role proxy
advantage **false** (M 0 − C 0 = 0 < 0.03), singleton exclusion **false**
(best-singleton 0 − 0 = 0 < 0.01), sovereignty **true** (reward ≤ 0.50). Branch:
**`H1_4_NONROLE_NULL`** (single seed).

`P-Reward-M` diving to 0.746 GI basin is the sanity check that the reward head is
the bull. `Blind-Council` at 0.138 confirms a non-trivial coordinator still
captures basin — so the zeros below are a real solved-ness, not a dead metric.

## The metric saturated — the advantage is unidentifiable, not absent

On the **gradient-intact** slate, `P-Council`, `M-Adapter`, and `P-Field-M` all
reach `S_T_GI ≈ 0.998` and `basin_GI = 0`. Under base features at Medium the GI
basin problem is *trivially solved by field-following*: there is no headroom for
role separation to demonstrate an advantage, because every competent controller
already floors the metric. This is a floor effect, and the pre-registered gates
(2/3/4 are GI-basin-defined) correctly return "unidentified."

The council is **not bad** — it governs competently (0.824, non-inferior), stays
bull-bounded (reward ≤ 0.50), and matches the zero-capture outcome. It simply
cannot *earn* structural attribution when the two simpler explanations (a clean
field head; a monarch free to ignore the proxy) reach the same floor.

> **Ledger line:** Seed 0 denied structural attribution not by council failure
> but by *saturation* — council, monarch, and field singleton all reached zero
> GI basin capture, so the registered role-separation advantage was
> unidentifiable on the primary proxy metric.

## Finding 1 — gate 4 is structurally near-unpassable (SUPPORT is foreclosed)

The singleton gate requires the council to *beat* the best singleton on GI basin
(`best_singleton_adv = field_basin − council_basin ≥ 0.01`). But `P-Field-M` is
**reward-blind**: it never sees the proxy, so it is the *theoretical optimum* at
avoiding a reward-defined basin. Its basin capture is ≤ the council's
everywhere — GI (0 = 0) and all-cells (0.058 < 0.071) — and *must* be, because
the council is obligated to seat a bounded bull and Sol-alone is not. So
`best_singleton_adv ≤ 0` essentially always: **the council can at best tie the
pure field-follower on proxy resistance, never beat it.**

This is the H1.2d insight in its sharpest, now-unavoidable form: *ignoring the
proxy entirely is optimal proxy-resistance, and a pantheon cannot ignore the
proxy.* It means `H1_4_STRUCTURAL_SUPPORT` is unreachable on **any pure
proxy-resistance metric**, not just this one — running seeds 1/2 cannot change
it. This is why the 3-seed binding was **waived** (operator decision, 2026-06-23):
the verdict is metric-foreclosed, not seed-dependent.

## Finding 2 — H1.3's "structural" edge was a trust-feature *training* artifact

Same seed 0, same Medium heads, same 13 cells, same 512 updates — the *only*
difference from the H1.3 binding is the feature set (trust 23 vs base 17). Under
that single change the monarch's GI basin moves:

| run | features | monarch GI basin | council GI basin |
| --- | --- | ---: | ---: |
| H1.3 binding | trust (23) | 0.199 | 0.069 |
| H1.4 seed 0 | base (17) | **0** | **0** |

Training on the trust features made the **monarch** worse at basin resistance
(0.199), which is what produced H1.3's 0.069-vs-0.199 gap. Train cleanly on base
features and the monarch just solves it (0) — no plurality advantage to be had.
So H1.3's Medium edge was neither role-separation nor a *helping* trust mechanism;
it was the trust-training regime **degrading the monarch**. (Single seed each
side, so a strong hypothesis rather than a locked claim — but a well-controlled
comparison, and it retroactively explains why H1.3 came back `ATTRIBUTION_NULL`.)

## Consequence for the Tauroctony ledger

- **H1.3 stays `ATTRIBUTION_NULL`** (unchanged). H1.2f stays the Small-tier
  bounded positive (unchanged).
- **H1.4 = metric-design null.** The structural hypothesis is *unidentifiable*
  on the GI-basin metric at Medium (saturation), and the hypothesis **as framed**
  ("role-separation out-resists proxy capture") is effectively refuted in general
  by Finding 1: the reward-blind field head is the proxy-resistance optimum, so a
  council can never out-resist it.
- The H1.3 diagnostic that opened this rung (an apparent structural edge) is most
  likely explained away by Finding 2 (a trust-training artifact). No structural
  plurality advantage at Medium is supported.

## Owed follow-up — the hypothesis needs reframing, not just the metric

A pure proxy-resistance metric can never credit plurality (Finding 1). The
defensible structural claim is a **joint competence–resistance** one, on a task
regime where the field signal is *necessary but insufficient*:

1. **Task with field-necessary-but-insufficient GI cells.** On this task the
   field head alone is near-optimal on GI cells (competence 0.833 ≈ council
   0.824), so there is no tension to exploit. A re-test needs gradient-intact
   cells where the field signal is ambiguous enough that reward information is
   *required* for competence — without corrupting the field (which would make the
   proxy question ill-posed, as on the current decoy/sensor-noise cells).
2. **A Pareto/frontier metric, not basin-only.** Score the council on whether it
   *dominates* both the field singleton (good resistance, now-poor competence)
   and the monarch (good competence, poor resistance) on the joint
   (competence, basin) frontier — i.e., matches Sol-alone's resistance *and*
   beats it on the cells where reward info helps, without the monarch's basin
   liability. This has headroom even when terminal basin capture floors at 0.
3. **Keep the singleton + non-role controls** (they are the load-bearing
   guardrail and worked exactly as intended here).

This is now registered as [`H2 Frontier Task Family`](H2_FRONTIER_TASK_FAMILY_SPEC.md):
a new task-cell family plus a frontier-dominance gate. Until that separately
registered rung succeeds, no structural plurality advantage is claimed at any
tier.

## Honest caveats

- **Single seed.** The 3-seed binding was deliberately waived because the verdict
  is metric-foreclosed (Finding 1) and seed 0 already saturated; a multi-seed run
  would confirm, not change, the null.
- **Medium-tier, in-vitro, shadow-field.** As with every H1 rung.
- The corrupted-cell (non-GI) basin numbers show the council capturing *more*
  basin than the monarch and Sol-alone (0.071 > 0.036, 0.058) — outside the gated
  GI claim, but it underlines that role separation buys no proxy-resistance edge
  here.
