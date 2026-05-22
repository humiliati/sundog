# Sundog Isotrophy - K_facet v0.3h Ledger

Working hook:

> Twenty structural-zero receipts and one quarantine that explains itself.
> The audit chain catches itself working.

Status: 2026-05-22, corrected after the O_617 orientation-aware deep dive.
This ledger stages the v0.3h K_facet result for public communication and
external review. The internal methodology + result writeup remains the
authoritative source.

This is **not** a roadmap. It is a staging document that translates the v0.3h
technical result into a Sundog-discipline-shaped public surface, with explicit
claim boundary and falsification surface. The deep research roadmap continues
to live at [`sundog_v_isotrophy.md`](sundog_v_isotrophy.md).

## Anchor Documents

- [`internal/anniversary/kfacet_v03h_writeup.md`](../internal/anniversary/kfacet_v03h_writeup.md)
  - v0.3h methodology + result + reproducibility surface. Authoritative.
- [`internal/anniversary/kfacet_v03h_o617_deep_dive.md`](../internal/anniversary/kfacet_v03h_o617_deep_dive.md)
  - six-probe O_617 bridge anatomy. Corrected attribution: O_617 is a clean
  opposite-strict row; its quarantine comes from a bridge direction outside the
  valid D3 representation, not an admission weakness.
- [`internal/anniversary/kfacet-runner-spec.md`](../internal/anniversary/kfacet-runner-spec.md)
  - runner spec for the audit chain.
- [`sundog_v_isotrophy.md`](sundog_v_isotrophy.md)
  - deep research roadmap covering the full Z3-choreography to Z2-piano-trio
  descent program.
- [`/isotrophy`](../isotrophy.html)
  - public-facing page rendering the v0.3h verdict and audit chain in plain
  English with explicit claim boundary.

## The Load-Bearing Statement

The single sentence that all public copy must trace back to:

> v0.3h resolves 20 strict catalog rows as structural zeros. The sole
> quarantined row, O_617, is a clean opposite-strict row whose bridge direction
> sits outside the valid D3 representation; it is not a Gamma_i audit-chain
> failure and not an admission weakness.

In compact form: **"20/21 structural zeros plus one quarantined O_617
defective-D3 bridge - not a closed 21/21 theorem-facing result."**

The phrasing is load-bearing. The 20/21 and the quarantine appear in the same
sentence. O_617 is named. Any communication that collapses this to "21/21" or
to "Sundog proved the choreography isotrophy theorem" silently retires the
discipline that makes the result interesting.

## What v0.3h Is And Is Not

### What v0.3h Is

- A **pre-registered three-stage audit chain**: sentinel/Gamma runner,
  adaptive-floor reprocessor, and bridge audit on the 21 strict G.2
  single-curve choreographies at `m3 = 1`.
- A **named class of artifact**: the structural-zero receipt. For 20 rows,
  `c_i = d_i = 0` by construction because the standard D3 sector is absent in
  `ker(M_i - I)`. Not by tolerance. Not by post-hoc pruning.
- A **named quarantine**: O_617. The bridge vector is physically valid and
  tangent to the `(E, |L|)` level set, but admitting it produces an invalid
  odd `E(1)` residual and `sigma3^3-I = 3.96e-2` at the bridge-admitted
  kernel. The defect lives in the representation at that boundary, not in the
  Gamma_i audit chain.
- A **method that generalizes**. The receipt-first, closure-relative,
  adaptive-floor, bridge-audit template is the first Sundog audit-chain
  instance applied to theorem-adjacent mathematics rather than a MuJoCo
  controller or a chat widget.

### What v0.3h Is Not

- **Not a closed 21/21 theorem-facing result.** The team explicitly disclaimed
  it. The audit chain is intact; the theorem is not closed.
- **Not a proof of Sundog.** Structural zeros at `m3 = 1` for the strict G.2
  catalog are a specific, defensible, narrow finding.
- **Not a weak-admission story.** The corrected deep dive shows O_617 is
  opposite-strict with admission residual `1.01e-8`. The earlier `1.62e-1`
  figure was its canonical residual, diagnostic-only for an opposite-strict
  row.
- **Not a near-trivial bridge.** The v2 signed isotypic check corrects the
  label: O_617's bridge is near the sign sector (`<v,F_beta v> = -0.9999997`),
  with a small `E` contamination residue from imperfect `sigma3` closure. The
  catalog-wide separator finds no valid standard `E` direction in any row.
- **Not the wider isotrophy program.** The broader Z3-to-Z2 piano-trio descent
  program remains live in [`sundog_v_isotrophy.md`](sundog_v_isotrophy.md).
- **Not an interactive workbench.** The K_facet test is a one-shot audit on a
  fixed catalog; the public surface is the static narrative at `/isotrophy`.

## Falsification Surface

The v0.3h verdict can be falsified or weakened in three named ways.

1. **An eligible row is shown to contain a valid standard D3 block after all.**
   That would invalidate that row's structural-zero receipt. The contract is
   the pre-registered guard set: D3 leakage `<= 1e-3`, gap ratio `<= 1e-3`,
   and first-rejected singular value `>= 1e-3`.
2. **O_617's defect is shown to live in the Gamma_i audit chain rather than in
   the bridge representation.** That would reopen the bridge-audit verdict.
   The six-probe deep dive is the surface to attack.
3. **A row resolved at the adaptive floor is shown to depend on a floor choice
   outside the pre-registered guards.** That would make the receipt conditional
   on the floor rather than structural.

## Evaluation Criteria

Reviewers should look for:

- Pre-registered constants and outcome categories that pre-date row
  interpretation. See `kfacet-runner-spec.md`.
- A receipt directory for every row referenced by the audit chain. See
  `results/isotrophy/k-facet-v03-*`.
- A specific, falsifiable attribution for the quarantined row. See
  `kfacet_v03h_o617_deep_dive.md`.
- A public-facing summary that preserves the same load-bearing phrasing as the
  internal writeup. See [`/isotrophy`](../isotrophy.html).

If any of those are missing or weakened, this ledger is wrong and should be
updated.

## Brand And Promo Notes

Brand and promo copy may reference v0.3h as the theorem-adjacent instance of
the same audit-chain discipline used in the photometric controller and the
trace-conditioned chat experiment.

Standing rules:

1. **The 20/21 and the quarantine appear in the same sentence.**
2. **O_617 is named.**
3. **"Audit chain is intact; theorem-facing result is not closed"** is the
   load-bearing distinction.
4. **"Structural-zero receipt"** is the named artifact class.
5. **No "21/21" framing.**
6. **No weak-admission framing for O_617.** It is clean opposite-strict; the
   quarantine is a defective-D3 bridge at the kernel boundary.
7. **No near-T framing for O_617.** Its bridge is near-S/sign-isotypic; the
   apparent `E(1)` is a contamination residue, not a valid standard block.

## Cross-References

- [`PROMO_HIGHLIGHTS.md`](PROMO_HIGHLIGHTS.md)
- [`BRAND_POSITIONING.md`](BRAND_POSITIONING.md)
- [`SUNDOG_V_CAPSET.md`](SUNDOG_V_CAPSET.md)
- [`SEO_AND_SOCIAL_READINESS_ROADMAP.md`](SEO_AND_SOCIAL_READINESS_ROADMAP.md)
- [`SUNDOG_V_THREEBODY.md`](SUNDOG_V_THREEBODY.md)

## Inspection Trail

- 2026-05-22 - v0.3h writeup and O_617 deep dive shipped to
  `internal/anniversary/`.
- 2026-05-22 - corrected O_617 attribution after orientation-aware admission
  residual check: opposite-strict admission residual `1.01e-8`, canonical
  residual `1.62e-1` diagnostic-only.
- 2026-05-22 - corrected O_617 WHY-dive isotypic label after direct signed
  check and v2 separator: near-S/sign-isotypic, not near-T/trivial.
- 2026-05-22 - `/isotrophy` public-facing page filed at root.
- 2026-05-22 - promo and brand notes filed.

Open follow-ups:

- SEO readiness matrix treatment for `/isotrophy`.
- Homepage pillar or rail card for `/isotrophy`.
- A public-facing v0.4 plan, if any.
