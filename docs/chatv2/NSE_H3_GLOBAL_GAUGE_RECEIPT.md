# NSE-H3 Global-Gauge Probe — Receipt

> 2026-07-07. Read of the frozen `NSE_H3_GLOBAL_GAUGE_SPEC.md`. Post-processing on
> banked samples (no integration). Non-promotional; finite, sampled-support,
> proxy-relative (`NSE-H1-PROXY-ONLY` carried), no C1 promotion, no-publish.

## Regression (precondition): PASS

| anchor | FVE_vw | eq-wt median R² | state perm | control acc vs maj (margin) | control perm |
| --- | --- | --- | --- | --- | --- |
| G=200 | **0.9994** | 0.7107 | −0.0008 ✓ | 0.9995 vs 0.6997 (**0.300**) ✓ | clean ✓ |
| G=300 | **0.9916** | 0.6059 | −0.0017 ✓ | 0.9954 vs 0.7312 (**0.264**) ✓ | clean ✓ |

G=200 reproduces the banked ~0.99 marginal read (comparability ✓). G=300's first
global read is also on the marginal side of the frozen 0.99 line (by 0.0016 —
reported as measured). Both estimator-valid, both control-powered.

## Verdict: `NSE-H3-GLOBAL-REGIME2-NONMARGINAL`

The full matched-Re G=675 attractor (registered stride-4 subsample, n=50k), in
the coverage-free global gauge:

| read | value | gate |
| --- | --- | --- |
| **FVE(Q_K \| Φ_K), variance-weighted** | **0.6322** | < 0.99 (frozen marginal line) — **non-marginal** |
| state residual (energy norm) | **0.368** | vs anchors' 0.001 / 0.008 |
| enstrophy-norm FVE | 0.7041 | reported |
| **equal-weight median per-component R²** | **0.0039** | the *typical* high-mode component is unpredictable from Φ_K (anchors: 0.71 / 0.61) |
| state permutation control | −0.0014 | valid ✓ |
| **control: action from Φ_K** | acc **0.8866** vs majority 0.6923 (**margin 0.194**) | ≥ 0.10 powered ✓ |
| control permutation | 0.6960 vs 0.6960 | clean ✓ |

**Terciles (reported):** control holds in every band — including the tails the
fiber apparatus could never read: lowE acc 0.918/maj 0.872, midE 0.927/0.818,
highE **0.820/0.593 (margin 0.227 — strongest in the high-energy tail)**.
`R²(E_high)` stays 0.72–0.83 per band: the aggregate high-band *energy* remains
fairly predictable while the *configuration* (per-component) is free.

## What this is: the non-marginal regime-2 cell

The anchors' long-standing caveat was marginality: FVE ≈ 0.99 means the attractor
is nearly a graph over Φ_K, and the certified twin-state non-injectivity lives in
a ~1% residual. **At matched-Re G=675 the residual is ~37% energy-weighted and the
typical component is entirely free — while the action remains determined (margin
0.19, powered, permutation-clean, in every energy band).** State genuinely
under-determined + control sufficient = regime-2 without the marginality caveat —
the separation the non-marginal probe (hidim G=1000) went looking for, found at
the matched-Re forcing-move cell.

The full-arc picture now closes coherently: **the same attractor width that
coverage-walled every fiber apparatus is the genuine state-freedom** the witness
wants. In the fiber gauge that width is an obstruction (sparse ε_K-fibers →
DEFER/SLIVER, mode-band-only positive); in the global gauge it is the signal
(FVE 0.63). Two gauges, one geometry.

## Honest caveats (load-bearing)

- **FVE is estimator-relative** (HGB, the frozen validated instrument): a stronger
  estimator could raise it. Cross-cell comparison is apples-to-apples (same
  estimator, same n, permutation-gated everywhere), and the anchor↔G=675 gap
  (0.999 → 0.632) dwarfs plausible estimator slack.
- **Control is powered but not anchor-tight** (0.887 vs the anchors' 0.995+): the
  action is determined by Φ_K with real error at this cell — reported as measured;
  the low-E band's margin is thin (0.046) because that band is 87% majority.
- One cell, one forcing move, proxy-relative labels, 32×32 Galerkin,
  sampled support. Not a theorem, not promotion, no infinite-dimensional claim.

## Ledger

- `NSE-H3-GLOBAL-REGIME2-NONMARGINAL` stands beside the fiber verdicts (different
  gauge; nothing overturned): frozen DEFER / A SLIVER / B fenced / MIDE clean
  mode-band positive / **global non-marginal full-attractor positive**.
- Parked successor: the K-sweep at G=675 via state-recon (m_det bracket at the
  deeper cell) — own registration, new runs.

Cross-refs: `NSE_H3_GLOBAL_GAUGE_SPEC.md`, `NSE_H3_MIDE_SUBREGIME_RECEIPT.md`,
`PDE_C1_NONMARGINAL_PROBE.md` (the doctrine this fulfills),
`results/proof/c1-h3-kf3-g675-adaptive/global_gauge_manifest.json`,
`results/proof/c1-relative-reg-g{200,300}/global_gauge_manifest.json`.
