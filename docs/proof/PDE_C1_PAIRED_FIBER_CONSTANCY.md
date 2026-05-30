# PDE C1 Paired Fiber-Constancy Exhibit

> Proof-track pre-registration, 2026-05-29. Sharpens the C1 Reading-2
> **regime-2 witness** from a matched-radius composition into a **paired**
> test on the same pairs. Status: pre-registered, harness extended,
> verdict-bearing runs pending. Not a Navier-Stokes existence/smoothness
> claim; finite-Galerkin, sampled-support, numerical.

## 1. The gap this closes

The certified regime-2 witness today is a composition of **two population
statistics at a matched radius** `epsilon_K`:

- **State-insufficiency** — the twin-state certificate
  ([`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md)):
  within `epsilon_K` signature-balls the complementary high modes `Q_K`
  vary (non-injectivity on the sampled support).
- **Control-sufficiency** — the kNN/disintegration read
  ([`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md)):
  within `epsilon_K` signature-balls the proxy action is near-constant
  (`mean_minority -> 0`).

Both are aggregate facts about `epsilon_K`-neighbourhoods, computed on the
same support at the same radius. They are **not** yet a statement about the
**same pairs**. A skeptical reviewer can ask:

> Maybe the action-homogeneity is carried by the low-mode-similar
> sub-population, while the genuinely high-mode-**separated** pairs straddle
> the safety-trigger boundary. Your two statistics never look at one pair at
> a time, so they do not exclude this.

That is exactly the Reading-2 fiber criterion
([`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
section "Fiber Criterion"):

```text
Phi_K(x0) = Phi_K(x1)  =>  pi*(x0) = pi*(x1)   (mu-a.e.)
```

The cleanest direct test of that implication is on the certified
non-injective pairs themselves: take the witness pairs (signature-near,
high-mode-separated) and ask whether each pair shares the proxy action.

## 2. Construction

Reuse the twin-state machinery verbatim. For the sampled support, a
**witness pair** `(i, j)` is one the certificate already counts:

```text
||Phi_K(x_i) - Phi_K(x_j)|| <= epsilon_K        (signature-near)
||Q_K(x_i)  - Q_K(x_j) ||   >= delta_H          (high-mode separated)
```

Let `a(x) = 1[lookahead_max(x) > E_max]` be the registered binary proxy
action (the same label the kNN read scores; `E_max` is the held-out
quantile-calibrated threshold of the cell). Define the **witness-pair action
disagreement fraction**:

```text
D_witness = #{ unique witness pairs (i,j) : a(x_i) != a(x_j) }
            / #{ unique witness pairs }
```

`a` is symmetric in the pair, so directed and unique de-duplication give the
same fraction up to asymmetric k-NN membership; both are reported, the
**unique** fraction is the headline. A **candidate-pair** disagreement
`D_candidate` (signature-near, *without* the high-mode-separation filter) is
reported as a comparator: if `D_witness ~ D_candidate`, high-mode separation
does not induce action disagreement, which is the regime-2 claim.

## 3. Pre-registered procedure

The harness extension is **additive only** — it adds per-pair action
agreement to the existing `twin-state` adjudicator and does not alter any
primary twin-state computation. The runs therefore **reproduce the existing
certificates bit-for-bit** (same seeds, same `E_max`, same witness pairs)
and add the paired read in the same pass. The reproduced primary fields are
a built-in no-regression check.

Two regimes, mirroring the two-regime witness:

```text
# G = 200 (k_f = 2, K = 3, d = 18)  -- reproduces c1-kolmogorov-v5-twin-state
python scripts/pde_c1_kolmogorov_cell.py --preset lock_v5 \
  --adjudicator twin-state --out results/proof/c1-paired-fiber-g200

# G = 300 (k_f = 2, K = 3, d = 18)  -- reproduces c1-rg-v1-g300-twin-state
python scripts/pde_c1_kolmogorov_cell.py --preset lock_v7_g300 \
  --adjudicator twin-state --out results/proof/c1-paired-fiber-g300
```

Each is ~22 min (G=200) / ~42 min (G=300), seed `20260528`, `sample_count =
50000`. Background tasks per the inline-compute rule.

## 4. Pre-registered verdict

Threshold is the **existing** `delta_action = 0.10` (the same fiber-
incompatibility tolerance used by the bin and kNN adjudicators) — no new
goalpost. The actual `D_witness` is expected to be far smaller (the kNN
`mean_minority -> 0`); the threshold is the pre-committed pass line, the
margin is reported.

| `paired_fiber_verdict` | condition |
| --- | --- |
| `PAIRED_FIBER_CONSTANCY_POSITIVE` | certificate stands, proxy non-constant, `D_witness <= delta_action` |
| `PDE-C1-PAIRED-NEG` | certificate stands, proxy non-constant, `D_witness > delta_action` |
| `PAIRED_FIBER_DEFERRED_VACUITY` | proxy structurally constant (`damp = 0` or `no_op = 0`) on this run |
| `PAIRED_FIBER_UNDEFINED` | no twin-state certificate / no witness pairs |
| `SMOKE_ONLY` | non-verdict-bearing preset |

The secondary verdict **never overrides** the primary `TWIN_STATE_CERTIFIED`
verdict; it is an additional read on the same run.

**The negative is real.** `PDE-C1-PAIRED-NEG` is the named-negative boundary
(regime 3) from the Postulate-1 sidecar, instantiated on the witnessed
pairs: certified non-injective fibers that carry **incompatible** optimal
actions would show the low-band signature is *not* control-sufficient. If it
fires it is filed, not rescued by a post-hoc objective change.

## 5. What this does and does not establish

**Does (on success):** composes state-insufficiency and control-sufficiency
on the **same** certified pairs, closing the matched-radius-aggregate gap and
directly instantiating the Reading-2 fiber criterion on the non-injective
support. Serves sub-goals (2) "sufficient for the action, not for
reconstruction" and the core of (3) "characterize the null set" — the
witness high-distance percentiles (already in the certificate) quantify the
null coordinate `Q_K` as thick, while `D_witness ~ 0` quantifies the action
coordinate as degenerate, on the same conditional measure `P(. | Phi_K)`.

**Does not:** it is finite-Galerkin, sampled-support, numerical, and
proxy-specific (one objective family, one forcing geometry). It is not a
theorem about the infinite-dimensional NSE attractor and does not by itself
discharge external review. C1 stays **PROVISIONAL, UNPROMOTED**.

## 7. Result (2026-05-29) — POSITIVE at both regimes

Both deterministic re-runs returned `PAIRED_FIBER_CONSTANCY_POSITIVE`, and
both reproduced their twin-state certificate **bit-for-bit** (verdict,
witness-pair count, `damp_fraction`, `epsilon_K`, `E_max`, `delta_H`,
witness high-distance p50 all identical to the original) — so the paired
read is provably additive on the *exact* certified pairs, not a re-derived
or drifted sample.

| quantity | G=200 (`lock_v5`) | G=300 (`lock_v7_g300`) |
| --- | --- | --- |
| `paired_fiber_verdict` | `PAIRED_FIBER_CONSTANCY_POSITIVE` | `PAIRED_FIBER_CONSTANCY_POSITIVE` |
| `D_witness` (unique) | **0.0367** (25460 / 693795) | **0.0382** (36003 / 942834) |
| `D_witness` (directed) | 0.0360 | 0.0376 |
| `D_candidate` (comparator) | 0.0319 (1263121 pairs) | 0.0290 (1318748 pairs) |
| `D_witness - D_candidate` | +0.0048 | +0.0092 |
| `delta_action` threshold | 0.10 (clears ~2.7x) | 0.10 (clears ~2.6x) |
| no-regression vs certificate | bit-for-bit | bit-for-bit |

Results: `results/proof/c1-paired-fiber-g200/`, `…-g300/`.

**Reading.** `D_witness` is ~3.7-3.8%, not literally zero — a small set of
state-separated signature-near pairs do straddle the safety-trigger
boundary. The decisive fact is that `D_witness` is **within ~1 point of
`D_candidate`** in both regimes: high-mode (`Q_K`) separation adds almost
nothing to action disagreement. So the residual non-constancy is a
**signature-space boundary layer** — pairs whose look-ahead sits right at
`E_max` flip the label regardless of high modes — and **not** something the
unresolved state drives. This is the Reading-2 regime-2 claim sharpened:
the proxy action is (a.e.) blind to `Q_K`; what little non-constancy exists
is governed by signature proximity near the decision surface, which is
exactly the coordinate the controller is meant to use. It composes the
state-insufficiency and control-sufficiency reads on the **same** certified
pairs, in both Grashof regimes — closing the matched-radius-aggregate gap.

**Honest residual.** This does not pin the ~0.5-0.9 point witness-minus-
candidate gap to signature distance vs `Q_K` definitively; a matched-
signature-distance control (compare `Q_K`-separated vs `Q_K`-close pairs at
equal signature distance) would. Registered as a follow-up; not required for
the POSITIVE verdict. Scope is unchanged: finite-Galerkin, sampled-support,
two Grashof points at `k_f = 2`. **C1 stays PROVISIONAL, UNPROMOTED.**

## 6. Cross-references

- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
  — the fiber criterion this exhibit tests directly.
- [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md)
  — the state-insufficiency half; this exhibit reuses its witness pairs.
- [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md)
  — the control-sufficiency half (matched-radius aggregate) this sharpens.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) — the ledger.
