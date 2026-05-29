# Yang-Mills Pre-Registration Holding Pen

Roadmap: [`SUNDOG_V_YANG_MILLS.md`](../../SUNDOG_V_YANG_MILLS.md)
Lit-pass: [`YANG_MILLS_LITPASS_MEMO.md`](../../YANG_MILLS_LITPASS_MEMO.md)

Filed: **2026-05-29 (PT)**

Status: **Phase 0 open - no domain lock filed**. This directory exists so the
first Yang-Mills pre-registration has a stable home. It is not itself a frozen
experiment spec and admits no runner code.

## Required Next Artifact

The next file should be:

```text
P0_DOMAIN_AND_RECEIPT_LOCK.md
```

It must freeze, before any run:

- gauge group and dimension;
- lattice sizes and boundary condition;
- action and coupling slate;
- ensemble source or generator algorithm;
- burn-in, thinning, and autocorrelation handling if generated locally;
- primary gauge-invariant signature vocabulary;
- held-out observable/regime labels;
- metadata, raw-link, gauge-fixed, random, coupling-stratified, permutation,
  and gauge-randomized controls;
- output paths and receipt template;
- compute cap and staged commands under the repository ten-minute rule;
- positive, metadata-only, negative, and inconclusive branches.

## Guardrail

No Yang-Mills code run is admitted until `P0_DOMAIN_AND_RECEIPT_LOCK.md` exists
and explicitly says it is frozen. Exploratory notebooks or scratch checks, if
ever needed, must be labelled exploratory and cannot be cited as receipts.
