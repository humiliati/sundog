# P-vs-NP Direction-B — Gate-0 leg-(d) preflight (measured boundary)

> 2026-06-05. **Verdict: `leg_d_not_available` — emergent capacity-relative one-wayness from a
> trained body is NOT supported on the chatv2 (D2) substrate.** A model-free ~$0/CPU-minutes
> preflight plus a short in-harness confirmation. This converts `SUNDOG_CERTIFICATE_PROBLEM.md`
> §6 RISK-4 ("imports hardness, does not demonstrate it") from an *admission* into a *measured
> boundary*. Companion to [`SUNDOG_CERTIFICATE_PROBLEM.md`](SUNDOG_CERTIFICATE_PROBLEM.md).

## The question

The lane's syndrome/ISD certificate (Candidate A) **imports** its hardness (SIS/ISD assumption) and
its lossiness (parity-check `H` algebra). Direction B asked whether the same lossy-quotient +
cost-asymmetry could **emerge from a trained body** rather than be imported. A 6-agent design panel
(2026-06-05) decomposed the property into four legs — (a) state-insufficiency, (b)
control-sufficiency, (c) de-confound, (d) cost-asymmetry/one-wayness — and concluded legs (a)+(b)+(c)
are **already emergent** on a trained body (chatv2 pair-XOR, banked), so the entire dispute is **leg
(d)**, and leg (d) fights the **gradient-invertibility barrier**. Gate-0 is the cheapest test of
whether leg (d) can emerge on the recommended substrate D2.

## Substrate (D2)

chatv2 pair-XOR latents (`latent="computed"`, arity=2, H=8, bpc=24, δ=0.45; each `z_i` is a
parity-channel aggregate, provably NOT linearly input-decodable) + a frozen public GF(2) syndrome
head `σ = H_pub·z` (rank `m=5`, so `2^(H−m)=8` preimages per syndrome → lossy by algebra).
Runner: `scripts/pvnp_directionb_gate0.py`. Receipts: `results/pvnp/directionb-gate0/`.

## Results

Recovery det = `(cv_acc − majority)/(1 − majority)`, n=3000.

| feature set → target | z (linear) | z (MLP) | σ (linear) | secret |
| --- | ---: | ---: | ---: | ---: |
| raw input (de-confound floor) | **−0.006** | **+0.929** | −0.030 | −0.012 |
| untrained TinyGPT body (random) | −0.018 | +0.044 | −0.023 | −0.022 |
| generic ReLU random features | +0.153 | +0.633 | −0.032 | −0.046 |
| **secret from σ** (lossiness check) | — | — | — | **−0.002** |

In-harness confirmation (`gate0_confirm.json`): a **trained** body (2500 steps) recovers `z`
linearly at **median det 0.310** (per-latent up to 0.91; `acc_mean 0.703`) vs the **untrained floor
−0.018** → `Δ(trained − untrained) ≈ +0.33`, still rising (chatv2 full-train banks `z1_acc ≈ 0.94`).

## Reading

1. **Substrate valid.** De-confound holds (raw-linear z = −0.006 ≈ chance) and the functional is
   present (raw-MLP z = **0.929**). z is computable but not linearly leaked.
2. **The one-way object's hardness/lossiness is imported.** `σ = H_pub·z` is lossy *by algebra* —
   the secret is unrecoverable from σ (det **−0.002**, 8 preimages/syndrome). No body shapes this;
   the σ→light-witness hardness is the syndrome-decoding object measured in
   `results/pvnp/certificate-syndrome-v1` (no body involved).
3. **The body cannot be the one-way part.** Control-sufficiency *forces* a trained body to expose `z`
   (untrained −0.018 → trained **+0.31**, up to 0.91). A representation that must *use* `z` to decide
   Safe linearly *contains* `z`. **One-wayness and control-sufficiency are contradictory for the
   body's `z`.** The untrained-body-hides-z reading (−0.018) is a red herring — that's an untrained
   net, not protection; generic random features already partially crack the parity (+0.153).
4. The Δ the whole direction hinges on — `inversion-cost(trained body) − inversion-cost(raw
   abstraction)` — is structurally **≤ 0** here (training *helps* the attacker), the opposite of the
   positive Δ emergence requires.

> The script's mechanical verdict string `gate0_inconclusive_survives_to_B1` keyed only on
> "untrained body linearly exposes z" (which was at-floor, ambiguous). The adjudicated lane verdict,
> with the confirmation, is the **kill**: leg-(d) emergence is not available.

## Verdict & disposition

**`leg_d_not_available`** — emergent capacity-relative one-wayness from a trained body is not
supported on D2. Do **not** spend GPU on B1→B3. (D1/LDT was also predicted dead by the smooth-encoder
argument and is hard-blocked today at rollout 0.324≪0.999; D3/VQ is the just-shelved JEPA-0D
bottleneck structure.)

**What stands.** The lane's genuinely non-imported, emergent-from-a-trained-body asset is the
**(a)+(b)+(c) bundle** — state-insufficiency + control-sufficiency + de-confound — already banked on
chatv2 and which never needed leg (d) to beat Candidate A. Direction B's honest endpoint is this
measured boundary, not a chase across the gradient barrier.

## Artifacts

- `scripts/pvnp_directionb_gate0.py` — the model-free Gate-0 preflight
- `results/pvnp/directionb-gate0/gate0.json` — the no-training measurement
- `results/pvnp/directionb-gate0/gate0_confirm.json` — the trained-body control-sufficiency confirmation

---

*Sundog Research Lab — P-vs-NP Direction-B Gate-0. Measured boundary: one-wayness is the imported
leg; the trained body supplies the emergent state-insufficiency + control-sufficiency + de-confound,
not the hardness. R1 toy; kill-gated R&D.*
