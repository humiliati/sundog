# Lattice Build-Gate — Remote-GPU Runbook

> 2026-06-03. Turnkey operator steps for running the Phase-1 build-gate
> (`scripts/lattice_ldt_model.py --mode build-gate`) on a rented GPU instead of
> the local box. **Hyperbolic-primary, vendor-generic** (the SSH/scp workflow is
> identical on RunPod / Vast / Lambda). **Execution is gated on owner greenlight**;
> this doc makes the run mechanical when the window opens. Nothing here is a
> result — the build-gate verdict is, and it is adjudicated + committed per §7.

## 0. What this runs (and why offload)

The build-gate trains the LDT reimplementation on the 1K Sudoku-Extreme regime and
scores it with the I5 rollout ([`PHASE1_I5_ROLLOUT_CONTRACT.md`](PHASE1_I5_ROLLOUT_CONTRACT.md)):
`build_gate_pass iff rollout_exact_rate >= 0.999`. Until pass, **no body/fiber
number is a B-layer result** ([`PROMOTE_GATE.md`](PROMOTE_GATE.md) R1).

The model is tiny (**798,346 params, d=128**); the paper trains it in ~4 min on a
B200. We *have* a local CUDA GPU, so offloading is **not** about compute we lack —
it's about (a) freeing the local box for the wall-clock-long I2–I5 tuning loop, (b)
running tuning attempts in parallel on cheap throwaway instances, (c) a faster GPU
for the cap-bound rollout-eval. A single GPU is enough; **do not provision a
cluster.**

## 1. Sizing + cost (verify current pricing at sign-up)

Hyperbolic on-demand, hourly, no contract
([pricing](https://costbench.com/software/ai-gpu-cloud/hyperbolic/),
[hyperbolic.ai](https://www.hyperbolic.ai/)):

| GPU | $/hr (Apr 2026) | use |
| --- | ---: | --- |
| RTX 4090 24 GB | ~$0.50 | cheapest; the model fits trivially |
| A100 80 GB | ~$1.80 | **recommended** (throughput for rollout-eval) |
| H100 SXM 80 GB | ~$3.20 | only for max eval speed |

Our envelope: **~0.5–3 GPU-h per build-gate attempt**, **~5–20 GPU-h for the whole
I2–I5 loop** → **≈ $3–10 (4090) / $10–36 (A100)**. VRAM is a non-issue (<10 MB
weights; tiny activations). 1 GPU.

## 2. Provision (one-time setup + per-run rent)

One-time ([renting FAQ](https://docs.hyperbolic.xyz/docs/renting-faq)):
1. Account at <https://app.hyperbolic.ai/>, verify email.
2. Billing → add funds (~$25 to start).
3. Settings → SSH Public Key → paste `~/.ssh/id_ed25519.pub` (generate with
   `ssh-keygen -t ed25519` if needed).
4. (Optional) install the CLI: <https://github.com/HyperbolicLabs/hyperbolic-cli>.

Per run — rent **one** GPU (web dashboard `app.hyperbolic.ai/gpus`, or CLI; confirm
exact flags against [rent-gpus docs](https://docs.hyperbolic.xyz/docs/rent-gpus)):
```bash
hyperbolic ondemand                                              # browse availability
hyperbolic rent ondemand --instance-type virtual-machine --gpu-count 1   # rent 1 GPU
# note the instance IP from the dashboard; ready in a few min (up to ~25)
ssh ubuntu@<instance-ip>
```

## 3. Get the code (at a clean build-gate commit — provenance)

```bash
# on the instance
git clone <sundog-repo-url> sundog && cd sundog
git checkout <BUILD_GATE_COMMIT>          # the commit whose manifest will be pinned
git rev-parse HEAD                          # record it; must match manifest.gitCommit
```
Run from a **clean** commit so `manifest.gitCommit` is meaningful (the runner stamps
it). The dataset is gitignored, so it does **not** arrive with the clone — see §4.

## 4. Get the data (transfer a ~21 MB head-slice — result-identical)

The full dataset is 718 MB, but the loader only reads the **first 100k train rows**
as its sample pool (`max(n_train*100, 100000)`) plus `limit_test` test rows. So a
head-slice is **byte-identical in result** to shipping the whole file for the default
build-gate (`n_train=1000`, `max_eval ≤ 10000`):

```bash
# FROM LOCAL (where the dataset lives): slice + ship ~21 MB instead of 718 MB
head -n 100001 docs/lattice/Soduko-Extreme/train.csv > /tmp/train.csv   # header + 100k rows
head -n  10001 docs/lattice/Soduko-Extreme/test.csv  > /tmp/test.csv     # header + 10k rows
ssh ubuntu@<ip> 'mkdir -p ~/sundog/docs/lattice/Soduko-Extreme'
scp /tmp/train.csv /tmp/test.csv ubuntu@<ip>:~/sundog/docs/lattice/Soduko-Extreme/
```
Size the slice to ≥ `max(n_train*100, 100000)` train rows and ≥ `max_eval` test rows
if you raise either. *(Alternative: re-download Sudoku-Extreme from the HRM/TRM source
on the instance — verify the URL.)* **Do not commit the dataset from the instance** —
it is gitignored for a reason.

> Alignment note (pre-existing, not introduced here): the CSV is source-grouped, so
> the first-100k pool may be single-source-biased — the exact 1K subset is a
> build-gate alignment param flagged in the loader, resolved by amendment if the
> verdict turns on it. Slicing changes nothing vs the default loader (both read the
> first 100k).

## 5. Environment (pin for reproducibility)

```bash
python -V                                   # 3.12 locally; match major.minor if you can
pip install torch==2.5.1                     # match our local 2.5.1+cu121; pick the cuXXX
                                             #   wheel matching the instance CUDA driver
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect True
```
numpy is **not** required (the runner is torch-only). Record the torch build —
`manifest.torch` stamps it.

## 6. Run (fail-fast, then scale)

```bash
cd ~/sundog
# FAIL-FAST first: small eval to get a fast read before the full 10k-puzzle eval
python scripts/lattice_ldt_model.py --mode build-gate \
  --data-dir docs/lattice/Soduko-Extreme \
  --out results/lattice/build-gate-sudoku-extreme \
  --max-eval 1000 --seed 0
```
Watch the printed `result`: `branch`, `rollout_exact_rate`, `one_shot_exact_rate`,
`stop_reason_counts`, `diagnostics`. **Red flag:** a high `node_cap_fraction` means
the model is weak and the rollout is cap-bound (slow + not solving) — stop, the
verdict is `build_gate_partial`/`fail`, don't burn time on the full 10k eval. If
`--max-eval 1000` looks promising (`rollout_exact_rate` near 1), re-run with
`--max-eval 10000` for the binding read. Frozen contract params (`theta_drop=0.5`,
`theta_cls=0.6`, caps 64/4096) are in code + stamped in the manifest; **do not edit
them on the instance** (contract §11 — changes are amendments).

## 7. Pull the receipt + adjudicate + commit (locally)

```bash
# FROM LOCAL
scp -r ubuntu@<ip>:~/sundog/results/lattice/build-gate-sudoku-extreme ./results/lattice/
```
The receipt = `manifest.json` (verdict + provenance: `gitCommit`, `device`, `torch`,
`rolloutExactRate`, `stop_reason_counts`, `diagnostics`) + `rollout_per_puzzle.jsonl`.
`results/lattice/` is **gitignored**, so the *verdict* is committed as a short receipt
doc (the lane convention — verdict committed, raw artifacts not):

- **`build_gate_pass`** → file `docs/lattice/BUILD_GATE_RECEIPT.md` (verdict +
  pinned manifest provenance + the rollout diagnostics) and **only then** does
  Phase 2 (B2) open. The chatv2 audit team reviews the receipt before any B-layer
  number is read.
- **`build_gate_partial` / `build_gate_fail`** → **do not tune I2–I5 in place**
  (contract §11). File an amendment naming which inference ([I2] reinject, [I3] loss,
  [I4] conflict target, [I5] rollout policy) is being changed and why, then re-run.

## 8. Teardown (stop the meter)

```bash
hyperbolic terminate <instance-id>          # or destroy via the dashboard
```
Hourly billing stops at termination. Nothing persists on the instance that isn't in
the pulled receipt + the committed code.

## 9. Provenance + reproducibility checklist (before a `pass` is believed)

- [ ] ran at a **clean** build-gate commit; `manifest.gitCommit` matches `git rev-parse HEAD`;
- [ ] `manifest.torch` recorded (a different GPU/torch build can perturb float — note it);
- [ ] `seed` recorded (default 0); `n_train`/`max_eval` recorded in the result;
- [ ] for a **promotable** `pass`: re-run once (same commit/seed, fresh instance) and
      confirm the verdict is stable — a build-gate pass that doesn't survive a re-run
      is not a pass;
- [ ] `rolloutContract` + `rolloutVersion` + thresholds present in the manifest
      (the runner writes them; confirms no silent param drift).

## 10. Discipline / what this does NOT change

- The **verdict and its gates are identical** wherever compute runs — the manifest
  carries the provenance, the audit team reviews the verdict, the promote-gate
  ladder is unchanged. Remote GPU is an *infra* choice, not a scientific one.
- **No leakage/secret concern** (public Sudoku-Extreme + our own model — unlike the
  ARC eval-blind discipline). Nothing sensitive goes to the instance; just code +
  public data. Still: don't paste credentials into the box beyond the SSH key.
- This runbook does **not** authorize the run — owner greenlight + a confirmed GPU
  window do. Expect a first `build_gate_partial` → the amendment-driven I2–I5 loop.

## 11. Cross-references

- [`PHASE1_I5_ROLLOUT_CONTRACT.md`](PHASE1_I5_ROLLOUT_CONTRACT.md) — the rollout the
  build-gate scores with; §11 amendment rule for a partial.
- [`PROMOTE_GATE.md`](PROMOTE_GATE.md) — R0→R1; build-gate pass is the R1 entry gate.
- [`../SUNDOG_V_LATTICE.md`](../SUNDOG_V_LATTICE.md) — Phase 1 (build-gate) + the
  audit-team handoff; `chatv2/LATTICE_HANDOFF.md` §0 for what transfers from chatv2.
