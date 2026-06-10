# Prior-Art Re-Search Receipt — Syndrome-Decoding Paper (2026-06-10)

Re-run of the negative prior-art searches behind the "to our knowledge, first" hedges in
[`SYNDROME_DECODING_FORMALIZATION.md`](SYNDROME_DECODING_FORMALIZATION.md) /
[`lipics/syndrome-decoding.tex`](lipics/syndrome-decoding.tex), per the drafting-notes
gate. Method per claim, result, and disposition. **Re-run once more in the week before
the actual submission deadline** (this receipt is a snapshot, not a permanent clearance).

## Verdict: all first-ness claims STAND; 2 wording corrections + 2 survey additions applied

| # | Claim checked | Method | Result | Disposition |
|---|---|---|---|---|
| 1 | (C2) No prior mechanization of Garey–Johnson `3SAT ≤ 3DM` or `3DM ≤ X3C` in any proof assistant | Web searches: "3SAT 3DM reduction proof assistant mechanized", ""Garey-Johnson" OR "truth-setting" reduction formalized Isabelle Coq", AFP-domain search | No formalization found anywhere; only informal complexity papers | **Stands** |
| 2 | (C1) No prior mechanization of syndrome-decoding NP-hardness (BMvT78 spine) in any assistant | Web search "syndrome decoding NP-hardness formalization proof assistant"; coding-theory formalization sweep | Nothing; all formal coding-theory work is decoder/code correctness | **Stands** |
| 3 | `poly-reductions` scope claim (§7): covers SAT→IS/VC/Clique/SetCover + HC variants + VC→FNS, nothing toward 3DM/X3C/subset-sum/coding; repo-cited (no canonical paper) | Fetched current repo README (github.com/wimmers/poly-reductions) | Reduction list **exactly matches** the paper's description; no 3DM/X3C/subset-sum/partition/knapsack/coding; README cites no paper | **Stands** (repo-cite retained) |
| 4 | [Sim26] non-overlap + figures | arXiv abs/2601.15571 | Now **v4 (2026-03-01)**; coNP/Σ₂ᴾ/PP/PSPACE for "physical counting"; no 3SAT/3DM/X3C/coding; 28,863 lines / 1,252 theorems (paper's "~28.9k" ✓) | **Stands** (citation date already says rev. Mar 2026) |
| 5 | [ELWT26] non-overlap | arXiv abs/2605.16523 | Still **v1 (2026-05-15)**; distance-certificate verification (reduces distance *to* SAT), no hardness mechanization; no venue acceptance shown | **Stands** |
| 6 | "mathlib has no parity-check / syndrome / generator / linear-code primitives" | **grep on pinned mathlib v4.30.0** (local checkout under `sundogcert/.lake/packages/mathlib`): `LinearCode\|ParityCheck\|GeneratorMatrix\|syndrome\|errorCorrect` → 0 files. **GitHub code search on current master** (`gh api search/code`): LinearCode 0, ParityCheck 0, GeneratorMatrix 0, syndrome+decoding 0 | Verified on pinned **and** master | **Stands**, now grep-verified per the drafting rule |
| 7 | "mathlib offers no machine model" (§1, §4.3) | grep pinned mathlib `Computability/` | **WRONG as worded**: mathlib HAS TM models (TM0/TM1/TM2) and `TM2ComputableInPolyTime` (`Mathlib/Computability/TuringMachine/Computable.lean`) — a stand-alone polytime-computability definition whose composition is still `proof_wanted`; no classes, no SAT, no resource-bounded reductions | **CORRECTED** in §1 + §4.3 (MD + tex): claim is now "no complexity classes / no reduction framework", with the TM + polytime nuance stated. Referee-bait removed |
| 8 | Lean coding-theory survey completeness (§7) | Searches "Lean coding theory", "ArkLib Reed-Solomon Lean", "Cotoleta Hagiwara" | Two missing precedents: **Cotoleta** (Hagiwara–Nakano–Kong, ISITA 2016 — repetition + Hamming(7,4), early Lean) and **ArkLib** (Verified-zkEVM, Lean 4 + mathlib — RS proximity infra for SNARKs, in progress). Both correctness-oriented, no decision problems / hardness | **ADDED** to §7 + references ([HNK16], [ArkLib]) in MD + tex |
| 9 | New Lean complexity work since draft (beyond [Sim26]) | Search "Lean 4 NP-completeness 2026 complexity" | A Berkeley Sp26 course (CS 294-268) covers NP-reductions in Lean (teaching, not a library); FormalizedFormalLogic undecidability work (not Karp reductions); nothing overlapping | **Stands** |
| 10 | Cook–Levin status ("Coq + Isabelle, not yet Lean") | Same searches; poly-reductions README (has a `Cook_Levin/` dir building on the Isabelle line) | No Lean Cook–Levin found | **Stands** |

## Venue timeline facts established (for the submission plan)

- **ITP 2026: deadline PASSED** — abstracts 2026-02-12, papers 2026-02-19; conference
  2026-07-26/29, Lisbon (FLoC). LIPIcs proceedings.
- **CPP 2027**: co-located with POPL 2027 (Mexico City, ~mid-Jan 2027). CFP not yet
  posted; CPP 2025/2026 pattern ⇒ abstracts ~Sep 9–10, papers ~Sep 16–17, 2026.
  Lightweight double-blind; **acmart** (sigplan) template, not LIPIcs.
- **ITP 2027**: expect deadline ~Feb 2027; LIPIcs — current build ready as-is.

## Sources

- https://github.com/wimmers/poly-reductions (README, reduction list)
- https://arxiv.org/abs/2601.15571 (Sim26 v4) · https://arxiv.org/abs/2605.16523 (ELWT26 v1)
- https://github.com/Verified-zkEVM/ArkLib · https://ieeexplore.ieee.org/document/7840479 (Cotoleta, ISITA 2016)
- https://itp-conference-2026.github.io/ (ITP 2026 dates) · https://popl26.sigplan.org/home/CPP-2026 (CPP pattern)
- Local: grep over `sundogcert/.lake/packages/mathlib` (pinned v4.30.0); `gh api search/code` over `leanprover-community/mathlib4` master.
