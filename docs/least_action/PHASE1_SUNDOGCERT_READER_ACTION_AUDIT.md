# Phase 1 Reader-Action Audit — `sundogcert` public surface

> First concrete output of [`SUNDOG_V_LEAST_ACTION.md`](../SUNDOG_V_LEAST_ACTION.md).
> Internal working artifact (docs-only; NOT a public page — Phase 6 gate not triggered).
> Target = the named Current Pain Point: the public Lean certificate surface the
> author cannot reload after a week. Dated 2026-06-17.

**Targets audited (two of the three Phase 1 artifacts):**
1. `sundogcert/README.md` — the public theorem-map front door.
2. [`docs/SUNDOG_V_CERTIFICATE_LEAN.md`](../SUNDOG_V_CERTIFICATE_LEAN.md) — the lane ledger.
   (Third proposed target — "one recent proof note hard to reload" — deferred to a follow-up pass.)

This audit produces the Phase 1 exit deliverable: a reader-action map + a ranked edit list.
The Euler-pass rewrites themselves are Phase 2 and are NOT performed here.

---

## Headline catch (status-ambiguity cost, load-bearing)

**The two public docs disagree on their own central count.**

| source | examples | kinds of math | includes `AuditCost`? |
|---|---|---|---|
| `sundogcert/README.md` (module table + closing para) | **7** | **6** | yes (named "a seventh worked example") |
| `sundogcert/METHOD.md` (per README citation) | **7** | **6** | yes |
| `docs/SUNDOG_V_CERTIFICATE_LEAN.md` (title, intro, table, status) | **6** | **5** | **no — absent from the table** |

Each doc is internally consistent; they are inconsistent *with each other*. The certificate
ledger predates `AuditCost.lean` (the HS7 audit-game / ∀-verifier-blindness example, landed
2026-06-10/11) and was never re-bumped. A cold reader who opens both cannot tell which number
is right — and the ledger is simply stale, a full worked-example behind the repo. This is the
discipline's first catch: a self-legibility-after-delay failure, exactly the pre-registered risk.

---

## Reader-action map — `B / Φ / T / A / I / F / R`

Scored for the README (the front door). ✅ recoverable by a cold reader; ⚠️ recoverable but costs
rereading or a private-context hop; ❌ not recoverable without help.

| sym | role | where it lives | score | note |
|---|---|---|---|---|
| `B` | hidden body | the secret `s`; the `\|F\|ᵏ` preimage bodies per syndrome | ⚠️ | the word "body" never appears — reader must map *secret* → *body*. One notation-hop. |
| `Φ` | shadow / signature | the syndrome `He = H(sG+e)` | ✅ | stated cleanly, with the lossiness ("loses `k·log\|F\|` bits"). |
| `T` | transform | parity-check multiply; the `O(m·n)` cost model | ✅ | `verifyCost_le`, tied to the deployed 8,192-op `[128,64]` check. |
| `A` | accepted action | `accept ⟹ Safe`; the three-valued verifier | ✅ | "the witness *is* the proof"; spoofing structurally impossible. |
| `I` | imports / wall | decoding hardness (ISD/SIS); + NP class, poly-time-ness, Cook–Levin | ✅✅ | **the strongest coordinate** — the wall is stated up front in a blockquote and never overclaimed. This is the surface's best feature; preserve it. |
| `F` | falsifier / boundary | *implicit only* — the reject-bound looseness as "the shadow of the hardness assumption"; the trust surface (scheme defs + meaning of *Safe*) | ❌→⚠️ | **the weakest coordinate.** Axiom-clean proofs "can't fail," so a cold reader asked *what would void this?* stalls. The real falsifiable surface = "is the cost model honest?" + "is *Safe* defined right?" lives in the **sibling ledger**, not the README. No crisp F at the front door. |
| `R` | receipt / replay | `lake exe cache get; lake build`; `#print axioms`; build-enforced `AxiomAudit.lean` | ✅✅ | referee-free, self-enforcing. Excellent. |

**Where a cold reader first needs private context:** the 22-row module table — it is sorted by
*file dependency order* (the reduction-chain plumbing `VarWheel`/`ClauseGadget`/`ThreeDMReindex`
sits at the same visual weight as the headline cores), so the reader cannot see which three or
four rows carry the claim and which are gadget machinery.

**What causes the most rereading:** the certificate ledger's taxonomy paragraph (lines 38–63),
which narrates "the first two… the third… the fourth… the fifth… four of the six…" — the author
reasoning *about* the example set rather than carrying the reader through it. High cross-reference
cost; a reader must hold the whole numbered list in working memory to parse one paragraph.

---

## Ranked edit list (for the Phase 2 Euler pass)

1. **Reconcile the count (correctness, do first).** Bump `SUNDOG_V_CERTIFICATE_LEAN.md` to **seven
   examples / six kinds of math**: title, intro blockquote, the worked-examples table (add the
   `AuditCost` row — "finite decidability / audit game"; core = proved-cheap audit + ∀-verifier
   blindness `pooled_channel_blind`; wall = the pooled-mean channel model), the taxonomy paragraph,
   and the Status line. **Entangled with the branding fork** (the title rename is public copy) — hold
   for owner greenlight. *This is the discipline's first shippable win.*
2. **Name `B` as "body."** One clause at first use of the secret `s` in the README: "the secret `s`
   (the hidden *body*)." Closes the only notation-hop in the otherwise-clean `B`.
3. **Add a crisp `F` at the front door.** One sentence in the README naming the trust surface AS the
   falsifier: "What could void this is not the proofs (the kernel re-checks them) but the *statement*:
   the honesty of the cost model and the definition of *Safe* — that is the entire falsifiable
   surface." Lifts F from ❌ to ✅ at the point of entry.
4. **Split the module table by role.** Three groups: **headline cores** (`Certificate`…`CheckCost`,
   `CertWall`), **method demonstrations** (the seven worked examples), **reduction-chain machinery**
   (`SATReduction*`, `VarWheel`, `ClauseGadget`, `ThreeDM*`, gadget files). Lets the reader find the
   load-bearing rows without reading 22 lines of equal weight.
5. **Replace the taxonomy paragraph with the chart.** Swap the lines-38–63 narration for a small
   "which examples share the `Φ→T→A` shape" table (example | kind of math | shared shape? | its own
   import) — the Meta-Generality Rule applied to the ledger's own example set. Turns the highest
   reread-cost block into a scannable map.

Edits 2–5 are truth-preserving Euler-pass rewrites (no claim moves). Edit 1 carries a factual
correction and a public-copy rename, so it waits on the branding decision below.

---

## Exit-criterion check

Phase 1 exit for these two artifacts: **met** — each has a reader-action map (above) and a ranked
edit list (above). The pre-registered Phase 1 falsifier (a cold reader still cannot name import /
falsifier / receipt after the pass) is testable only after the Phase 2 Euler pass runs; `I` and `R`
already pass at the front door, `F` is the one coordinate the pass must fix.
