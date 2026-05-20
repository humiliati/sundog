# Isotrophy Handoff Note — Updated with Pair-ID Catalog Spec (v0.3 thread)

**Date**: 2026-05-19 (post-audit, pair-ID signed off)
**Status**: Pair-ID + Gauge-Cocycle Catalog spec landed. F_beta-structural chain argument drafted (open analytical item). New teammate onboarding notes included.

## Fidelity Audit Summary (for new teammate)
Verified faithful on disk (code, npm scripts, main doc sections 545–646, parent note 808-811, this handoff). 

**Two caveats**:
1. Raw CSV receipts gitignored/protected. Canonical per-row data lives in `docs/sundog_v_isotrophy.md` table (lines ~624-646). Reproduce via `npm run isotrophy:tau12:cases`.
2. **Open items not yet in docs** (conversation has advanced two moves beyond v0.3a section):
   - (a) Pair-ID + Gauge-Cocycle Catalog spec (now landed below).
   - (b) (E,|L|)-singleton + F_β-structural chain argument (drafted in §4 below). This upgrades the catalog-asymmetric prediction from "≥5" to "**all 21**" and implies a manifest-level F_β structural cocycle with tau uniformly active. v0.3 derivation is gated on the induced-rep formula, not on a per-row tau decision.

**Cleanest onboarding path for new teammate**: Start at `docs/sundog_v_isotrophy.md` §G.2 RESOLVED → v0.2 retirement → v0.3 DESIGN DECISION → v0.3a case-split → v0.3b F_beta pair-ID. Then read this handoff note for the post-v0.3a/v0.3b threads. The pair-ID question is now receipt-backed; tau is a manifest-level constant, not a per-row flag.

## Landed: Pair-ID + Gauge-Cocycle Catalog Specification (Signed Off)

**Precise spec** (implementation-ready; no code until signed — now signed).

### Scope
All 21 strict single-curve choreographies in supplementary-A (13 canonical-strict + 8 opposite-strict). 0 endo under (12)-up-to-O(3)-gauge.

### 1. Bare-(12)-Image IC Computation
For IC_i = (q₁, q₂, q₃, v₁, v₂, v₃):
```
IC'_i := (12) · IC_i  = (q₂, q₁, q₃, v₂, v₁, v₃)
```
Pure label permutation. Cheap, exact (mod original precision). Preserves E and |L|.

### 2. (E, |L|)-Matching Filter (Validated Invariants Pipeline)
Search j ≠ i s.t.
|E_j − E(IC'_i)| < ε_E  and  ||L|_j − |L|(IC'_i)| < ε_L
where ε_E, ε_L are **exactly** the thresholds locked in the singleton (E,|L|) verification.

### 3. Verification
- Must flip classification (canonical ↔ opposite).
- Pass validated detector-path closure-relative residual test on IC'_i (same thresholds as original 21).

### 4. Gauge Cocycle (if verified partner)
Use established Procrustes alignment on post-(12) positions (incl. Φ_{T/2} if required by α_i) vs. candidate IC_j to extract (R_{12,i}, φ_{12,i}).

Record per row:
- partner_row_index or null
- is_catalog_asymmetric (bool)
- If paired: R, φ, residuals, diffs
- Consistency check: pairing is involution; orientation flip respected.

### Updated Prediction (incorporating singleton uniqueness)
**All 21 rows are expected to be catalog-asymmetric.**  
Rationale (see §F_beta chain below): (E,|L|) values are unique across the 21 (singleton verification). Bare (12) preserves invariants, so IC'_i can only possibly match row i on (E,|L|). But 0 endo + orientation flip rules out self-pairing. Hence **zero partners inside supplementary-A**. The earlier "≥5" (from 13-8 count) is superseded; the structural ansatz makes the stronger global result.

This simplifies the induced-rep d_i input: no row requires a separate missing-partner IVP. The partner orbit's variational structure is obtained from the parent monodromy by linear F_β-conjugation, `rho(R_pi) * M_i^-1 * rho(R_pi)^-1`, with τ acting as the anti-symplectic flip on momenta. A single sentinel re-integration may be useful later as a robustness cross-check, but it is not the per-row workflow.

**Output schema**: Structured records (JSON/CSV/Markdown table) consumable by d_i derivation. Include sanity invariants (no self-pairs, flip respected, all asymmetric).

**Sign-off status**: ✅ Signed off, audited, and receipt-confirmed. Ready for the induced-representation derivation; do not add a per-row tau flag.

## Open Analytical Item: F_β-Structural Chain Argument (Draft for Review)

**Claim**: The combination of (E,|L|)-singleton verification + Liao ansatz structure forces **all 21 catalog-asymmetric under bare (12)**, and reduces the cocycle to a **uniform F_β structural template with tau active as a schema constant**. Current receipt schema (spatial_parity, R_i, phi_i + Bragg cross-check) keeps per-row gauge data, but the τ component itself is not a row variable.

### Why the catalog is not closed under bare (12)
Liao's ansatz uses the **composite structural relabel**
F_β := ((12), τ, R_π)
where:
- (12) = body permutation,
- τ = specific time shift (typically tied to half-period or choreography symmetry point),
- R_π = 180° spatial rotation (aligns frame, preserves or interacts with canonical orientation).

The supplementary-A catalog consists of strict single-curve choreographies **generated/found/filtered inside this ansatz**. Therefore the set of *represented frames* (the specific ICs stored, with their phases and orientations) is closed under the action/conjugation by F_β (by construction of the search).

Bare (12) alone is **not** F_β. It is F_β composed with the inverse compensators τ⁻¹ and R_π⁻¹. Applying bare (12) to a catalog IC therefore produces an image whose *natural frame* (starting phase + orientation after the permutation) lies **outside** the ansatz's canonical frames.

### Singleton verification closes the trap
The validated (E,|L|)-singleton check established that the 21 rows have **distinct (E, |L|)** to integration precision (no two share the same invariants within tolerance).

Since bare (12) is Hamiltonian symmetry → preserves (E, |L|) exactly, the image IC'_i has **identical invariants to row i**.

Therefore, in the (E, |L|)-matching step, the *only* possible row that could ever satisfy the tolerance for IC'_i is row i itself.

But:
- 0 endo (no fixed points under (12)-up-to-gauge) rules out j = i.
- Orientation flip (canonical ↔ opposite) provides independent confirmation that self-matching is impossible anyway.

**Conclusion**: No j ≠ i exists with matching (E, |L|). Every bare-(12) image fails the filter → **all 21 rows are catalog-asymmetric**. Their partners live outside supplementary-A (in frames not representable under the current ansatz without the compensating τ + R_π).

### Consequence for induced-rep d_i and cocycle schema
- No per-row pair-ID variability from "inside catalog" cases.
- Uniform treatment: for every C_i, the missing partner's variational structure is obtained from C_i's monodromy by linear F_β-conjugation: `M_(12*C_i) = rho(R_pi) * M_i^-1 * rho(R_pi)^-1`, with τ acting as the anti-symplectic flip on momenta. No separate IVP is required.
- The identification back to a frame at C_i(0) is always realized by the **uniform F_β** (which already bundles the (12) + τ + R_π) composed with any additional per-row gauge cocycle (R_i, φ_i) already present in the receipt.
- Therefore the twist operator α_i must treat τ as a schema-level constant: it is active for all 21 rows because F_β is structural for all 21 rows, not a per-row switch.

**Required schema extension** (landed in the F_beta pair-ID receipt):
Add one manifest-level schema line, not 21 per-row values: `structural_cocycle = F_β = ((12), tau-active, R_pi)`. A per-row `tau_flag` is forbidden because `identity` would imply F_β is not structural on that row, contradicting the ansatz. The v0.3a receipt schema (spatial_parity, R_i, phi_i + phi/(T/2) Bragg) can keep per-row gauge data, but the τ component is uniformly active.

This is the load-bearing structural follow-up flagged earlier. The review accepted the chain with the two tweaks above. `npm run isotrophy:fbeta:pair-id` then confirmed the "all asymmetric" outcome and emitted the uniform cocycle records at `results/isotrophy/k-facet-v03-fbeta-pair-id-21strict/`.

## Next Steps
1. Done: reviewed and signed off with the two tweaks above.
2. Done: implemented and ran the pair-ID runner; result was 21/21 catalog-asymmetric and 21/21 F_beta closure-tight.
3. Done: receipt schema carries the manifest-level structural cocycle, not a per-row τ flag.
4. Done: landed the receipt summary in `docs/sundog_v_isotrophy.md` and the canonical handoff note.
5. Next: proceed to v0.3 single-track d_i formula with the full induced-representation track (no endomorphism branch).

**Receipt lock reminder**: induced-only remains robust under full O(3) parity coverage. tau12_Z wins 6 residuals, floor 2.8e7, none near closure.

---

*This file created/updated in artifacts/ as durable record. In live repo, merge equivalent content into the canonical docs and internal/anniversary/isotrophy_handoff_note.md.*
