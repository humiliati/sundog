# Structural Failure Coincidence — Cut 2 C5 Publication-Plumbing Freeze

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Sibling freezes: [`P2_CUT2_C2A_NUMERIC_FREEZE.md`](P2_CUT2_C2A_NUMERIC_FREEZE.md) · [`P2_CUT2_C3A_NUMERIC_FREEZE.md`](P2_CUT2_C3A_NUMERIC_FREEZE.md) · [`P2_CUT2_C4A_AUDIT_FREEZE.md`](P2_CUT2_C4A_AUDIT_FREEZE.md)
Filed: **2026-05-16 (PT)**. Status: **C5 FILED FOR AUDIT — HOLD FOR
EXECUTION**. Freezes the publication-plumbing **structure + the
operational guard-artifact obligations**; the concrete write-path
manifest and the runnable guard script are the maintainer's pre-run fill
(no fabrication); a tripped guard **blocks (`PUBLICATION_PLUMBING_VIOLATION`),
never re-scopes**. With C5 filed, **all pre-registration conditions are
filed**. Cut-2 execution remains HELD on the concrete fills + the single
joint P2-spec admission re-run. No harness written; nothing run.

## Purpose

The publication-plumbing seam was opened in prose in the 2026-05-16
`P2_RUN_SPEC.md` / `P2_SPEC_ADMISSION.md` C5 amendments (the Cut-2
harness may write only under `results/structural-failure/cut2-*/`; a
pre/post guard scoped to the shipping surfaces must be clean; any
violation ⇒ `PUBLICATION_PLUMBING_VIOLATION`, never PASS). C5 makes that
**operational** — propagating the C4-A ruling that a *principle* is not
an artifact: the guard must be a concrete, runnable, frozen object, not
a description.

## 1. The frozen write allowlist (default-deny)

A Cut-2 run may create/modify files **only** under:

```
results/structural-failure/cut2-*/
```

Everything else in the working tree is **forbidden** for the duration of
a Cut-2 run. This is stated as an **allowlist**, not a blocklist of
shipping surfaces.

## 2. C5 keystone load-bearing self-seal (surfaced adversarially)

The C5 analog of C2-A §5 / C3-A §2 / C4-A §4. A guard that asserts
"clean within an *enumerated set of shipping paths*" (a **blocklist**)
can be silently **under-scoped**: forget one shipping surface — or add a
new one later — and a real publication write slips through as "clean."
That is the publication-plumbing self-seal.

Therefore the guard is **default-deny / allowlist-complement, frozen
before the harness exists**: a full-tree
`git status --porcelain` (equivalently `git diff --exit-code` over the
whole tree) must show **zero** changes outside
`results/structural-failure/cut2-*/`. The guard's scope is the *entire
tree minus the tiny allowlist* — it **cannot be narrowed** to admit a
write, because anything not in the allowlist trips it by construction. A
tripped guard ⇒ terminal `PUBLICATION_PLUMBING_VIOLATION`; the scope is
**never** re-curated post-hoc to make a run "clean" (A3).

## 3. Operational artifacts (propagated C4-A discipline)

Each a concrete, reproducible, frozen object — not prose:

- **Frozen allowed-write-path manifest** — a hashable file pinning the
  exact glob allowlist (`results/structural-failure/cut2-*/`) and the
  default-deny rule; its hash is recorded so the manifest itself cannot
  be silently widened.
- **Runnable pre/post guard script** — snapshots full-tree state before
  the Cut-2 run; after the run asserts the change set ⊆ the manifest
  allowlist; emits `PUBLICATION_PLUMBING_VIOLATION` and a non-zero exit
  on **any** out-of-allowlist change. Deterministic, re-runnable by a
  third party.
- **Verdict wiring** — `PUBLICATION_PLUMBING_VIOLATION` is **terminal
  and dominates**: it overrides any D1∧D2∧D3 / four-quantity pass; a run
  that trips it can never be reported as a Cut-2 PASS.

## 4. Provenance-tagged freeze (A3)

`[G]` immutable/operational boundary; `[E]` pre-registered engineering
artifact (amend-only, justified, never post-results).

| artifact | role | provenance |
| --- | --- | --- |
| write allowlist `results/structural-failure/cut2-*/` | the only writable location | **[G]** (default-deny boundary) |
| shipping surfaces (README, root `*.html`, `docs/`, `chat/`, `public/data/`, `dist/`) | must be untouched by a Cut-2 run | **[G]** (facts: these ship) |
| hashable write-path manifest | pins the allowlist | **[E]** operational artifact |
| runnable pre/post guard script | enforces default-deny, emits the violation | **[E]** operational artifact |
| `PUBLICATION_PLUMBING_VIOLATION` terminal-dominant wiring | verdict precedence | **[G]** (cannot be downgraded) |

`scripts/copy-site-docs.mjs` is **explicitly left untouched** —
hardening it is a separate, larger decision and is out of C5 scope (as
the original C5 amendment already stated). The prereg folder may still
ship to `dist/` on a normal build; C5 only prevents the Cut-2
*experiment* from being the thing that writes a shipping surface.

## 5. Honest couplings / ordering

- C5 is **independent of the numeric fills** (C2-A/C3-A/C4-A `[E]`
  values) — which is why it can be frozen now; but the **single joint
  admission re-run still gates on all of C2-A, C3-A, C4-A, *and* C5
  together**.
- C5's guard **wraps any Cut-2 run regardless of the cut** (closed-form
  or, later, the Cut-3 rendered escalation); it is not cut-specific.
- C5 does not move any geometry/receipt boundary or any threshold; it is
  pure write-path hygiene.

## Cut-2 C5 binding rules

1. The guard is **default-deny / allowlist-complement** over the full
   tree; a blocklist or curated-path-subset guard ⇒ **void**.
2. The write-path manifest + guard script are frozen here, hashable, and
   runnable before any Cut-2 instantiation; a prose-only guard is not
   admission-sufficient.
3. `PUBLICATION_PLUMBING_VIOLATION` is terminal and dominates any pass;
   downgrading it to a warning ⇒ **void**.
4. The guard scope is **never** re-curated after a run to make it
   "clean" (A3); a violation blocks (append-only redesign).

## Explicit non-bindings (cannot satisfy C5)

- A blocklist of shipping paths (under-scopable) instead of the
  default-deny allowlist-complement.
- A guard that checks only a curated path subset rather than the full
  tree minus the allowlist.
- Re-scoping/curating the guard after seeing what the harness wrote.
- A prose description instead of a runnable, hashable guard artifact.
- Treating `PUBLICATION_PLUMBING_VIOLATION` as non-terminal.

## Open items

C5 files the structure + the operational guard-artifact obligations. The
concrete write-path manifest (hashed) and the runnable pre/post guard
script are the maintainer's pre-run fill (no fabrication). **With C5
filed, every pre-registration condition is filed.** The *only* remaining
gates to the single joint P2-spec admission re-run are the maintainer's
concrete fills landing: C2-A/C3-A/C4-A `[E]` values + operational
artifacts + receipt tables, and the C5 manifest + guard script.

After all of C2-A, C3-A, C4-A, C5 are filed **with concrete
values/artifacts/receipts landed**, the P2-spec admission check **must
be re-run** as one audit of the whole discriminating cut; only on
**ADMIT** may a Cut-2 harness be built or run. Public-Language
Constraint remains fully in force: no `CONFIRMED` /
traceability-success / theorem language anywhere (including the rail).

## Honest prior (unchanged)

Write-path hygiene does not change the science: a real inverse-free ESC
controller against a biased signal with a tempting reachable decoy keeps
the likely honest outcome at **D / BOUNDARY FOUND** — the Proxy-Collapse
confirmation avenue (`debunked.md`, P1 §C). **B** is earned *only* by a
measured refusal of the tempting decoy at the quantified in-sample cost
**and** emergent failure coincident with L1/L2/L3. Either is a clean
result; the in-between is not.

## Audit Notes

*(reviewer space — append-only below)*
