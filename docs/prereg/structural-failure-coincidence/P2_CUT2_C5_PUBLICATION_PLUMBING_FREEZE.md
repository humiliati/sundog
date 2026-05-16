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

**2026-05-16 (PT) — Codex audit.** Direction accepted; C5 is **not yet
execution-closing** until the manifest and runnable guard land. The
default-deny / allowlist-complement design is the right anti-self-seal:
it avoids an under-scoped shipping-surface blocklist, and
`PUBLICATION_PLUMBING_VIOLATION` correctly dominates any scientific
pass. Two implementation holds are load-bearing. (1) The guard must
define its baseline semantics: either require a clean full-tree baseline
before Cut-2 starts, or snapshot the pre-run tree and reject only **new**
out-of-allowlist changes. Given this repo's normal dirty-workflow risk,
the choice must be explicit and reproducible; otherwise pre-existing
dirty files can either cause false violations or hide new writes. (2)
`git diff --exit-code` alone is insufficient: it misses untracked files,
and ordinary `git status --porcelain` can miss ignored files. The guard
script must include tracked modifications, untracked files, and any
ignored-but-public/shipping paths in its full-tree check, with
normalized repo-relative paths and symlink/junction escape rejection
before applying the `results/structural-failure/cut2-*/` allowlist. No
harness/controller run.

**2026-05-16 (PT) — maintainer. Operational artifacts landed (concrete
fill).** The C5 manifest and runnable guard now exist on disk, with
hashes pinned below. The 2026-05-16 Codex audit's two implementation
holds are addressed in the artifact bodies themselves; this append
records the artifacts and their hashes — it does not change the freeze
above.

*Hold (1) — baseline semantics chosen and frozen.* **Snapshot mode** is
the elected semantics: pre-run, the guard captures a SHA-256 of every
file outside the allowlist (tracked + untracked + ignored); post-run,
it re-hashes the same paths and flags any new path, removed path, or
content-hash delta. Snapshot mode tolerates a normal dirty workflow
while still catching pre-existing-file overwrites because the snapshot
records content hashes, not just presence. Semantics frozen here:
amending requires an append-only justification, never a post-results
flip.

*Hold (2) — full-tree coverage with normalization and symlink-escape
rejection.* The guard walks the entire working tree (excluding only
`.git/` and `node_modules/`, which never ship and would add only hashing
cost). It uses `lstat` at scan time and `realpath` to reject any
symlink whose real path escapes the repo root **before** the allowlist
check runs. Repo-relative paths are normalized to POSIX form
(`a/b/c`, not `a\\b\\c`) so the allowlist regex applies identically on
Windows and Linux.

*Manifest immutability.* The guard's `check` step reads the current
manifest's canonical-JSON SHA-256 and refuses to validate if the
snapshot's recorded canonical-SHA differs. This means the manifest
itself cannot be silently widened between snapshot and check —
attempting to do so is its own `PUBLICATION_PLUMBING_VIOLATION`. The
canonical-JSON hash is used (sorted keys, no whitespace) so pretty-print
reformatting doesn't drift the hash.

*Pinned artifacts (paths repo-relative, hashes SHA-256, 2026-05-16
PT).*

| artifact | path | sha256 |
| --- | --- | --- |
| C5 write-path manifest | `results/structural-failure/cut2-prereg/c5-write-path-manifest.json` | raw `7e51bc304ead2b057b6d0ab575f9ee666437bd6ec58ae3f2fe279e3e5e2ab3bf` · canonical `bfa2dd666068d0c180a46ad4469f122a9c34a4970b7348c54a716f7309b3b40f` |
| C5 guard script | `scripts/cut2-publication-plumbing-guard.mjs` | `5e85928384140f584a55e75df7ed4795f209426b3ba1603a13e0d717e95ffa06` |

The **canonical** manifest SHA is the one the guard's `check` step
verifies against; the raw SHA is recorded for human spot-checks of the
on-disk pretty-printed file. Pretty-print reformatting that leaves the
canonical bytes unchanged is **not** a manifest drift.

*Usage (frozen as the Cut-2-step wrapper).*

```
# pre-run, snapshot full tree state
node scripts/cut2-publication-plumbing-guard.mjs snapshot \
     --out results/structural-failure/cut2-<step>/c5-pre-snapshot.json

# ... Cut-2 step runs here; harness writes only under
# results/structural-failure/cut2-*/ per C5/§1; doc filings are by-hand
# under docs/prereg/structural-failure-coincidence/ and are NOT
# admissible from within the run ...

# post-run, diff against the snapshot
node scripts/cut2-publication-plumbing-guard.mjs check \
     --snapshot results/structural-failure/cut2-<step>/c5-pre-snapshot.json
```

Non-zero exit + `PUBLICATION_PLUMBING_VIOLATION` from the `check` step
is **terminal**: the Cut-2 step is VOID and may not be reported as
PASS, regardless of D1∧D2∧D3 / four-quantity outcome.

*Smoke-test verification status.* The guard script's hash utilities
(`hash-file`, `hash-canonical-json`) were exercised in the authoring
environment and match independent SHA-256 of the same bytes; the
manifest and the script both round-trip cleanly through them. A
**full** snapshot+check round-trip was attempted but exceeded the
authoring environment's 45-second sandbox time budget: the repo's
`results/` tree is ~67k files / ~3.9 GB outside the allowlist by
design, and hashing all of it takes longer than the sandbox allows.
This is **expected and intentional**: the freeze chose full-tree
coverage over a blocklist precisely to make the guard un-narrowable.
The round-trip will be verified on the Windows host (which has no
sandbox timeout) before the first real Cut-2 step.

*Scope deliberately unchanged.* `scripts/copy-site-docs.mjs` is **still
not** modified — hardening it remains a separate, larger decision out
of C5 scope, per the original C5 amendment and this freeze's §4.

*Note on a stray scratch file.* The C5 smoke-test left one zero-byte
stray file inside the C5 allowlist:
`results/structural-failure/cut2-prereg/_smoke_test_scratch_DELETE_FROM_WINDOWS.txt`.
It exists because the authoring sandbox can rename but not unlink
files under the Windows mount; it has been truncated to 0 bytes and
moved inside the allowlist so it does not pollute the working tree or
trip the guard. It is safe to delete from the Windows side at any time
and is not a Cut-2 artifact.

Justification: closes the C5 operational-artifact obligations (manifest
+ runnable guard) with hashes pinned and semantics frozen in writing.
No frozen body edited; no threshold/boundary moved; Public-Language
Constraint remains fully in force. Cut-2-execute remains HELD on the
joint admission re-run with C2-A / C3-A / C4-A receipts also landed.

**2026-05-16 (PT) — Codex audit of Wave-1 C5 concrete fill.**
Operational artifacts verified as filed: `hash-canonical-json` for the
C5 manifest returned `bfa2dd666068d0c180a46ad4469f122a9c34a4970b7348c54a716f7309b3b40f`,
`hash-file` for the manifest returned
`7e51bc304ead2b057b6d0ab575f9ee666437bd6ec58ae3f2fe279e3e5e2ab3bf`,
and `hash-file` for the guard script returned
`5e85928384140f584a55e75df7ed4795f209426b3ba1603a13e0d717e95ffa06`,
matching the pinned table above. The implementation resolves the prior
C5 audit holds: snapshot semantics are explicit, tracked/untracked/
ignored file coverage is implemented by filesystem traversal rather
than `git diff`, paths are normalized, and symlink escapes are rejected.
Two joint-admission notes remain. First, the full snapshot/check
round-trip is deliberately host-run because this repo's full tree
exceeds the authoring sandbox's short timeout; admission should require
that host receipt before the first Cut-2 step. Second, the broad
`results/structural-failure/cut2-*/` allowlist intentionally includes
`cut2-prereg`; therefore C5 is the publication-plumbing guard, not the
immutability guard for other prereg artifacts stored there. C4/C2/C3
artifact immutability must still be checked by their own pinned hashes
at the joint admission re-run. No harness/controller run.
