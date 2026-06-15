<!-- markdownlint-disable MD029 -->
# claude merge rules

Contract for how Claude handles branch merges in the Veritate repo. This doc lives next to `claude_preflight.md` so it can be referenced or removed cleanly. If anything in `claude_preflight.md` conflicts with this doc, this doc wins for the merge surface only.

## rules

1. **Merging requires explicit user permission per merge.** Even if the merge looks clean, run no `git merge`, branch-crossing rebase, or equivalent integration without the user's specific go-ahead for that specific merge. Auto-mode does not relax this. This is the single hardest rule in this doc.
2. Branch flow is strictly one-directional: `experimental` → `dev` → `main`. `main` ↔ `experimental` direct merges never happen under any circumstance.
3. `dev` is canonical and is UAT for `experimental`. Canonical README, canonical config, and canonical training reference all live on `dev`. `experimental` is the working forward-edge.
4. Files that do NOT carry across an `experimental` → `dev` merge:
    - `veritate_mri/wiki/build_notes/*` — per-build, stays on its branch.
    - `versions.json` — never bumps as part of a merge; each branch keeps its own version line.
5. Files that DO carry across: code, fixes, improvements. Take both sides' additive changes wherever they don't logically conflict. Surface only genuine "same line, different intent" content conflicts to the user for resolution.
6. End state target: after an `experimental` ↔ `dev` merge, the two branches are at the same commit — neither ahead nor behind.
7. Every merge is preservation, not selection. Do not silently drop improvements or fixes from either side.
8. **Merges happen only on local throwaway branches.** Never run `git merge` directly on `dev`, `experimental`, or `main`. Always `git checkout -b <merge-branch>` from the target first, do the merge work there, and only fast-forward the real branch after the user approves the result. The throwaway branch is local-only and never pushed to origin without an explicit per-action permission. If anything goes sideways, `git branch -D <merge-branch>` leaves zero trace.
9. **Tests pass before any merge to `dev`.** Run `pytest tests/` on the throwaway merge branch before fast-forwarding `dev`. Any failing test blocks the merge. If the merge integrates new functionality, a test for that functionality lands in the same merge commit (per preflight rule 52).

## procedure

When the user proposes a merge:

1. Inspect read-only: divergence count, file conflicts, location collisions, conflict surface. Report to the user.
2. Wait for the explicit "merge it" before any mutating command.
3. Create a throwaway branch off the target per rule 8.
4. During resolution, apply rule 4 (drop wiki and `versions.json` from the experimental side; keep dev's).
5. Apply rule 5: take both sides' additive changes; only surface genuine conflicts to the user.
6. Run `pytest tests/` on the throwaway branch (rule 9). If any test fails, fix or abort. Do not commit a merge with failing tests.
7. Get user approval on the resolved tree before committing the merge.
8. Fast-forward the real branch only after the merge commit has been reviewed.
9. Final state target per rule 6.
