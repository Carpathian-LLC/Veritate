# claude preflight

Rule index. The rules themselves moved to where they act: hard gates are hooks that fire
whether or not anyone remembers them, standards are skills that load with the work that
needs them. Read this file only to resolve a citation ("rule 12") or to see what enforces
what. Numbers are stable and are cited across `worklog.md`, `ideas.md` and `handoff.md`.

Veritate is a platform for training AI models on consumer hardware. That is the whole
product. Anything that does not serve training, evaluating, or running models is out of
scope - push back once naming this rule, propose a separate project, and proceed only if
the user confirms after the pushback. Standing decision. The chat product was extracted
2026-08-20 (`CHAT_HANDOFF.md`); the Generation tab is the only conversational surface.

| rules | subject | where they live now |
|---|---|---|
| 1-4, 7 | communication: short, direct, no padding, no self-attribution | `CLAUDE.md` (always loaded) |
| 5, 6 | pushback required; never declare a goal impossible | skill `veritate-persist` + Stop hook |
| 8, 9 | do only what was asked; keep a todo list | `CLAUDE.md` |
| 10, 11 | irreversible and shared-state actions get confirmed first | no hook - confirm before acting |
| 12 | never stage, commit, push, or bump a version | hook `guard_write.py` (versions.json); git is the user's to run |
| 13 | no training launches outside the dashboard route | skill `veritate-research` + the `documentation.md` checklist |
| 14 | read the ledgers before designing an experiment | skill `veritate-research` |
| 15 | dispatched agents: one scoped task, no edits outside it | `CLAUDE.md` |
| 16 | no merge without permission for that specific merge | no hook - the user runs every git operation |
| 17-30 | code standards, module ownership, one trainer, kernels | skill `veritate-code` |
| 31-33 | tests, determinism, run-it-before-claiming-it | skill `veritate-tests` + hooks `after_edit.py`, `persist.py` |
| 34-38 | five doc files, same-diff contract updates, doc voice | skill `veritate-docs` + hook `guard_write.py` |
| 39, 40 | lab entries for every attempt; never mint a wren name | skill `veritate-research` |

Documentation set, unchanged: `documentation.md` is the single platform reference;
`successes.md` / `failures.md` / `ideas.md` are the research ledgers. `worklog.md`,
`handoff.md` and `lab/` are logs, not documentation. There are no other locations.

A rule that turns out to be wrong is edited where it lives - the skill file or the hook -
not appended here. This file only maps numbers to places.
