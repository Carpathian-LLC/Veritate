# Agent-facing documentation

Files agents must read before working in this repo. The repo-root [claude_preflight.md](../../claude_preflight.md) wins over everything here when they conflict.

## Required reads

- [coding_roe.md](coding_roe.md) — rules 100–128 on writing code in this repo. Cited from preflight rule 20a. Lean code, no defensive bloat, measure before optimizing.
- [claude_merge.md](claude_merge.md) — branch-merge rules. Cited from preflight rule 45. Its rule 1 is absolute: no merge without explicit user permission for that specific merge.
- [agent_roe.md](agent_roe.md) — general agent rules of engagement.

## Updating

These files describe the contract agents follow. If a rule needs to change, update the file and update the corresponding preflight rule in the same change. Don't write rationale or history into the rule files — they're imperative contracts, not design diaries.

## Voice

Same voice rule as the rest of `documentation/`: developer-to-developer, no "the user", no narrative of how the rule came to be. State the rule and how to apply it.
