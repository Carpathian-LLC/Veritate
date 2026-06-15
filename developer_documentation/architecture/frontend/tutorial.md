# Tutorial overlay

## What it is

Onboarding walkthrough: spotlight overlays with tooltip cards that explain each tab and key panel. Auto-starts for first-time users; can be re-run from settings.

## How it works

File: [veritate_mri/web/tutorial.js](../../../veritate_mri/web/tutorial.js) + [tutorial.css](../../../veritate_mri/web/tutorial.css). IIFE module.

- Reads `tutorial_enabled` and `tutorial_completed` from `/settings`. Auto-starts when enabled and not completed.
- `window.Tutorial.start()` runs it manually.
- Steps walk through tabs in order; each step has a spotlighted element (DOM selector), a card with text, and Next/Skip controls.
- On completion, POSTs `tutorial_completed: true` to `/settings`.

## Dependencies

- `/settings` GET/POST for the persistence flags.
- DOM selectors targeting elements in `index.html`. Renaming or removing tabs without updating the tutorial steps causes the spotlight to point at nothing.

## Pitfalls

- Tutorial state is per-machine (settings live in `data/mri_settings.json`). Wiping the machine resets it.
- When adding a new tab, decide whether the tutorial should cover it. If yes, add a step in `tutorial.js` referencing the new tab's selector.
