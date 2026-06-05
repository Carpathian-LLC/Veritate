# Standalone modules

## What they are

Self-contained JavaScript panels that mount themselves into the dashboard without requiring edits to `index.js`. Each is an IIFE (immediately invoked function expression) that hooks into the existing tab system via `hashchange` and click events, exposes nothing globally, and can be deleted as a unit.

## The pattern

```js
(function () {
  "use strict";

  // private state
  const state = { ... };

  function mount() { ... }
  function onTabSwitch() { ... }

  document.addEventListener("DOMContentLoaded", () => {
    window.addEventListener("hashchange", onTabSwitch);
    document.querySelectorAll(".tab").forEach(t => {
      t.addEventListener("click", () => setTimeout(onTabSwitch, 30));
    });
    onTabSwitch();
  });
})();
```

The IIFE keeps all state out of the global namespace. The module mounts into a host element in `index.html` (e.g., `<div id="coralBody"></div>`) and registers its own listeners.

## Current modules

- **[prune.js](../../../veritate_mri/web/prune.js)** — neuron-pruning panel that lives inside the Models tab. Calls `/pruning/report` and `/pruning/generate_plugin`.
- **[tutorial.js](../../../veritate_mri/web/tutorial.js)** — onboarding walkthrough that spotlights elements with overlay cards. Auto-starts when `tutorial_enabled && !tutorial_completed`.
- **[coral_lab.js](../../../veritate_mri/web/coral_lab.js)** — three-column run comparison for the coral merge experiment. Polls `/runs` and `/run/<name>/csv` every 5s.

## Wire-up to index.html

A standalone module needs three things in `index.html`:

1. A `<link rel="stylesheet" href="/static/module.css">` and `<script src="/static/module.js" defer>` in `<head>`.
2. An optional `<div class="tab" data-tab="X">` if it owns a tab.
3. A `<div class="tab-body" data-tab="X">` with a host element the module mounts into.

If the module owns a tab, `"X"` also has to be added to the `valid` array in [index.js:2098](../../../veritate_mri/web/index.js#L2098) — otherwise `activateTab` falls back to `generation` and the tab body never activates.

See [coral_lab.md](coral_lab.md) for a concrete example with deletion instructions.

## Dependencies

- [tab_system.md](tab_system.md) — modules hook the tab system to know when to mount and run.
- The route(s) each module consumes. The convention is to reuse existing routes when possible (Coral Lab uses `/runs` and `/run/<name>/csv`) rather than adding new ones per module.

## Pitfalls

- Modules that own a tab must update the `valid` array — easy to forget. The tab will appear in the bar but clicking it silently routes to `generation`.
- `addEventListener("hashchange", ...)` doesn't fire if the hash doesn't change. Also bind a tab `click` handler with a small `setTimeout` so the module sees re-activations of the same tab.
- Each module's CSS should namespace its class names (e.g., `.coral-*`) to avoid collisions with `index.css`.
