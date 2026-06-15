# Tab system

## What it is

Hash-based navigation between top-level dashboard tabs. One tab visible at a time; clicking a tab sets `location.hash` and a single function flips the active class on the matching tab + tab-body.

## How it works

Tabs in HTML at [index.html:63–70](../../../veritate_mri/web/index.html#L63):

```html
<div class="tabs">
  <div class="tab" data-tab="generation">Generation</div>
  <div class="tab" data-tab="training">Training</div>
  ...
</div>
```

Tab bodies use the same `data-tab` attribute; CSS rules at [index.css:58–59](../../../veritate_mri/web/index.css#L58) show only the body whose data-tab matches the active class.

Activation logic at [index.js:2097–2140](../../../veritate_mri/web/index.js#L2097):

1. `activateTab(name)` validates `name` against a hardcoded `valid` array.
2. Toggles `.active` on all `.tab` and `.tab-body` elements by matching `data-tab`.
3. Branches per-tab to call init/cleanup (start polling, mount lazy content, stop SSE streams).

Click handler at [index.js:2142–2146](../../../veritate_mri/web/index.js#L2142) sets `location.hash = t.dataset.tab`. Hashchange listener at [index.js:2156](../../../veritate_mri/web/index.js#L2156) drives the actual `activateTab` call.

## The `valid` allowlist

Hardcoded array at [index.js:2098](../../../veritate_mri/web/index.js#L2098):

```js
const valid = ["generation", "learning", "training", "wiki", "logs", "settings"];
if (!valid.includes(name)) name = "generation";
```

Unknown tab names silently fall back to `generation`. **Adding a new tab requires adding its name here** — without it, the tab body never gets the `.active` class and the new tab appears to do nothing.

## Dependencies

- [index.html](../../../veritate_mri/web/index.html) — declares `.tab` and `.tab-body` elements.
- [index.css:55–59](../../../veritate_mri/web/index.css#L55) — `.tab-body { display: none }` + `.active { display: block }`.
- Each tab's per-tab init function (`startTrainPolling`, `ensureLearningLoaded`, `ensureWikiLoaded`, etc.).

## Pitfalls

- The `valid` array is the sole source of truth for valid tab names. Don't rely on `data-tab` to advertise routability.
- Hash navigation doesn't push to history if the same tab is clicked twice — listen to both `hashchange` and `click` if you need an event every activation.
- Tab init runs synchronously inside `activateTab`. Heavy work (loading large data, fitting many canvases) belongs inside a `requestAnimationFrame` callback, as the existing branches do.

## See also

- [data_flow.md](data_flow.md) — what each tab's init function fetches.
- [standalone_modules.md](standalone_modules.md) — IIFE modules that hook the tab system from outside `index.js`.
