# Wiki tab

## What it is

Browser for the project's markdown wiki: build notes, experiments, project log entries.

## How it works

Markup at [index.html:1645–1658](../../../veritate_mri/web/index.html#L1645).

- Subtabs (categories) populate from `/wiki/categories`.
- Per-category entry list populates from `/wiki/entries/<category>`.
- Selecting an entry fetches its markdown body and renders it in `#wikiEntry`.
- `ensureWikiLoaded()` runs on tab activation and is idempotent — only the first activation does the fetch.

Backend reads from a wiki directory; see [wiki_routes.py](../../../veritate_mri/routes/wiki_routes.py).

`GET /wiki/<category>/<slug>/page` ([wiki_routes.py](../../../veritate_mri/routes/wiki_routes.py#L73)) returns a standalone styled HTML render (`_wiki_page_html`) of a single entry for opening in a new browser tab, distinct from the JSON entry route the tab itself consumes. Per-setting pages live in the `settings/` wiki category (`veritate_mri/data/wiki/settings/<slug>.md`) and are the target of the training form's "learn more" links (see [training_tab.md](training_tab.md)).

## Dependencies

- `/wiki/categories`, `/wiki/entries/<cat>`, `/wiki/entry/<id>`, and `/wiki/<cat>/<slug>/page` routes.
- Reader [wiki.py](../../../veritate_mri/readers/wiki.py).

## Pitfalls

- Entries are markdown; the renderer is a small client-side library and may not match GitHub-flavored markdown exactly. Code blocks and tables work; nested lists with mixed indentation can render weirdly.
- The wiki is read-only from the dashboard. Editing happens on disk via normal text editors.
