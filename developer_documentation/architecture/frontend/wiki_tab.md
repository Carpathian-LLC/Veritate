# Wiki tab

## What it is

Browser for the in-app markdown wiki: API reference, extension authoring, per-setting pages, build notes, and concepts.

## How it works

Markup at [index.html:1645-1658](../../../veritate_mri/web/index.html#L1645).

- Subtabs (categories) populate from `GET /wiki`, which returns `{"categories": [...]}`.
- Per-category entry list populates from `GET /wiki/<category>`, which returns `{"category": ..., "entries": [...]}`.
- Selecting an entry fetches `GET /wiki/<category>/<slug>` and renders the returned entry body in `#wikiEntry`.
- `ensureWikiLoaded()` runs on tab activation and is idempotent: only the first activation does the fetch.

Entries are read from `veritate_mri/data/wiki/<category>/<slug>.md` by [wiki.py](../../../veritate_mri/readers/wiki.py); the routes are in [wiki_routes.py](../../../veritate_mri/routes/wiki_routes.py).

`GET /wiki/<category>/<slug>/page` ([wiki_routes.py](../../../veritate_mri/routes/wiki_routes.py#L73)) returns a standalone styled HTML render (`_wiki_page_html`) of a single entry for opening in a new browser tab, distinct from the JSON entry route the tab itself consumes. Per-setting pages live in the `settings/` wiki category (`veritate_mri/data/wiki/settings/<slug>.md`) and are the target of the training form's "learn more" links (see [training_tab.md](training_tab.md)).

## Dependencies

- `GET /wiki`, `GET /wiki/<category>`, `GET /wiki/<category>/<slug>`, and `GET /wiki/<category>/<slug>/page` routes.
- Reader [wiki.py](../../../veritate_mri/readers/wiki.py).

## Pitfalls

- A missing category returns 404 from `GET /wiki/<category>`, and a missing entry returns 404 from `GET /wiki/<category>/<slug>`; both carry an `error` field rather than an empty body.
- Entries render through a safe markdown subset. Headings, lists, fenced code, blockquotes, tables, inline code, bold, italic, and links render; nested lists with mixed indentation do not.
- The wiki is read-only from the dashboard. Editing happens on disk via normal text editors.
