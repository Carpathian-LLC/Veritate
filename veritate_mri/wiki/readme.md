# wiki

Markdown content tree served by the MRI server and pulled by the public site.

## layout

```
veritate_mri/wiki/
  <category>/
    <slug>.md
```

- `<category>` and `<slug>` must match `[a-z0-9_]` and `[a-z0-9_\-]` respectively.
- Each `.md` starts with optional frontmatter:

```
---
title: Short title
date: 2026-05-05
tags: [build, vnni]
summary: One sentence shown in the listing.
---

Body markdown.
```

## endpoints

- `GET /wiki` lists categories.
- `GET /wiki/<category>` lists entries (frontmatter only).
- `GET /wiki/<category>/<slug>` returns frontmatter, raw body, and rendered HTML.

## adding content

Drop a new `.md` under the appropriate category. Commit. The local server picks it up on the next request; the public site picks it up on the next repo pull.
