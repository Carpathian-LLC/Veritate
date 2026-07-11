# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - in-app wiki: categories, per-category entry lists, single entries.
# veritate_mri/routes/wiki_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import html as _html

from flask import Response

from readers import wiki as wiki_reader

# ------------------------------------------------------------------------------------
# Constants

_WIKI_PAGE_CSS = (
    ":root{color-scheme:dark}"
    "body{margin:0;background:#0e1116;color:#c9d1d9;"
    "font:15px/1.6 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}"
    ".wrap{max-width:760px;margin:0 auto;padding:2.5rem 1.5rem 4rem}"
    "h1,h2,h3{color:#f0f6fc;line-height:1.25}"
    "h1{font-size:1.6rem;border-bottom:1px solid #21262d;padding-bottom:.4rem}"
    "a{color:#58a6ff}"
    "code{background:#161b22;padding:.1em .35em;border-radius:4px;font-size:.9em}"
    "pre{background:#161b22;padding:1rem;border-radius:6px;overflow-x:auto}"
    "pre code{background:none;padding:0}"
    "table{border-collapse:collapse;width:100%}"
    "th,td{border:1px solid #21262d;padding:.4rem .6rem;text-align:left}"
    "blockquote{border-left:3px solid #30363d;margin:0;padding:.2rem 1rem;color:#8b949e}"
    ".src{color:#6e7681;font-size:.8rem;margin-top:2.5rem;border-top:1px solid #21262d;padding-top:.8rem}"
)

# ------------------------------------------------------------------------------------
# Functions

def _wiki_page_html(entry):
    title = _html.escape(str(entry.get("title") or entry.get("slug") or "wiki"))
    cat   = _html.escape(str(entry.get("category") or ""))
    slug  = _html.escape(str(entry.get("slug") or ""))
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{title} · Veritate wiki</title><style>{_WIKI_PAGE_CSS}</style></head>"
        "<body><div class=\"wrap\">" + (entry.get("body_html") or "") +
        f"<div class=\"src\">Veritate wiki · {cat}/{slug}</div></div></body></html>"
    )


def register(app):
    @app.route("/wiki")
    def wiki_index():
        return {"categories": wiki_reader.list_categories()}

    @app.route("/wiki/<category>")
    def wiki_category(category):
        entries = wiki_reader.list_entries(category)
        if entries is None:
            return ({"error": f"category not found: {category}"}, 404)
        return {"category": category, "entries": entries}

    @app.route("/wiki/<category>/<slug>")
    def wiki_entry(category, slug):
        entry = wiki_reader.load_entry(category, slug)
        if entry is None:
            return ({"error": f"entry not found: {category}/{slug}"}, 404)
        return entry

    @app.route("/wiki/<category>/<slug>/page")
    def wiki_entry_page(category, slug):
        """Standalone HTML render of a wiki entry, for opening in a new tab from
        a setting's `learn more` link."""
        entry = wiki_reader.load_entry(category, slug)
        if entry is None:
            return (f"entry not found: {category}/{slug}", 404)
        return Response(_wiki_page_html(entry), mimetype="text/html")
