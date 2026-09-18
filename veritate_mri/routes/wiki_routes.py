# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - in-app wiki: serves repo-root documentation.md as a table of contents, the
#   whole rendered doc, and standalone per-section pages for "learn more" links.
# veritate_mri/routes/wiki_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import html as _html

from flask import Response
from readers import wiki as wiki_reader

# ------------------------------------------------------------------------------------
# Constants

# Mirrors veritate_mri/web/wiki.css so a `learn more` tab looks like the dashboard
# it was opened from: same palette, prose in a proportional face, code in mono.
_WIKI_PAGE_CSS = (
    ":root{color-scheme:dark;--bg:#08090c;--line:#1e2330;--text:#d6d8db;--dim:#6f7480;"
    "--soft:#9aa4b5;--accent:#6aa6ff;--warm:#ffae5d}"
    "*{box-sizing:border-box}"
    "body{margin:0;background:var(--bg);color:var(--text);"
    "font:13.5px/1.7 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}"
    ".wrap{max-width:78ch;margin:0 auto;padding:36px 22px 64px}"
    "h1,h2,h3,h4{letter-spacing:0}"
    "h1{color:#fff;font-size:21px;font-weight:600;margin:0 0 6px;padding-bottom:12px;"
    "border-bottom:1px solid var(--line)}"
    "h2{color:#fff;font-size:16px;font-weight:600;margin:38px 0 4px;padding-top:16px;"
    "border-top:1px solid var(--line)}"
    "h3{color:var(--accent);font-size:13.5px;font-weight:600;margin:24px 0 2px}"
    # The page is one section, so its own heading is the page title whatever level it is.
    ".wrap>:first-child{color:#fff;font-size:21px;font-weight:600;margin:0 0 6px;"
    "padding-bottom:12px;border-bottom:1px solid var(--line);border-top:none;padding-top:0}"
    "h4{color:var(--soft);font-size:12.5px;font-weight:600;margin:16px 0 2px}"
    "p{margin:10px 0}ul,ol{margin:10px 0;padding-left:20px}li{margin:4px 0}"
    "strong{color:#fff;font-weight:600}em{color:var(--soft);font-style:italic}"
    "a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}"
    "hr{border:none;border-top:1px solid var(--line);margin:20px 0}"
    "code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;color:var(--warm);"
    "background:#0a0c12;border-radius:3px;padding:1px 4px}"
    "pre{background:#06070a;border:1px solid var(--line);border-radius:4px;padding:12px 14px;"
    "overflow-x:auto;margin:12px 0}"
    "pre code{background:none;padding:0;color:var(--text);font-size:12px}"
    "blockquote{margin:14px 0;padding:8px 14px;border-left:2px solid var(--accent);"
    "background:rgba(106,166,255,.06);color:var(--soft);border-radius:0 3px 3px 0}"
    "table{width:100%;border-collapse:collapse;margin:14px 0;"
    "font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px}"
    "th{color:var(--accent);font-size:10px;letter-spacing:.1em;text-transform:uppercase;font-weight:600;"
    "padding:6px 10px;border-bottom:1px solid var(--line)}"
    "td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top;color:var(--soft)}"
    ".wiki-xref-path{color:var(--dim);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px}"
    ".src{color:var(--dim);font-size:10px;letter-spacing:.14em;text-transform:uppercase;margin-top:44px;"
    "border-top:1px solid var(--line);padding-top:12px}"
)

# ------------------------------------------------------------------------------------
# Functions

def _wiki_page_html(slug, title, body_html):
    title_esc = _html.escape(title)
    slug_esc  = _html.escape(slug)
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{title_esc} · Veritate wiki</title><style>{_WIKI_PAGE_CSS}</style></head>"
        "<body><div class=\"wrap\">" + (body_html or "") +
        f"<div class=\"src\">Veritate wiki · {slug_esc}</div></div></body></html>"
    )


def _section_title(slug):
    for s in wiki_reader.toc():
        if s["slug"] == slug:
            return s["title"]
    return slug


def register(app):
    @app.route("/wiki")
    def wiki_index():
        return {"sections": wiki_reader.toc()}

    @app.route("/wiki/doc")
    def wiki_doc():
        return {"body_html": wiki_reader.doc_html()}

    @app.route("/wiki/<slug>/page")
    @app.route("/wiki/<category>/<slug>/page")
    def wiki_section_page(slug, category=None):
        """Standalone HTML render of one documentation.md section, for opening in
        a new tab from a setting's `learn more` link. `category` is accepted for
        URL compatibility with existing links and otherwise ignored."""
        del category
        body_html = wiki_reader.section_html(slug)
        if body_html is None:
            return (f"section not found: {slug}", 404)
        return Response(_wiki_page_html(slug, _section_title(slug), body_html), mimetype="text/html")
