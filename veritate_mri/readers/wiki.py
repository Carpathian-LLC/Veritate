# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - reader for the single-file platform doc at repo-root documentation.md.
# - splits the file into ## / ### sections by heading, caches by (path, mtime).
# - renders a safe markdown subset to html; headings get id=<slug> for anchor jumps.
# veritate_mri/readers/wiki.py
# ------------------------------------------------------------------------------------
# Imports:

import html
import os
import re

from . import paths as paths_mod

# ------------------------------------------------------------------------------------
# Constants

SECTION_HEADING_LEVELS = (2, 3)

SLUG_KEEP_RE = re.compile(r"[^a-z0-9_]")

HEADING_RE  = re.compile(r"^(#{1,6})\s+(.*)$")
HRULE_RE    = re.compile(r"^-{3,}\s*$")
ULIST_RE    = re.compile(r"^[-*]\s+(.*)$")
OLIST_RE    = re.compile(r"^(\d+)\.\s+(.*)$")
FENCE_RE    = re.compile(r"^```\s*([a-zA-Z0-9_\-]*)\s*$")
QUOTE_RE    = re.compile(r"^>\s?(.*)$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

INLINE_CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE        = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE      = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
LINK_RE        = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
AUTOLINK_RE    = re.compile(r"(?<![\"'>=])\b(https?://[^\s<]+)")

INLINE_PLACEHOLDER = "\x00CODE\x00"

_doc_cache = {"key": None, "text": "", "sections": []}

# ------------------------------------------------------------------------------------
# Functions

def slugify(text):
    s = text.strip().lower().replace(" ", "_")
    return SLUG_KEEP_RE.sub("", s)


def load_doc():
    path = paths_mod.documentation_path()
    mtime = os.path.getmtime(path)
    key = (path, mtime)
    if _doc_cache["key"] == key:
        return _doc_cache["sections"]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sections = _parse_sections(text)
    _doc_cache["key"] = key
    _doc_cache["text"] = text
    _doc_cache["sections"] = sections
    return sections


def _parse_sections(text):
    lines = text.split("\n")
    headings = []
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line.strip())
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))
    seen = set()
    sections = []
    for idx, (line_no, level, title) in enumerate(headings):
        if level not in SECTION_HEADING_LEVELS:
            continue
        slug = slugify(title)
        if slug in seen:
            continue
        seen.add(slug)
        end = len(lines)
        for later_no, later_level, _ in headings[idx + 1:]:
            if later_level <= level:
                end = later_no
                break
        sections.append({
            "slug": slug, "title": title, "level": level,
            "body": "\n".join(lines[line_no:end]),
        })
    return sections


def toc():
    return [{"slug": s["slug"], "title": s["title"], "level": s["level"]} for s in load_doc()]


def section_html(slug):
    for s in load_doc():
        if s["slug"] == slug:
            return render_markdown(s["body"])
    return None


def doc_html():
    load_doc()
    return render_markdown(_doc_cache["text"])


def render_markdown(text):
    if not text:
        return ""
    out = []
    lines = text.split("\n")
    i = 0
    in_para = []
    in_ul = False
    in_ol = False
    quote_buf = []

    def flush_para():
        if in_para:
            out.append("<p>" + _inline(" ".join(in_para)) + "</p>")
            in_para.clear()

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>"); in_ul = False
        if in_ol:
            out.append("</ol>"); in_ol = False

    def flush_quote():
        if quote_buf:
            inner = render_markdown("\n".join(quote_buf))
            out.append("<blockquote>" + inner + "</blockquote>")
            quote_buf.clear()

    while i < len(lines):
        line = lines[i]
        m = FENCE_RE.match(line)
        if m:
            flush_para(); close_lists(); flush_quote()
            lang = m.group(1) or ""
            j = i + 1
            buf = []
            while j < len(lines) and not FENCE_RE.match(lines[j]):
                buf.append(lines[j])
                j += 1
            cls = f' class="lang-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>" + html.escape("\n".join(buf)) + "</code></pre>")
            i = j + 1
            continue
        stripped = line.strip()
        if not stripped:
            flush_para(); close_lists(); flush_quote()
            i += 1
            continue
        m = QUOTE_RE.match(stripped)
        if m:
            flush_para(); close_lists()
            quote_buf.append(m.group(1))
            i += 1
            continue
        flush_quote()
        if _is_table_start(lines, i):
            flush_para(); close_lists()
            j, html_block = _render_table(lines, i)
            out.append(html_block)
            i = j
            continue
        m = HEADING_RE.match(stripped)
        if m:
            flush_para(); close_lists()
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            out.append(f'<h{level} id="{slugify(heading_text)}">{_inline(heading_text)}</h{level}>')
            i += 1
            continue
        if HRULE_RE.match(stripped):
            flush_para(); close_lists()
            out.append("<hr>")
            i += 1
            continue
        m = ULIST_RE.match(stripped)
        if m:
            flush_para()
            if in_ol: out.append("</ol>"); in_ol = False
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append("<li>" + _inline(m.group(1)) + "</li>")
            i += 1
            continue
        m = OLIST_RE.match(stripped)
        if m:
            flush_para()
            if in_ul: out.append("</ul>"); in_ul = False
            if not in_ol: out.append("<ol>"); in_ol = True
            out.append("<li>" + _inline(m.group(2)) + "</li>")
            i += 1
            continue
        close_lists()
        in_para.append(stripped)
        i += 1
    flush_para(); close_lists(); flush_quote()
    return "\n".join(out)


def _is_table_start(lines, i):
    if i + 1 >= len(lines): return False
    head = lines[i].strip()
    sep  = lines[i + 1].strip()
    if "|" not in head: return False
    return bool(TABLE_SEP_RE.match(sep))


def _split_row(line):
    s = line.strip()
    if s.startswith("|"): s = s[1:]
    if s.endswith("|"):   s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _render_table(lines, i):
    head_cells = _split_row(lines[i])
    sep_cells  = _split_row(lines[i + 1])
    aligns = []
    for c in sep_cells:
        left  = c.startswith(":")
        right = c.endswith(":")
        if left and right: aligns.append("center")
        elif right:        aligns.append("right")
        else:              aligns.append("left")
    while len(aligns) < len(head_cells): aligns.append("left")
    j = i + 2
    rows = []
    while j < len(lines):
        s = lines[j].strip()
        if not s or "|" not in s: break
        rows.append(_split_row(lines[j]))
        j += 1
    parts = ["<table>", "<thead><tr>"]
    for k, c in enumerate(head_cells):
        parts.append(f'<th style="text-align:{aligns[k]}">{_inline(c)}</th>')
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for k, c in enumerate(row):
            a = aligns[k] if k < len(aligns) else "left"
            parts.append(f'<td style="text-align:{a}">{_inline(c)}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return j, "".join(parts)


def _inline(text):
    spans = []
    def stash_code(m):
        spans.append("<code>" + html.escape(m.group(1)) + "</code>")
        return INLINE_PLACEHOLDER
    text = INLINE_CODE_RE.sub(stash_code, text)
    text = html.escape(text)
    text = BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = ITALIC_RE.sub(r"<em>\1</em>", text)
    def link_sub(m):
        href = m.group(2)
        label = m.group(1)
        if href.startswith(("http://", "https://")):
            return f'<a href="{href}" target="_blank" rel="noopener">{label}</a>'
        if href.startswith(("/", "#")):
            return f'<a href="{href}">{label}</a>'
        # Relative path inside the repo: keep the label visible, show
        # the path muted alongside it. We don't link because the wiki
        # is rendered in the dashboard, not on disk.
        return f'<span class="wiki-xref">{label} <span class="wiki-xref-path">{href}</span></span>'
    text = LINK_RE.sub(link_sub, text)
    text = AUTOLINK_RE.sub(r'<a href="\1" target="_blank" rel="noopener">\1</a>', text)
    idx = [0]
    def restore(_m):
        s = spans[idx[0]]; idx[0] += 1
        return s
    return re.sub(INLINE_PLACEHOLDER, restore, text)
