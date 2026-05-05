# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - reader for the wiki content tree under veritate_mri/wiki/<category>/<slug>.md.
# - parses yaml-ish frontmatter and renders a safe markdown subset to html.
# - frontmatter keys: title, date, tags, summary. all optional.
# veritate_mri/readers/wiki.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import re
import html

from . import paths as paths_mod

# ------------------------------------------------------------------------------------
# Constants

FRONTMATTER_DELIM     = "---"
FRONTMATTER_LIST_OPEN = "["
FRONTMATTER_LIST_CLOSE = "]"
KV_SEP                = ":"
LIST_SEP              = ","

CATEGORY_RE = re.compile(r"^[a-z0-9_]+$")
SLUG_RE     = re.compile(r"^[a-z0-9_\-]+$")

HEADING_RE  = re.compile(r"^(#{1,6})\s+(.*)$")
HRULE_RE    = re.compile(r"^-{3,}\s*$")
ULIST_RE    = re.compile(r"^[-*]\s+(.*)$")
OLIST_RE    = re.compile(r"^(\d+)\.\s+(.*)$")
FENCE_RE    = re.compile(r"^```\s*([a-zA-Z0-9_\-]*)\s*$")

INLINE_CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE        = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE      = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
LINK_RE        = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
AUTOLINK_RE    = re.compile(r"(?<![\"'>=])\b(https?://[^\s<]+)")

INLINE_PLACEHOLDER = "\x00CODE\x00"

DEFAULT_TITLE = "(untitled)"

# ------------------------------------------------------------------------------------
# Functions

def _safe_category(c):
    return bool(c) and bool(CATEGORY_RE.match(c))


def _safe_slug(s):
    return bool(s) and bool(SLUG_RE.match(s))


def list_categories():
    root = paths_mod.wiki_root()
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p): continue
        if not _safe_category(name): continue
        out.append({"name": name, "n_entries": _count_entries(p)})
    return out


def _count_entries(category_dir):
    n = 0
    for fn in os.listdir(category_dir):
        if fn.endswith(paths_mod.WIKI_ENTRY_SUFFIX):
            n += 1
    return n


def list_entries(category):
    if not _safe_category(category):
        return None
    cdir = paths_mod.wiki_category_dir(category)
    if not os.path.isdir(cdir):
        return None
    out = []
    for fn in os.listdir(cdir):
        if not fn.endswith(paths_mod.WIKI_ENTRY_SUFFIX): continue
        slug = fn[: -len(paths_mod.WIKI_ENTRY_SUFFIX)]
        if not _safe_slug(slug): continue
        path = os.path.join(cdir, fn)
        meta = _read_meta(path)
        meta["slug"]   = slug
        meta["_mtime"] = os.path.getmtime(path)
        out.append(meta)
    out.sort(key=lambda r: (r.get("date") or "", r["_mtime"], r["slug"]), reverse=True)
    for r in out:
        r.pop("_mtime", None)
    return out


def load_entry(category, slug):
    if not _safe_category(category) or not _safe_slug(slug):
        return None
    p = paths_mod.wiki_entry_path(category, slug)
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    fm, body = _split_frontmatter(text)
    meta = _parse_frontmatter(fm)
    meta["slug"]      = slug
    meta["category"]  = category
    meta["body_md"]   = body
    meta["body_html"] = render_markdown(body)
    return meta


def _read_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    fm, _ = _split_frontmatter(text)
    return _parse_frontmatter(fm)


def _split_frontmatter(text):
    lines = text.split("\n")
    if not lines or lines[0].strip() != FRONTMATTER_DELIM:
        return ({}, text)
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == FRONTMATTER_DELIM:
            end = i
            break
    if end is None:
        return ({}, text)
    fm = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1:])
    return (fm, body.lstrip("\n"))


def _parse_frontmatter(fm):
    if isinstance(fm, dict):
        return dict(fm)
    out = {"title": None, "date": None, "tags": [], "summary": None}
    if not fm:
        return out
    for raw in fm.split("\n"):
        line = raw.rstrip()
        if not line.strip(): continue
        if KV_SEP not in line: continue
        k, v = line.split(KV_SEP, 1)
        k = k.strip().lower()
        v = v.strip()
        if k == "tags":
            out["tags"] = _parse_list(v)
        elif k in ("title", "date", "summary"):
            out[k] = _strip_quotes(v)
    if out["title"] is None:
        out["title"] = DEFAULT_TITLE
    return out


def _strip_quotes(v):
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _parse_list(v):
    if v.startswith(FRONTMATTER_LIST_OPEN) and v.endswith(FRONTMATTER_LIST_CLOSE):
        v = v[1:-1]
    return [_strip_quotes(p.strip()) for p in v.split(LIST_SEP) if p.strip()]


def render_markdown(text):
    if not text:
        return ""
    out = []
    lines = text.split("\n")
    i = 0
    in_para = []
    in_ul = False
    in_ol = False

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

    while i < len(lines):
        line = lines[i]
        m = FENCE_RE.match(line)
        if m:
            flush_para(); close_lists()
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
            flush_para(); close_lists()
            i += 1
            continue
        m = HEADING_RE.match(stripped)
        if m:
            flush_para(); close_lists()
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2).strip())}</h{level}>")
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
    flush_para(); close_lists()
    return "\n".join(out)


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
        if not (href.startswith("http://") or href.startswith("https://")
                or href.startswith("/") or href.startswith("#")):
            return m.group(0)
        return f'<a href="{href}" target="_blank" rel="noopener">{m.group(1)}</a>'
    text = LINK_RE.sub(link_sub, text)
    text = AUTOLINK_RE.sub(r'<a href="\1" target="_blank" rel="noopener">\1</a>', text)
    idx = [0]
    def restore(_m):
        s = spans[idx[0]]; idx[0] += 1
        return s
    text = re.sub(INLINE_PLACEHOLDER, restore, text)
    return text
