/* Developed by Carpathian, LLC. Distribution Not Authorized. */
/* veritate_mri/web/wiki.js */

// Wiki tab: renders repo-root documentation.md as a sticky contents rail plus
// one scrolling article. index.js calls window.Wiki.load() when the tab opens.

(function () {
  "use strict";

  const SPY_OFFSET = 120;   // px below the viewport top that counts as "reading here"

  let loaded  = false;
  let loading = false;
  let links   = [];         // { a, heading } in document order, headings resolved after render
  let active  = null;
  let queued  = false;

  function by_id(id) { return document.getElementById(id); }

  function render_toc(sections) {
    const toc = by_id("wikiToc");
    toc.textContent = "";
    links = [];
    if (!sections.length) {
      toc.innerHTML = '<div class="wiki-nav-empty">documentation.md has no sections.</div>';
      return;
    }
    links = sections.map(s => {
      const a = document.createElement("a");
      a.className = "lvl" + s.level;
      a.textContent = s.title;
      a.dataset.slug = s.slug;
      a.addEventListener("click", () => {
        const h = document.getElementById(s.slug);
        if (h) h.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      toc.appendChild(a);
      return { a: a, heading: null };
    });
  }

  // Keep a group visible when it matches, or when any section under it does.
  function apply_filter(q) {
    const needle = q.trim().toLowerCase();
    let shown = 0;
    let group = null;
    let group_hits = 0;
    const close_group = () => {
      if (group) group.a.classList.toggle("hidden", group_hits === 0);
    };
    links.forEach(l => {
      const hit = !needle || l.a.textContent.toLowerCase().includes(needle);
      if (l.a.classList.contains("lvl2")) {
        close_group();
        group = l;
        group_hits = hit ? 1 : 0;
        if (hit) shown++;
        return;
      }
      l.a.classList.toggle("hidden", !hit);
      if (hit) { group_hits++; shown++; }
    });
    close_group();
    const empty = by_id("wikiNavEmpty");
    if (empty) empty.style.display = shown ? "none" : "block";
  }

  function set_active(link) {
    if (link === active) return;
    if (active) active.a.classList.remove("active");
    active = link;
    if (!active) return;
    active.a.classList.add("active");
    const rail = by_id("wikiToc");
    const top  = active.a.offsetTop;
    const bot  = top + active.a.offsetHeight;
    if (top < rail.scrollTop || bot > rail.scrollTop + rail.clientHeight) {
      rail.scrollTop = top - rail.clientHeight / 2;
    }
  }

  function spy() {
    queued = false;
    let current = null;
    for (const l of links) {
      if (!l.heading) continue;
      if (l.heading.getBoundingClientRect().top - SPY_OFFSET > 0) break;
      current = l;
    }
    set_active(current || links.find(l => l.heading) || null);
  }

  function on_scroll() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(spy);
  }

  function load() {
    if (loaded || loading) return;
    loading = true;
    by_id("wikiArticle").innerHTML = '<div class="wiki-msg">loading documentation.md…</div>';
    Promise.all([fetch("/wiki").then(r => r.json()), fetch("/wiki/doc").then(r => r.json())])
      .then(([toc, doc]) => {
        by_id("wikiArticle").innerHTML = doc.body_html || "";
        render_toc(toc.sections || []);
        links.forEach(l => { l.heading = document.getElementById(l.a.dataset.slug); });
        by_id("wikiFilter").addEventListener("input", e => apply_filter(e.target.value));
        window.addEventListener("scroll", on_scroll, { passive: true });
        spy();
        loaded = true;
      })
      .catch(e => {
        const msg = document.createElement("div");
        msg.className = "wiki-msg";
        msg.textContent = "failed to load the wiki: " + e;
        by_id("wikiArticle").replaceChildren(msg);
      })
      .finally(() => { loading = false; });
  }

  window.Wiki = { load: load };
})();
