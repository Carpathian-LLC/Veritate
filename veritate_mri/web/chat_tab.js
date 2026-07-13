// Developed by Carpathian, LLC. Distribution Not Authorized.
// chat_tab.js : the front-door chat (hybrid.html) ported into the dashboard's
// Chat tab. Same features and same backend endpoints (/hybrid/*). Wrapped in an
// IIFE so none of its globals leak into the dashboard; DOM lookups are scoped to
// #chatMount and use the "ch-" id prefix. chatTabInit() is called by
// activateTab("chat") the first time the tab is opened (lazy network init).
(function () {
  const SETTINGS_KEY = "veritate_chat_settings_v1";
  const CONVO_KEY = "veritate_chat_convo_v1";
  const CLOUD_ID = "cloud";
  const TEACHER_ID = "teacher";
  const TEACHER_PREFIX = "teacher:";
  const VERITATE_DOCS_ID = "veritate_docs";   // grounds the public model on the platform docs
  const WRAP_FIT_GAP = 16;      // px of breathing room below the composer
  const WRAP_MIN_HEIGHT = 200;  // px floor so a very short window still shows log + composer

  const mount = document.getElementById("chatMount");
  if (!mount) return;
  const $ = id => mount.querySelector("#ch-" + id);
  const log = $("log"), q = $("q"), send = $("send");

  const state = Object.assign(
    { model: VERITATE_DOCS_ID, backend: "pytorch", use_rag: false, use_logs: false },
    loadSettings());
  let cCapable = new Set();    // models with a veritate.bin (c-engine runnable)
  let hasCorpus = false, nFiles = 0, busy = false;
  let remoteById = new Map();  // remote model id -> {label, group} from /hybrid/models
  let convo = loadConvo();     // {summary, turns} conversation memory, server-compacted

  function loadSettings() {
    try { return JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") || {}; }
    catch (_) { return {}; }
  }
  function saveSettings() {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(state)); } catch (_) {}
  }
  function loadConvo() {
    try { const c = JSON.parse(localStorage.getItem(CONVO_KEY) || "{}") || {};
          return { summary: c.summary || "", turns: Array.isArray(c.turns) ? c.turns : [] }; }
    catch (_) { return { summary: "", turns: [] }; }
  }
  function saveConvo() {
    try { localStorage.setItem(CONVO_KEY, JSON.stringify(convo)); } catch (_) {}
  }
  function newChat() {
    convo = { summary: "", turns: [] }; saveConvo();
    lastCtx = null;
    log.innerHTML = '<div class="hint-row">Ask a question. Pick one of your trained models in settings, or let the cloud answer.</div>';
    updateCtxRing(0); q.focus();
  }

  // Gauge is driven entirely by the server's per-model `context` meter on each reply.
  const CTX_CIRC = 2 * Math.PI * 14;
  let lastCtx = null;   // last server context meter {turns, chars, char_limit, pct}; null until a reply lands
  function updateCtxRing(pct) {
    const arc = $("ctxArc"); if (!arc) return;
    const p = Math.max(0, Math.min(1, pct || 0));
    arc.style.strokeDashoffset = (CTX_CIRC * (1 - p)).toFixed(2);
    arc.style.stroke = p >= 0.9 ? "#e0a45a" : "var(--accent)";
    $("ctxPct").textContent = Math.round(p * 100) + "%";
    $("ctxRing").querySelector("title").textContent = `context ${Math.round(p * 100)}% full`;
  }
  function ctxDetail() {
    if (lastCtx) return `${lastCtx.chars} / ${lastCtx.char_limit} chars · ${lastCtx.turns} turns`;
    return "send a message to measure";
  }
  function toggleCtxPop() {
    const pop = $("ctxPop");
    if (pop.hidden) { $("ctxPopDetail").textContent = ctxDetail(); pop.hidden = false; }
    else pop.hidden = true;
  }

  function findModelOption(arg) {
    if (!arg) return null;
    const a = arg.toLowerCase(), opts = Array.from($("modelSel").options);
    return opts.find(o => o.value.toLowerCase() === a)
        || opts.find(o => o.textContent.toLowerCase().includes(a)) || null;
  }
  function handleSlash(text) {
    const parts = text.slice(1).trim().split(/\s+/);
    const cmd = (parts.shift() || "").toLowerCase(), arg = parts.join(" ");
    if (cmd === "new") { newChat(); return true; }
    if (cmd === "help") {
      bubble("bot", "commands: /new (clear conversation), /model <name>, /rag on|off, /help");
      return true;
    }
    if (cmd === "rag") {
      state.use_rag = /^(on|1|true|yes)$/i.test(arg); renderControls();
      bubble("bot", `grounding ${state.use_rag ? "on" : "off"}`); return true;
    }
    if (cmd === "model") {
      const opt = findModelOption(arg);
      if (!opt) { bubble("bot", `no model matches "${arg}"`); return true; }
      state.model = opt.value; renderControls();
      bubble("bot", `model: ${opt.textContent}`); return true;
    }
    bubble("bot", `unknown command: /${cmd}. try /help`); return true;
  }

  function bubble(cls, text) {
    const hint = log.querySelector(".hint-row"); if (hint) hint.remove();
    const d = document.createElement("div");
    d.className = "msg " + cls; d.textContent = text;
    log.appendChild(d); d.scrollIntoView({ block: "end" });
    return d;
  }

  function isRemote() {
    return state.model === CLOUD_ID || state.model === TEACHER_ID
        || state.model === VERITATE_DOCS_ID || state.model.startsWith(TEACHER_PREFIX);
  }

  function activeLabel() {
    if (state.model === VERITATE_DOCS_ID) return "Veritate (platform docs)";
    const r = remoteById.get(state.model);
    if (r) return r.label;
    if (state.model === CLOUD_ID) return "Carpathian AI (public)";
    return state.model;
  }

  function renderControls() {
    const sel = $("modelSel");
    if (sel.value !== state.model) sel.value = state.model;
    $("activeModel").textContent = activeLabel();
    $("engineRow").style.display = isRemote() ? "none" : "";
    $("ragRow").style.display = (state.model === VERITATE_DOCS_ID) ? "none" : "";

    const cOk = cCapable.has(state.model);
    for (const b of $("engineSeg").children) {
      const eng = b.dataset.engine;
      b.disabled = (eng === "c" && !cOk);
      b.classList.toggle("on", eng === state.backend);
    }
    if (state.backend === "c" && !cOk) { state.backend = "pytorch"; renderControls(); return; }
    $("engineHint").textContent = cOk ? "" : "C engine needs a veritate.bin export for this model.";

    const chk = $("ragChk");
    chk.checked = state.use_rag && hasCorpus;
    chk.disabled = !hasCorpus;
    $("logChk").checked = state.use_logs;
    renderKb();
    saveSettings();
  }

  function renderKb() {
    $("kbUpload").disabled = busy;
    $("kbHint").textContent = busy ? "adding…"
      : (nFiles > 0 ? `${nFiles} file${nFiles === 1 ? "" : "s"} in your knowledge base, plus the Veritate platform docs. Add more anytime.`
         : (hasCorpus ? "Veritate platform docs are loaded. Tick grounding to ask about the platform, or add your own files."
                      : "No knowledge base yet. Upload a .txt to enable retrieval."));
  }

  function refreshHealth() {
    return fetch("/hybrid/health").then(r => r.json()).then(d => {
      hasCorpus = !!d.has_corpus; nFiles = d.n_files || 0;
    }).catch(() => {});
  }

  function upload() {
    const f = $("kbFile").files[0];
    if (!f || busy) return;
    busy = true; renderKb();
    const fd = new FormData(); fd.append("file", f);
    fetch("/hybrid/kb/upload", { method: "POST", body: fd }).then(r => r.json()).then(d => {
      busy = false;
      if (!d.ok) { renderKb(); $("kbHint").textContent = "error: " + (d.error || "failed"); return; }
      $("kbFile").value = ""; $("kbFileName").textContent = "no file selected";
      refreshHealth().finally(() => {
        renderControls();
        $("kbHint").textContent = `added ${d.filename} — ${nFiles} file${nFiles === 1 ? "" : "s"} indexed`;
      });
    }).catch(() => { busy = false; renderKb(); $("kbHint").textContent = "upload failed"; });
  }

  function addGroup(sel, label, opts) {
    if (!opts.length) return;
    const g = document.createElement("optgroup"); g.label = label;
    for (const o of opts) {
      const opt = document.createElement("option");
      opt.value = o.value; opt.textContent = o.text;
      g.appendChild(opt);
    }
    sel.appendChild(g);
  }

  function fillModels(models, remote) {
    const sel = $("modelSel");
    sel.innerHTML = "";
    remoteById = new Map(remote.map(r => [r.id, r]));

    addGroup(sel, "Veritate", [{ value: VERITATE_DOCS_ID, text: "Veritate (platform docs)" }]);

    addGroup(sel, "Local (trained)", models.map(m => {
      const params = m.n_params ? ` · ${(m.n_params / 1e6).toFixed(0)}M` : "";
      return { value: m.name, text: `${m.name} (step ${m.step}${params})` };
    }));

    const groups = [];
    for (const r of remote) {
      let g = groups.find(x => x.label === r.group);
      if (!g) { g = { label: r.group, opts: [] }; groups.push(g); }
      g.opts.push({ value: r.id, text: r.label });
    }
    for (const g of groups) addGroup(sel, g.label, g.opts);

    const names = models.map(m => m.name);
    const valid = state.model === VERITATE_DOCS_ID || names.includes(state.model) || remoteById.has(state.model);
    if (!valid) state.model = VERITATE_DOCS_ID;
    renderControls();
  }

  function init() {
    Promise.all([
      refreshHealth(),
      fetch("/pytorch-models").then(r => r.json()).catch(() => ({ models: [] })),
      fetch("/c-models").then(r => r.json()).catch(() => ({ models: [] })),
      fetch("/hybrid/models").then(r => r.json()).catch(() => ({ models: [] })),
    ]).then(([_h, pt, c, rm]) => {
      cCapable = new Set((c.models || []).map(m => m.name));
      fillModels(pt.models || [], rm.models || []);
      updateCtxRing(0);
    });
  }

  // Terminal `done` frame: lock in the canonical answer, store compacted memory,
  // update the context ring, render retrieval sources.
  function applyDone(pending, d) {
    pending.textContent = d.answer || pending.textContent || "(no answer)";
    if (d.memory) {
      convo = { summary: d.memory.summary || "",
                turns: Array.isArray(d.memory.turns) ? d.memory.turns : [] };
      saveConvo();
    }
    lastCtx = d.context || null;
    updateCtxRing(lastCtx ? lastCtx.pct : 0);
    if (d.sources && d.sources.length) {
      const s = document.createElement("div"); s.className = "src";
      s.innerHTML = "<b>retrieved</b> · ";
      s.appendChild(document.createTextNode(
        d.sources.map(x => x.score == null ? x.text : `${x.text} (${x.score})`).join(" · ")));
      pending.appendChild(s);
    }
    pending.scrollIntoView({ block: "end" });
  }

  // Streams the reply token-by-token like the Generation tab: reads the SSE body
  // of /hybrid/chat/stream and paints each `delta` as it arrives.
  async function ask() {
    const text = q.value.trim(); if (!text) return;
    if (text[0] === "/" && handleSlash(text)) { q.value = ""; q.focus(); return; }
    bubble("user", text); q.value = ""; send.disabled = true;
    const pending = bubble("bot", "");
    pending.innerHTML = '<span class="spin"></span>';
    let acc = "", started = false;
    try {
      const r = await fetch("/hybrid/chat/stream", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, model: state.model,
                               backend: state.backend, use_rag: state.use_rag, use_logs: state.use_logs,
                               history: convo.turns, summary: convo.summary }),
      });
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n\n")) >= 0) {
          const line = buf.slice(0, nl).replace(/^data: /, "");
          buf = buf.slice(nl + 2);
          if (!line) continue;
          const ev = JSON.parse(line);
          if (ev.kind === "delta") {
            if (!started) { pending.textContent = ""; started = true; }
            acc += ev.text; pending.textContent = acc;
            pending.scrollIntoView({ block: "end" });
          } else if (ev.kind === "error") {
            pending.textContent = "error: " + (ev.error || "failed");
          } else if (ev.kind === "done") {
            applyDone(pending, ev);
          }
        }
      }
      if (pending.querySelector(".spin")) pending.textContent = "(no answer)";
    } catch (_) {
      if (!started) pending.textContent = "request failed";
    } finally {
      send.disabled = false; q.focus();
    }
  }

  // Size the panel to the space actually below the chrome (topbar + tabs + an
  // on/off HUD) so the composer never drops below the fold. Guarded so the
  // ResizeObserver settles instead of looping.
  function fitWrap() {
    const w = mount.querySelector(".wrap");
    if (!w || !w.offsetParent) return;
    const target = Math.max(WRAP_MIN_HEIGHT,
                            window.innerHeight - w.getBoundingClientRect().top - WRAP_FIT_GAP);
    if (Math.abs(parseFloat(w.style.height || "0") - target) < 1) return;
    w.style.height = target + "px";
  }

  let wired = false;
  // Called by activateTab("chat") on every open. Refits the panel each time;
  // wires listeners + kicks off the network init once, refocuses thereafter.
  window.chatTabInit = function () {
    fitWrap();
    if (wired) { q.focus(); return; }
    wired = true;

    window.addEventListener("resize", fitWrap);
    if (window.ResizeObserver) new ResizeObserver(fitWrap).observe(document.body);

    $("newChat").addEventListener("click", newChat);
    $("gear").addEventListener("click", () => $("settings").classList.toggle("open"));
    $("settingsClose").addEventListener("click", () => $("settings").classList.remove("open"));
    $("ctxRing").addEventListener("click", e => { e.stopPropagation(); toggleCtxPop(); });
    document.addEventListener("click", e => {
      const pop = $("ctxPop");
      if (pop && !pop.hidden && !e.target.closest(".ctxwrap")) pop.hidden = true;
    });
    $("modelSel").addEventListener("change", e => { state.model = e.target.value; renderControls(); });
    $("engineSeg").addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b || b.disabled) return;
      state.backend = b.dataset.engine; renderControls();
    });
    $("ragChk").addEventListener("change", e => { state.use_rag = e.target.checked; saveSettings(); });
    $("logChk").addEventListener("change", e => { state.use_logs = e.target.checked; saveSettings(); });
    $("kbFile").addEventListener("change", e => {
      const f = e.target.files[0]; $("kbFileName").textContent = f ? f.name : "no file selected";
    });
    $("kbUpload").addEventListener("click", upload);
    send.addEventListener("click", ask);
    q.addEventListener("keydown", e => { if (e.key === "Enter") ask(); });

    init(); q.focus();
  };
})();
