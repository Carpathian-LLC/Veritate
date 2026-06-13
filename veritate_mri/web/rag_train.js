/* Developed by Carpathian, LLC. Distribution Not Authorized. */
/* veritate_mri/web/rag_train.js */

// Logic for the "Answer from context (RAG)" action of the Training tab's train
// flow. Lists pytorch checkpoints as base-model choices and runs rag SFT
// (/rag/train), which auto-builds the corpus from the configured teacher when
// missing, then polls /rag/status for job state + a log tail. The markup is
// static in index.html (#ragTrainPanel); index.js toggles its visibility per flow.

(function () {
  "use strict";

  const MODELS_URL = "/pytorch-models";
  const TRAIN_URL  = "/rag/train";
  const STATUS_URL = "/rag/status";
  const STOP_URL   = "/rag/stop";

  const POLL_MS = 1500;
  const DEFAULT_STEPS = 1500;
  const DEFAULT_FACTS = 200;

  let pollTimer = null;

  function el(tag, attrs, html) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }

  function msg(text, kind) {
    const m = document.getElementById("ragMsg");
    if (!m) return;
    m.textContent = text || "";
    m.className = "rag-msg" + (kind ? " " + kind : "");
  }

  function wire() {
    const trainBtn = document.getElementById("ragTrain");
    if (!trainBtn || trainBtn.dataset.wired) return;
    trainBtn.dataset.wired = "1";
    trainBtn.addEventListener("click", train);
    const stop = document.getElementById("ragStop");
    if (stop) stop.addEventListener("click", stop_job);
    load_models();
    poll();
  }

  function stop_job() {
    const go = (ok) => {
      if (!ok) return;
      fetch(STOP_URL, { method: "POST" }).then(r => r.json()).then(() => poll()).catch(() => {});
    };
    if (window.confirmDialog) window.confirmDialog("Stop the rag job? The latest checkpoint is kept.", "Stop").then(go);
    else go(window.confirm("Stop the rag job?"));
  }

  function load_models() {
    fetch(MODELS_URL).then(r => r.json()).then(d => {
      const sel = document.getElementById("ragModel");
      sel.innerHTML = "";
      const rows = (d.models || []);
      if (!rows.length) { sel.appendChild(el("option", { value: "" }, "no models yet")); return; }
      for (const m of rows) {
        const label = m.name + "  (step " + m.step + (m.plugin ? ", " + m.plugin : "") + ")";
        sel.appendChild(el("option", { value: m.name }, label));
      }
    }).catch(() => msg("could not load model list", "err"));
  }

  function set_buttons(disabled) {
    document.getElementById("ragTrain").disabled = disabled;
    const stop = document.getElementById("ragStop");
    if (stop) stop.style.display = disabled ? "" : "none";
  }

  function train() {
    const sel = document.getElementById("ragModel");
    let source = sel.value;
    if (!source) {
      for (const o of sel.options) { if (o.value) { source = o.value; sel.value = o.value; break; } }
    }
    if (!source) { msg("no base model available; train or import a checkpoint first", "err"); return; }
    const nameEl = document.getElementById("ragName");
    let name = (nameEl.value || "").trim();
    if (!name) { name = source + "_rag"; nameEl.value = name; }
    let steps = parseInt(document.getElementById("ragSteps").value || "0", 10);
    if (!steps || steps < 1) steps = DEFAULT_STEPS;
    const facts = parseInt(document.getElementById("ragFacts").value || "0", 10) || DEFAULT_FACTS;
    set_buttons(true);
    msg("starting rag training…");
    fetch(TRAIN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: source, name: name, steps: steps, n_facts: facts }),
    }).then(r => r.json()).then(d => {
      if (!d.ok) { msg(d.error || "train failed", "err"); set_buttons(false); return; }
      msg(d.auto_built ? "building corpus, then training…" : "training started.", "ok");
      poll();
    }).catch(() => { msg("train request failed", "err"); set_buttons(false); });
  }

  function render_status(d) {
    const job = document.getElementById("ragJob");
    const log = document.getElementById("ragLog");
    const phase = d.phase ? d.phase : "none";
    job.textContent = "job: " + phase + "  status: " + (d.status || "idle") +
      (d.exit_code != null ? "  exit: " + d.exit_code : "");
    log.textContent = d.log || "";
    log.scrollTop = log.scrollHeight;
  }

  function poll() {
    if (pollTimer) clearTimeout(pollTimer);
    fetch(STATUS_URL).then(r => r.json()).then(d => {
      if (!d.ok) return;
      render_status(d);
      if (d.running) {
        set_buttons(true);
        pollTimer = setTimeout(poll, POLL_MS);
      } else {
        set_buttons(false);
      }
    }).catch(() => {});
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})();
