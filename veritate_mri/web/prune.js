/* Developed by Carpathian, LLC. Distribution Not Authorized. */
/* veritate_mri/web/prune.js */

// Self-contained neuron-pruning panel. Mounts into the Models tab, lists vanilla
// checkpoints, runs /pruning/report (dead-neuron analysis + size estimate), and
// /pruning/generate_plugin (writes a width-pruned trainer). No index.js edits.

(function () {
  "use strict";

  const MODELS_URL = "/pytorch-models";
  const REPORT_URL = "/pruning/report";
  const GEN_URL    = "/pruning/generate_plugin";

  let plan = null;
  let cur  = { model: "", step: 0 };

  function el(tag, attrs, html) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }

  function msg(text, kind) {
    const m = document.getElementById("pruneMsg");
    if (!m) return;
    m.textContent = text || "";
    m.className = "prune-msg" + (kind ? " " + kind : "");
  }

  function build_panel() {
    const panel = el("div", { class: "panel", id: "prunePanel" });
    panel.innerHTML =
      '<h2>neuron pruning <em>drop dead FFN units, keep the live ones</em></h2>' +
      '<div class="body">' +
      '<div class="prune-controls">' +
      '<label>model <select id="pruneModel"><option value="">loading…</option></select></label>' +
      '<label>step <input class="prune-step" id="pruneStep" type="number" min="0" value="0"></label>' +
      '<button id="pruneAnalyze" type="button">analyze</button>' +
      '<button id="pruneGenerate" type="button" disabled>generate pruned model</button>' +
      '</div>' +
      '<div class="prune-summary" id="pruneSummary"></div>' +
      '<div id="pruneTableWrap"></div>' +
      '<div class="prune-msg" id="pruneMsg"></div>' +
      '</div>' +
      '<p class="desc">Width-pruning scores each FFN unit by post-activation magnitude over sampled ' +
      'windows, keeps the most active fraction per layer, and writes a smaller trainer you can run or ' +
      'export. Works on vanilla byte-level decoders (not Mixture-of-Experts). Run analyze first, then ' +
      'generate.</p>';
    return panel;
  }

  function mount() {
    const learning = document.querySelector('.tab-body[data-tab="learning"]');
    if (!learning || document.getElementById("prunePanel")) return;
    learning.insertBefore(build_panel(), learning.firstChild);
    document.getElementById("pruneModel").addEventListener("change", on_model_change);
    document.getElementById("pruneAnalyze").addEventListener("click", analyze);
    document.getElementById("pruneGenerate").addEventListener("click", generate);
    load_models();
  }

  function load_models() {
    fetch(MODELS_URL).then(r => r.json()).then(d => {
      const sel = document.getElementById("pruneModel");
      sel.innerHTML = "";
      const rows = (d.models || []);
      if (!rows.length) { sel.appendChild(el("option", { value: "" }, "no models yet")); return; }
      for (const m of rows) {
        const label = m.name + "  (step " + m.step + (m.plugin ? ", " + m.plugin : "") + ")";
        const o = el("option", { value: m.name }, label);
        o.dataset.step = m.step;
        sel.appendChild(o);
      }
      on_model_change();
    }).catch(() => msg("could not load model list", "err"));
  }

  function on_model_change() {
    const sel = document.getElementById("pruneModel");
    const opt = sel.options[sel.selectedIndex];
    if (opt && opt.dataset.step) document.getElementById("pruneStep").value = opt.dataset.step;
    reset_results();
  }

  function reset_results() {
    plan = null;
    document.getElementById("pruneSummary").innerHTML = "";
    document.getElementById("pruneTableWrap").innerHTML = "";
    document.getElementById("pruneGenerate").disabled = true;
    msg("");
  }

  function analyze() {
    const model = document.getElementById("pruneModel").value;
    const step  = parseInt(document.getElementById("pruneStep").value || "0", 10);
    if (!model) { msg("pick a model", "err"); return; }
    reset_results();
    msg("analyzing activations…");
    const url = REPORT_URL + "?model=" + encodeURIComponent(model) + "&step=" + step;
    fetch(url).then(r => r.json()).then(d => {
      if (!d.ok) { msg(d.error || "analyze failed", "err"); return; }
      cur = { model: d.model, step: d.step };
      plan = d.plan;
      render_report(d);
      document.getElementById("pruneGenerate").disabled = false;
      msg("analysis ready. review the plan, then generate.", "ok");
    }).catch(() => msg("analyze request failed", "err"));
  }

  function render_report(d) {
    document.getElementById("pruneSummary").innerHTML =
      "<span>dead neurons: <b>" + d.dead_pct + "%</b></span>" +
      "<span>size: <b>" + d.size_mb_before + " MB</b> -> <b>" + d.size_mb_after + " MB</b></span>" +
      "<span>params: <b>" + d.n_params.toLocaleString() + "</b> -> <b>" +
        d.n_params_after.toLocaleString() + "</b></span>" +
      "<span>corpus: <b>" + d.corpus + "</b>  samples: <b>" + d.samples + "</b></span>";
    const rows = d.per_layer.map(function (e) {
      const keepPct = Math.round(e.keep * 100);
      return "<tr><td>" + e.layer + "</td><td>" + e.alive + " / " + e.total + "</td><td>" +
        (e.alive_frac * 100).toFixed(1) + "%</td><td>" + keepPct + "%</td>" +
        '<td><span class="prune-bar" style="width:' + Math.max(2, keepPct) + 'px"></span></td></tr>';
    }).join("");
    document.getElementById("pruneTableWrap").innerHTML =
      '<table class="prune-table"><thead><tr><th>layer</th><th>alive / total</th>' +
      '<th>alive %</th><th>keep %</th><th>keep</th></tr></thead><tbody>' + rows + "</tbody></table>";
  }

  function generate() {
    if (!plan) { msg("run analyze first", "err"); return; }
    document.getElementById("pruneGenerate").disabled = true;
    msg("generating pruned trainer…");
    fetch(GEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: cur.model, step: cur.step, plan: plan }),
    }).then(r => r.json()).then(d => {
      if (!d.ok) { msg(d.error || "generate failed", "err"); document.getElementById("pruneGenerate").disabled = false; return; }
      msg("created pruned trainer: " + d.plugin_id + " (open the Training tab to run it).", "ok");
    }).catch(() => { msg("generate request failed", "err"); document.getElementById("pruneGenerate").disabled = false; });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();
