/* Developed by Carpathian, LLC. Distribution Not Authorized. */
/* veritate_mri/web/generation_images.js */

// The Generation tab's image layout. The tab has one selector, what to generate: text
// (chat / agent / autocomplete and the byte-level MRI, all in index.js) or images, which
// swaps the whole tab for this module: a composer that draws with an image model through
// POST /images/generate, the MRI of that one picture (its passes, the order its cells were
// decided, how sure each cell was; from the route's trace), and the checkpoint MRI of the
// model it drew with (ImageMri). Markup lives in index.html under #imgGenView; the kind
// switch puts `is-images` on the tab body and generation_images.css hides the text layout.
// The picked kind persists per browser. Self-contained: no index.js symbols.

(function () {
  "use strict";

  const KIND_KEY = "vt:generation:kind";
  const SOURCE_MODES = ["variation", "inpaint", "expand"];
  const state = { models: [], busy: false, mri: null };

  const el = id => document.getElementById(id);
  const esc = s => String(s).replace(/[<>&]/g, c => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]));
  const num = (id, dflt) => { const v = parseFloat(el(id).value); return isFinite(v) ? v : dflt; };
  const pct = v => typeof v === "number" && isFinite(v) ? (v * 100).toFixed(1) + "%" : "—";

  function say(msg, color) {
    const status = el("imgGenStatus");
    status.textContent = msg;
    status.style.color = color || "var(--dim)";
  }

  // ---- text | images -----------------------------------------------------------------
  function storedKind() {
    try { return localStorage.getItem(KIND_KEY) === "images" ? "images" : "text"; } catch (_e) { return "text"; }
  }

  function setKind(kind) {
    try { localStorage.setItem(KIND_KEY, kind); } catch (_e) { /* per-browser convenience only */ }
    document.querySelector('.tab-body[data-tab="generation"]').classList.toggle("is-images", kind === "images");
    const radio = document.querySelector(`input[name="genKind"][value="${kind}"]`);
    if (radio) radio.checked = true;
    if (kind === "images") { loadModels(); return; }
    el("genKindHint").textContent = "";
    if (state.mri) state.mri.hide();
  }

  // ---- the models that can draw ------------------------------------------------------
  function loadModels() {
    return fetch("/images/models").then(r => r.json()).then(d => {
      state.models = (d && d.models) || [];
      const sel = el("imgGenModel");
      const cur = sel.value;
      sel.innerHTML = state.models.length
        ? state.models.map(m => `<option value="${esc(m.name)}">${esc(m.name)} (${m.height}x${m.width})</option>`).join("")
        : '<option value="">- no image models yet -</option>';
      if (cur && state.models.some(m => m.name === cur)) sel.value = cur;
      const n = state.models.length;
      el("imgGenCount").textContent = n ? `${n} image model${n === 1 ? "" : "s"}` : "";
      el("genKindHint").textContent = n ? "" : "no image models yet: train one in the Training tab (Train an image model)";
      fillSteps();
      followModel();
    }).catch(() => {});
  }

  function fillSteps() {
    const m = state.models.find(x => x.name === el("imgGenModel").value);
    const steps = m ? m.steps.slice().sort((a, b) => b - a) : [];
    el("imgGenStep").innerHTML = steps.map((s, i) => `<option value="${s}">${s}${i === 0 ? " (latest)" : ""}</option>`).join("");
  }

  // The checkpoint MRI follows the picked model: what the model drawing knows.
  function followModel() {
    if (!state.mri) state.mri = ImageMri.create("imgGenMri");
    const name = el("imgGenModel").value;
    if (name) state.mri.show(name);
    else state.mri.hide();
  }

  function modeChanged() {
    const mode = el("imgGenMode").value;
    el("imgGenSourceRow").classList.toggle("on", SOURCE_MODES.includes(mode));
    el("imgGenStrengthWrap").classList.toggle("off", mode !== "variation");
    el("imgGenExpandWrap").classList.toggle("off", mode !== "expand");
    el("imgGenRectWrap").classList.toggle("off", mode !== "inpaint");
    el("imgGenPrompt").disabled = mode === "unconditional";
  }

  function readSource() {
    const file = el("imgGenSource").files[0];
    if (!file) return Promise.resolve(null);
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error("could not read the photo"));
      reader.readAsDataURL(file);
    });
  }

  // ---- draw one picture --------------------------------------------------------------
  function run() {
    if (state.busy) return;
    const model = el("imgGenModel").value;
    if (!model) { say("train an image model first (Training tab, Train an image model)", "var(--hot)"); return; }
    const mode = el("imgGenMode").value;
    readSource().then(source => {
      if (SOURCE_MODES.includes(mode) && !source) { say("pick a source photo for this mode", "var(--hot)"); return; }
      const body = {
        model, mode, trace: true,
        step: parseInt(el("imgGenStep").value, 10) || undefined,
        caption: el("imgGenPrompt").value || "",
        image: source || undefined,
        strength: num("imgGenStrength", 0.6),
        expand: num("imgGenExpand", 0.6),
        rect: [num("imgGenX0", 0.25), num("imgGenY0", 0.25), num("imgGenX1", 0.75), num("imgGenY1", 0.75)],
        passes: Math.round(num("imgGenPasses", 8)),
        temperature: num("imgGenTemp", 1.0),
        seed: Math.round(num("imgGenSeed", 0)),
      };
      state.busy = true;
      el("imgGenRun").disabled = true;
      say(`generating (${body.passes} passes) ...`, "var(--warm)");
      return fetch("/images/generate", { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify(body) })
        .then(r => r.json())
        .then(d => {
          if (!d.ok) { say(d.error || "generation failed", "var(--hot)"); return; }
          showResult(d, body);
          say(`${d.height}x${d.width} in ${d.seconds}s on ${d.device}`, "var(--data-pos)");
        });
    }).catch(e => say("failed: " + (e && e.message || e), "var(--hot)"))
      .finally(() => { state.busy = false; el("imgGenRun").disabled = false; });
  }

  // The picture, its numbers, and the MRI of how it formed.
  function showResult(d, body) {
    const src = "data:image/png;base64," + d.png;
    el("imgGenOut").src = src;
    const save = el("imgGenSave");
    save.href = src;
    save.download = `${d.model}_step${d.step}_${d.mode}_seed${body.seed}.png`;
    el("imgGenOutWrap").classList.add("on");
    const tr = d.trace;
    const meanConf = tr.confidence_map.reduce((a, b) => a + b, 0) / Math.max(1, tr.confidence_map.length);
    const kpi = (v, k, title) => `<div class="imri-kpi" title="${esc(title)}"><div class="v">${v}</div><div class="k">${k}</div></div>`;
    el("imgGenKpis").innerHTML =
      kpi(`${d.regenerated}<span class="imgl-of"> / ${d.code_bytes}</span>`, "cells drawn", "code bytes the model decided; the rest were kept from the source photo") +
      kpi(tr.passes.length, "passes", "forward passes it took; each commits the cells it is surest of") +
      kpi(pct(meanConf), "sure", "the model's own probability on its picks, averaged over the picture") +
      kpi(`${tr.codes_used}<span class="imgl-of">/255</span>`, "codes used", "distinct codes in the picture; a handful is a collapse");
    renderTrace(tr);
  }

  function renderTrace(tr) {
    const C = ImageMri.charts, colors = ImageMri.colors;
    const [gh, gw] = tr.grid;
    const n = tr.passes.length;
    const labels = tr.passes.map(p => "pass " + p.pass);
    el("imgGenTrace").innerHTML = `<div class="imri">
      <div class="imri-section">how this picture formed <span>one forward per pass; grey cells were still undecided</span></div>
      <div class="panel"><div class="body">
        <div class="imgg-passes">${tr.passes.map(p => `<figure><img src="data:image/png;base64,${p.png}" alt="pass ${p.pass}"><figcaption>pass ${p.pass}: ${p.committed} cells, ${pct(p.confidence)} sure</figcaption></figure>`).join("")}</div>
      </div></div>
      <div class="imri-grid">
        <div class="panel"><h2>the order it formed <em>which pass decided each cell</em></h2>
          <div class="body"><div class="imri-heatwrap">${C.cellHeat(C.gridRows(tr.commit_pass_map, gh, gw), { lo: 1, hi: n, fmt: v => "pass " + v })}
            <div class="imri-side">${C.scaleBar("pass 1", `pass ${n}`)}<span class="meta">blue cells were decided first, orange last. A model that has learned structure commits the layout early and the detail late.</span></div></div></div></div>
        <div class="panel"><h2>how sure it was <em>per cell, when it committed</em></h2>
          <div class="body"><div class="imri-heatwrap">${C.cellHeat(C.gridRows(tr.confidence_map, gh, gw), { lo: 0, hi: 1, fmt: v => pct(v) + " sure" })}
            <div class="imri-side">${C.scaleBar("unsure", "sure")}<span class="meta">the probability the model gave its own pick, averaged over the cell's planes. Kept cells from a source photo count as certain.</span></div></div></div></div>
        <div class="panel"><h2>cells per pass <em>and how sure each pass was</em></h2>
          <div class="body"><div class="imri-chart small">${C.barChart(tr.passes.map(p => p.committed), labels, colors[2])}</div>
            <div class="imri-chart small">${C.barChart(tr.passes.map(p => p.confidence || 0), labels, colors[4])}</div>
            <div class="imri-legend"><span><i style="background:${colors[2]}"></i>cells committed</span><span><i style="background:${colors[4]}"></i>mean confidence of those cells</span><span>the schedule is cosine: few cells first, most in the last passes</span></div></div></div>
      </div>
    </div>`;
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!el("imgGenView")) return;
    document.querySelectorAll('input[name="genKind"]').forEach(r => r.addEventListener("change", () => setKind(r.value)));
    el("imgGenModel").addEventListener("change", () => { fillSteps(); followModel(); });
    el("imgGenMode").addEventListener("change", modeChanged);
    el("imgGenRun").addEventListener("click", run);
    modeChanged();
    setKind(storedKind());
  });
})();
