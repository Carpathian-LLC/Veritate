/* Developed by Carpathian, LLC. Distribution Not Authorized. */
/* veritate_mri/web/image_live.js */

// The Training tab's live view of an image run. The byte-model panels below it (register
// fluency, comprehension, concepts, lens drift) mean nothing for a model whose output is
// pixels, so the host takes their place while a picture model trains. Reads
// /images/live/<name>: which stage the trainer is in and how far, on which device (written
// from the first second, long before train.csv has a row), the checkpoints on disk, the
// latest probe, and the run log tail. Figures come from ImageMri.charts.
// The caller decides WHICH run is live and passes it to setActive; this module owns the
// polling, the parsing and the rendering. Self-contained: no index.js symbols.

(function () {
  "use strict";

  const TRAINER_ID = "native/image_trainer";
  const POLL_RUNNING_MS = 2500;
  const POLL_IDLE_MS = 10000;
  const STAGES = [["decode", "decode pictures"], ["codec", "fit codec"], ["encode", "encode corpus"], ["train", "train"]];

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function fmtDur(s) {
    if (s == null || !isFinite(s)) return "";
    s = Math.max(0, Math.round(s));
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m " + (s % 60) + "s";
    return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
  }

  function fmtRate(r) { return r >= 10 ? Math.round(r).toLocaleString() : r.toFixed(1); }

  function parseCsv(text) {
    const out = { train: [], val: [] };
    if (!text) return out;
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 2) return out;
    const ix = Object.fromEntries(lines[0].split(",").map((h, i) => [h.trim(), i]));
    const rows = [];
    for (const ln of lines.slice(1)) {
      const c = ln.split(",");
      const step = parseInt(c[ix.step], 10), loss = parseFloat(c[ix.loss]);
      if (!isFinite(step) || !isFinite(loss)) continue;
      rows.push({
        step, loss,
        lr: parseFloat(c[ix.lr]), gn: parseFloat(c[ix.grad_norm]), tps: parseFloat(c[ix.tok_per_s]),
        val: (c[ix.split] || "").trim().endsWith("val"),
      });
    }
    // An older run's rows sit ahead of this one's in the same file. The file is written in
    // order and a run's steps never go backwards, so the last place the step DROPS is where
    // the current run begins. Truncating the ordered stream rather than the train list alone
    // is what keeps a previous run's val rows (whose step numbers are higher) out of the
    // chart, and keeps this run's starting-weights val row (logged at the resume step, before
    // its first train row) in it.
    let start = 0;
    for (let i = 1; i < rows.length; i++) if (rows[i].step < rows[i - 1].step) start = i;
    for (const r of rows.slice(start)) {
      const point = { step: r.step, loss: r.loss, lr: r.lr, gn: r.gn, tps: r.tps };
      if (r.val) out.val.push(point); else out.train.push(point);
    }
    return out;
  }

  function stageBars(progress, state) {
    return STAGES.map(([key, label]) => {
      const s = (progress && progress.stages && progress.stages[key]) || { state: "pending" };
      let detail = "&mdash;", fill = 0, cls = s.state;
      if (s.state === "done") { detail = `&#10003; ${fmtDur(s.seconds)}`; fill = 100; }
      else if (s.state === "skipped") { detail = "reused"; fill = 100; }
      else if (s.state === "running") {
        fill = s.total ? 100 * s.done / s.total : 0;
        detail = `${Number(s.done || 0).toLocaleString()} / ${Number(s.total || 0).toLocaleString()}`
          + (s.rate ? ` &middot; ${fmtRate(s.rate)}/s` : "")
          + (s.eta_s != null && state === "running" ? ` &middot; ${fmtDur(s.eta_s)} left` : "");
        if (state !== "running") cls = state === "failed" ? "failed" : "pending";
      }
      return `<div class="imgl-stage ${cls}"><div class="imgl-stage-h"><b>${label}</b><span>${detail}</span></div><div class="imgl-bar"><i style="width:${fill.toFixed(1)}%"></i></div></div>`;
    }).join("");
  }

  // how the run computes: the precision it measured fastest, the attention path, compiled or not
  function speedChips(notes) {
    return [
      notes.precision ? `<span class="imgl-chip imgl-note" title="autocast precision this run computes in (auto picks the half precision the GPU measures fastest; the run log has the rates)${notes.ns_dtype ? `; Muon orthogonalizes in ${esc(notes.ns_dtype)}` : ""}">${esc(notes.precision)}</span>` : "",
      notes.attention && notes.attention !== "sdpa" ? `<span class="imgl-chip imgl-note" title="attention written out in half precision instead of sdpa's fp32 fallback: 1.8x faster and half the memory on Apple GPUs">explicit attention</span>` : "",
      notes.compiled ? `<span class="imgl-chip imgl-note" title="the training forward runs through torch.compile (measured 1.54x on an M2)">compiled</span>` : "",
    ].join("");
  }

  function probeFigures(name, data, probe, ckEvery, probeEvery) {
    const firstAt = probeEvery > 0 ? Math.min(probeEvery, ckEvery || probeEvery) : ckEvery;
    if (!probe || probe.step == null) {
      return `<div class="meta">the first pictures appear at step ${firstAt ? firstAt.toLocaleString() : "?"}${probeEvery ? ` and every ${probeEvery.toLocaleString()} steps after (the <i>pictures every</i> setting)` : " (the first checkpoint)"}.</div>`;
    }
    const charts = ImageMri.charts;
    const u = f => `/images/mri/${encodeURIComponent(name)}/${probe.step}/${f}?${probe.step}`;
    const has = f => (probe.files || []).includes(f);
    const nPlanes = probe.planes || (data.geometry && data.geometry.planes) || 1;
    const passes = (probe.pass_committed || []).length;
    const nSamp = 8 + ((probe.caption_samples || []).length);
    const [gh, gw] = probe.grid || [0, 0];
    const formation = Array.isArray(probe.commit_pass_map) && gh && gw
      ? charts.cellHeat(charts.gridRows(probe.commit_pass_map, gh, gw), { lo: 1, hi: probe.formation_passes || passes || 8, cell: 10, fmt: v => "pass " + v })
      : (has("formation.png") ? charts.photoTile(u("formation.png"), probe, 2, "formation order") : "");
    charts.wireLightbox();
    return `<div class="imgl-figs">`
      + (has("samples.png") ? `<h4>drawn from nothing &middot; same seeds every probe &middot; click a picture to enlarge</h4>${charts.photoTile(u("samples.png"), probe, nSamp, "samples")}` : "")
      + (has("passes.png") ? `<h4>how one picture forms &middot; ${passes ? passes + " passes, " : ""}grey cells still undecided</h4>${charts.photoTile(u("passes.png"), probe, passes || 8, "decode passes")}` : "")
      + ((formation || has("planes.png")) ? `<div class="imgl-row">`
        + (formation ? `<div><h4>which pass decided each cell</h4>${formation}${charts.scaleBar("pass 1", "last pass")}</div>` : "")
        + (has("planes.png") ? `<div><h4>coarse to fine &middot; 1, 2, &hellip; ${nPlanes} planes</h4>${charts.photoTile(u("planes.png"), probe, nPlanes, "coarse to fine")}</div>` : "")
        + `</div>` : "")
      + (has("fill.png") ? `<h4>held-out pictures: original &middot; half hidden &middot; the model's fill</h4>${charts.photoTile(u("fill.png"), probe, 3, "fill test")}` : "")
      + `<div class="meta">probe at step ${Number(probe.step).toLocaleString()}${probe.fill_accuracy != null ? ` &middot; fill accuracy ${(100 * probe.fill_accuracy).toFixed(1)}%` : ""}${probe.codes_used != null ? ` &middot; ${probe.codes_used} / 255 codes in use` : ""}${probe.colour_match != null ? ` &middot; colour match ${(100 * probe.colour_match).toFixed(0)}%` : ""}${probe.detail_ratio != null ? ` &middot; detail ${(100 * Math.min(1, probe.detail_ratio)).toFixed(0)}% of the codec ceiling` : ""} &middot; <a href="#" data-imgl-open>full brain view &rsaquo;</a></div></div>`;
  }

  // ------------------------------------------------------------------------------------
  // The view

  function create(hostId, deps) {
    const d = deps || {};
    const state = { active: false, name: null, data: null, csv: null, busy: false, lastFetch: 0 };
    const host = () => document.getElementById(hostId);

    // Mirrors save.compose_name(name, size) so the view can find the run's model dir from
    // the runner's args before config.json exists.
    function runName(args) {
      if (!args) return null;
      if (args.resume) return String(args.resume);
      if (!args.name || !args.size) return null;
      const slug = d.slugify(String(args.name)), size = String(args.size);
      if (!slug) return null;
      return (slug === size || slug.endsWith("_" + size) || slug.endsWith(size)) ? slug : slug + "_" + size;
    }

    function setActive(on, name) {
      const box = host();
      if (!box) return;
      if (on && name !== state.name) {
        state.name = name; state.data = null; state.csv = null; state.lastFetch = 0;
        render();
      }
      if (on === state.active) return;
      state.active = on;
      box.style.display = on ? "block" : "none";
      if (d.onActive) d.onActive(on);
      if (!on) { state.name = null; state.data = null; state.csv = null; }
    }

    function load() {
      const name = state.name;
      if (!name || state.busy) return;
      const now = Date.now();
      const hot = !state.data || state.data.running;
      if (now - state.lastFetch < (hot ? POLL_RUNNING_MS : POLL_IDLE_MS)) return;
      state.busy = true; state.lastFetch = now;
      Promise.all([
        fetch(`/images/live/${encodeURIComponent(name)}?` + now, { cache: "no-store" }).then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`/run/${encodeURIComponent(name)}/csv?` + now, { cache: "no-store" }).then(r => r.ok ? r.text() : "").catch(() => ""),
      ]).then(([live, csv]) => {
        state.busy = false;
        if (!state.active || state.name !== name) return;
        if (live && live.ok) state.data = live;
        state.csv = parseCsv(csv);
        render();
      }).catch(() => { state.busy = false; });
    }

    function kpiCards(data, progress, runState) {
      const tr = (progress && progress.stages && progress.stages.train) || {};
      if (!tr.state || tr.state === "pending") return "";
      const notes = (progress && progress.notes) || {};
      const csv = state.csv || { train: [], val: [] };
      const last = csv.train.length ? csv.train[csv.train.length - 1] : null;
      const lastVal = csv.val.length ? csv.val[csv.val.length - 1] : null;
      const seq = notes.seq || (data.geometry && data.geometry.seq) || 0;
      const kpi = (v, k, title) => `<div class="imgl-kpi" title="${esc(title || "")}"><div class="v">${v}</div><div class="k">${k}</div></div>`;
      let out = kpi(`${Number(tr.done || 0).toLocaleString()}<span class="imgl-of"> / ${Number(tr.total || 0).toLocaleString()}</span>`, "step", "training steps done");
      if (last) out += kpi(last.loss.toFixed(3), "loss", "masked-fill loss on the training batch. Lower is better; it should fall steadily.");
      if (lastVal) out += kpi(lastVal.loss.toFixed(3), `val loss<span class="imgl-of"> @ ${lastVal.step.toLocaleString()}</span>`, "the same loss on held-out pictures");
      if (tr.step_s) out += kpi(tr.step_s >= 10 ? tr.step_s.toFixed(0) + "s" : tr.step_s.toFixed(2) + "s", "per step", "wall-clock seconds per optimizer step over the last log window (one step = one batch of pictures)");
      if (last && seq) out += kpi(fmtRate(last.tps / seq), "pictures / s", "training throughput");
      if (last && isFinite(last.lr)) out += kpi(last.lr.toExponential(1), "learning rate", "");
      if (tr.eta_s != null && runState === "running") out += kpi(fmtDur(tr.eta_s), "time left", "from the current step rate");
      if (notes.fill_accuracy != null) out += kpi((100 * notes.fill_accuracy).toFixed(1) + "%", `fill accuracy<span class="imgl-of"> @ ${Number(notes.probe_step).toLocaleString()}</span>`, "of hidden cells the model fills in exactly, on held-out pictures (the checkpoint probe)");
      if (notes.grad_accum > 1) out += kpi(`${notes.micro_batch}<span class="imgl-of"> &times; ${notes.grad_accum}</span>`, "pictures per forward", "the whole batch did not fit in memory in one forward, so each step is several smaller forwards with the gradients added up. Same batch, same result, slower.");
      if (notes.params) out += kpi(d.fmtParams(notes.params), "parameters", "");
      if (notes.records) out += kpi(Number(notes.records).toLocaleString(), "training pictures", "records in the corpus");
      return out;
    }

    function render() {
      const box = host();
      if (!box) return;
      const name = state.name || "";
      const data = state.data;
      if (!data) {
        box.innerHTML = `<div class="imgl"><div class="panel"><h2>image run <em>${esc(name)}</em></h2><div class="body"><div class="meta">waiting for the run to report&hellip;</div></div></div></div>`;
        return;
      }
      const p = data.progress || null;
      const runState = p ? p.state : (data.running ? "running" : "idle");
      const stateColor = { running: "var(--warm)", done: "var(--data-pos)", stopped: "var(--dim)", failed: "var(--hot)" }[runState] || "var(--dim)";
      const device = p && p.device;
      const devChip = !device ? "" : device === "cpu"
        ? `<span class="imgl-chip imgl-cpu" title="no GPU was available to this run">CPU</span>`
        : `<span class="imgl-chip imgl-gpu" title="codec fit, corpus encode and training run on the GPU. Decoding JPEGs is CPU work by nature; it is the first stage only.">GPU &middot; ${esc(device)}</span>`;
      const elapsed = p ? fmtDur((p.ended || Date.now() / 1000) - p.started) : "";
      const tr = (p && p.stages && p.stages.train) || {};
      const training = !!(tr.state && tr.state !== "pending");
      const kpis = kpiCards(data, p, runState);

      const ck = data.checkpoint_steps || [];
      const runArgs = d.runArgs() || {};
      const ckEvery = parseInt(runArgs.ckpt_every, 10) || 0;
      const nextIn = ckEvery && training ? ckEvery - ((tr.done || 0) % ckEvery) : 0;
      const ago = data.last_checkpoint_at ? fmtDur(Date.now() / 1000 - data.last_checkpoint_at) + " ago" : "";
      const saved = ck.length
        ? `<b style="color:var(--data-pos)">model saved</b> &middot; ${ck.length} checkpoint${ck.length === 1 ? "" : "s"} in models/${esc(name)}/ &middot; last at step ${Number(ck[ck.length - 1]).toLocaleString()}${ago ? ` (${ago})` : ""}${runState === "running" && nextIn ? ` &middot; next save in ${nextIn.toLocaleString()} steps` : ""}`
        : training ? `<b style="color:var(--warm)">not saved yet</b>${ckEvery ? ` &middot; first checkpoint at step ${ckEvery.toLocaleString()}` : ""}` : `<span>nothing to save yet &middot; the model is created after the corpus is encoded</span>`;

      let panels = "";
      if (training) {
        const csv = state.csv || { train: [], val: [] };
        const svg = ImageMri.charts.svgLine([
          { name: "train", color: ImageMri.colors[1], points: csv.train.map(r => [r.step, r.loss]) },
          { name: "val", color: ImageMri.colors[3], points: csv.val.map(r => [r.step, r.loss]) },
        ]);
        const lp = data.latest_probe;
        const probe = probeFigures(name, data, lp, ckEvery, parseInt(runArgs.probe_every, 10) || 0);
        panels = `<div class="grid r2">
      <div class="panel"><h2>loss <em>train and held-out, by step</em></h2><div class="body"><div class="imgl-chart">${svg}</div><div class="imgl-legend"><span><i style="background:${ImageMri.colors[1]}"></i>train</span><span><i style="background:${ImageMri.colors[3]}"></i>held-out</span></div></div></div>
      <div class="panel"><h2>what it draws now <em>${lp && lp.step != null ? "checkpoint " + Number(lp.step).toLocaleString() : "waiting for a checkpoint"}</em></h2><div class="body">${probe}</div></div>
    </div>`;
      }
      const tail = (data.log_tail || []);
      const log = tail.length ? `<div class="panel"><h2>run log <em>last lines</em></h2><div class="body"><pre class="imgl-log">${esc(tail.join("\n"))}</pre></div></div>` : "";

      box.innerHTML = `<div class="imgl">
    <div class="panel">
      <h2>image run <em>${esc(name)}</em><span class="imgl-right">${devChip}${speedChips((p && p.notes) || {})}<b style="color:${stateColor}">${esc(runState)}</b>${elapsed ? `<span class="meta">${elapsed}</span>` : ""}</span></h2>
      <div class="body">
        ${runState === "failed" ? `<div class="imgl-fail">${esc(p.message || "the run failed; see the run log")}</div>` : ""}
        <div class="imgl-stages">${stageBars(p, runState)}</div>
        ${p && p.message && runState !== "failed" ? `<div class="imgl-msg">${esc(p.message)}</div>` : ""}
        <div class="imgl-saved">${saved}${ck.length && runState !== "running" ? `<button type="button" class="imgl-continue" data-imgl-continue title="open the Training tab with this model picked to continue">continue this model &rsaquo;</button>` : ""}</div>
        ${kpis ? `<div class="imgl-kpis">${kpis}</div>` : ""}
      </div>
    </div>
    ${panels}
    ${log}
  </div>`;
      const cont = box.querySelector("[data-imgl-continue]");
      if (cont) cont.addEventListener("click", () => d.onContinue(name));
      const open = box.querySelector("[data-imgl-open]");
      if (open) open.addEventListener("click", e => { e.preventDefault(); d.onOpenBrain(name); });
    }

    return {
      runName,
      setActive,
      load,
      render,
      get active() { return state.active; },
      get name() { return state.name; },
    };
  }

  // parseCsv is exported for the same reason it is tested: a run's whole loss curve comes
  // through it, and a silent mis-parse looks like a flat chart rather than an error.
  window.ImageLive = { create, TRAINER_ID, parseCsv };
})();
