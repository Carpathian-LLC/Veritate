/* ============================================================================
 *  CORAL LAB — STANDALONE DASHBOARD (DELETABLE)
 *  Developed by Carpathian, LLC. Distribution Not Authorized.
 *
 *  Three-column comparison for the Coral Merge experiment. Polls /runs and
 *  /run/<name>/csv every 5s; renders side-by-side training metrics for any
 *  three runs the user picks. No edits to index.js required.
 *
 *  To remove: delete this file + coral_lab.css + the two coral_lab references
 *  in index.html (link + script tags) + the tab + tab-body marked
 *  DELETABLE-CORAL in index.html.
 * ========================================================================== */

(function () {
  "use strict";

  const POLL_MS = 5000;

  const SLOTS = [
    { id: "A",       role: "Coral-A (constituent)",  defaultPrefix: "coral_a_" },
    { id: "B",       role: "Coral-B (constituent)",  defaultPrefix: "coral_b_" },
    { id: "CMP",     role: "Baseline / Blend",       defaultPrefix: "coral_" },
  ];

  const COLORS = {
    A:   "#7fd1ff",
    B:   "#f0a45a",
    CMP: "#a98bff",
    train: "rgba(255,255,255,0.7)",
    val:   "#ff7878",
  };

  const state = {
    runs:  [],
    pick:  { A: "", B: "", CMP: "" },
    data:  { A: [], B: [], CMP: [] },
    timer: null,
    mounted: false,
  };

  function $(id) { return document.getElementById(id); }

  function el(tag, attrs, html) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }

  function fmt(v, decimals) {
    if (v == null || isNaN(v)) return "—";
    if (typeof v !== "number") return String(v);
    if (decimals == null) decimals = 3;
    return v.toFixed(decimals);
  }

  function fmtSci(v) {
    if (v == null || isNaN(v) || v === 0) return "—";
    return v.toExponential(2);
  }

  function fmtTime(seconds) {
    if (!seconds || isNaN(seconds)) return "—";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    if (m < 60) return m + "m " + s + "s";
    const h = Math.floor(m / 60);
    return h + "h " + (m % 60) + "m";
  }

  // ------------------------------------------------------------------------
  // Data fetch

  async function fetchRunsList() {
    try {
      const r = await fetch("/runs");
      const d = await r.json();
      state.runs = (d && d.runs) ? d.runs : [];
    } catch (e) {
      state.runs = [];
    }
  }

  async function fetchRunCsv(name) {
    if (!name) return [];
    try {
      const r = await fetch("/run/" + encodeURIComponent(name) + "/csv");
      if (!r.ok) return [];
      const txt = await r.text();
      return parseCsv(txt);
    } catch (e) {
      return [];
    }
  }

  function parseCsv(text) {
    const lines = text.split("\n").filter(l => l.trim().length > 0);
    if (lines.length === 0) return [];
    const header = lines[0].split(",");
    const idx = {
      step:      header.indexOf("step"),
      split:     header.indexOf("split"),
      loss:      header.indexOf("loss"),
      lr:        header.indexOf("lr"),
      grad_norm: header.indexOf("grad_norm"),
      tok_per_s: header.indexOf("tok_per_s"),
      wall_s:    header.indexOf("wall_s"),
      seed:      header.indexOf("seed"),
    };
    const out = [];
    for (let i = 1; i < lines.length; i++) {
      const c = lines[i].split(",");
      if (c.length < header.length) continue;
      out.push({
        step:      parseInt(c[idx.step],      10),
        split:     c[idx.split],
        loss:      parseFloat(c[idx.loss]),
        lr:        parseFloat(c[idx.lr]),
        grad_norm: parseFloat(c[idx.grad_norm]),
        tok_per_s: parseFloat(c[idx.tok_per_s]),
        wall_s:    parseFloat(c[idx.wall_s]),
        seed:      parseInt(c[idx.seed], 10),
      });
    }
    return out;
  }

  // ------------------------------------------------------------------------
  // Drawing

  function fitCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    canvas.width  = Math.floor(rect.width  * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }

  function drawSeries(canvas, series, opts) {
    const fit = fitCanvas(canvas);
    if (!fit) return;
    const { ctx, w, h } = fit;
    opts = opts || {};
    const pad = { l: 36, r: 8, t: 8, b: 20 };
    const iw = w - pad.l - pad.r;
    const ih = h - pad.t - pad.b;

    ctx.fillStyle = opts.bg || "#05070a";
    ctx.fillRect(0, 0, w, h);

    const all = series.flatMap(s => s.data);
    if (all.length === 0) {
      ctx.fillStyle = "#3a4050";
      ctx.font = "11px monospace";
      ctx.fillText("no data", w / 2 - 24, h / 2);
      return;
    }

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const p of all) {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    }
    if (xMax === xMin) xMax = xMin + 1;
    const yRange = yMax - yMin || 1;
    yMin -= yRange * 0.05;
    yMax += yRange * 0.05;

    // gridlines
    ctx.strokeStyle = "#161a24";
    ctx.lineWidth = 1;
    ctx.font = "9px monospace";
    ctx.fillStyle = "#3a4050";
    for (let i = 0; i <= 4; i++) {
      const yv = yMin + (yMax - yMin) * (i / 4);
      const y  = pad.t + ih - (ih * (yv - yMin) / (yMax - yMin));
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + iw, y);
      ctx.stroke();
      ctx.fillText(yv.toFixed(2), 2, y + 3);
    }

    // x-axis labels
    ctx.fillText(String(Math.floor(xMin)), pad.l, h - 4);
    ctx.fillText(String(Math.floor(xMax)), pad.l + iw - 28, h - 4);

    // axis line
    ctx.strokeStyle = "#1a1f2b";
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, pad.t + ih);
    ctx.lineTo(pad.l + iw, pad.t + ih);
    ctx.stroke();

    // series
    for (const s of series) {
      if (!s.data.length) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.dashed ? 1 : 1.5;
      if (s.dashed) ctx.setLineDash([4, 3]);
      else ctx.setLineDash([]);
      ctx.beginPath();
      let started = false;
      for (const p of s.data) {
        const x = pad.l + iw * (p.x - xMin) / (xMax - xMin);
        const y = pad.t + ih - ih * (p.y - yMin) / (yMax - yMin);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  // ------------------------------------------------------------------------
  // Render

  function renderSlotPanel(slot) {
    const rows = state.data[slot.id];
    const name = state.pick[slot.id];
    const trainRows = rows.filter(r => r.split === "train");
    const valRows   = rows.filter(r => r.split === "val");
    const last      = trainRows[trainRows.length - 1];
    const lastVal   = valRows[valRows.length - 1];

    const col = $("coralCol" + slot.id);
    if (!col) return;

    if (!name) {
      col.classList.add("empty");
      col.innerHTML =
        '<div class="coral-col-head">' +
          '<span class="coral-col-name">— not selected —</span>' +
          '<span class="coral-col-role">' + slot.role + '</span>' +
        '</div>' +
        '<div class="coral-empty">pick a run above</div>';
      return;
    }
    col.classList.remove("empty");

    col.innerHTML =
      '<div class="coral-col-head">' +
        '<span class="coral-col-name">' + name + '</span>' +
        '<span class="coral-col-role">' + slot.role + '</span>' +
      '</div>' +
      '<div class="coral-stats">' +
        '<span class="k">step</span><span class="v">'      + (last ? last.step : "—") + '</span>' +
        '<span class="k">train loss</span><span class="v">' + fmt(last && last.loss, 4) + '</span>' +
        '<span class="k">val loss</span><span class="v">'   + fmt(lastVal && lastVal.loss, 4) + '</span>' +
        '<span class="k">lr</span><span class="v">'         + fmtSci(last && last.lr) + '</span>' +
        '<span class="k">grad norm</span><span class="v">'  + fmt(last && last.grad_norm, 3) + '</span>' +
        '<span class="k">tok/s</span><span class="v">'      + fmt(last && last.tok_per_s, 0) + '</span>' +
        '<span class="k">wall</span><span class="v">'       + fmtTime(last && last.wall_s) + '</span>' +
        '<span class="k">samples</span><span class="v">'    + rows.length + '</span>' +
      '</div>' +
      '<canvas class="coral-chart" id="coralChart' + slot.id + '"></canvas>';

    const canvas = $("coralChart" + slot.id);
    if (canvas) {
      drawSeries(canvas, [
        { color: COLORS.train, data: trainRows.map(r => ({ x: r.step, y: r.loss })) },
        { color: COLORS.val,   data: valRows.map(r   => ({ x: r.step, y: r.loss })) },
      ]);
    }
  }

  function renderCombined() {
    const canvas = $("coralCombinedLoss");
    if (!canvas) return;
    const series = SLOTS.map(slot => {
      const rows = state.data[slot.id].filter(r => r.split === "train");
      return {
        color: COLORS[slot.id],
        data:  rows.map(r => ({ x: r.step, y: r.loss })),
      };
    });
    const valSeries = SLOTS.map(slot => {
      const rows = state.data[slot.id].filter(r => r.split === "val");
      return {
        color:  COLORS[slot.id],
        dashed: true,
        data:   rows.map(r => ({ x: r.step, y: r.loss })),
      };
    });
    drawSeries(canvas, series.concat(valSeries));
  }

  function renderPickers() {
    SLOTS.forEach(slot => {
      const sel = $("coralPick" + slot.id);
      if (!sel) return;
      const cur = state.pick[slot.id];
      sel.innerHTML = '<option value="">— none —</option>' +
        state.runs.map(r =>
          '<option value="' + r.name + '"' + (r.name === cur ? " selected" : "") + '>' + r.name + '</option>'
        ).join("");
    });
  }

  function renderAll() {
    renderPickers();
    SLOTS.forEach(renderSlotPanel);
    renderCombined();
  }

  // ------------------------------------------------------------------------
  // Polling loop

  async function pollOnce() {
    await fetchRunsList();
    const pulls = SLOTS.map(async slot => {
      const name = state.pick[slot.id];
      state.data[slot.id] = name ? await fetchRunCsv(name) : [];
    });
    await Promise.all(pulls);
    renderAll();
  }

  function startPolling() {
    if (state.timer) return;
    pollOnce();
    state.timer = setInterval(pollOnce, POLL_MS);
  }

  function stopPolling() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  // ------------------------------------------------------------------------
  // Mount

  function autoPickDefaults() {
    // Auto-select obvious candidates by name prefix on first mount.
    const byPrefix = (pfx) => state.runs
      .filter(r => r.name.startsWith(pfx))
      .map(r => r.name);

    const aOpts   = byPrefix("coral_a_");
    const bOpts   = byPrefix("coral_b_");
    const cmpOpts = byPrefix("coral_baseline_").concat(byPrefix("coral_blend_"));
    if (!state.pick.A   && aOpts.length)   state.pick.A   = aOpts[0];
    if (!state.pick.B   && bOpts.length)   state.pick.B   = bOpts[0];
    if (!state.pick.CMP && cmpOpts.length) state.pick.CMP = cmpOpts[0];
  }

  function bindPickerEvents() {
    SLOTS.forEach(slot => {
      const sel = $("coralPick" + slot.id);
      if (!sel) return;
      sel.addEventListener("change", () => {
        state.pick[slot.id] = sel.value;
        try { localStorage.setItem("coral.pick." + slot.id, sel.value); } catch (e) {}
        pollOnce();
      });
    });
  }

  function restorePicks() {
    try {
      SLOTS.forEach(slot => {
        const v = localStorage.getItem("coral.pick." + slot.id);
        if (v) state.pick[slot.id] = v;
      });
    } catch (e) {}
  }

  function buildScaffold() {
    const host = $("coralBody");
    if (!host) return;
    host.innerHTML =
      '<div class="coral-wrap">' +
        '<div class="coral-head">' +
          '<h2>coral lab <em>three-column comparison for the Coral Merge experiment — standalone, deletable</em></h2>' +
          '<span id="coralStatus" class="coral-msg">idle</span>' +
        '</div>' +

        '<div class="coral-recipe">' +
          'Coral Merge: align two same-shape models’ FFN basis (Hungarian on activation correlation), splice with per-matrix scalar coefficients (~88 floats), then dual-teacher distill-refine on the mixed corpus. Output is the same shape as either constituent. Compares against a from-scratch model trained on the mixed corpus.' +
          '<details>' +
            '<summary>workflow</summary>' +
            '<div style="margin-top:6px;line-height:1.6">' +
              '1. Train <code>coral_a_tinystories_30m</code> on tinystories.<br>' +
              '2. Train <code>coral_b_distill_v1_30m</code> on distill_v1.<br>' +
              '3. Train <code>coral_baseline_50m</code> on distill_v1_mix_tinystories.<br>' +
              '4. Run <code>tools/coral/merge.py</code> to produce <code>coral_blend_30m</code>.<br>' +
              '5. Compare blend vs baseline on val loss across the mixed corpus.<br>' +
              '<br>Full spec at <code>~/Documents/GitHub/Agent-Documents/Veritate/coral_merge_spec.md</code>.' +
            '</div>' +
          '</details>' +
        '</div>' +

        '<div class="coral-pickers">' +
          SLOTS.map(slot =>
            '<label>' + slot.role +
              '<select id="coralPick' + slot.id + '"></select>' +
            '</label>'
          ).join("") +
        '</div>' +

        '<div class="coral-cols">' +
          SLOTS.map(slot => '<div class="coral-col empty" id="coralCol' + slot.id + '"></div>').join("") +
        '</div>' +

        '<div class="coral-multichart">' +
          '<h3>combined train loss (solid) + val loss (dashed)</h3>' +
          '<canvas id="coralCombinedLoss"></canvas>' +
          '<div class="coral-multi-legend">' +
            SLOTS.map(slot =>
              '<span><span class="swatch" style="background:' + COLORS[slot.id] + '"></span>' + slot.id + '</span>'
            ).join("") +
          '</div>' +
        '</div>' +
      '</div>';
  }

  function mount() {
    if (state.mounted) return;
    if (!$("coralBody")) return; // tab body not in DOM yet
    state.mounted = true;
    buildScaffold();
    restorePicks();
    bindPickerEvents();
    fetchRunsList().then(() => {
      autoPickDefaults();
      renderPickers();
      pollOnce();
    });
  }

  function onTabSwitch() {
    const active = document.querySelector('.tab-body.active');
    if (active && active.dataset.tab === "coral") {
      mount();
      startPolling();
    } else {
      stopPolling();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      window.addEventListener("hashchange", onTabSwitch);
      // also catch tab clicks (hash may not change if same hash)
      document.querySelectorAll(".tab").forEach(t => {
        t.addEventListener("click", () => setTimeout(onTabSwitch, 30));
      });
      onTabSwitch();
    });
  } else {
    window.addEventListener("hashchange", onTabSwitch);
    document.querySelectorAll(".tab").forEach(t => {
      t.addEventListener("click", () => setTimeout(onTabSwitch, 30));
    });
    onTabSwitch();
  }
})();
