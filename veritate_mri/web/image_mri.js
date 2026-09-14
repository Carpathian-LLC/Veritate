/* Developed by Carpathian, LLC. Distribution Not Authorized. */
/* veritate_mri/web/image_mri.js */

// The image-model MRI: every probed checkpoint of an image model (GET /images/mri/<model>)
// as what it draws, how a picture forms, what is inside, and how that moved over training,
// plus "prompt it" at every checkpoint. One view per host element, so two tabs can show
// two models: the Models tab mounts one for the picked timeline, the Generation tab's
// image layout mounts its own for the model it draws with. Self-contained: no index.js
// symbols. Styles in image_mri.css.
//   const view = ImageMri.create("hostId", { onContinue(name) });   // onContinue optional
//   view.show(name); view.hide(); view.render(); view.active; view.model; view.step
// ImageMri.charts exposes the figure helpers (svgLine, barChart, cellHeat, scaleBar,
// photoTile, gridRows, wireLightbox) and ImageMri.colors the series palette, so another
// module draws the same figures for one generation or one live run.

(function () {
  "use strict";

  const PLAY_MS = 1200;              // ms per checkpoint when playing through the probes
  const PROMPT_POLL_MS = 1500;
  const POLL_MS = 4000;              // while a run is training the shown model
  const LAYER_COLS = 8;              // image_probe.LAYER_COLS: tiles per row in layers.png / attention.png
  const CSS_TILE = 160;              // css px per probe tile: crisp on a 2x display, full size on click
  const FILM_TILE = 112, FILM_GAP = 4;
  const COLORS = ["#5dff9b", "#5db8ff", "#ffae5d", "#ff5d8f", "#c77dff", "#ffe45d", "#5dffe4", "#ff7a5d"];
  const imgCache = {};               // url -> Image, shared by every view (same probe pngs)

  function esc(s) { return String(s).replace(/[<>&]/g, c => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c])); }

  function svgLine(series, opts) {
    // series: [{name, color, points: [[x, y], ...]}]; one shared x (step) axis.
    const W = 600, H = 150, L = 42, R = 10, T = 10, B = 22;
    const pts = series.flatMap(s => s.points);
    if (!pts.length) return `<svg viewBox="0 0 ${W} ${H}"><text x="${W / 2}" y="${H / 2}" fill="#6f7480" font-size="11" text-anchor="middle">no data yet</text></svg>`;
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = opts && opts.y0 !== undefined ? opts.y0 : Math.min(...ys), y1 = opts && opts.y1 !== undefined ? opts.y1 : Math.max(...ys);
    if (y1 === y0) { y1 = y0 + 1; }
    const sx = x => x1 === x0 ? (L + (W - L - R) / 2) : L + (x - x0) / (x1 - x0) * (W - L - R);
    const sy = y => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
    const fmt = v => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);
    let out = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
    for (const f of [0, 0.5, 1]) {
      const y = T + f * (H - T - B);
      out += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#1e2330" stroke-width="1"/>`;
      out += `<text x="${L - 4}" y="${y + 3}" fill="#6f7480" font-size="9" text-anchor="end">${fmt(y1 - f * (y1 - y0))}</text>`;
    }
    const xname = (opts && opts.xname) || "step";
    out += `<text x="${L}" y="${H - 6}" fill="#6f7480" font-size="9">${xname} ${x0.toLocaleString()}</text>`;
    out += `<text x="${W - R}" y="${H - 6}" fill="#6f7480" font-size="9" text-anchor="end">${x1.toLocaleString()}</text>`;
    for (const s of series) {
      if (!s.points.length) continue;
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
      out += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
      const last = s.points[s.points.length - 1];
      out += `<circle cx="${sx(last[0]).toFixed(1)}" cy="${sy(last[1]).toFixed(1)}" r="3" fill="${s.color}"/>`;
    }
    return out + `</svg>`;
  }

  function barChart(values, labels, color) {
    const W = 600, H = 150, L = 10, B = 22, T = 10;
    if (!values || !values.length) return `<svg viewBox="0 0 ${W} ${H}"><text x="${W / 2}" y="${H / 2}" fill="#6f7480" font-size="11" text-anchor="middle">no data yet</text></svg>`;
    const max = Math.max(1e-9, ...values);
    const bw = (W - 2 * L) / values.length;
    let out = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
    values.forEach((v, i) => {
      const h = (v / max) * (H - T - B);
      out += `<rect x="${(L + i * bw + 2).toFixed(1)}" y="${(H - B - h).toFixed(1)}" width="${(bw - 4).toFixed(1)}" height="${h.toFixed(1)}" fill="${color}" opacity="0.85"/>`;
      out += `<text x="${(L + i * bw + bw / 2).toFixed(1)}" y="${H - 7}" fill="#6f7480" font-size="9" text-anchor="middle">${labels[i]}</text>`;
      out += `<text x="${(L + i * bw + bw / 2).toFixed(1)}" y="${(H - B - h - 3).toFixed(1)}" fill="#a6b0bf" font-size="9" text-anchor="middle">${v.toFixed(2)}</text>`;
    });
    return out + `</svg>`;
  }

  function b64ToBytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // The same-seed samples' codes at a checkpoint, as one Uint8Array per sample.
  function sampleCodes(s) {
    if (!s || !s.sample_codes_b64 || !s.code_bytes) return null;
    if (!s._codes) {
      const all = b64ToBytes(s.sample_codes_b64);
      const n = Math.floor(all.length / s.code_bytes);
      s._codes = Array.from({ length: n }, (_, i) => all.subarray(i * s.code_bytes, (i + 1) * s.code_bytes));
    }
    return s._codes;
  }

  // Churn between two checkpoints: which cells of the same-seed samples changed. map = share
  // of samples whose cell changed (any plane), per plane and overall change fractions, and a
  // per-sample cell mask for outlining.
  function churnBetween(prev, cur) {
    const a = sampleCodes(prev), b = sampleCodes(cur);
    if (!a || !b) return null;
    const n = Math.min(a.length, b.length), cb = cur.code_bytes, planes = cur.planes || 1, cell = Math.floor(cb / planes);
    if (!n || !cell) return null;
    const map = new Float32Array(cell), perPlane = new Array(planes).fill(0), changed = [];
    let all = 0;
    for (let i = 0; i < n; i++) {
      const ch = new Uint8Array(cell);
      for (let k = 0; k < cb; k++) {
        if (a[i][k] !== b[i][k]) { all++; perPlane[Math.floor(k / cell)]++; ch[k % cell] = 1; }
      }
      for (let c = 0; c < cell; c++) if (ch[c]) map[c]++;
      changed.push(ch);
    }
    for (let c = 0; c < cell; c++) map[c] /= n;
    return { map, perPlane: perPlane.map(v => v / (n * cell)), all: all / (n * cb), changed, n };
  }

  function gridRows(flat, gh, gw) {
    const rows = [];
    for (let y = 0; y < gh; y++) rows.push(Array.from(flat.slice(y * gw, (y + 1) * gw)));
    return rows;
  }

  // Fixed-cell heat map, never stretched: every cell is `cell` css px square, so a 16x16
  // grid is a small crisp square and a 12x12 head grid reads as a table. `kind`: "heat"
  // (blue low -> orange high), "div" (green positive, red negative, symmetric about 0).
  // `lo`/`hi` pin the scale (a pass count, a 0..1 share); otherwise the data's range.
  function cellHeat(rows, opts) {
    opts = opts || {};
    if (!rows || !rows.length || !rows[0].length) return `<span class="meta">${esc(opts.empty || "no data yet")}</span>`;
    const nr = rows.length, nc = rows[0].length;
    const cell = opts.cell || Math.max(8, Math.min(16, Math.floor(224 / Math.max(nr, nc))));
    const rowLabel = opts.rowLabel || null, colLabel = opts.colLabel || null;
    const L = rowLabel ? 30 : 1, T = 1, B = colLabel ? 16 : 1, R = 1;
    const W = L + nc * cell + R, H = T + nr * cell + B;
    const flat = rows.flat().filter(v => typeof v === "number" && isFinite(v));
    if (!flat.length) return `<span class="meta">${esc(opts.empty || "no data yet")}</span>`;
    let lo = opts.lo !== undefined ? opts.lo : Math.min(...flat), hi = opts.hi !== undefined ? opts.hi : Math.max(...flat);
    if (opts.kind === "div") { hi = Math.max(1e-6, ...flat.map(Math.abs)); lo = -hi; }
    if (hi <= lo) hi = lo + 1;
    const col = v => {
      if (opts.kind === "div") { const t = Math.max(-1, Math.min(1, v / hi)); return t >= 0 ? `rgba(93,255,155,${(0.12 + 0.88 * t).toFixed(2)})` : `rgba(255,93,143,${(0.12 - 0.88 * t).toFixed(2)})`; }
      const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
      return `rgb(${Math.round(40 + 215 * t)},${Math.round(90 + 70 * t)},${Math.round(230 - 210 * t)})`;
    };
    const fmt = opts.fmt || (v => Number(v).toFixed(2));
    let out = `<svg class="imri-heat" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
    rows.forEach((r, i) => {
      if (rowLabel) out += `<text x="${L - 3}" y="${T + i * cell + cell / 2 + 3}" fill="#6f7480" font-size="9" text-anchor="end">${rowLabel(i)}</text>`;
      r.forEach((v, j) => {
        const label = (rowLabel ? rowLabel(i) + " " : "") + (colLabel ? colLabel(j) : `row ${i + 1}, col ${j + 1}`);
        out += `<rect x="${L + j * cell}" y="${T + i * cell}" width="${cell - 1}" height="${cell - 1}" fill="${typeof v === "number" && isFinite(v) ? col(v) : "#1e2330"}"><title>${label}: ${typeof v === "number" ? fmt(v) : "?"}</title></rect>`;
      });
    });
    if (colLabel) for (let j = 0; j < nc; j++) out += `<text x="${L + j * cell + cell / 2}" y="${H - 4}" fill="#6f7480" font-size="9" text-anchor="middle">${colLabel(j)}</text>`;
    return out + `</svg>`;
  }

  // The legend under a heat: a gradient bar with the two ends named.
  function scaleBar(loText, hiText, kind) {
    const grad = kind === "div" ? "linear-gradient(90deg, rgb(255,93,143), #1e2330, rgb(93,255,155))" : "linear-gradient(90deg, rgb(40,90,230), rgb(255,160,20))";
    return `<span class="imri-scale"><span>${loText}</span><i style="background:${grad}"></i><span>${hiText}</span></span>`;
  }

  // A probe png shown at a fixed tile size. New probes hold the frame itself per tile (THUMB
  // 320, nothing resampled) and are shown at CSS_TILE, so they stay sharp on a 2x
  // display; older 96 px probes show at 96, which is all the pixels they have. `cols` is the
  // png's tile count across, which fixes the css width so the browser never stretches it.
  // Click opens the png at full size.
  function tileCss(s) { return Math.min(CSS_TILE, (s && s.thumb) || 96); }
  function photoTile(url, s, cols, alt) {
    const thumb = (s && s.thumb) || 96, gap = (s && s.gap) || 4, css = tileCss(s);
    const w = Math.round((cols * thumb + (cols + 1) * gap) * css / thumb);
    return `<img class="imri-photo" src="${url}" alt="${esc(alt || "")}" style="width:${w}px" data-full="${url}" title="click to enlarge">`;
  }

  function openLightbox(url) {
    const old = document.querySelector(".imri-lightbox");
    if (old) old.remove();
    const box = document.createElement("div");
    box.className = "imri-lightbox";
    box.innerHTML = `<img src="${url}" alt="">`;
    box.addEventListener("click", () => box.remove());
    document.body.appendChild(box);
  }
  let lightboxWired = false;
  function wireLightbox() {
    if (lightboxWired) return;
    lightboxWired = true;
    document.addEventListener("click", e => {
      const img = e.target && e.target.closest ? e.target.closest("img.imri-photo") : null;
      if (img) { e.preventDefault(); openLightbox(img.dataset.full || img.src); }
    });
    document.addEventListener("keydown", e => { if (e.key === "Escape") { const lb = document.querySelector(".imri-lightbox"); if (lb) lb.remove(); } });
  }

  function loadImage(url, cb) {
    let img = imgCache[url];
    if (img && img.complete && img.naturalWidth) { cb(img); return; }
    if (!img) { img = new Image(); imgCache[url] = img; img.src = url; }
    img.addEventListener("load", () => cb(img), { once: true });
  }

  function create(hostId, opts) {
    opts = opts || {};
    const host = document.getElementById(hostId);
    const st = { active: false, model: null, data: null, step: null, timer: null, running: false,
                 follow: true, playing: false, playTimer: null, sample: 0, fillPic: 0,
                 prompt: { model: null, state: null, timer: null, photo: null, photoName: "" } };

    function show(name) {
      st.active = true;
      if (st.timer) { clearTimeout(st.timer); st.timer = null; }
      if (st.model !== name) { play(false); st.model = name; st.data = null; st.step = null; st.follow = true; }
      load();
    }

    function hide() {
      st.active = false;
      if (st.timer) { clearTimeout(st.timer); st.timer = null; }
      play(false);
      st.model = null; st.data = null;
      host.innerHTML = "";
    }

    function load() {
      const name = st.model;
      if (!name || !st.active) return;
      Promise.all([
        fetch(`/images/mri/${encodeURIComponent(name)}?` + Date.now(), { cache: "no-store" }).then(r => r.json()),
        fetch("/trainers").then(r => r.json()).catch(() => null),
      ]).then(([d, tr]) => {
        if (!st.active || st.model !== name) return;
        st.data = d && d.ok ? d : null;
        const run = tr && tr.running;
        st.running = !!(run && run.status === "running" && run.args && run.args.name &&
          (`${String(run.args.name).toLowerCase().replace(/[^a-z0-9]+/g, "_")}_${run.args.size}` === name || run.args.resume === name));
        render();
        if (st.running) st.timer = setTimeout(load, POLL_MS);
      }).catch(() => {});
    }

    function render() {
      const d = st.data;
      wireLightbox();
      // Two children: the main view is rebuilt on every refresh; the prompt panel is built once
      // per model so typing in it survives the polls while a run is training.
      let box = host.querySelector("[data-imri-main]");
      if (!box) {
        host.innerHTML = `<div data-imri-main></div><div data-imri-prompt></div>`;
        box = host.querySelector("[data-imri-main]");
      }
      if (!d) { box.innerHTML = `<div class="panel"><div class="body meta">could not load this model's image probe.</div></div>`; return; }
      const steps = d.steps || [];
      const name = d.model;
      promptMount(name, (d.checkpoint_steps || []).length > 0);
      if (!steps.length) {
        box.innerHTML = `<div class="imri"><div class="panel"><h2>${esc(name)} <em>image model</em></h2><div class="body meta">` +
          `${d.checkpoint_steps && d.checkpoint_steps.length ? `${d.checkpoint_steps.length} checkpoints, none probed yet` : "no probe yet"} &mdash; the probe runs every <i>pictures every</i> steps and at every checkpoint${st.running ? " (training now)" : ""}.</div></div></div>`;
        return;
      }
      if (st.follow || st.step === null || !steps.some(s => s.step === st.step)) st.step = steps[steps.length - 1].step;
      const cur = steps.find(s => s.step === st.step) || steps[steps.length - 1];
      const g = d.geometry || {};
      const planes = cur.planes || g.planes || 1;
      const layers = cur.layers || (cur.attention_entropy_per_layer || []).length || 0;
      const url = (step, f) => `/images/mri/${encodeURIComponent(name)}/${step}/${f}?${step}`;
      const has = (s, f) => (s.files || []).includes(f);
      const num = v => typeof v === "number" && isFinite(v);
      const series = (key, color, label, pick) => ({ name: label, color, points: steps.map(s => [s.step, pick ? pick(s) : s[key]]).filter(q => num(q[1])) });
      const kpi = (v, k, title) => `<div class="imri-kpi" title="${esc(title || "")}"><div class="v">${v}</div><div class="k">${k}</div></div>`;
      const pct = v => num(v) ? (v * 100).toFixed(1) + "%" : "&mdash;";
      const photo = (file, cols, alt, fallback) => has(cur, file) ? photoTile(url(cur.step, file), cur, cols, alt) : `<span class="meta">${fallback || "not in this probe"}</span>`;
      const section = (title, sub) => `<div class="imri-section">${title}${sub ? ` <span>${sub}</span>` : ""}</div>`;
      const legend = html => `<div class="imri-legend">${html}</div>`;
      const needVal = "needs held-out pictures (a val bin)";
      const curIdx = steps.findIndex(s => s.step === cur.step);
      const prev = curIdx > 0 ? steps[curIdx - 1] : null;
      const [gh, gw] = cur.grid || [0, 0];
      const tileCols = Math.min(LAYER_COLS, layers || LAYER_COLS);

      // -- over training -------------------------------------------------------------------
      const acc = series("fill_accuracy", COLORS[0], "all");
      const perPlane = Array.from({ length: planes }, (_, p) => ({
        name: `plane ${p}${p === 0 ? " (structure)" : p === planes - 1 ? " (detail)" : ""}`, color: COLORS[(p + 1) % COLORS.length],
        points: steps.map(s => [s.step, (s.fill_accuracy_per_plane || [])[p]]).filter(q => num(q[1])),
      }));
      const ratios = ["0.25", "0.5", "0.75", "1.0"];
      const byRatio = ratios.map((r, i) => ({ name: `${Math.round(parseFloat(r) * 100)}% hidden`, color: COLORS[i % COLORS.length],
        points: steps.map(s => [s.step, (s.loss_by_hidden_fraction || {})[r]]).filter(q => num(q[1])) }));
      const ent = cur.attention_entropy_per_layer || [];
      const heads = cur.attention_entropy_per_head || [];
      const agree = cur.lens_agreement_per_layer || [];
      const lensAcc = cur.lens_accuracy_per_layer || [];
      const norms = cur.residual_norm_per_layer || [];
      const cal = cur.calibration || [];
      const passesN = (cur.pass_committed || []).length;
      const novelty = cur.novelty_per_sample || [];
      const layerIdx = arr => arr.map((v, i) => [i + 1, v]);
      const commit = cur.commit_layer;
      const churn = prev ? churnBetween(prev, cur) : null;
      const churnSeries = steps.slice(1).map((s, i) => { const c = churnBetween(steps[i], s); return c ? { step: s.step, c } : null; }).filter(Boolean);
      const churnAll = { name: "all planes", color: COLORS[2], points: churnSeries.map(x => [x.step, x.c.all]) };
      const churnPlanes = Array.from({ length: planes }, (_, p) => ({ name: `plane ${p}`, color: COLORS[(p + 3) % COLORS.length], points: churnSeries.map(x => [x.step, x.c.perPlane[p]]).filter(q => num(q[1])) }));
      const firstMap = steps.find(s => Array.isArray(s.loss_map) && s.loss_map.length);
      const improved = firstMap && Array.isArray(cur.loss_map) && cur.loss_map.length === firstMap.loss_map.length && cur !== firstMap && gh && gw
        ? gridRows(cur.loss_map.map((v, i) => firstMap.loss_map[i] - v), gh, gw) : null;
      const nSamples = Math.min(8, (sampleCodes(cur) || []).length || 8);
      const captionN = (cur.caption_samples || []).length;
      // how it forms: the pass each plane commits in, what forms first, how far attention reaches
      const commitPlanes = cur.commit_pass_per_plane || [];
      const commitSeries = Array.from({ length: planes }, (_, p) => ({ name: `plane ${p}`, color: COLORS[(p + 1) % COLORS.length],
        points: steps.map(s => [s.step, (s.commit_pass_per_plane || [])[p]]).filter(q => num(q[1])) })).filter(s => s.points.length);
      const structure = { name: "structure", color: COLORS[1], points: steps.map(s => [s.step, (s.fill_accuracy_per_plane || [])[0]]).filter(q => num(q[1])) };
      const colour = series("colour_match", COLORS[3], "colour");
      const detail = { name: "detail", color: COLORS[2], points: steps.map(s => [s.step, num(s.detail_ratio) ? Math.min(1, s.detail_ratio) : NaN]).filter(q => num(q[1])) };
      const reach = cur.attention_distance_per_layer || [];
      const meanOf = a => (a && a.length) ? a.reduce((x, y) => x + y, 0) / a.length : NaN;
      const reachSeries = series("attention_distance_mean", COLORS[6], "mean reach", s => meanOf(s.attention_distance_per_layer));
      // maps drawn from the numbers (crisp at any size), the pngs only as a fallback
      const formationRows = Array.isArray(cur.commit_pass_map) && gh && gw ? gridRows(cur.commit_pass_map, gh, gw) : null;
      const lossRows = Array.isArray(cur.loss_map) && gh && gw ? gridRows(cur.loss_map, gh, gw) : null;
      const confRows = Array.isArray(cur.confidence_map) && gh && gw ? gridRows(cur.confidence_map, gh, gw) : null;
      const nPass = cur.formation_passes || passesN || 8;

      box.innerHTML = `<div class="imri">
        <div class="panel">
          <h2>${esc(name)} <em>image model &middot; ${g.height || "?"}&times;${g.width || "?"} px &middot; ${g.image_code_bytes || "?"} bytes/picture &middot; ${planes} planes &middot; ${layers || "?"} layers${st.running ? ' &middot; <span class="imri-live">training now</span>' : ""}</em>${!st.running && opts.onContinue && (d.checkpoint_steps || []).length ? `<button type="button" class="imri-continue" data-imri-continue title="open the Training tab with this model picked to continue from its last checkpoint">continue training &rsaquo;</button>` : ""}</h2>
          <div class="body">
            <div class="imri-head"><span>pictures: <b>${esc(d.image_set || "?")}</b></span><span>codec: <b>${esc(d.codec || "?")}</b></span><span>probes: <b>${steps.length}</b></span><span>showing step <b>${cur.step.toLocaleString()}</b></span></div>
            <div class="imri-strip" data-imri-strip>${steps.map(s => `<button type="button" data-imri-step="${s.step}" class="${s.step === cur.step ? "on" : ""}" title="step ${s.step}">${has(s, "samples.png") ? `<img src="${url(s.step, "samples.png")}" alt="">` : ""}<span>${s.step.toLocaleString()}</span></button>`).join("")}</div>
            <div class="imri-scrub"><button type="button" data-imri-prev title="previous probe">&#9664;</button>
              <input type="range" data-imri-range min="0" max="${steps.length - 1}" value="${curIdx}" title="scrub through the probes">
              <button type="button" data-imri-next title="next probe">&#9654;</button>
              <button type="button" data-imri-play class="${st.playing ? "on" : ""}">${st.playing ? "&#10074;&#10074; pause" : "&#9654; play"}</button>
              <label><input type="checkbox" data-imri-follow ${st.follow ? "checked" : ""}> follow latest</label>
              <span>probe ${curIdx + 1} of ${steps.length}${prev ? ` &middot; since step ${prev.step.toLocaleString()}` : ""}</span></div>
            <div class="imri-kpis">
              ${kpi(pct(cur.fill_accuracy), "fill accuracy", "of hidden cells filled in exactly, half the picture hidden, on held-out pictures")}
              ${kpi(pct(cur.mean_confidence), "confidence", "the model's own probability on its picks for hidden cells")}
              ${kpi(num(cur.expected_calibration_error) ? (cur.expected_calibration_error * 100).toFixed(1) + "<span style='font-size:11px;color:var(--dim)'> pts</span>" : "&mdash;", "calibration error", "gap between confidence and accuracy; 0 means it knows what it knows")}
              ${kpi(num(cur.novelty_mean) ? pct(cur.novelty_mean) : "&mdash;", "novelty", "cells of a sample that differ from the closest training picture; 0% is a copy")}
              ${kpi(commit ? `${commit}<span style="font-size:11px;color:var(--dim)"> / ${layers}</span>` : "&mdash;", "decides at layer", "first layer whose picture already matches the final answer on 90% of hidden cells")}
              ${kpi(num(cur.codes_used) ? `${cur.codes_used}<span style="font-size:11px;color:var(--dim)">/255</span>` : "&mdash;", "codes in use", "a collapse shows as a handful")}
              ${kpi(ent.length ? meanOf(ent).toFixed(2) : "&mdash;", "attention spread", "0 focused, 1 uniform, averaged over layers")}
              ${kpi(churn ? pct(1 - churn.all) : "&mdash;", "settled since last", "share of the same-seed samples' cells unchanged since the previous probe; rising toward 100% is convergence")}
              ${kpi(num(cur.colour_match) ? pct(cur.colour_match) : "&mdash;", "colour match", "how closely the samples' palette matches the held-out pictures; 100% is the same palette")}
              ${kpi(num(cur.detail_ratio) ? pct(Math.min(1, cur.detail_ratio)) : "&mdash;", "detail", "sharpness of the samples against the codec's own reconstructions, the most detail a sample can have")}
            </div>
          </div>
        </div>

        ${section("what it draws", "same seeds at every probe, so one draw evolves")}
        <div class="panel">
          <div class="body">
            <div class="imri-fig">${photo("samples.png", nSamples + captionN, "samples")}
            ${legend(`<span>${nSamples} from nothing${captionN ? `; then from held-out captions: ${cur.caption_samples.map(c => `&ldquo;${esc(c)}&rdquo;`).join(", ")}` : ""}</span>`)}</div>
            ${has(cur, "nearest.png") ? `<div class="imri-fig"><h4>closest training picture to each sample &middot; copying or inventing?</h4>${photo("nearest.png", nSamples, "nearest training pictures")}
              ${legend(novelty.map(v => `<span>${pct(v)} new</span>`).join(""))}</div>` : ""}
          </div>
        </div>
        <div class="panel">
          <h2>one draw through training <em>the same seed at every probe &middot; orange outlines: cells that changed since the probe before</em></h2>
          <div class="body"><div class="imri-film">
            <div class="pick"><span>sample</span><select data-imri-sample>${Array.from({ length: nSamples }, (_, i) => `<option value="${i}" ${i === st.sample ? "selected" : ""}>${i + 1}</option>`).join("")}</select><span>click a frame to jump to that probe</span></div>
            <canvas data-imri-film-samples></canvas>
            <div class="pick"><span>held-out picture</span><select data-imri-fillpic>${[0, 1, 2, 3].map(i => `<option value="${i}" ${i === st.fillPic ? "selected" : ""}>${i + 1}</option>`).join("")}</select><span>its completion at every probe (original at the left)</span></div>
            <canvas data-imri-film-fill></canvas>
          </div></div>
        </div>

        ${section("how a picture forms", "one sample, pass by pass")}
        <div class="panel">
          <div class="body"><div class="imri-fig">${photo("passes.png", passesN || nPass, "decode passes")}
            ${passesN ? legend(cur.pass_committed.map((n, i) => `<span>pass ${i + 1}: ${n} cells${num((cur.pass_confidence || [])[i]) ? `, ${pct(cur.pass_confidence[i])} sure` : ""}</span>`).join("")) : ""}
            ${legend(`<span>grey cells are still undecided. The most confident cells are committed first and each pass sees the last one's decisions; a model that has learned structure commits the layout early and the detail late.</span>`)}</div></div>
        </div>
        <div class="imri-grid">
          <div class="panel"><h2>the order it forms <em>which pass decided each cell</em></h2>
            <div class="body"><div class="imri-heatwrap">${formationRows ? cellHeat(formationRows, { lo: 1, hi: nPass, fmt: v => "pass " + v }) : photo("formation.png", 2, "formation order")}
              <div class="imri-side">${scaleBar("pass 1", `pass ${nPass}`)}<span class="meta">blue cells were decided in the first passes, orange in the last. Structure should go blue before detail does.</span></div></div>
            ${commitPlanes.length ? `<div class="imri-chart small">${barChart(commitPlanes, commitPlanes.map((_, p) => "plane " + p), COLORS[2])}</div>${legend(`<span>mean commit pass per plane (of ${nPass}); plane 0 is structure, the last plane detail</span>`)}` : ""}
            ${commitSeries.length > 0 && commitSeries[0].points.length > 1 ? `<div class="imri-chart small">${svgLine(commitSeries, { y0: 1, y1: Math.max(2, nPass) })}</div>${legend(commitSeries.map(s => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("") + `<span>over training: the structure plane should commit earlier as the model learns</span>`)}` : ""}</div></div>
          <div class="panel"><h2>coarse to fine <em>the first sample from 1, 2, &hellip; ${planes} planes</em></h2>
            <div class="body"><div class="imri-fig"><div class="cols" style="max-width:${(planes * tileCss(cur) + (planes + 1) * 4)}px">${Array.from({ length: planes }, (_, i) => `<span>${i + 1} plane${i ? "s" : ""}</span>`).join("")}</div>${photo("planes.png", planes, "coarse to fine")}
            ${legend(`<span>each residual plane adds one byte per cell: the left tile is plane 0 alone (layout), the right tile the whole picture. The difference across the row is the detail the model still has to learn.</span>`)}</div></div></div>
          <div class="panel"><h2>churn <em>what is still changing</em></h2>
            <div class="body">${churn && gh && gw ? `<div class="imri-heatwrap">${cellHeat(gridRows(churn.map, gh, gw), { lo: 0, hi: 1, fmt: v => pct(v) + " of samples changed" })}
              <div class="imri-side">${scaleBar("settled", "still moving")}<span class="meta">each cell: share of the ${churn.n} same-seed samples whose code there changed since step ${prev.step.toLocaleString()}. Structure settles first; a region that keeps flickering late is one the model has not learned.</span></div></div>` : `<span class="meta">needs two probes</span>`}
            ${churnSeries.length ? `<div class="imri-chart small">${svgLine([churnAll, ...churnPlanes], { y0: 0, y1: 1 })}</div>${legend(`<span><i style="background:${churnAll.color}"></i>cells changed since the previous probe</span>` + churnPlanes.map(p => `<span><i style="background:${p.color}"></i>${p.name}</span>`).join("") + `<span>falling toward 0 is convergence; a jump is a phase change</span>`)}` : ""}</div></div>
          <div class="panel"><h2>where it improved <em>loss per cell, first probe minus now</em></h2>
            <div class="body">${improved ? `<div class="imri-heatwrap">${cellHeat(improved, { kind: "div", fmt: v => (v >= 0 ? "easier by " : "harder by ") + Math.abs(v).toFixed(3) })}
              <div class="imri-side">${scaleBar("harder", "easier", "div")}<span class="meta">since step ${firstMap.step.toLocaleString()}. Solid green everywhere is learning; a red centre with green edges means it learned borders and backgrounds before subjects.</span></div></div>` : `<span class="meta">needs two probes with held-out pictures</span>`}</div></div>
        </div>

        ${section("inside the model", "held-out pictures, half the cells hidden")}
        <div class="imri-grid">
          <div class="panel"><h2>can it complete a real picture</h2>
            <div class="body"><div class="imri-fig"><div class="cols" style="max-width:${3 * tileCss(cur) + 16}px"><span>original</span><span>hidden</span><span>filled</span></div>${photo("fill.png", 3, "fill test", needVal)}</div></div></div>
          <div class="panel"><h2>how sure it is <em>and whether that is earned</em></h2>
            <div class="body"><div class="imri-heatwrap">${confRows ? cellHeat(confRows, { lo: 0, hi: 1, fmt: v => pct(v) + " sure" }) : photo("confidence.png", 4, "confidence map", needVal)}
              <div class="imri-side">${scaleBar("unsure", "sure")}<span class="meta">the model's probability on its pick for each hidden cell of the first held-out picture (known cells count as certain)</span></div></div>
            ${cal.length ? `<div class="imri-chart small">${barChart(cal.map(b => num(b.accuracy) ? b.accuracy : 0), cal.map(b => `${Math.round(b.lo * 100)}&ndash;${Math.round(b.hi * 100)}%`), COLORS[4])}</div>${legend(`<span>accuracy of the cells in each confidence band (${cal.map(b => b.n).join(" / ")} cells). Calibrated means the bars climb with the bands.</span>`)}` : ""}</div></div>
          <div class="panel"><h2>through the layers <em>what it would draw if it stopped at layer 1, 2, &hellip; ${layers || ""}</em></h2>
            <div class="body"><div class="imri-fig">${photo("layers.png", tileCols, "logit lens", needVal)}
            ${legend(`<span>the residual after each block, read through the model's own output head. Where the final picture first appears is where the decision is made${commit ? ` (layer ${commit})` : ""}.</span>`)}</div>
            ${agree.length ? `<div class="imri-chart small">${svgLine([{ name: "agreement with final", color: COLORS[1], points: layerIdx(agree) }, { name: "accuracy", color: COLORS[0], points: layerIdx(lensAcc) }], { y0: 0, y1: 1, xname: "layer" })}</div>${legend(`<span><i style="background:${COLORS[1]}"></i>agreement with the final layer</span><span><i style="background:${COLORS[0]}"></i>accuracy against the real picture</span>`)}` : ""}</div></div>
          <div class="panel"><h2>where it struggles <em>and how much each layer carries</em></h2>
            <div class="body"><div class="imri-heatwrap">${lossRows ? cellHeat(lossRows, { fmt: v => "loss " + v.toFixed(2) }) : photo("cell_loss.png", 2, "loss per cell", needVal)}
              <div class="imri-side">${scaleBar("easy", "hard")}<span class="meta">loss per cell over the held-out pictures${num(cur.centre_loss) ? `: centre ${cur.centre_loss.toFixed(2)} vs edge ${cur.edge_loss.toFixed(2)}` : ""}</span></div></div>
            ${norms.length ? `<div class="imri-chart small">${barChart(norms, norms.map((_, i) => "L" + (i + 1)), COLORS[2])}</div>${legend(`<span>mean residual norm after each block. A healthy net grows it steadily; a flat tail means late layers add little.</span>`)}` : ""}</div></div>
          <div class="panel"><h2>where it looks <em>from the centre cell, one map per layer</em></h2>
            <div class="body"><div class="imri-fig">${photo("attention.png", tileCols, "attention", needVal)}</div>
            ${ent.length ? `<div class="imri-chart small">${barChart(ent, ent.map((_, i) => "L" + (i + 1)), "#5db8ff")}</div>${legend(`<span>attention spread per layer &mdash; lower is more focused</span>`)}` : ""}</div></div>
          <div class="panel"><h2>attention by head <em>focused (blue) to uniform (orange)</em></h2>
            <div class="body"><div class="imri-heatwrap">${cellHeat(heads, { lo: 0, hi: 1, cell: 14, rowLabel: i => "L" + (i + 1), colLabel: j => "h" + j, empty: needVal })}
              <div class="imri-side"><span class="meta">heads specialise as the model learns: a trained net shows a mix of sharp local heads and broad context heads; all-orange has not yet learned where to look.</span></div></div></div></div>
          <div class="panel"><h2>how far it looks <em>attention reach per layer, in cells</em></h2>
            <div class="body">${reach.length ? `<div class="imri-chart small">${barChart(reach, reach.map((_, i) => "L" + (i + 1)), COLORS[6])}</div>` : `<span class="meta">${needVal}</span>`}
            ${reachSeries.points.length > 1 ? `<div class="imri-chart small">${svgLine([reachSeries])}</div>` : ""}
            ${legend(`<span>attention-weighted distance from a cell to the cells it reads, over heads${gh && gw ? ` (grid ${gh}&times;${gw}; a uniform head reads ~${(0.52 * Math.hypot(gh, gw)).toFixed(1)} cells away)` : ""}. Untrained attention reads far; trained texture layers settle near 1-2 cells while a few layers stay global.</span>`)}</div></div>
        </div>

        ${section("over training", `${steps.length} probes`)}
        <div class="imri-grid imri-grid-3">
          <div class="panel"><h2>what forms first <em>structure, colour, detail</em></h2>
            <div class="body"><div class="imri-chart small">${svgLine([structure, colour, detail], { y0: 0, y1: 1 })}</div>
            ${legend(`<span><i style="background:${structure.color}"></i>structure: plane-0 fill accuracy</span><span><i style="background:${colour.color}"></i>colour match</span><span><i style="background:${detail.color}"></i>detail vs the codec ceiling</span>`)}</div></div>
          <div class="panel"><h2>fill accuracy <em>per plane</em></h2>
            <div class="body"><div class="imri-chart small">${svgLine([acc, ...perPlane], { y0: 0, y1: 1 })}</div>
            ${legend(`<span><i style="background:${COLORS[0]}"></i>all</span>` + perPlane.map(p => `<span><i style="background:${p.color}"></i>${esc(p.name)}</span>`).join(""))}</div></div>
          <div class="panel"><h2>loss by how much is hidden</h2>
            <div class="body"><div class="imri-chart small">${svgLine(byRatio)}</div>
            ${legend(byRatio.map(p => `<span><i style="background:${p.color}"></i>${p.name}</span>`).join(""))}</div></div>
          <div class="panel"><h2>confidence <em>sure, and right to be?</em></h2>
            <div class="body"><div class="imri-chart small">${svgLine([series("mean_confidence", COLORS[4], "confidence"), series("expected_calibration_error", COLORS[3], "calibration error")], { y0: 0, y1: 1 })}</div>
            ${legend(`<span><i style="background:${COLORS[4]}"></i>confidence</span><span><i style="background:${COLORS[3]}"></i>calibration error (lower is better)</span>`)}</div></div>
          <div class="panel"><h2>novelty <em>copying or inventing</em></h2>
            <div class="body"><div class="imri-chart small">${svgLine([series("novelty_mean", COLORS[5], "novelty")], { y0: 0, y1: 1 })}</div>
            ${legend(`<span>share of a sample's cells that differ from its closest training picture; falling toward 0 is memorisation</span>`)}</div></div>
          <div class="panel"><h2>decision depth <em>which layer settles the answer</em></h2>
            <div class="body"><div class="imri-chart small">${svgLine([series("commit_layer", COLORS[1], "decides at layer")], { y0: 0, y1: Math.max(1, layers) })}</div>
            ${legend(`<span>early on the last layers do all the work; as the net learns, the answer forms earlier</span>`)}</div></div>
          <div class="panel"><h2>codes in use <em>a collapse shows as a handful</em></h2>
            <div class="body"><div class="imri-chart small">${svgLine([series("codes_used", COLORS[2], "codes")], { y0: 0, y1: 255 })}</div></div></div>
          <div class="panel"><h2>codec ceiling <em>the codec's own reconstruction</em></h2>
            <div class="body"><div class="imri-fig">${photo("recon.png", 4, "codec reconstruction", needVal)}
            ${legend(`<span>the model cannot draw sharper than this; blur here is the codec's, not the model's</span>`)}</div></div></div>
        </div>
      </div>`;

      // scrubber + films (event handlers live on the elements; the strip buttons use delegation)
      const q = sel => box.querySelector(sel);
      const cont = q("[data-imri-continue]");
      if (cont) cont.addEventListener("click", () => opts.onContinue(name));
      const range = q("[data-imri-range]");
      if (range) range.addEventListener("input", () => { st.follow = false; stepTo(parseInt(range.value, 10)); });
      const prevBtn = q("[data-imri-prev]"), nextBtn = q("[data-imri-next]"), playBtn = q("[data-imri-play]"), follow = q("[data-imri-follow]");
      if (prevBtn) prevBtn.addEventListener("click", () => { st.follow = false; stepTo(curIdx - 1); });
      if (nextBtn) nextBtn.addEventListener("click", () => { st.follow = false; stepTo(curIdx + 1); });
      if (playBtn) playBtn.addEventListener("click", () => { play(!st.playing); render(); });
      if (follow) follow.addEventListener("change", () => { st.follow = follow.checked; if (follow.checked) stepTo(steps.length - 1); });
      const sampleSel = q("[data-imri-sample]"), fillSel = q("[data-imri-fillpic]");
      if (sampleSel) sampleSel.addEventListener("change", () => { st.sample = parseInt(sampleSel.value, 10) || 0; render(); });
      if (fillSel) fillSel.addEventListener("change", () => { st.fillPic = parseInt(fillSel.value, 10) || 0; render(); });
      const filmS = q("[data-imri-film-samples]"), filmF = q("[data-imri-film-fill]");
      if (filmS) {
        const outlines = steps.map((s, i) => { if (!i) return null; const c = churnBetween(steps[i - 1], s); return c && c.changed[st.sample] ? c.changed[st.sample] : null; });
        filmstrip(filmS, steps.map(s => ({ s, col: st.sample, row: 0 })), name, "samples.png", gh, gw, outlines);
      }
      if (filmF) {
        // the original once (left), then the completion at every probe
        const frames = [{ s: cur, col: 0, row: st.fillPic, label: "original" }].concat(steps.map(s => ({ s, col: 2, row: st.fillPic })));
        filmstrip(filmF, frames, name, "fill.png", gh, gw, null);
      }
    }

    function stepTo(idx) {
      const steps = (st.data && st.data.steps) || [];
      if (!steps.length) return;
      idx = Math.max(0, Math.min(steps.length - 1, idx));
      st.step = steps[idx].step;
      st.follow = idx === steps.length - 1 && st.follow;
      render();
    }

    function play(on) {
      if (st.playTimer) { clearInterval(st.playTimer); st.playTimer = null; }
      st.playing = on;
      if (!on) return;
      const steps = (st.data && st.data.steps) || [];
      let idx = steps.findIndex(s => s.step === st.step);
      if (idx >= steps.length - 1) idx = -1;
      st.follow = false;
      st.playTimer = setInterval(() => {
        const all = (st.data && st.data.steps) || [];
        idx++;
        if (idx >= all.length) { play(false); render(); return; }
        st.step = all[idx].step;
        render();
      }, PLAY_MS);
    }

    // A filmstrip: one tile out of one probe png per frame, left to right, drawn at FILM_TILE
    // css px on a canvas sized for the display's pixel ratio (crisp on 2x). `frames` are
    // {s: step record, col, row, label?}; `outlines[i]` (a gh*gw cell mask) outlines the cells
    // that changed since the frame before. Click a frame to jump to its checkpoint.
    function filmstrip(canvas, frames, name, file, gh, gw, outlines) {
      const T = FILM_TILE, G = FILM_GAP, n = frames.length;
      const dpr = window.devicePixelRatio || 1;
      const Wc = n * (T + G) + G, Hc = T + 2 * G + 14;
      canvas.width = Math.round(Wc * dpr); canvas.height = Math.round(Hc * dpr);
      canvas.style.width = Wc + "px"; canvas.style.height = Hc + "px";
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#0e1016";
      ctx.fillRect(0, 0, Wc, Hc);
      frames.forEach((f, i) => {
        const s = f.s, x = G + i * (T + G);
        ctx.fillStyle = "#6f7480"; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
        ctx.fillText(f.label || String(s.step), x + T / 2, Hc - 3);
        if (!(s.files || []).includes(file)) return;
        const tile = s.thumb || 96, sg = s.gap || 4;
        loadImage(`/images/mri/${encodeURIComponent(name)}/${s.step}/${file}?${s.step}`, img => {
          ctx.drawImage(img, sg + f.col * (tile + sg), sg + f.row * (tile + sg), tile, tile, x, G, T, T);
          if (outlines && outlines[i] && gh && gw) {
            ctx.strokeStyle = "rgba(255,174,93,0.95)"; ctx.lineWidth = 1;
            const cw = T / gw, chh = T / gh;
            for (let c = 0; c < gh * gw; c++) if (outlines[i][c]) ctx.strokeRect(x + (c % gw) * cw + 0.5, G + Math.floor(c / gw) * chh + 0.5, cw - 1, chh - 1);
          }
          if (!f.label && s.step === st.step) { ctx.strokeStyle = "#5db8ff"; ctx.lineWidth = 2; ctx.strokeRect(x + 1, G + 1, T - 2, T - 2); }
        });
      });
      canvas.onclick = e => {
        const rect = canvas.getBoundingClientRect();
        const i = Math.floor(((e.clientX - rect.left) * (Wc / rect.width) - G) / (T + G));
        if (i >= 0 && i < n && !frames[i].label) { st.follow = false; st.step = frames[i].s.step; render(); }
      };
    }

    // ---- prompt it: the same words at every checkpoint --------------------------------
    function promptMount(name, hasCheckpoints) {
      const panel = host.querySelector("[data-imri-prompt]");
      if (!panel) return;
      const ps = st.prompt;
      if (ps.model === name && panel.firstChild) { promptRender(); return; }
      if (ps.timer) { clearTimeout(ps.timer); ps.timer = null; }
      ps.model = name; ps.state = null; ps.photo = null; ps.photoName = "";
      panel.innerHTML = `<div class="imri"><div class="panel">
        <h2>prompt it <em>the same words at every checkpoint &middot; and how much the words steer</em></h2>
        <div class="body"><div class="imri-prompt">
          <textarea data-imp-text rows="2" placeholder="describe a picture &mdash; the model draws it at each saved checkpoint, same seed, so you see the words take hold over training"></textarea>
          <div class="ctl">
            <label>seed <input type="number" data-imp-seed value="0"></label>
            <label>passes <input type="number" data-imp-passes value="8" min="1" max="64"></label>
            <label style="cursor:pointer">photo <input type="file" accept="image/*" data-imp-file style="width:170px"></label><span data-imp-photo></span>
            <button type="button" data-imp-one ${hasCheckpoints ? "" : "disabled"}>draw at this checkpoint</button>
            <button type="button" class="primary" data-imp-all ${hasCheckpoints ? "" : "disabled"}>draw at every checkpoint</button>
            <button type="button" data-imp-stop style="display:none">stop</button>
            <span data-imp-status class="meta"></span>
          </div>
          <div data-imp-results></div>
        </div></div></div></div>`;
      const q = sel => panel.querySelector(sel);
      q("[data-imp-file]").addEventListener("change", e => {
        const f = e.target.files && e.target.files[0];
        if (!f) { ps.photo = null; ps.photoName = ""; q("[data-imp-photo]").textContent = ""; return; }
        const reader = new FileReader();
        reader.onload = () => { ps.photo = String(reader.result); ps.photoName = f.name; q("[data-imp-photo]").textContent = `${f.name} attached: variation of it, guided by the words`; };
        reader.readAsDataURL(f);
      });
      q("[data-imp-one]").addEventListener("click", () => promptStart([st.step]));
      q("[data-imp-all]").addEventListener("click", () => promptStart(null));
      q("[data-imp-stop]").addEventListener("click", () => { fetch(`/images/mri/${encodeURIComponent(name)}/prompt/stop`, { method: "POST" }).catch(() => {}); });
      promptRender();
    }

    function promptStart(steps) {
      const panel = host.querySelector("[data-imri-prompt]");
      const ps = st.prompt, name = ps.model;
      if (!panel || !name) return;
      const q = sel => panel.querySelector(sel);
      const caption = (q("[data-imp-text]").value || "").trim();
      const body = { caption, seed: parseInt(q("[data-imp-seed]").value, 10) || 0, passes: parseInt(q("[data-imp-passes]").value, 10) || 8,
                     mode: ps.photo ? "variation" : "text", image: ps.photo || undefined, steps: steps && steps[0] != null ? steps : undefined };
      if (!caption && !ps.photo) { q("[data-imp-status]").textContent = "type some words (or attach a photo) first"; return; }
      q("[data-imp-status]").textContent = "starting…";
      fetch(`/images/mri/${encodeURIComponent(name)}/prompt`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
        .then(r => r.json()).then(d => {
          if (!d.ok) { q("[data-imp-status]").textContent = d.error || "could not start"; return; }
          ps.state = d.state; promptRender(); promptPoll();
        }).catch(e => { q("[data-imp-status]").textContent = "failed: " + e; });
    }

    function promptPoll() {
      const ps = st.prompt, name = ps.model;
      if (ps.timer) { clearTimeout(ps.timer); ps.timer = null; }
      if (!name) return;
      fetch(`/images/mri/${encodeURIComponent(name)}/prompt/status?` + Date.now(), { cache: "no-store" }).then(r => r.json()).then(d => {
        if (st.prompt.model !== name) return;
        ps.state = d.state || null;
        promptRender();
        if (ps.state && ps.state.status === "running") ps.timer = setTimeout(promptPoll, PROMPT_POLL_MS);
      }).catch(() => {});
    }

    function promptRender() {
      const panel = host.querySelector("[data-imri-prompt]");
      const ps = st.prompt;
      if (!panel) return;
      const q = sel => panel.querySelector(sel);
      const job = ps.state, status = q("[data-imp-status]"), out = q("[data-imp-results]"), stop = q("[data-imp-stop]");
      if (!status || !out) return;
      const running = !!(job && job.status === "running");
      if (stop) stop.style.display = running ? "" : "none";
      ["[data-imp-one]", "[data-imp-all]"].forEach(sel => { const b = q(sel); if (b) b.disabled = running; });
      if (!job || job.status === "idle") { status.textContent = ""; out.innerHTML = ""; return; }
      const results = job.results || [];
      const total = (job.steps || []).length;
      status.textContent = running ? `drawing ${results.length} / ${total} checkpoints… (two pictures each: with the words and without)`
        : job.status === "failed" ? `failed: ${job.error}` : `${results.length} checkpoint${results.length === 1 ? "" : "s"} &middot; seed ${job.seed} &middot; ${job.passes} passes${job.caption ? ` &middot; "${job.caption}"` : ""}`.replace(/&middot;/g, "·");
      const pct = v => (v * 100).toFixed(0) + "%";
      const steer = results.map(r => [r.step, r.steering]);
      out.innerHTML = `<div class="results">${results.map(r => `<div class="res ${r.step === st.step ? "on" : ""}" data-imp-step="${r.step}" title="step ${r.step}: with the words (large), without them (small), same seed">
          <img src="data:image/png;base64,${r.png}" alt="with words"><img class="small" src="data:image/png;base64,${r.uncond_png}" alt="without words">
          <span>step ${Number(r.step).toLocaleString()}${job.caption ? ` &middot; words moved ${pct(r.steering)}` : ""}</span></div>`).join("")}</div>
        ${job.caption && results.length > 1 ? `<div class="imri-chart">${svgLine([{ name: "steering", color: COLORS[6], points: steer }], { y0: 0, y1: 1 })}</div>
        <div class="imri-legend"><span><i style="background:${COLORS[6]}"></i>caption influence: share of cells the words changed against the same seed without them. Rising over training means the model is learning to listen; flat near 0 means the captions it trained on did not teach it these words.</span></div>` : ""}`;
      out.querySelectorAll("[data-imp-step]").forEach(el => el.addEventListener("click", () => { st.follow = false; st.step = parseInt(el.dataset.impStep, 10); render(); }));
    }

    host.addEventListener("click", e => {
      const btn = e.target && e.target.closest ? e.target.closest("[data-imri-step]") : null;
      if (!btn || !st.active) return;
      st.step = parseInt(btn.dataset.imriStep, 10);
      render();
    });

    return { show, hide, render, get active() { return st.active; }, get model() { return st.model; },
             get step() { return st.step; } };
  }

  window.ImageMri = { create, colors: COLORS,
                      charts: { svgLine, barChart, cellHeat, scaleBar, photoTile, gridRows, wireLightbox } };
})();
