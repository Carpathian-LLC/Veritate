/* Developed by Carpathian, LLC. Distribution Not Authorized. */
/* veritate_mri/static/index.js */

const $ = id => document.getElementById(id);

// ---- canvases ----
const cFfn  = $("cFfn"),  ctxFfn  = cFfn.getContext("2d");
const cTop  = $("cTop"),  ctxTop  = cTop.getContext("2d");
const cTel    = $("cTel"),    ctxTel    = cTel.getContext("2d");
const cFlow   = $("cFlow"),   ctxFlow   = cFlow.getContext("2d");
const cLetMs  = $("cLetMs"),  ctxLetMs  = cLetMs.getContext("2d");
const cDecisive = $("cDecisive"), ctxDecisive = cDecisive.getContext("2d");
const cConfBar   = $("cConfBar"),   ctxConfBar   = cConfBar.getContext("2d");
const cConfTrend = $("cConfTrend"), ctxConfTrend = cConfTrend.getContext("2d");

// ---- state ----
let evtSrc = null;
let meta = null;
let frames = [];
let currentFrame = -1;
let live = true;
let promptBytes = [];
let generatedBytes = [];
let coactState = { pairs: new Map(), nFrames: 0 };

// ---- utility ----

// Friendly message for the two errors the UI sees when the Python backend
// is offline: WebKit fetch failure ("TypeError: Load failed" / "Failed to fetch")
// and the SyntaxError thrown when response.json() runs on an empty/HTML body.
function _backendErrMsg(e) {
  const s = String(e && e.message || e || "");
  if (e instanceof TypeError && /load failed|failed to fetch|networkerror/i.test(s)) {
    return "backend offline. relaunch with python run.py";
  }
  if (e instanceof SyntaxError) {
    return "backend offline or returned non-JSON. relaunch with python run.py";
  }
  return s || String(e);
}

function fitCanvas(c) {
  if (c.offsetParent === null) return { w: 0, h: 0 };
  const dpr = window.devicePixelRatio || 1;
  const cssW = c.clientWidth, cssH = c.clientHeight || parseInt(c.getAttribute("height"), 10);
  c.width = Math.floor(cssW * dpr); c.height = Math.floor(cssH * dpr);
  c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w: cssW, h: cssH };
}

function colorRamp(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.floor(8 + (255 - 8) * Math.pow(t, 1.4));
  const g = Math.floor(12 + (210 - 12) * Math.pow(t, 1.4));
  const b = Math.floor(20 + (255 - 20) * Math.pow(t, 0.85));
  return `rgb(${r},${g},${b})`;
}

// region-aware color ramp: cool blue for sensory (L0-L3),
// warm orange for association (L4-L8), hot red for output (L9-L11)
function regionRamp(t, layer) {
  t = Math.max(0, Math.min(1, t));
  const region = regionLabel(layer).cls;
  if (region === "b-sense") {
    const r = Math.floor(18 + (110 - 18) * Math.pow(t, 1.5));
    const g = Math.floor(34 + (210 - 34) * Math.pow(t, 1.2));
    const b = Math.floor(50 + (255 - 50) * Math.pow(t, 0.85));
    return `rgb(${r},${g},${b})`;
  }
  if (region === "b-assoc") {
    const r = Math.floor(50 + (255 - 50) * Math.pow(t, 0.85));
    const g = Math.floor(34 + (180 - 34) * Math.pow(t, 1.3));
    const b = Math.floor(18 + (90 - 18)  * Math.pow(t, 1.6));
    return `rgb(${r},${g},${b})`;
  }
  const r = Math.floor(60 + (255 - 60) * Math.pow(t, 0.85));
  const g = Math.floor(20 + (95 - 20)  * Math.pow(t, 1.6));
  const b = Math.floor(28 + (95 - 28)  * Math.pow(t, 1.6));
  return `rgb(${r},${g},${b})`;
}

let _layerCount = 12;
function setLayerCount(n) {
  if (!n || n < 1) return;
  _layerCount = n;
  _buildFfnLegend(n);
}
function _regionBounds(n) {
  const s = Math.max(1, Math.round(n / 3));
  return { sense_end: s - 1, assoc_end: n - s - 1, total: n };
}
function _buildFfnLegend(n) {
  const el = $("ffnRegionLegend");
  if (!el) return;
  const b = _regionBounds(n);
  const range = (a, c) => a === c ? `L${a}` : `L${a}&ndash;L${c}`;
  el.innerHTML = `
    <span class="leg b-sense"><b>${range(0, b.sense_end)} sensory</b><span class="range">cool blue</span><em>raw byte features, surface patterns</em></span>
    <span class="leg b-assoc"><b>${range(b.sense_end + 1, b.assoc_end)} association</b><span class="range">warm orange</span><em>concepts, syntax, semantics</em></span>
    <span class="leg b-out"><b>${range(b.assoc_end + 1, n - 1)} output</b><span class="range">hot red</span><em>commits to a specific next byte</em></span>
  `;
}

function regionLabel(layer) {
  const b = _regionBounds(_layerCount);
  if (layer <= b.sense_end) return { name: "sensory", cls: "b-sense" };
  if (layer <= b.assoc_end) return { name: "association", cls: "b-assoc" };
  return { name: "output", cls: "b-out" };
}

function glyphFor(b) {
  if (b === 10) return { txt: "↵",  cls: "glyph nl" };
  if (b === 32) return { txt: "␣",  cls: "glyph np" };
  if (b === 9)  return { txt: "→",  cls: "glyph nl" };
  if (b >= 33 && b < 127) return { txt: String.fromCharCode(b), cls: "glyph" };
  return { txt: b.toString(16).padStart(2,"0"), cls: "glyph np" };
}

function byteToCh(b) {
  if (b === 10) return "\n";
  if (b >= 32 && b < 127) return String.fromCharCode(b);
  return "·";
}

function tapeChar(b) {
  if (b === 10) return "↵";
  if (b === 32) return "·";
  if (b === 9)  return "→";
  if (b >= 33 && b < 127) return String.fromCharCode(b);
  return "·";
}

function escapeTape(s) {
  return s.replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
}

function updateScrubTape(elId, frames, currentFrame, promptBytes) {
  const el = $(elId);
  if (!el) return;
  if (currentFrame < 0 || !frames || !frames.length) {
    el.innerHTML = `<span class="tape-empty">no token selected</span>`;
    return;
  }
  const ctxN = 14;
  const generatedBs = frames.map(f => f.byte);
  const tape = (promptBytes || []).concat(generatedBs.slice(0, currentFrame + 1));
  const tail = tape.slice(Math.max(0, tape.length - ctxN));
  const beforeCurrent = tail.slice(0, -1);
  const cur = tail[tail.length - 1];
  const nextFrame = frames[currentFrame + 1];
  const nextByte = nextFrame ? nextFrame.byte : null;
  const ctxStr = beforeCurrent.map(tapeChar).join("");
  let html = `<span class="tape-ctx">${escapeTape(ctxStr)}</span>` +
             `<span class="tape-cur">${escapeTape(tapeChar(cur))}</span>`;
  if (nextByte !== null) {
    html += `<span class="tape-arr">→</span>` +
            `<span class="tape-next">${escapeTape(tapeChar(nextByte))}</span>`;
  } else {
    html += `<span class="tape-arr">→</span><span class="tape-next">?</span>`;
  }
  el.innerHTML = html;
}

// ---- drawers (parameterized by canvas/element so multiple tabs can reuse) ----
function drawFfn(c, ctx, ffnFull) {
  const { w, h } = fitCanvas(c);
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);
  const L = ffnFull.length, B = ffnFull[0].length;
  const padL = 24;
  const cellW = (w - padL) / B, cellH = h / L;
  const hover = c.__hover;
  for (let l = 0; l < L; l++) {
    for (let b = 0; b < B; b++) {
      const isHover = hover && hover.layer === l && hover.bucket === b;
      if (isHover) {
        ctx.shadowColor = "#39ff14"; ctx.shadowBlur = 8;
        ctx.fillStyle = "#39ff14";
      } else {
        ctx.fillStyle = regionRamp(ffnFull[l][b] / 255, l);
      }
      ctx.fillRect(padL + b * cellW, l * cellH + 1, Math.max(1, cellW - 0.3), Math.max(1, cellH - 2));
      if (isHover) ctx.shadowBlur = 0;
    }
    // region-tinted layer label
    const r = regionLabel(l);
    ctx.fillStyle = r.cls === "b-sense" ? "#5dc8ff" : r.cls === "b-assoc" ? "#ffae5d" : "#ff5d5d";
    ctx.font = "9px ui-monospace,monospace";
    ctx.fillText("L" + l, 4, l * cellH + cellH * 0.65);
  }
}

function attachFfnHover(c, ctx, getFrame) {
  c.style.cursor = "pointer";
  c.addEventListener("mousemove", (e) => {
    const frame = getFrame();
    if (!frame || !frame.ffn_full || !frame.ffn_full.length) return;
    const rect = c.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const padL = 24;
    const L = frame.ffn_full.length, B = frame.ffn_full[0].length;
    const cellW = (rect.width - padL) / B, cellH = rect.height / L;
    let layer = -1, bucket = -1;
    if (x >= padL && y >= 0 && y < rect.height) {
      layer = Math.floor(y / cellH);
      bucket = Math.floor((x - padL) / cellW);
      if (layer < 0 || layer >= L || bucket < 0 || bucket >= B) { layer = -1; bucket = -1; }
    }
    const next = (layer < 0) ? null : { layer, bucket };
    const prev = c.__hover;
    if ((!prev && !next) || (prev && next && prev.layer === next.layer && prev.bucket === next.bucket)) return;
    c.__hover = next;
    drawFfn(c, ctx, frame.ffn_full);
  });
  c.addEventListener("mouseleave", () => {
    if (!c.__hover) return;
    c.__hover = null;
    const frame = getFrame();
    if (frame && frame.ffn_full && frame.ffn_full.length) drawFfn(c, ctx, frame.ffn_full);
  });
}

function drawSaturation(c, ctx, sat) {
  const { w, h } = fitCanvas(c);
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);
  if (!sat || !sat.length) {
    ctx.fillStyle = "#8a8f9a";
    ctx.font = "12px ui-monospace,monospace";
    ctx.fillText("post-GELU saturation not measured for this checkpoint.", 12, 22);
    ctx.fillStyle = "#6f7480"; ctx.font = "11px ui-monospace,monospace";
    ctx.fillText("Per-token frames are absent — only probe-step neuron snapshots exist.", 12, 42);
    ctx.fillText("To populate: re-run training with dump_generation enabled (Rule 4 dumps).", 12, 60);
    return;
  }
  const L = sat.length;
  const maxSat = Math.max(...sat);
  // happy path when QAT did its job: every layer is well below the int8 budget.
  // a near-black grid with twelve "0.000%" labels reads as broken; surface the
  // good news instead.
  if (maxSat <= 0) {
    ctx.fillStyle = "#10241a"; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#5dff9b"; ctx.font = "13px ui-monospace,monospace";
    ctx.fillText("0% across all " + L + " layers — no INT8 clipping pressure", 14, 26);
    ctx.fillStyle = "#a8e8be"; ctx.font = "11px ui-monospace,monospace";
    ctx.fillText("post-GELU activations all sit inside the ±3.97 INT8 budget.", 14, 48);
    ctx.fillText("This is the QAT happy path: quantization is essentially free here.", 14, 66);
    ctx.fillStyle = "#6f7480"; ctx.font = "10px ui-monospace,monospace";
    ctx.fillText("Threshold = 127 / scale 32 = 3.97. Higher saturation = pressure to clip.", 14, h - 10);
    return;
  }
  const padL = 32, padR = 70;
  const cellH = h / L;
  const norm = Math.max(0.01, maxSat);
  for (let l = 0; l < L; l++) {
    const t = Math.min(1, sat[l] / norm);
    const r = Math.floor(20 + 230 * Math.pow(t, 0.85));
    const g = Math.floor(20 + 60 * Math.pow(t, 1.5));
    const b = Math.floor(30 + 10 * Math.pow(t, 1.2));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(padL, l * cellH + 1, w - padL - padR, Math.max(1, cellH - 2));
    const region = regionLabel(l);
    ctx.fillStyle = region.cls === "b-sense" ? "#5dc8ff" : region.cls === "b-assoc" ? "#ffae5d" : "#ff5d5d";
    ctx.font = "9px ui-monospace,monospace";
    ctx.fillText("L" + l, 4, l * cellH + cellH * 0.65);
    ctx.fillStyle = "#d6d8db";
    ctx.font = "10px ui-monospace,monospace";
    ctx.fillText((sat[l] * 100).toFixed(3) + "%", w - padR + 4, l * cellH + cellH * 0.65);
  }
}

function drawQuantKl(c, ctx, checkpoints, currentIdx) {
  const { w, h } = fitCanvas(c);
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);
  const points = (checkpoints || []).map((ck, i) => ({
    idx: i,
    step: (typeof ck.effective_step === "number") ? ck.effective_step : ck.step,
    kl: (typeof ck.quant_kl_bits === "number") ? ck.quant_kl_bits : null,
    qat: !!ck.is_qat,
  }));
  const valid = points.filter(p => p.kl !== null);
  if (valid.length === 0) {
    ctx.fillStyle = "#8a8f9a"; ctx.font = "12px ui-monospace,monospace";
    ctx.fillText("FP32 vs INT8 logit divergence not yet computed for this run.", 12, 24);
    ctx.fillStyle = "#6f7480"; ctx.font = "11px ui-monospace,monospace";
    ctx.fillText("Each checkpoint needs both an FP32 forward and a post-hoc INT8 quantized", 12, 46);
    ctx.fillText("forward; the training-time probe doesn't run the INT8 pass.", 12, 62);
    ctx.fillStyle = "#6aa6ff"; ctx.font = "11px ui-monospace,monospace";
    ctx.fillText("(quant_kl artifact is written by the unified save helper at each checkpoint.)", 12, 88);
    ctx.fillStyle = "#6f7480"; ctx.font = "10px ui-monospace,monospace";
    ctx.fillText("(populates quant_kl_bits per checkpoint; the chart picks it up on next reload)", 12, h - 10);
    return;
  }
  const padL = 44, padR = 14, padT = 10, padB = 22;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const minStep = Math.min(...valid.map(p => p.step));
  const maxStep = Math.max(...valid.map(p => p.step));
  const stepRange = Math.max(1, maxStep - minStep);
  const minKl = 0;
  const maxKl = Math.max(0.05, ...valid.map(p => p.kl)) * 1.1;
  const xOf = step => padL + (plotW * (step - minStep) / stepRange);
  const yOf = kl   => padT + plotH * (1 - (kl - minKl) / (maxKl - minKl));
  // gridlines
  ctx.strokeStyle = "#15192a"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (plotH * i / 4);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    const klVal = maxKl - (maxKl - minKl) * i / 4;
    ctx.fillStyle = "#6f7480"; ctx.font = "9px ui-monospace,monospace";
    ctx.fillText(klVal.toFixed(2), 4, y + 3);
  }
  // axis labels
  ctx.fillStyle = "#6f7480"; ctx.font = "9px ui-monospace,monospace";
  ctx.fillText("KL bits", 4, padT - 2);
  ctx.fillText("step " + minStep.toLocaleString(), padL, h - 6);
  const lastLabel = "step " + maxStep.toLocaleString();
  ctx.fillText(lastLabel, padL + plotW - ctx.measureText(lastLabel).width, h - 6);
  // line
  ctx.strokeStyle = "#6aa6ff"; ctx.lineWidth = 1.5;
  ctx.beginPath();
  valid.forEach((p, i) => {
    const x = xOf(p.step), y = yOf(p.kl);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // current-checkpoint vertical bar
  if (currentIdx >= 0 && currentIdx < points.length) {
    const cur = points[currentIdx];
    if (cur.kl !== null) {
      const cx = xOf(cur.step);
      ctx.strokeStyle = "rgba(93, 255, 155, 0.55)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, padT); ctx.lineTo(cx, padT + plotH); ctx.stroke();
    }
  }
  // points (dot color = QAT vs FP32)
  for (const p of valid) {
    const x = xOf(p.step), y = yOf(p.kl);
    ctx.fillStyle = (p.idx === currentIdx) ? "#5dff9b" : (p.qat ? "#ffae5d" : "#5dc8ff");
    ctx.beginPath(); ctx.arc(x, y, p.idx === currentIdx ? 4 : 3, 0, Math.PI * 2); ctx.fill();
  }
  // store geometry for click handler
  c.__klGeom = { points: valid, padL, padR, plotW, padT, plotH, minStep, stepRange };
}

function attachQuantKlClick(c, onPick) {
  c.style.cursor = "pointer";
  c.addEventListener("click", e => {
    const g = c.__klGeom;
    if (!g) return;
    const rect = c.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const step = g.minStep + (x - g.padL) / g.plotW * g.stepRange;
    let best = g.points[0], bestDist = Infinity;
    for (const p of g.points) {
      const d = Math.abs(p.step - step);
      if (d < bestDist) { bestDist = d; best = p; }
    }
    onPick(best.idx);
  });
}

function drawTopNeurons(c, ctx, ffnTop) {
  const { w, h } = fitCanvas(c);
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);
  const L = ffnTop.length, K = ffnTop[0].length;
  const padL = 28;
  const cellW = (w - padL) / K, cellH = h / L;
  let maxV = 0.001;
  for (const layer of ffnTop) for (const n of layer) if (n.v > maxV) maxV = n.v;
  const hover = c.__hover;
  for (let l = 0; l < L; l++) {
    const r = regionLabel(l);
    ctx.fillStyle = r.cls === "b-sense" ? "#5dc8ff" : r.cls === "b-assoc" ? "#ffae5d" : "#ff5d5d";
    ctx.font = "9px ui-monospace,monospace";
    ctx.fillText("L" + l, 4, l * cellH + cellH * 0.65);
    for (let k = 0; k < K; k++) {
      const n = ffnTop[l][k];
      const t = n.v / maxV;
      const isHover = hover && hover.layer === l && hover.k === k;
      if (isHover) {
        ctx.shadowColor = "#39ff14"; ctx.shadowBlur = 10;
        ctx.fillStyle = "#39ff14";
      } else {
        ctx.fillStyle = regionRamp(t, l);
      }
      ctx.fillRect(padL + k * cellW + 1, l * cellH + 1, cellW - 2, cellH - 2);
      if (isHover) ctx.shadowBlur = 0;
      if (cellW > 50 && cellH > 18) {
        ctx.fillStyle = isHover ? "#000" : (t > 0.55 ? "#000" : "#9aa4b5");
        ctx.font = isHover ? "bold 10px ui-monospace,monospace" : "10px ui-monospace,monospace";
        ctx.fillText("#" + n.id, padL + k * cellW + 4, l * cellH + cellH * 0.7);
      }
    }
  }
}

function attachTopNeuronsHover(c, ctx, getFrame) {
  c.style.cursor = "pointer";
  c.addEventListener("mousemove", (e) => {
    const frame = getFrame();
    if (!frame || !frame.ffn_top || !frame.ffn_top.length) return;
    const rect = c.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const padL = 28;
    const L = frame.ffn_top.length, K = frame.ffn_top[0].length;
    const cellW = (rect.width - padL) / K, cellH = rect.height / L;
    let layer = -1, k = -1;
    if (x >= padL && y >= 0 && y < rect.height) {
      layer = Math.floor(y / cellH);
      k = Math.floor((x - padL) / cellW);
      if (layer < 0 || layer >= L || k < 0 || k >= K) { layer = -1; k = -1; }
    }
    const next = (layer < 0) ? null : { layer, k };
    const prev = c.__hover;
    if ((!prev && !next) || (prev && next && prev.layer === next.layer && prev.k === next.k)) return;
    c.__hover = next;
    drawTopNeurons(c, ctx, frame.ffn_top);
  });
  c.addEventListener("mouseleave", () => {
    if (!c.__hover) return;
    c.__hover = null;
    const frame = getFrame();
    if (frame && frame.ffn_top && frame.ffn_top.length) drawTopNeurons(c, ctx, frame.ffn_top);
  });
}

function drawLetterMs(c, ctx, allFrames, currentIdx, allBytes, promptLen) {
  const { w, h } = fitCanvas(c);
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);
  if (allFrames.length < 1) {
    ctx.fillStyle = "#6f7480"; ctx.font = "11px ui-monospace,monospace";
    ctx.fillText("no frames yet", 8, h / 2);
    return;
  }
  const N = allFrames.length;
  const padL = 36, padT = 8, padB = 18;
  const plotW = w - padL - 8, plotH = h - padT - padB;
  let maxMs = 0.001;
  for (const f of allFrames) if (f.fwd_ms > maxMs) maxMs = f.fwd_ms;
  ctx.strokeStyle = "#171b24"; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = padT + (plotH * g) / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
  }
  ctx.fillStyle = "#6f7480"; ctx.font = "10px ui-monospace,monospace";
  for (let g = 0; g <= 4; g++) {
    const ms = maxMs * (1 - g / 4);
    const y = padT + (plotH * g) / 4;
    ctx.fillText(ms.toFixed(1), 4, y + 3);
  }
  const barW = Math.max(1, plotW / N);
  for (let i = 0; i < N; i++) {
    const v = allFrames[i].fwd_ms / maxMs;
    const bh = Math.max(1, plotH * v);
    const x = padL + i * barW;
    const y = padT + plotH - bh;
    ctx.fillStyle = i === currentIdx ? "#39ff14" : "#3a4660";
    ctx.fillRect(x, y, Math.max(1, barW - 0.5), bh);
  }
  if (currentIdx >= 0 && currentIdx < N) {
    const x = padL + currentIdx * barW + barW / 2;
    ctx.shadowColor = "#39ff14"; ctx.shadowBlur = 8;
    ctx.strokeStyle = "#39ff14"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    ctx.shadowBlur = 0;
  }
  ctx.fillStyle = "#6f7480"; ctx.font = "10px ui-monospace,monospace";
  ctx.fillText("ms per byte (max " + maxMs.toFixed(1) + ")", padL, h - 4);
}

function updateLetterStats(frames, idx, generatedBs) {
  if (idx < 0 || idx >= frames.length) {
    $("lsByte").textContent = "—";
    $("lsMs").textContent = "—";
    $("lsSurp").textContent = "—";
    $("lsEnt").textContent = "—";
    $("lsFrame").textContent = "—";
    return;
  }
  const f = frames[idx];
  const b = generatedBs[idx] ?? f.byte ?? 0;
  let ch = byteToCh(b);
  if (b === 10) ch = "↵";
  else if (b === 32) ch = "␣";
  else if (b === 9) ch = "→";
  $("lsByte").textContent = `${ch}  (0x${b.toString(16).padStart(2,"0")})`;
  $("lsMs").textContent   = f.fwd_ms.toFixed(2) + " ms";
  $("lsSurp").textContent = f.surprise_bits != null ? f.surprise_bits.toFixed(2) : "—";
  $("lsEnt").textContent  = f.entropy_bits != null ? f.entropy_bits.toFixed(2) : "—";
  $("lsFrame").textContent = `${idx + 1} / ${frames.length}`;
}

function drawTelemetry(c, ctx, allFrames, currentIdx) {
  const { w, h } = fitCanvas(c);
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);
  if (allFrames.length < 1) return;
  const N = allFrames.length;
  const series = ["surprise_bits", "entropy_bits", "fwd_ms"];
  const stroke = { surprise_bits: "#ff5d5d", entropy_bits: "#5dc8ff", fwd_ms: "#9aa4b5" };
  const padL = 36, padT = 8, padB = 18;
  const plotW = w - padL - 8, plotH = h - padT - padB;
  ctx.strokeStyle = "#171b24"; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = padT + (plotH * g) / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
  }
  for (const s of series) {
    let maxV = 0.001;
    for (const f of allFrames) if (f[s] > maxV) maxV = f[s];
    ctx.strokeStyle = stroke[s]; ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = padL + (N === 1 ? plotW / 2 : (plotW * i) / (N - 1));
      const y = padT + plotH * (1 - allFrames[i][s] / maxV);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  if (currentIdx >= 0) {
    const x = padL + (N === 1 ? plotW / 2 : (plotW * currentIdx) / (N - 1));
    ctx.shadowColor = "#39ff14"; ctx.shadowBlur = 8;
    ctx.strokeStyle = "#39ff14"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    ctx.shadowBlur = 0;
  }
  ctx.fillStyle = "#6f7480"; ctx.font = "10px ui-monospace,monospace";
  ctx.fillText("token →", padL, h - 4);
  ctx.fillStyle = "#ff5d5d"; ctx.fillText("● surprise",     w - 270, 12);
  ctx.fillStyle = "#5dc8ff"; ctx.fillText("● uncertainty",  w - 180, 12);
  ctx.fillStyle = "#9aa4b5"; ctx.fillText("● latency",      w - 80,  12);
}

function drawFlow(c, ctx, infoFlow, T, allBytes) {
  // wrap to multiple rows so the byte text never disappears as the sequence grows.
  const cssW = c.clientWidth || c.parentElement.clientWidth || 800;
  const padX = 4;
  const minCellW = 14;
  const maxCellW = 22;
  const cellsPerRow = Math.max(1, Math.min(T, Math.floor((cssW - padX * 2) / minCellW)));
  const rows = Math.max(1, Math.ceil(T / cellsPerRow));
  const cellH = 22;
  const rowGap = 4;
  const padTop = 4;
  const padBot = 6;
  const targetH = padTop + rows * cellH + Math.max(0, rows - 1) * rowGap + padBot;
  if (Math.abs((c.clientHeight || 0) - targetH) > 1) c.style.height = targetH + "px";

  const { w, h } = fitCanvas(c);
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);

  const weights = new Array(T).fill(0);
  for (const e of infoFlow) weights[e.p] = Math.max(weights[e.p], e.w);
  const startIdx = Math.max(0, allBytes.length - T);

  const cellW = Math.min(maxCellW, Math.max(minCellW, Math.floor((w - padX * 2) / cellsPerRow)));
  for (let p = 0; p < T; p++) {
    const row = Math.floor(p / cellsPerRow);
    const col = p % cellsPerRow;
    const x = padX + col * cellW;
    const y = padTop + row * (cellH + rowGap);
    ctx.fillStyle = colorRamp(weights[p]);
    ctx.fillRect(x, y, Math.max(1, cellW - 1), cellH);
    const b = allBytes[startIdx + p] ?? 0;
    const ch = byteToCh(b);
    const glyph = ch === "\n" ? "↵" : (ch === " " ? "·" : ch);
    ctx.fillStyle = weights[p] > 0.55 ? "#000" : "#d6d8db";
    ctx.font = "11px ui-monospace,monospace";
    const tw = ctx.measureText(glyph).width;
    ctx.fillText(glyph, x + (cellW - tw) / 2, y + cellH * 0.7);
  }
}

function drawCandidates(elId, cand) {
  const max = Math.max(...cand.map(c => c.p));
  let html = "<table>";
  for (const c of cand) {
    const g = glyphFor(c.b);
    const wPct = (c.p / max * 100).toFixed(0);
    html += `<tr>
      <td style="width:34px"><span class="${g.cls}">${g.txt}</span></td>
      <td style="width:42px;color:var(--dim)">0x${c.b.toString(16).padStart(2,"0")}</td>
      <td><span class="bar" style="width:${Math.max(2, wPct * 2.0)}px"></span></td>
      <td style="text-align:right;color:var(--dim);width:54px">${(c.p * 100).toFixed(1)}%</td>
    </tr>`;
  }
  html += "</table>";
  $(elId).innerHTML = html;
}

function drawList(elId, vals, cls) {
  const max = Math.max(...vals, 0.001);
  let html = "";
  for (let l = 0; l < vals.length; l++) {
    const wPct = vals[l] / max * 100;
    html += `<li><b>L${l}</b><span class="bar ${cls}" style="width:${Math.max(2, wPct * 1.7)}px"></span><span class="stat" style="margin-left:6px">${vals[l].toFixed(2)}</span></li>`;
  }
  $(elId).innerHTML = html;
}

function regionRowClass(layer) {
  if (layer <= 3) return "region-sense";
  if (layer <= 8) return "region-assoc";
  return "region-output";
}

function renderLabelPill(label, opts) {
  if (!label || !label.category) return "";
  const cat = label.category;
  const trig = label.trigger;
  let inner;
  let titleHint = `${cat} pattern`;
  if (cat === "single") {
    inner = `'${escapeTape(trig || "?")}'`;
    titleHint = `single byte detector — peaks land on '${trig}'`;
  } else if (cat === "word") {
    inner = `&ldquo;${escapeTape(trig || "")}&rdquo;`;
    titleHint = `word detector — peaks land inside '${trig}'`;
  } else if (cat === "bigram" || cat === "trigram" || cat === "4gram" || cat === "5gram" || cat === "6gram" || cat === "7gram") {
    // n-gram substring near peak
    inner = `~${escapeTape(trig || "")}`;
    titleHint = `${cat} pattern — '${trig}' near peak`;
  } else {
    // class label (vowel / consonant / digit / punct / whitespace)
    inner = cat;
    titleHint = `byte-class detector — peak bytes are ${cat}`;
  }
  const confStr = (opts && opts.showConf && label.confidence != null)
    ? `<span class="conf">${Math.round(label.confidence * 100)}%</span>`
    : "";
  return `<span class="neuron-label cat-${cat}" title="${titleHint}, ${Math.round((label.confidence||0)*100)}% of probe stories">${inner}${confStr}</span>`;
}

function drawDlaTable(elId, entries) {
  if (!entries || !entries.length) {
    $(elId).innerHTML = `<div style="color:var(--dim);font-size:11px;padding:8px 0">no attribution data &mdash; restart the server to enable.</div>`;
    return;
  }
  let html = `<table class="dla-table"><thead><tr>
    <th style="text-align:left">layer</th>
    <th style="text-align:left">neuron</th>
    <th>act</th>
    <th>weight</th>
    <th>contrib</th>
  </tr></thead><tbody>`;
  for (const e of entries) {
    const cls = regionRowClass(e.layer);
    const cclass = e.contrib >= 0 ? "contrib-pos" : "contrib-neg";
    const labelPill = e.label ? renderLabelPill(e.label) : "";
    html += `<tr class="${cls}" data-layer="${e.layer}" data-neuron="${e.neuron}">
      <td class="layer-cell">L${e.layer}</td>
      <td class="neuron-cell">#${e.neuron}${labelPill}</td>
      <td>${e.act.toFixed(3)}</td>
      <td>${e.w.toFixed(3)}</td>
      <td class="${cclass}">${e.contrib >= 0 ? "+" : ""}${e.contrib.toFixed(3)}</td>
    </tr>`;
  }
  html += `</tbody></table>`;
  const el = $(elId);
  el.innerHTML = html;
  el.querySelectorAll("tbody tr").forEach(tr => {
    tr.addEventListener("click", () => {
      showNeuronModal(parseInt(tr.dataset.layer, 10), parseInt(tr.dataset.neuron, 10));
    });
    tr.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      const L = parseInt(tr.dataset.layer, 10);
      const N = parseInt(tr.dataset.neuron, 10);
      ablateAndRegen(L, N);
    });
  });
}

// v8: ablate a single FFN neuron and regenerate. sets the inputs the click
// handler reads, then triggers the same #go path so behavior matches.
function ablateAndRegen(layer, neuron) {
  const lEl = $("ablLayer"), nEl = $("ablNeuron");
  if (!lEl || !nEl) return;
  lEl.value = String(layer);
  nEl.value = String(neuron);
  const goBtn = $("go");
  if (goBtn && !goBtn.disabled) goBtn.click();
}

function drawDecisionTrace(suffix, frame) {
  const pickedByteEl = $("dlaPickedByte" + suffix);
  const argmaxByteEl = $("dlaArgmaxByte" + suffix);
  if (pickedByteEl) pickedByteEl.textContent = frame.byte != null ? glyphFor(frame.byte).txt + " (0x" + frame.byte.toString(16).padStart(2,"0") + ")" : "—";
  if (argmaxByteEl) argmaxByteEl.textContent = frame.argmax_byte != null ? glyphFor(frame.argmax_byte).txt + " (0x" + frame.argmax_byte.toString(16).padStart(2,"0") + ")" : "—";
  drawDlaTable("dlaPickedTable" + suffix, frame.dla_picked || []);
  drawDlaTable("dlaArgmaxTable" + suffix, frame.dla_argmax || []);
  if (suffix === "") drawDlaCand(frame);
}

let _dlaCandSelected = 0;

function drawDlaCand(frame) {
  const tabsEl = $("dlaCandTabs");
  const tableEl = $("dlaCandTable");
  const statusEl = $("dlaCandAblationStatus");
  if (!tabsEl || !tableEl) return;
  const cand = frame.dla_cand || [];
  if (!cand.length) {
    tabsEl.innerHTML = `<span style="color:var(--dim);font-size:11px">no per-candidate DLA &mdash; engine pre-v8 or pytorch backend without v8 fields.</span>`;
    tableEl.innerHTML = "";
    if (statusEl) statusEl.textContent = "";
    return;
  }
  if (_dlaCandSelected >= cand.length) _dlaCandSelected = 0;
  let html = "";
  for (let i = 0; i < cand.length; i++) {
    const b = cand[i].b;
    const g = glyphFor(b);
    const cls = (i === _dlaCandSelected) ? "dla-cand-tab dla-cand-active" : "dla-cand-tab";
    html += `<button class="${cls}" data-i="${i}" style="padding:3px 8px;font:inherit;font-size:11px;background:${i === _dlaCandSelected ? '#1b2838' : '#0a0c12'};border:1px solid var(--line);color:var(--text);border-radius:3px;cursor:pointer">${g.txt} <span style="color:var(--dim)">0x${b.toString(16).padStart(2,'0')}</span></button>`;
  }
  tabsEl.innerHTML = html;
  tabsEl.querySelectorAll(".dla-cand-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      _dlaCandSelected = parseInt(btn.dataset.i, 10);
      drawDlaCand(frame);
    });
  });
  drawDlaTable("dlaCandTable", cand[_dlaCandSelected].entries || []);
  if (statusEl) {
    const a = frame.ablation;
    statusEl.textContent = (a && a.layer >= 0 && a.neuron >= 0)
      ? `ablation active for this token: L${a.layer} N${a.neuron}` : "";
  }
}

function confColor(v) {
  // red 0 -> yellow 0.5 -> green 1.0 ramp
  v = Math.max(0, Math.min(1, v));
  if (v < 0.5) {
    const t = v / 0.5;
    const r = 230, g = Math.round(60 + (200 - 60) * t), b = 60;
    return `rgb(${r},${g},${b})`;
  }
  const t = (v - 0.5) / 0.5;
  const r = Math.round(230 - (230 - 60) * t), g = 200, b = Math.round(60 + (90 - 60) * t);
  return `rgb(${r},${g},${b})`;
}

function drawConfidence(frame, allFrames, currentIdx) {
  const valEl = $("confValue");
  const conf = (frame && typeof frame.confidence === "number") ? frame.confidence : null;
  if (valEl) {
    if (conf == null) valEl.innerHTML = `<span style="color:var(--dim)">confidence data not present — generate a token</span>`;
    else valEl.innerHTML = `confidence: <b style="color:${confColor(conf)}">${(conf * 100).toFixed(1)}%</b>`;
  }
  // big bar
  {
    const c = cConfBar, ctx = ctxConfBar;
    const { w, h } = fitCanvas(c);
    ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);
    if (conf != null) {
      ctx.fillStyle = "#1a1d24"; ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = confColor(conf);
      ctx.fillRect(0, 0, Math.max(2, w * conf), h);
      ctx.fillStyle = "#0c0d10"; ctx.fillRect(Math.floor(w * 0.5), 0, 1, h);
    }
  }
  // four sub-bars
  const compEl = $("confComponents");
  if (compEl) {
    if (conf == null) {
      compEl.innerHTML = "";
    } else {
      const items = [
        { k: "margin",            v: frame.margin,           norm: Math.max(0, Math.min(1, frame.margin / 6.0)) },
        { k: "entropy",           v: frame.entropy,          norm: Math.max(0, Math.min(1, frame.entropy)) },
        { k: "lens consistency",  v: frame.lens_consistency, norm: Math.max(0, Math.min(1, frame.lens_consistency)) },
        { k: "residual stab",     v: frame.residual_stab,    norm: Math.max(0, Math.min(1, (frame.residual_stab + 1) / 2)) },
      ];
      let html = "";
      for (const it of items) {
        const pct = (it.norm * 100).toFixed(0);
        const col = confColor(it.norm);
        html += `<div style="font-size:11px">
                   <div style="display:flex;justify-content:space-between;color:var(--dim)">
                     <span>${it.k}</span><span>${(it.v).toFixed(3)}</span>
                   </div>
                   <div style="height:8px;background:#1a1d24;margin-top:2px">
                     <div style="height:8px;width:${pct}%;background:${col}"></div>
                   </div>
                 </div>`;
      }
      compEl.innerHTML = html;
    }
  }
  // mini line chart over time
  {
    const c = cConfTrend, ctx = ctxConfTrend;
    const { w, h } = fitCanvas(c);
    ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);
    if (!allFrames || allFrames.length === 0) return;
    const padL = 20, padR = 6, padT = 6, padB = 14;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    ctx.strokeStyle = "#23262d"; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT + plotH * 0.5); ctx.lineTo(padL + plotW, padT + plotH * 0.5);
    ctx.stroke();
    ctx.fillStyle = "#6f7480"; ctx.font = "9px ui-monospace,monospace";
    ctx.fillText("1.0", 2, padT + 8);
    ctx.fillText("0.5", 2, padT + plotH * 0.5 + 3);
    ctx.fillText("0.0", 2, padT + plotH);
    const N = allFrames.length;
    const dx = N > 1 ? plotW / (N - 1) : 0;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#5dc8ff";
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < N; i++) {
      const f = allFrames[i];
      if (!f || typeof f.confidence !== "number") continue;
      const x = padL + i * dx;
      const y = padT + plotH * (1 - f.confidence);
      if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    }
    ctx.stroke();
    if (currentIdx >= 0 && currentIdx < N) {
      const f = allFrames[currentIdx];
      if (f && typeof f.confidence === "number") {
        const x = padL + currentIdx * dx;
        const y = padT + plotH * (1 - f.confidence);
        ctx.fillStyle = confColor(f.confidence);
        ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill();
      }
    }
  }
}

function drawDecisiveness(c, ctx, decisiveness) {
  const { w, h } = fitCanvas(c);
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);
  if (!decisiveness || !decisiveness.length) {
    ctx.fillStyle = "#6f7480"; ctx.font = "11px ui-monospace,monospace";
    ctx.fillText("decisiveness data not present — generate a token", 10, 18);
    return;
  }
  const L = decisiveness.length;
  const padL = 28, padR = 12, padT = 22, padB = 20;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const barW = plotW / L;
  // region band: tinted backgrounds + region labels
  const senseN = 4, assocN = 5, outputN = L - senseN - assocN;
  ctx.fillStyle = "rgba(93, 200, 255, 0.06)";
  ctx.fillRect(padL, padT, barW * senseN, plotH);
  ctx.fillStyle = "rgba(255, 174, 93, 0.06)";
  ctx.fillRect(padL + barW * senseN, padT, barW * assocN, plotH);
  ctx.fillStyle = "rgba(255, 93, 93, 0.06)";
  ctx.fillRect(padL + barW * (senseN + assocN), padT, barW * outputN, plotH);
  // region labels above the bars
  ctx.font = "9px ui-monospace,monospace";
  ctx.fillStyle = "#5dc8ff";
  ctx.fillText("SENSORY", padL + 4, 13);
  ctx.fillStyle = "#ffae5d";
  ctx.fillText("ASSOCIATION", padL + barW * senseN + 4, 13);
  ctx.fillStyle = "#ff5d5d";
  ctx.fillText("OUTPUT", padL + barW * (senseN + assocN) + 4, 13);
  // y-axis label
  ctx.fillStyle = "#6f7480";
  ctx.fillText("decisiveness ×", 4, 13);
  // bars
  const max = Math.max(...decisiveness, 0.001);
  for (let l = 0; l < L; l++) {
    const t = decisiveness[l] / max;
    const barH = Math.max(1, plotH * t);
    const region = l <= 3 ? "#5dc8ff" : l <= 8 ? "#ffae5d" : "#ff5d5d";
    ctx.fillStyle = region;
    ctx.globalAlpha = 0.7;
    ctx.fillRect(padL + l * barW + 2, padT + plotH - barH, barW - 4, barH);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#6f7480";
    ctx.font = "9px ui-monospace,monospace";
    ctx.fillText("L" + l, padL + l * barW + Math.max(2, (barW - 14) / 2), padT + plotH + 12);
    ctx.fillStyle = "#d6d8db";
    ctx.fillText(decisiveness[l].toFixed(1), padL + l * barW + 2, padT + plotH - barH - 2);
  }
}

function drawLens(elId, lens, finalByte) {
  if (!lens || lens.length === 0) { $(elId).innerHTML = ""; return; }
  // commit layer: first layer whose top-1 matches the eventually-sampled byte
  let commitL = -1;
  if (finalByte != null) {
    for (let l = 0; l < lens.length; l++) {
      if (lens[l] && lens[l][0] && lens[l][0].b === finalByte) { commitL = l; break; }
    }
  }
  // peak-confidence layer (largest top-1 prob)
  let peakL = 0, peakP = 0;
  for (let l = 0; l < lens.length; l++) {
    const p = (lens[l] && lens[l][0]) ? lens[l][0].p : 0;
    if (p > peakP) { peakP = p; peakL = l; }
  }

  const HIGHLIGHT = "#f0d65a"; // gold — matches --highlight, used here as "this row matches sampled byte"
  const finalGlyph = finalByte != null ? glyphFor(finalByte) : null;
  let header = "";
  if (finalGlyph) {
    header = `<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--soft);margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid var(--line)">
      <span>sampled byte <span class="${finalGlyph.cls}" style="background:${HIGHLIGHT};color:#000;font-weight:700">${finalGlyph.txt}</span></span>
      ${commitL >= 0 ? `<span>first matched at <b style="color:${HIGHLIGHT}">L${commitL}</b></span>` : `<span style="color:var(--dim)">no layer's top-1 matched the sampled byte</span>`}
      <span>highest top-1 confidence at <b style="color:var(--accent)">L${peakL}</b> @ ${(peakP*100).toFixed(1)}%</span>
    </div>`;
  }

  // region band labels
  const regionBand = `<div style="display:flex;font-size:9px;letter-spacing:.12em;color:var(--dim);margin-bottom:4px;padding-left:30px">
    <span style="flex:4;color:var(--cool)">L0–L3 SENSORY</span>
    <span style="flex:5;color:var(--warm)">L4–L8 ASSOCIATION</span>
    <span style="flex:3;color:var(--hot)">L9–L11 OUTPUT</span>
  </div>`;

  let rows = "";
  for (let l = 0; l < lens.length; l++) {
    const top = lens[l] || [];
    const r = regionLabel(l);
    const tint = r.cls === "b-sense" ? "#5dc8ff" : r.cls === "b-assoc" ? "#ffae5d" : "#ff5d5d";
    const t1 = top[0] || {b: 0, p: 0};
    const top1Match = (finalByte != null && t1.b === finalByte);
    const barPct = Math.min(100, t1.p * 100 * 4);  // scale so 25% prob fills the bar
    const g1 = glyphFor(t1.b);
    const fillColor = top1Match ? HIGHLIGHT : tint;
    const top1TextColor = top1Match ? HIGHLIGHT : "var(--text)";
    let chips = `<span class="${g1.cls}" style="background:${fillColor};color:#000;font-weight:700;padding:1px 6px;min-width:22px">${g1.txt}</span>
                 <span class="stat" style="font-size:10px;color:${top1TextColor};font-weight:600;margin-right:6px">${(t1.p*100).toFixed(1)}%</span>`;
    for (let k = 1; k < top.length; k++) {
      const t = top[k];
      const gk = glyphFor(t.b);
      chips += `<span class="${gk.cls}" style="opacity:0.45;padding:0 4px;margin-right:2px">${gk.txt}</span><span class="stat" style="font-size:9px;color:var(--dim);margin-right:6px">${(t.p*100).toFixed(1)}</span>`;
    }
    const isCommit = (l === commitL);
    const rowAccent = isCommit ? `box-shadow: inset 0 0 0 1px ${HIGHLIGHT};` : "";
    rows += `<li style="display:flex;align-items:center;gap:8px;padding:4px 6px;border-radius:3px;margin-bottom:2px;background:#0a0c12;${rowAccent}">
      <b style="color:${tint};width:22px;font-size:10px">L${l}</b>
      <div style="flex:0 0 56px;height:8px;background:#1a2030;border-radius:2px;position:relative;overflow:hidden">
        <div style="position:absolute;left:0;top:0;bottom:0;width:${barPct}%;background:${fillColor};"></div>
      </div>
      <div class="lens-row" style="flex:1;display:flex;align-items:center;gap:0">${chips}</div>
      ${isCommit ? `<span style="color:${HIGHLIGHT};font-size:11px;font-weight:700" title="first layer matching sampled byte">✓</span>` : ''}
    </li>`;
  }
  $(elId).innerHTML = `${header}${regionBand}<ul class='layers' style="padding:0;margin:0">${rows}</ul>`;
}

function drawMemory(elId, mem) {
  if (!mem || mem.length === 0) {
    $(elId).innerHTML = `<div class="memory-item" style="color:var(--dim)">no memory probe loaded</div>`;
    return;
  }
  let html = "";
  for (const m of mem) {
    const safe = m.text.replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
    html += `<div class="memory-item"><span class="score">${m.score}</span>${safe}</div>`;
  }
  $(elId).innerHTML = html;
}

function drawFlowList(elId, flow) {
  if (!flow || flow.length === 0) { $(elId).innerHTML = ""; return; }
  let html = "";
  for (const f of flow.slice(0, 5)) {
    html += `<span class="stat" style="margin-right:14px">pos <b>${f.p}</b> · ${(f.w * 100).toFixed(0)}%</span>`;
  }
  $(elId).innerHTML = html;
}

// ---- generic response renderer ----
function renderResponseInto(el, promptBs, generatedBs, frameIdx, showCursor) {
  el.innerHTML = "";
  const ps = document.createElement("span"); ps.className = "pr";
  ps.textContent = String.fromCharCode(...promptBs.filter(b => b < 256));
  el.appendChild(ps);
  const slice = generatedBs.slice(0, frameIdx + 1);
  for (const b of slice) el.appendChild(document.createTextNode(byteToCh(b)));
  if (showCursor) {
    const cur = document.createElement("span"); cur.className = "cur"; cur.textContent = " ";
    el.appendChild(cur);
  }
  el.scrollTop = el.scrollHeight;
}

// ---- live tab render ----
function renderResponse() {
  renderResponseInto($("response"), promptBytes, generatedBytes, currentFrame, live);
}

function blankCanvas(c, ctx, msg) {
  const { w, h } = fitCanvas(c);
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#6f7480"; ctx.font = "11px ui-monospace,monospace";
  ctx.fillText(msg, 8, h / 2);
}

function _coactRegionClass(layer, totalLayers) {
  const t = totalLayers || 12;
  if (layer < t / 3)       return "cool";
  if (layer < (2 * t) / 3) return "warm";
  return "hot";
}

function resetLiveCoact() {
  coactState.pairs.clear();
  coactState.nFrames = 0;
  const el = $("liveCoact");
  if (el) el.innerHTML = `<div class="empty">waiting for tokens…</div>`;
}

function updateLiveCoact(frame) {
  const el = $("liveCoact");
  if (!el) return;
  if (!Array.isArray(frame.ffn_top) || !frame.ffn_top.length) return;
  const layers = frame.ffn_top.length;
  const fired = [];
  const k = 4;
  for (let L = 0; L < layers; L++) {
    const top = frame.ffn_top[L] || [];
    for (let i = 0; i < Math.min(k, top.length); i++) {
      const e = top[i];
      if (!e || typeof e.v !== "number" || Math.abs(e.v) < 0.5) continue;
      fired.push({ L, n: e.id });
    }
  }
  for (let i = 0; i < fired.length; i++) {
    for (let j = i + 1; j < fired.length; j++) {
      const a = fired[i], b = fired[j];
      const ka = `${a.L}:${a.n}`;
      const kb = `${b.L}:${b.n}`;
      const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
      coactState.pairs.set(key, (coactState.pairs.get(key) || 0) + 1);
    }
  }
  coactState.nFrames++;
  if (coactState.pairs.size === 0) {
    el.innerHTML = `<div class="empty">no co-firing pairs yet</div>`;
    return;
  }
  const top = [...coactState.pairs.entries()]
    .sort((x, y) => y[1] - x[1]).slice(0, 12);
  const maxCount = top[0][1] || 1;
  const html = top.map(([k, count]) => {
    const [pa, pb] = k.split("|");
    const [La, na] = pa.split(":").map(Number);
    const [Lb, nb] = pb.split(":").map(Number);
    const ra = _coactRegionClass(La, layers);
    const rb = _coactRegionClass(Lb, layers);
    const w  = (count / maxCount) * 100;
    return `<div class="pair">
      <span class="nid ${ra}" data-layer="${La}" data-neuron="${na}">L${La}·n${na}</span>
      <span class="arrow">↔</span>
      <span class="nid ${rb}" data-layer="${Lb}" data-neuron="${nb}">L${Lb}·n${nb}</span>
      <div class="meter"><div class="fill" style="width:${w.toFixed(1)}%"></div></div>
      <span class="count">${count}/${coactState.nFrames}</span>
    </div>`;
  }).join("");
  el.innerHTML = html;
  el.querySelectorAll(".nid").forEach(span => {
    span.addEventListener("click", () => {
      showNeuronModal(parseInt(span.dataset.layer, 10), parseInt(span.dataset.neuron, 10));
    });
  });
}

function render(frame) {
  if (!frame) return;
  highlightAsciiByte(frame.byte);
  const hasMri = Array.isArray(frame.ffn_full) && frame.ffn_full.length > 0;
  if (hasMri && frame.ffn_full.length !== _layerCount) setLayerCount(frame.ffn_full.length);
  if (hasMri) {
    drawFfn(cFfn, ctxFfn, frame.ffn_full);
    drawTopNeurons(cTop, ctxTop, frame.ffn_top);
    drawCandidates("cand", frame.cand);
    drawList("res", frame.res, "good");
    drawList("contrib", frame.contrib, "warm");
    drawLens("lens", frame.lens, frame.byte);
    drawDecisionTrace("", frame);
    drawDecisiveness(cDecisive, ctxDecisive, frame.decisiveness);
    drawConfidence(frame, frames, currentFrame);
    const allBytes = promptBytes.concat(generatedBytes.slice(0, currentFrame + 1));
    drawFlow(cFlow, ctxFlow, frame.info_flow, frame.T, allBytes);
    drawFlowList("flowList", frame.info_flow);
    drawMemory("memory", frame.memory);
    if (live) updateLiveCoact(frame);
    drawTelemetry(cTel, ctxTel, frames, currentFrame);
  } else {
    const msg = "c backend: bytes only (no activations captured)";
    blankCanvas(cFfn,  ctxFfn,  msg);
    blankCanvas(cTop,  ctxTop,  msg);
    blankCanvas(cFlow, ctxFlow, msg);
    $("cand").innerHTML = `<div style="color:var(--dim);font-size:11px">${msg}</div>`;
    $("res").innerHTML = "";
    $("contrib").innerHTML = "";
    $("lens").innerHTML = `<div style="color:var(--dim);font-size:11px">${msg}</div>`;
    $("flowList").innerHTML = "";
    $("memory").innerHTML = `<div class="memory-item" style="color:var(--dim)">${msg}</div>`;
    drawTelemetry(cTel, ctxTel, frames, currentFrame);
  }
  drawLetterMs(cLetMs, ctxLetMs, frames, currentFrame, generatedBytes, promptBytes.length);
  updateLetterStats(frames, currentFrame, generatedBytes);
  $("frameLabel").textContent = `${currentFrame + 1} / ${frames.length}`;
  $("scrub").value = currentFrame;
  $("scrub").max = Math.max(0, frames.length - 1);
  $("scrub").disabled = frames.length === 0;
  updateScrubTape("scrubTape", frames, currentFrame, promptBytes);
}

function setMeta(m) {
  meta = m;
  promptBytes = m.prompt_bytes || [];
  if (m.layers) setLayerCount(m.layers);
  const prec = m.precision || "FP32";
  const precColor = prec.startsWith("QAT") ? "var(--warm)" : "var(--data-pos)";
  const cDir = m.c_model_dir || (m.c_model || null);
  const cModelChip = cDir
    ? `<span class="stat">model <b style="color:#39ff14" title="${m.c_model_path || ''}">${cDir}</b></span>`
    : `<span class="stat">model <b style="color:var(--dim)">unloaded</b></span>`;
  const cEngineChip = m.c_engine_version
    ? `<span class="stat">veritate <b title="${m.c_exe_path || ''}">${m.c_engine_version}</b></span>`
    : "";
  const cPrec = m.c_model_precision || null;
  const cPrecColor = cPrec ? (cPrec.startsWith("INT4") ? "#ff8aa0"
                              : cPrec.startsWith("INT8") ? "#39ff14"
                              : "var(--warm)") : "var(--dim)";
  const cPrecChip = cPrec
    ? `<span class="stat">c-precision <b style="color:${cPrecColor}" title="bin v${m.c_model_bin_version || '?'} ${m.c_model_training || ''} ${m.c_model_activation || ''}">${cPrec}</b></span>`
    : "";
  $("modelMeta").innerHTML = `
    ${cEngineChip}
    ${cModelChip}
    ${cPrecChip}
    <span class="stat">precision <b style="color:${precColor}">${prec}</b></span>
    <span class="stat">params <b>${m.n_params.toLocaleString()}</b></span>
    <span class="stat">layers <b>${m.layers}</b></span>
    <span class="stat">heads <b>${m.heads}</b></span>
    <span class="stat">ffn <b>${m.ffn}</b></span>
  `;
}

// ---- addons ----
async function loadAddons() {
  const list = $("addonsList");
  if (!list) return;
  try {
    const r = await fetch("/addons");
    const data = await r.json();
    const addons = (data && data.addons) || [];
    if (!addons.length) {
      list.innerHTML = `<span class="stat" style="color:var(--dim)">none discovered</span>`;
      return;
    }
    list.innerHTML = addons.map(a => {
      const m = a.manifest || {};
      const desc = (m.description || "").replace(/"/g, "&quot;");
      return `<label title="${desc}" style="display:inline-flex;align-items:center;gap:4px;color:var(--text);cursor:pointer">
        <input type="checkbox" class="addon-toggle" data-id="${a.id}" style="vertical-align:middle">
        <span>${m.name || a.id}</span>
      </label>`;
    }).join("");
  } catch (e) {
    list.innerHTML = `<span class="stat" style="color:var(--hot)">${_backendErrMsg(e)}</span>`;
  }
}

function collectSelectedAddons() {
  return Array.from(document.querySelectorAll(".addon-toggle"))
    .filter(cb => cb.checked)
    .map(cb => cb.dataset.id);
}

loadAddons();

// ---- generation ----
$("go").addEventListener("click", () => {
  if (evtSrc) { evtSrc.close(); evtSrc = null; }
  if (replay.timer) { clearInterval(replay.timer); replay.timer = null; }
  frames = []; generatedBytes = []; currentFrame = -1; live = true;
  resetLiveCoact();
  setReplayMode("live");

  const prompt = $("prompt").value;
  const temp = $("temp").value, topk = $("topk").value, maxnew = $("maxnew").value;
  promptBytes = Array.from(new TextEncoder().encode(prompt));

  renderResponse();
  $("liveStats").innerHTML = `<span class="stat">thinking...</span>`;
  $("go").disabled = true; $("go").dataset.generating = "1"; $("stop").disabled = false;

  const backend = $("backend").value;
  const ablLayer  = parseInt(($("ablLayer")  || {value:"-1"}).value, 10);
  const ablNeuron = parseInt(($("ablNeuron") || {value:"-1"}).value, 10);
  const ablBadge  = $("ablBadge");
  if (ablBadge) ablBadge.style.display = (ablLayer >= 0 && ablNeuron >= 0) ? "" : "none";
  if (ablBadge && ablLayer >= 0 && ablNeuron >= 0) ablBadge.textContent = `ablated L${ablLayer} N${ablNeuron}`;
  const addonsSel = collectSelectedAddons();
  const addonsParam = addonsSel.length ? `&addons=${encodeURIComponent(addonsSel.join(","))}` : "";
  const url = `/generate?prompt=${encodeURIComponent(prompt)}&temperature=${temp}&top_k=${topk}&max_new=${maxnew}&backend=${backend}&ablate_layer=${ablLayer}&ablate_neuron=${ablNeuron}${addonsParam}`;
  evtSrc = new EventSource(url);
  const t0 = performance.now();
  evtSrc.onmessage = (e) => {
    if (!e.data) return;
    const ev = JSON.parse(e.data);
    if (ev.kind === "meta") { setMeta(ev); return; }
    if (ev.kind === "error") {
      // backend signaled an error mid-stream (e.g., c-engine pipe desync,
      // pytorch oom). surface it instead of silently hanging.
      const msg = ev.message || "(no message)";
      $("liveStats").innerHTML = `<span class="stat" style="color:var(--hot)">stream error: ${msg}</span>`;
      try { evtSrc.close(); } catch (_) {}
      evtSrc = null;
      $("go").disabled = false; $("go").dataset.generating = ""; $("stop").disabled = true; _applyGenerateGate();
      live = false;
      if (frames.length > 0) setReplayMode("ready");
      console.warn("[/generate] backend error event:", ev);
      return;
    }
    if (ev.kind === "token") {
      frames.push(ev);
      generatedBytes.push(ev.byte);
      if (live) {
        currentFrame = frames.length - 1;
        render(ev);
        renderResponse();
      } else {
        $("frameLabel").textContent = `${currentFrame + 1} / ${frames.length}`;
        $("scrub").max = Math.max(0, frames.length - 1);
      }
      const dt = (performance.now() - t0) / 1000;
      $("liveStats").innerHTML = `
        <span class="stat"><b>${frames.length}</b> bytes</span>
        <span class="stat"><b>${(frames.length/dt).toFixed(1)}</b> b/s</span>
        <span class="stat"><b>${ev.fwd_ms.toFixed(1)}</b> ms/forward</span>
        <span class="stat">surprise <b>${ev.surprise_bits.toFixed(1)}</b></span>
        <span class="stat">uncertainty <b>${ev.entropy_bits.toFixed(1)}</b></span>
      `;
    }
  };
  evtSrc.addEventListener("done", () => {
    evtSrc?.close(); evtSrc = null;
    $("go").disabled = false; $("go").dataset.generating = ""; $("stop").disabled = true; _applyGenerateGate();
    live = false; renderResponse();
    setReplayMode("ready");
  });
  evtSrc.onerror = () => {
    evtSrc?.close(); evtSrc = null;
    $("go").disabled = false; $("go").dataset.generating = ""; $("stop").disabled = true; _applyGenerateGate();
    if (frames.length > 0) setReplayMode("ready");
  };
});

$("stop").addEventListener("click", () => {
  evtSrc?.close(); evtSrc = null;
  $("go").disabled = false; $("stop").disabled = true;
  live = false; renderResponse();
  if (frames.length > 0) setReplayMode("ready");
});

$("scrub").addEventListener("input", (e) => {
  const idx = parseInt(e.target.value, 10);
  if (idx < 0 || idx >= frames.length) return;
  currentFrame = idx;
  live = (idx === frames.length - 1);
  render(frames[idx]);
  renderResponse();
  if (replay.mode === "playing") setReplayMode("paused");
});

// ---- live / replay button ----
const replay = { mode: "live", timer: null, msPerFrame: 80 };

function setReplayMode(mode) {
  replay.mode = mode;
  if (replay.timer) { clearInterval(replay.timer); replay.timer = null; }
  const btn = $("goLive");
  if (mode === "live") {
    btn.textContent = "live";
    btn.classList.remove("replay-active");
  } else if (mode === "ready") {
    btn.textContent = "▶ replay";
    btn.classList.remove("replay-active");
  } else if (mode === "playing") {
    btn.textContent = "⏸ pause";
    btn.classList.add("replay-active");
    replay.timer = setInterval(stepReplay, replay.msPerFrame);
  } else if (mode === "paused") {
    btn.textContent = "▶ resume";
    btn.classList.remove("replay-active");
  }
}

function stepReplay() {
  if (frames.length === 0) { setReplayMode("ready"); return; }
  if (currentFrame >= frames.length - 1) { setReplayMode("ready"); return; }
  currentFrame += 1;
  render(frames[currentFrame]);
  renderResponse();
}

$("goLive").addEventListener("click", () => {
  if (frames.length === 0) return;
  if (replay.mode === "live") {
    currentFrame = frames.length - 1;
    live = true;
    render(frames[currentFrame]);
    renderResponse();
    return;
  }
  if (replay.mode === "ready") {
    currentFrame = 0;
    live = false;
    render(frames[0]);
    renderResponse();
    setReplayMode("playing");
    return;
  }
  if (replay.mode === "playing") { setReplayMode("paused"); return; }
  if (replay.mode === "paused") {
    if (currentFrame >= frames.length - 1) currentFrame = 0;
    setReplayMode("playing");
    return;
  }
});

// ---- tabs ----
function activateTab(name) {
  const valid = ["generation", "learning", "training", "wiki", "logs", "settings"];
  if (!valid.includes(name)) name = "generation";
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x.dataset.tab === name));
  document.querySelectorAll(".tab-body").forEach(x => x.classList.toggle("active", x.dataset.tab === name));
  if (name === "learning") {
    ensureLearningLoaded();
    // mirror tier-1 panels for the picked timeline. one render per activation.
    if (learningTimelineName && classroomStateL.run !== learningTimelineName) {
      loadClassroomForLearning(learningTimelineName);
    }
    requestAnimationFrame(() => {
      [cFfnL, cTopL, cTelL, cSatL, cQuantKlL, cDecisiveL, cConfEvoL, cReadGradeL,
       cCoactL, cSurpriseL].forEach(fitCanvas);
      if (learningState.loaded) {
        drawQuantKl(cQuantKlL, ctxQuantKlL, learningState.meta.checkpoints, learningState.ckptIdx);
        renderLearning();
        if (typeof renderTier2ForLearning === "function") renderTier2ForLearning();
      }
      if (classroomStateL.loaded) {
        render_confidence_evo(classroomRefsL, classroomStateL.run, classroomStateL.steps, classroomStateL.confByStep);
        render_reading_level(classroomRefsL, classroomStateL.run, classroomStateL.gradesSteps, classroomStateL.gradesByStep, true, classroomStateL.config);
      }
    });
  } else if (name === "training") {
    startTrainPolling();
    trainStreamStart();
    requestAnimationFrame(() => {
      [cLossT, cLrT, cTpsT, cGnT, cConfEvoT, cReadGradeT].forEach(fitCanvas);
      if (trainLastText) parseAndRenderTrain(trainLastText);
      if (classroomState.loaded) {
        render_confidence_evo(classroomRefsT, classroomState.run, classroomState.steps, classroomState.confByStep);
        render_reading_level(classroomRefsT, classroomState.run, classroomState.gradesSteps, classroomState.gradesByStep, true, classroomState.config);
      }
    });
  } else if (name === "wiki") {
    stopTrainPolling();
    trainStreamStop();
    ensureWikiLoaded();
  } else {
    stopTrainPolling();
    trainStreamStop();
  }
}

document.querySelectorAll(".tab").forEach(t => {
  t.addEventListener("click", () => {
    location.hash = t.dataset.tab;
  });
});

// Module state for the brain-stream auto-subscribe in activateTab. Declared
// here (BEFORE the activateTab call below) because trainStreamStart/Stop read
// these bindings, and activateTab fires synchronously on page load if the
// hash is #training. let-bindings declared further down would be in the
// temporal dead zone.
var _trainStreamEvt = null;
var _trainStreamCount = 0;

window.addEventListener("hashchange", () => activateTab(location.hash.slice(1)));
setTimeout(() => activateTab(location.hash.slice(1)), 0);

// keep the max-bytes hint in sync with the prompt length. engine caps at V_SEQ=256.
const C_SEQ = 256;
function updateMaxHint() {
  const promptBs = (new TextEncoder().encode($("prompt").value)).length;
  const cap = Math.max(0, C_SEQ - promptBs);
  $("maxHint").textContent = `engine cap: ${cap} bytes (V_SEQ=${C_SEQ} − ${promptBs} prompt)`;
  $("maxnew").max = String(cap);
}
$("prompt").addEventListener("input", updateMaxHint);
updateMaxHint();

// probe server for backend availability and prefill model meta
function fmtMtime(t) {
  try { return new Date(t * 1000).toISOString().slice(0,10); }
  catch (e) { return ""; }
}

const _cModelMeta = {};

function _qatWarningFor(opt) {
  if (!opt) return "";
  const v = parseInt(opt.dataset.binVersion || "0", 10);
  if (v && v < 9) {
    return "pre-v9 binary. no INT8/QAT metadata. output may be incoherent if the model was not QAT-trained.";
  }
  const boost = opt.dataset.actBoost === "" ? null : parseInt(opt.dataset.actBoost, 10);
  if (boost !== null && !Number.isNaN(boost) && boost > 1) {
    return `model not QAT-refined (act_boost=${boost}). embeddings fall below INT8 resolution. output may be incoherent.`;
  }
  return "";
}

function refreshQatWarning() {
  const el = $("cQatWarning"), tx = $("cQatWarningText");
  if (!el || !tx) return;
  if ($("backend").value !== "c") { el.style.display = "none"; tx.textContent = ""; return; }
  const sel = $("cModel");
  const opt = sel && sel.selectedOptions && sel.selectedOptions[0];
  const msg = _qatWarningFor(opt);
  if (msg) { el.style.display = ""; tx.textContent = msg; el.title = msg; }
  else     { el.style.display = "none"; tx.textContent = ""; }
}

function applyBackendUI() {
  const isC = $("backend").value === "c";
  $("cModel").disabled = !isC || $("cModel").options.length === 0;
  refreshQatWarning();
  if (typeof _applyGenerateGate === "function") _applyGenerateGate();
}

function fillSelectFallback(selId, text) {
  const sel = $(selId);
  sel.innerHTML = "";
  const o = document.createElement("option");
  o.value = ""; o.textContent = text;
  sel.appendChild(o);
}

function refreshCModels() {
  return fetch("/c-models").then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }).then(d => {
    const sel = $("cModel");
    sel.innerHTML = "";
    if (!d.models || !d.models.length) {
      fillSelectFallback("cModel", "no veritate.bin found"); return;
    }
    for (const m of d.models) {
      const o = document.createElement("option");
      const prec = m.precision || "?";
      const train = m.training ? `/${m.training}` : "";
      o.title = m.description || "";
      o.value = m.bin_path;
      o.textContent = `[${prec}${train}] ${m.name}`;
      o.dataset.binVersion = String(m.bin_version || 0);
      o.dataset.actBoost   = (m.act_boost === null || m.act_boost === undefined) ? "" : String(m.act_boost);
      o.dataset.training   = m.training || "";
      if (m.is_current) o.selected = true;
      sel.appendChild(o);
    }
    refreshQatWarning();
  }).catch(err => {
    fillSelectFallback("cModel", `endpoint missing — restart server (${err.message || err})`);
  });
}

function postCConfig(body) {
  $("cConfigStatus").style.color = "var(--dim)";
  $("cConfigStatus").textContent = "switching…";
  return fetch("/c-config", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  }).then(r => r.json().then(j => ({ok: r.ok, j})))
    .then(({ok, j}) => {
      if (!ok || !j.ok) {
        $("cConfigStatus").style.color = "var(--hot)";
        $("cConfigStatus").textContent = j.error || "switch failed";
        return;
      }
      $("cConfigStatus").style.color = "var(--data-pos)";
      $("cConfigStatus").textContent = `loaded ${j.c_model_dir || "?"} on ${j.c_exe || "?"}`;
      return fetch("/meta").then(r => r.json()).then(m => {
        if (m.checkpoint) {
          setMeta({
            checkpoint: m.checkpoint, n_params: m.n_params,
            layers: m.layers, heads: m.heads, ffn: m.ffn,
            has_memory: m.has_memory, prompt_bytes: [],
            c_model: m.c_model, c_model_dir: m.c_model_dir, c_model_path: m.c_model_path,
            c_model_precision: m.c_model_precision, c_model_bin_version: m.c_model_bin_version,
            c_model_training: m.c_model_training, c_model_activation: m.c_model_activation,
            c_exe: m.c_exe, c_exe_path: m.c_exe_path,
            c_engine_version: m.c_engine_version, c_engine_label: m.c_engine_label,
          });
        }
      });
    })
    .catch(e => {
      $("cConfigStatus").style.color = "var(--hot)";
      $("cConfigStatus").textContent = `error: ${e}`;
    });
}

$("cModel").addEventListener("change", () => {
  refreshQatWarning();
  const v = $("cModel").value;
  if (v) postCConfig({model: v});
});
$("backend").addEventListener("change", applyBackendUI);

// "follow latest" — poll /c-models every 15s; if a fresher mtime than what's
// active appears, switch automatically. lets new training exports go live with
// no manual action.
let followLatestActiveMtime = 0;
function pollFollowLatest() {
  if (!$("followLatest").checked) return;
  fetch("/c-models").then(r => r.ok ? r.json() : null).then(d => {
    if (!d || !d.models || !d.models.length) return;
    const sorted = [...d.models].sort((a, b) => b.mtime - a.mtime);
    const top = sorted[0];
    if (!top.has_config) return;
    const cur = d.models.find(m => m.active);
    if (!cur || top.mtime <= (cur.mtime || 0)) return;
    if (top.mtime <= followLatestActiveMtime) return;
    followLatestActiveMtime = top.mtime;
    $("cConfigStatus").style.color = "var(--data-pos)";
    $("cConfigStatus").textContent = `auto-switching to ${top.name} (newer export)`;
    postCConfig({model: top.path}).then(refreshCModels);
  }).catch(() => {});
}
setInterval(pollFollowLatest, 15000);
$("followLatest").addEventListener("change", () => {
  if ($("followLatest").checked) pollFollowLatest();
});

refreshCModels().then(applyBackendUI);

fetch("/meta").then(r => r.json()).then(m => {
  const opt = $("backend").querySelector('option[value="c"]');
  if (m.c_backend_available) {
    opt.disabled = false;
    opt.textContent = "Veritate";
  } else {
    opt.textContent = "Veritate (not built — run build.bat)";
  }
  if (m.checkpoint && !meta) {
    setMeta({
      checkpoint: m.checkpoint, n_params: m.n_params,
      layers: m.layers, heads: m.heads, ffn: m.ffn,
      has_memory: m.has_memory, prompt_bytes: [],
      c_model: m.c_model, c_model_dir: m.c_model_dir, c_model_path: m.c_model_path,
      c_model_precision: m.c_model_precision, c_model_bin_version: m.c_model_bin_version,
      c_model_training: m.c_model_training, c_model_activation: m.c_model_activation,
      c_exe: m.c_exe, c_exe_path: m.c_exe_path,
      c_engine_version: m.c_engine_version, c_engine_label: m.c_engine_label,
    });
  }
  applyBackendUI();
}).catch(() => {});

// ============================================================
// LEARNING TAB
// ============================================================
const cFfnL  = $("cFfnL"),  ctxFfnL  = cFfnL.getContext("2d");
const cTopL  = $("cTopL"),  ctxTopL  = cTopL.getContext("2d");
const cTelL  = $("cTelL"),  ctxTelL  = cTelL.getContext("2d");
const cSatL  = $("cSatL"),  ctxSatL  = cSatL.getContext("2d");
const cQuantKlL = $("cQuantKlL"), ctxQuantKlL = cQuantKlL.getContext("2d");
const cDecisiveL = $("cDecisiveL"), ctxDecisiveL = cDecisiveL.getContext("2d");

const learningState = {
  loaded: false,
  meta: null,                  // timeline.json
  ckptIdx: 0,                  // which checkpoint is selected
  framesByStep: {},            // ckpt key -> { meta, frames }
  currentFrame: 0,             // token index within selected checkpoint
  promptBytes: [],
  _epoch: 0,                   // monotonic counter — drops stale selectCheckpoint awaits
};

function ckptKey(c) { return (c.stage || "default") + ":" + c.step; }
function ckptStageName(c) { return c.stage || (c.precision && c.precision !== "unknown" ? c.precision : "FP32"); }
function ckptEffStep(c) { return (typeof c.effective_step === "number") ? c.effective_step : c.step; }

function inferStages(meta) {
  if (meta.stages && meta.stages.length) return meta.stages;
  // back-compat: synthesize one stage per unique precision
  const seen = new Set();
  const out = [];
  for (const c of meta.checkpoints) {
    const name = ckptStageName(c);
    if (!seen.has(name)) {
      seen.add(name);
      out.push({ name, label: name, warm_start_step: null });
    }
  }
  return out;
}

function renderOutputEvolution() {
  const meta = learningState.meta;
  if (!meta) return;
  const list = $("ckptOutputs");
  const stages = inferStages(meta);
  // dynamic subtitle
  const sub = $("outputEvolutionSubtitle");
  if (sub) {
    const n = meta.checkpoints.length;
    if (stages.length <= 1) {
      sub.textContent = `${n} snapshots of the same brain learning to write`;
    } else {
      const labels = stages.map(s => s.label || s.name).join(" + ");
      sub.textContent = `${n} snapshots across ${stages.length} stages — ${labels}`;
    }
  }
  // group ckpts by stage
  const byStage = {};
  stages.forEach(s => { byStage[s.name] = []; });
  meta.checkpoints.forEach((c, i) => {
    const st = ckptStageName(c);
    if (!byStage[st]) byStage[st] = [];
    byStage[st].push({ ckpt: c, idx: i });
  });
  // union of effective steps
  const steps = [...new Set(meta.checkpoints.map(ckptEffStep))].sort((a, b) => a - b);
  const cols = stages.length;
  const rowStyle = `grid-template-columns: 90px repeat(${cols}, minmax(0, 1fr));`;
  let html = `<div class="ckpt-grid">`;
  // header
  html += `<div class="ckpt-grid-row is-head" style="${rowStyle}">`;
  html += `<div class="ckpt-grid-step-label" style="color:var(--text)">step</div>`;
  for (const s of stages) {
    const isQat = (s.label || s.name).toUpperCase().startsWith("QAT");
    const colCls = isQat ? "stage-col-qat" : "stage-col-fp32";
    html += `<div class="col ${colCls}">${escapeHtml(s.label || s.name)}</div>`;
  }
  html += `</div>`;
  // body
  const recentCount = 3;
  const recentStart = Math.max(0, steps.length - recentCount);
  steps.forEach((step, rowIdx) => {
    const recentCls = rowIdx >= recentStart ? " recent" : "";
    html += `<div class="ckpt-grid-row step-row${recentCls}" style="${rowStyle}">`;
    html += `<div class="ckpt-grid-step-label">${step.toLocaleString()}</div>`;
    for (const s of stages) {
      const ckpts = byStage[s.name] || [];
      const found = ckpts.find(({ ckpt }) => ckptEffStep(ckpt) === step);
      if (!found) {
        html += `<div class="ckpt-cell empty"></div>`;
      } else {
        const isQat = (found.ckpt.precision || "").toUpperCase().startsWith("QAT");
        const stageCls = isQat ? "qat" : "fp32";
        const stageTag = (found.ckpt.precision && found.ckpt.precision !== "unknown") ? found.ckpt.precision : "FP32";
        html += `
          <div class="ckpt-cell ${stageCls}" data-idx="${found.idx}">
            <div class="ckpt-cell-head">
              <span class="step-num">step ${found.ckpt.step.toLocaleString()}</span>
              <span class="stage-tag ${stageCls}">${escapeHtml(stageTag)}</span>
            </div>
            <div class="ckpt-cell-text"><span class="pr">${escapeHtml(meta.prompt)}</span>${escapeHtml(found.ckpt.output_text)}</div>
          </div>`;
      }
    }
    html += `</div>`;
  });
  html += `</div>`;
  list.innerHTML = html;
  list.querySelectorAll(".ckpt-cell:not(.empty)").forEach(cell => {
    cell.addEventListener("click", () => {
      const idx = parseInt(cell.dataset.idx, 10);
      selectCheckpoint(idx);
    });
  });
  highlightActiveCkptCell();
}

function highlightActiveCkptCell() {
  document.querySelectorAll(".ckpt-cell").forEach(c => c.classList.remove("active"));
  const cell = document.querySelector(`.ckpt-cell[data-idx="${learningState.ckptIdx}"]`);
  if (cell) cell.classList.add("active");
}

// dynamically resolved per timeline pick. used by selectCheckpoint to load
// individual step_NNNN.json files from the right directory.
var learningTimelineName = null;
var learningTimelinePathPrefix = "";

var learningTimelinesByName = {};
async function loadTimelinesList() {
  try {
    const r = await fetch("/timelines?" + Date.now(), { cache: "no-store" });
    const data = await r.json();
    const lines = data.timelines || [];
    learningTimelinesByName = {};
    for (const t of lines) learningTimelinesByName[t.name] = t;
    const sel = $("timelinePicker");
    if (!sel) return null;
    sel.innerHTML = lines.map(t => {
      const ago = ((Date.now() / 1000) - t.mtime);
      const agoTxt = ago < 60 ? Math.round(ago) + "s ago"
                   : ago < 3600 ? Math.round(ago / 60) + "m ago"
                   : ago < 86400 ? Math.round(ago / 3600) + "h ago"
                   : Math.round(ago / 86400) + "d ago";
      const hooksTxt = t.has_hooks
        ? `${t.n_checkpoints} checkpoints`
        : `${t.n_pt_checkpoints || 0} pt · no hooks yet`;
      return `<option value="${t.name}">${t.name}  ·  ${hooksTxt}  ·  ${agoTxt}</option>`;
    }).join("");
    if (lines.length === 0) {
      sel.innerHTML = '<option value="">no timelines found</option>';
      return null;
    }
    // restore previous if still present, else first
    const prev = learningTimelineName;
    if (prev && lines.some(t => t.name === prev)) {
      sel.value = prev;
    } else {
      sel.value = lines[0].name;
    }
    return sel.value;
  } catch (e) {
    console.error("timelines list failed", e);
    return null;
  }
}

function setTimelineActive(name) {
  if (!name) return;
  learningTimelineName = name;
  learningTimelinePathPrefix = `/timeline/${encodeURIComponent(name)}/`;
  // stop any running replay and invalidate any in-flight selectCheckpoint awaits
  // before clearing the cache (otherwise stale frames could publish into the new timeline).
  setReplayModeL("ready");
  learningState._epoch++;
  if (typeof tier2State !== "undefined") {
    tier2State.run = null;
    tier2State.coact = {};
    tier2State.surprise = null;
  }
  // reset learning state and reload
  learningState.loaded = false;
  learningState.meta = null;
  learningState.framesByStep = {};
  learningState.ckptIdx = 0;
  learningState.currentFrame = 0;
  ensureLearningLoaded();
  // classroom panels mirror the picked timeline. cleared cache so a re-pick of the same name re-renders.
  classroomStateL.run = null;
  loadClassroomForLearning(name);
}

async function ensureLearningLoaded() {
  if (learningState.loaded) return;
  // make sure timelines list is fresh and we have a selection
  if (!learningTimelineName) {
    const picked = await loadTimelinesList();
    if (picked) {
      learningTimelineName = picked;
      learningTimelinePathPrefix = `/timeline/${encodeURIComponent(picked)}/`;
    }
  }
  try {
    const r = await fetch(learningTimelinePathPrefix + "timeline.json?" + Date.now(), { cache: "no-store" });
    const meta = await r.json();
    learningState.meta = meta;
    learningState.promptBytes = Array.from(new TextEncoder().encode(meta.prompt));
    // summarize precision across checkpoints
    const precSet = new Set(meta.checkpoints.map(c => c.precision || "unknown"));
    let precSummary;
    if (precSet.size === 1) {
      const only = [...precSet][0];
      const color = only === "unknown" ? "var(--dim)" : (only.startsWith("QAT") ? "var(--warm)" : "var(--data-pos)");
      precSummary = `<span class="stat" style="margin-left:18px">precision <b style="color:${color}">${only}</b></span>`;
    } else {
      precSummary = `<span class="stat" style="margin-left:18px">precision <b style="color:var(--accent)">mixed (${[...precSet].join(", ")})</b></span>`;
    }
    const hasUnknown = precSet.has("unknown");
    const hint = hasUnknown
      ? `<div class="meta" style="margin-top:6px;color:var(--dim);font-size:11px">precision metadata not present for some checkpoints. Newer hook dumps include the precision tag automatically.</div>`
      : "";
    const tlEntry = learningTimelinesByName[learningTimelineName] || {};
    // Only fire the warning when the backend reports zero hook steps. A
    // model with some hooks (e.g. probe + lens but no generation yet) is
    // not "hookless" — those panels will still render.
    const noHooksWarn = (tlEntry.has_hooks === false)
      ? `<div class="meta" style="margin-top:8px;padding:8px 10px;border:1px solid var(--warm);border-radius:3px;color:var(--warm);font-size:11.5px;line-height:1.45">
          <b>No hook artifacts for this model yet.</b> Checkpoints (.pt) are present but the per-step probe / lens / classroom / generation dumps were not written. Either training has not reached its first save_checkpoint yet, or this trainer has not been ported to the <code>hook_spec()</code> contract (see documentation/hooks/contract.md). Outputs / quant-KL / classroom panels will stay empty until hooks land.
        </div>`
      : "";
    $("learningStatus").innerHTML = `
      <span class="stat">prompt <b>"${meta.prompt}"</b></span>
      <span class="stat" style="margin-left:18px">checkpoints <b>${meta.checkpoints.length}</b></span>
      <span class="stat" style="margin-left:18px">bytes per checkpoint <b>${meta.max_new}</b></span>
      ${precSummary}
      ${hint}
      ${noHooksWarn}
    `;
    // build outputs grid (one column per stage; rows ordered by effective_step)
    renderOutputEvolution();
    // slider
    const s = $("ckptSlider");
    s.min = 0; s.max = meta.checkpoints.length - 1; s.value = 0; s.disabled = false;
    if (!s.__bound) {
      s.addEventListener("input", e => selectCheckpoint(parseInt(e.target.value, 10)));
      s.__bound = true;
    }
    // quant KL chart (per-checkpoint trajectory)
    drawQuantKl(cQuantKlL, ctxQuantKlL, meta.checkpoints, 0);
    if (!cQuantKlL.__bound) {
      attachQuantKlClick(cQuantKlL, idx => selectCheckpoint(idx));
      cQuantKlL.__bound = true;
    }
    learningState.loaded = true;
    selectCheckpoint(0);
    // first activation: kick off the classroom mirror for the auto-picked timeline.
    // (the activateTab guard ran before learningTimelineName was set; load it now.)
    if (learningTimelineName && classroomStateL.run !== learningTimelineName) {
      loadClassroomForLearning(learningTimelineName);
    }
  } catch (e) {
    $("learningStatus").innerHTML = `<span style="color:var(--dim)">no hook data for this model. train a checkpoint or pick a different model.</span>`;
  }
}

function escapeHtml(s) {
  return s.replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
}

async function selectCheckpoint(idx) {
  const meta = learningState.meta;
  if (!meta || idx < 0 || idx >= meta.checkpoints.length) return;
  // stop any running replay before swapping checkpoints. otherwise the timer
  // keeps advancing currentFrame on the previous (or new) ckpt while we await
  // the fetch, leading to "starts in the middle and wraps".
  setReplayModeL("ready");
  // epoch token: drop stale fetch results from rapid slider drags. only the
  // most recent selectCheckpoint may publish state.
  learningState.ckptIdx = idx;
  const myEpoch = ++learningState._epoch;
  $("ckptSlider").value = idx;
  const c = meta.checkpoints[idx];
  const stageLabel = (c.precision && c.precision !== "unknown") ? c.precision : "FP32";
  $("ckptLabel").textContent = `${stageLabel} step ${c.step.toLocaleString()} (${idx + 1} / ${meta.checkpoints.length})`;
  highlightActiveCkptCell();
  drawQuantKl(cQuantKlL, ctxQuantKlL, meta.checkpoints, idx);
  // load frames if not cached
  const key = ckptKey(c);
  if (!learningState.framesByStep[key]) {
    try {
      const r = await fetch(learningTimelinePathPrefix + c.file);
      const j = await r.json();
      // probe-source ckpts (probe_step_*.json) carry layers/neurons but no per-token frames.
      // normalize so the scrubber + render paths don't crash on undefined frames.
      if (!Array.isArray(j.frames)) j.frames = [];
      learningState.framesByStep[key] = j;
    } catch (e) {
      console.error("failed to load " + c.file, e);
      learningState.framesByStep[key] = { frames: [] };
    }
  }
  if (myEpoch !== learningState._epoch) return;   // a newer pick superseded us
  learningState.currentFrame = 0;
  const data = learningState.framesByStep[key];
  const frames = data.frames || [];
  const ssL = $("scrubL");
  ssL.min = 0; ssL.max = Math.max(0, frames.length - 1); ssL.value = 0; ssL.disabled = frames.length === 0;
  $("goReplayL").disabled = frames.length === 0;
  setReplayModeL("ready");
  renderLearning();
  renderTier2ForLearning();
}


function renderLearning() {
  const meta = learningState.meta;
  if (!meta) return;
  const c = meta.checkpoints[learningState.ckptIdx];
  const data = learningState.framesByStep[ckptKey(c)];
  if (!data) return;
  const idx = learningState.currentFrame;
  const frame = (data.frames || [])[idx];
  if (!frame) {
    $("frameLabelL").textContent = `0 / 0`;
    updateScrubTape("scrubTapeL", [], -1, learningState.promptBytes);
    return;
  }
  const generatedBs = data.frames.map(f => f.byte);
  renderResponseInto($("responseL"), learningState.promptBytes, generatedBs, idx, false);
  $("frameLabelL").textContent = `${idx + 1} / ${data.frames.length}`;
  updateScrubTape("scrubTapeL", data.frames, idx, learningState.promptBytes);
  drawFfn(cFfnL, ctxFfnL, frame.ffn_full);
  drawSaturation(cSatL, ctxSatL, frame.saturation);
  drawTopNeurons(cTopL, ctxTopL, frame.ffn_top);
  drawCandidates("candL", frame.cand);
  drawList("resL", frame.res, "good");
  drawList("contribL", frame.contrib, "warm");
  drawLens("lensL", frame.lens, frame.byte);
  drawDecisionTrace("L", frame);
  drawDecisiveness(cDecisiveL, ctxDecisiveL, frame.decisiveness);
  drawMemory("memoryL", frame.memory);
  drawTelemetry(cTelL, ctxTelL, data.frames, idx);
}

$("scrubL").addEventListener("input", e => {
  learningState.currentFrame = parseInt(e.target.value, 10);
  if (replayL.mode === "playing") setReplayModeL("paused");
  renderLearning();
});

// ---- learning replay ----
const replayL = { mode: "ready", timer: null, msPerFrame: 80 };

function getCurrentLearningData() {
  if (!learningState.meta) return null;
  const ck = learningState.meta.checkpoints[learningState.ckptIdx];
  return learningState.framesByStep[ckptKey(ck)] || null;
}

function setReplayModeL(mode) {
  replayL.mode = mode;
  if (replayL.timer) { clearInterval(replayL.timer); replayL.timer = null; }
  const btn = $("goReplayL");
  if (mode === "ready")        { btn.textContent = "▶ replay"; btn.classList.remove("replay-active"); }
  else if (mode === "playing") { btn.textContent = "⏸ pause";  btn.classList.add("replay-active"); replayL.timer = setInterval(stepReplayL, replayL.msPerFrame); }
  else if (mode === "paused")  { btn.textContent = "▶ resume"; btn.classList.remove("replay-active"); }
}

function stepReplayL() {
  const data = getCurrentLearningData();
  if (!data || !data.frames.length) { setReplayModeL("ready"); return; }
  if (learningState.currentFrame >= data.frames.length - 1) { setReplayModeL("ready"); return; }
  learningState.currentFrame += 1;
  $("scrubL").value = learningState.currentFrame;
  renderLearning();
}

$("goReplayL").addEventListener("click", () => {
  const data = getCurrentLearningData();
  if (!data || !data.frames.length) return;
  if (replayL.mode === "ready") {
    learningState.currentFrame = 0;
    $("scrubL").value = 0;
    renderLearning();
    setReplayModeL("playing");
    return;
  }
  if (replayL.mode === "playing") { setReplayModeL("paused"); return; }
  if (replayL.mode === "paused") {
    if (learningState.currentFrame >= data.frames.length - 1) {
      learningState.currentFrame = 0;
      $("scrubL").value = 0;
      renderLearning();
    }
    setReplayModeL("playing");
    return;
  }
});

// ============================================================
// NEURON CLICK-TO-INSPECT MODAL
// ============================================================
function regionDescription(layerNum) {
  if (layerNum <= 3) {
    return {
      cls: "b-sense",
      name: "sensory cortex",
      blurb: "<b>L0–L3 process raw input features.</b> Bytes, n-grams, basic patterns: \"is this byte a vowel\", \"did we just see a space\", \"are we inside quotes\". A neuron firing here usually responds to a specific surface character or short prefix.",
    };
  }
  if (layerNum <= 8) {
    return {
      cls: "b-assoc",
      name: "association cortex",
      blurb: "<b>L4–L8 combine lower-level features into concepts.</b> Word boundaries, syntax, parts of speech, semantic categories. A neuron firing here usually responds to a meaning or grammatical role rather than a specific letter.",
    };
  }
  return {
    cls: "b-out",
    name: "prefrontal / output",
    blurb: "<b>L9–L11 commit to the next byte.</b> They project all the accumulated context into a specific byte prediction. A neuron firing here votes hard for or against specific bytes; this is where the actual choice gets made.",
  };
}

function regionChipClass(layerNum) {
  if (layerNum <= 3) return "region-sense";
  if (layerNum <= 8) return "region-assoc";
  return "region-output";
}

function renderStatsStrip(stats, currentActivation) {
  const cur = (currentActivation !== undefined && currentActivation !== null)
    ? currentActivation
    : (stats && stats.current_act !== null ? stats.current_act : null);
  let html = `<div class="modal-stats-strip">`;
  if (cur !== null && cur !== undefined) {
    html += `<span class="stat">current <b>${cur.toFixed(3)}</b></span>`;
  }
  if (stats && stats.probe_max) {
    html += `<span class="stat">probe max <b>${stats.probe_max.toFixed(3)}</b></span>`;
    if (cur !== null && cur !== undefined && stats.probe_max > 1e-6) {
      const pct = cur / stats.probe_max * 100;
      const pctColor = pct < 25 ? "var(--dim)" : pct < 60 ? "var(--warm)" : "var(--accent)";
      html += `<span class="stat">at <b style="color:${pctColor}">${pct.toFixed(0)}%</b> of peak</span>`;
    }
  }
  html += `</div>`;
  return html;
}

function renderByteAffinity(affinity) {
  if (!affinity || (!affinity.pos && !affinity.neg)) return "";
  const chip = e => `<span class="byte-chip"><span class="b">${glyphFor(e.b).txt}</span><span class="w">${e.w >= 0 ? "+" : ""}${e.w.toFixed(3)}</span></span>`;
  return `
    <div class="affinity-row">
      <div class="affinity-col pos">
        <h4>votes for</h4>
        <div>${(affinity.pos || []).map(chip).join("")}</div>
      </div>
      <div class="affinity-col neg">
        <h4>votes against</h4>
        <div>${(affinity.neg || []).map(chip).join("")}</div>
      </div>
    </div>`;
}

function renderCircuitChip(item, prefix) {
  const region = regionChipClass(item.layer);
  const wKey = item.contrib !== undefined ? item.contrib : item.w;
  const wClass = (wKey || 0) >= 0 ? "pos" : "neg";
  const wStr = (wKey >= 0 ? "+" : "") + wKey.toFixed(3);
  const actStr = (item.act !== undefined) ? `<span class="chip-w">act ${item.act.toFixed(2)}</span>` : "";
  const labelPill = item.label ? renderLabelPill(item.label) : "";
  return `<span class="circuit-chip ${region}" data-layer="${item.layer}" data-neuron="${item.neuron}" title="${prefix} L${item.layer} #${item.neuron}">
    <span class="chip-layer">L${item.layer}</span>
    <span class="chip-neuron">#${item.neuron}</span>
    ${labelPill}
    ${actStr}
    <span class="chip-w ${wClass}">${wStr}</span>
  </span>`;
}

function renderStoryWithPeak(text, peak_pos) {
  const safe = c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]);
  if (typeof peak_pos !== "number" || peak_pos < 0 || peak_pos >= text.length) {
    return text.replace(/[<>&]/g, safe);
  }
  const isWord = ch => /[a-zA-Z0-9'']/.test(ch);
  let wStart = peak_pos;
  while (wStart > 0 && isWord(text[wStart - 1])) wStart--;
  let wEnd = peak_pos + 1;
  while (wEnd < text.length && isWord(text[wEnd])) wEnd++;
  const ctxLeft  = 8;
  const ctxRight = 8;
  const start = Math.max(0, Math.min(wStart, peak_pos - ctxLeft));
  const end   = Math.min(text.length, Math.max(wEnd, peak_pos + 1 + ctxRight));
  const before = text.slice(0, start).replace(/[<>&]/g, safe);
  const ctxL   = text.slice(start, peak_pos).replace(/[<>&]/g, safe);
  const peakWordL = text.slice(wStart, peak_pos).replace(/[<>&]/g, safe);
  const peakWordC = text.slice(peak_pos, peak_pos + 1).replace(/[<>&]/g, safe);
  const peakWordR = text.slice(peak_pos + 1, wEnd).replace(/[<>&]/g, safe);
  const ctxLOnly = text.slice(start, wStart).replace(/[<>&]/g, safe);
  const ctxROnly = text.slice(wEnd, end).replace(/[<>&]/g, safe);
  const after = text.slice(end).replace(/[<>&]/g, safe);
  return `${before}<span class="peak-context">${ctxLOnly}<span class="peak-word">${peakWordL}<span class="peak-byte">${peakWordC}</span>${peakWordR}</span>${ctxROnly}</span>${after}`;
}

function extractPeakWord(text, peak_pos) {
  if (typeof peak_pos !== "number" || peak_pos < 0 || peak_pos >= text.length) return "";
  const isWord = ch => /[a-zA-Z0-9'']/.test(ch);
  if (!isWord(text[peak_pos])) return text[peak_pos];
  let s = peak_pos, e = peak_pos + 1;
  while (s > 0 && isWord(text[s - 1])) s--;
  while (e < text.length && isWord(text[e])) e++;
  return text.slice(s, e).toLowerCase();
}

function summarizeTriggers(stories) {
  if (!stories || !stories.length) return null;
  const counts = new Map();
  let total = 0;
  for (const s of stories) {
    const w = extractPeakWord(s.text || "", s.peak_pos);
    if (!w) continue;
    counts.set(w, (counts.get(w) || 0) + 1);
    total++;
  }
  if (total === 0) return null;
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  return { ranked, total };
}

async function showNeuronModal(layer, neuronId, currentActivation) {
  const r = regionLabel(layer);
  $("modalTitle").innerHTML = `Layer ${layer}, Neuron #${neuronId}<span class="region case b-${r.cls.replace('b-','')}" style="display:inline-block;background:${r.cls==='b-sense'?'#0e2230':r.cls==='b-assoc'?'#2a2010':'#2a1010'};color:${r.cls==='b-sense'?'#5dc8ff':r.cls==='b-assoc'?'#ffae5d':'#ff5d5d'}">${r.name}</span>`;
  $("modalBody").innerHTML = `<p class="meta">loading...</p>`;
  $("neuronModal").classList.remove("hidden");
  document.body.classList.add("no-scroll");
  try {
    const res = await fetch(`/neuron/${layer}/${neuronId}`);
    const data = await res.json();
    const region = regionDescription(layer);
    // append auto-derived label to title if we have one
    if (data.label) {
      $("modalTitle").innerHTML += renderLabelPill(data.label, { showConf: true });
    }

    const ptDown = data.pytorch_loaded === false;
    const ptErr  = data.pytorch_last_error || "";
    const ptHint = ptErr
      ? `<p class="modal-summary" style="color:var(--hot)">PyTorch backend failed to load. Server log says: <code>${escapeHtml(ptErr)}</code>. Restart the server after fixing, or click <b>load</b> in the Generation tab to retry.</p>`
      : `<p class="modal-summary" style="color:var(--warm)">PyTorch backend not loaded yet. If pytorch_load_mode is "always" the server will load it on startup &mdash; restart the server. Otherwise click <b>load</b> next to the backend selector in the Generation tab.</p>`;

    // tab 1: overview
    const overviewHtml = `
      <p class="modal-summary">${region.blurb}</p>
      <h4 style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);font-weight:600;margin:14px 0 4px">byte affinity</h4>
      ${ptDown
        ? ptHint
        : `<p class="modal-summary">When this neuron fires, it pushes the residual stream in a specific direction. Projected through the unembedding, that direction lights up some bytes (votes for) and pushes others down (votes against). Numbers are the per-unit-activation logit nudge.</p>
           ${renderByteAffinity(data.affinity)}`}`;

    // tab 2: circuit
    let circuitHtml = "";
    if (ptDown) {
      circuitHtml = ptHint;
    } else if (data.predecessors && data.predecessors.length) {
      circuitHtml += `<h4 style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);font-weight:600;margin:6px 0 4px">fed by: who drove this neuron to fire (current token)</h4>
        <p class="modal-summary">Earlier-layer neurons whose activation times write-direction contributed most to this neuron's pre-activation. Click a chip to inspect that upstream neuron. <span style="color:var(--data-pos)">Positive</span> = drove it up. <span style="color:var(--hot)">Negative</span> = pushed it down.</p>
        <div class="circuit-row">${data.predecessors.map(p => renderCircuitChip(p, "fed by")).join("")}</div>`;
    } else {
      circuitHtml += `<p class="modal-summary">No predecessor data: either this is layer 0 (no upstream FFN) or no token has been generated yet for the active model.</p>`;
    }
    if (data.successors && data.successors.length) {
      circuitHtml += `<h4 style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);font-weight:600;margin:18px 0 4px">feeds into &mdash; who listens to this neuron (static)</h4>
        <p class="modal-summary">Later-layer neurons whose read-direction aligns with this neuron's write-direction. The model is wired so that when this neuron fires, these neurons feel it. Doesn't depend on input.</p>
        <div class="circuit-row">${data.successors.map(s => renderCircuitChip(s, "feeds into")).join("")}</div>`;
    }

    let memoryHtml = "";
    let triggerSummaryHtml = "";
    if (data.stories && data.stories.length > 0) {
      const summary = summarizeTriggers(data.stories);
      if (summary) {
        const top = summary.ranked[0];
        const topText = (top[0] || "").replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
        const others = summary.ranked.slice(1, 5).map(([w, c]) => {
          const safe = (w || "").replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
          return `<span class="trigger-chip">${safe}<span class="trigger-count">×${c}</span></span>`;
        }).join("");
        triggerSummaryHtml = `
          <div class="trigger-summary">
            <div class="trigger-headline">
              <span class="trigger-label">Top trigger:</span>
              <span class="trigger-word">"${topText}"</span>
              <span class="trigger-frac">(${top[1]} of ${summary.total} top snippets)</span>
            </div>
            ${others ? `<div class="trigger-others"><span class="trigger-label">Also fires on:</span> ${others}</div>` : ""}
          </div>`;
      }
      memoryHtml = `<p class="modal-summary">Each row is a passage from the training corpus where this neuron fired hardest. The <span class="peak-word-demo">highlighted word</span> contains the activation peak. The score is the relative strength.</p>`;
      for (const s of data.stories) {
        const rendered = renderStoryWithPeak(s.text, s.peak_pos);
        memoryHtml += `<div class="modal-story has-peak"><span class="score">${s.score}</span>${rendered}</div>`;
      }
    } else {
      memoryHtml = `<p class="modal-summary">No training memory yet for this neuron. The memory probe runs at every checkpoint via <code>dump_generation</code>; if this is empty, the latest checkpoint either had not run yet or the neuron never fired meaningfully on the probe corpus.</p>`;
    }

    $("modalBody").innerHTML = `
      ${renderStatsStrip(data.stats || {}, currentActivation)}
      ${triggerSummaryHtml}
      <div class="modal-tabs">
        <div class="modal-tab active" data-tab="memory">What it fires on</div>
        <div class="modal-tab" data-tab="overview">What bytes it votes for</div>
        <div class="modal-tab" data-tab="circuit">Wiring</div>
      </div>
      <div class="modal-tab-body active" data-tab="memory">${memoryHtml}</div>
      <div class="modal-tab-body" data-tab="overview">${overviewHtml}</div>
      <div class="modal-tab-body" data-tab="circuit">${circuitHtml}</div>`;

    // wire tabs
    $("modalBody").querySelectorAll(".modal-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        const name = tab.dataset.tab;
        $("modalBody").querySelectorAll(".modal-tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
        $("modalBody").querySelectorAll(".modal-tab-body").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
      });
    });
    // wire circuit chips
    $("modalBody").querySelectorAll(".circuit-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const L = parseInt(chip.dataset.layer, 10);
        const n = parseInt(chip.dataset.neuron, 10);
        showNeuronModal(L, n);
      });
    });
  } catch (e) {
    $("modalBody").innerHTML = `<p class="meta" style="color:var(--hot)">failed to load: ${e}</p>`;
  }
}

function closeModal() {
  $("neuronModal").classList.add("hidden");
  document.body.classList.remove("no-scroll");
}
function closeConceptModal() {
  $("conceptModal").classList.add("hidden");
  if ($("neuronModal").classList.contains("hidden")) document.body.classList.remove("no-scroll");
}
function openQatHelpModal() {
  const sel = $("cModel");
  const opt = sel && sel.selectedOptions && sel.selectedOptions[0];
  let name = "your model";
  if (opt && opt.value) {
    const txt = opt.textContent || "";
    const m = txt.match(/\]\s*(.+)$/);
    name = (m ? m[1] : txt).trim() || "your model";
  }
  ["qatHelpModelName", "qatHelpModelName2", "qatHelpModelName3"].forEach(id => {
    const el = $(id); if (el) el.textContent = name;
  });
  $("qatHelpModal").classList.remove("hidden");
  document.body.classList.add("no-scroll");
}
function closeQatHelpModal() {
  $("qatHelpModal").classList.add("hidden");
  if ($("neuronModal").classList.contains("hidden") && $("conceptModal").classList.contains("hidden")) {
    document.body.classList.remove("no-scroll");
  }
}
// Generic modal. Resolves with the value of the button the user clicked,
// or null if dismissed (backdrop click / Escape / close button). Buttons:
// [{label, value, primary?}]. Body is HTML.
// Options:
//   nonDismissable - disables backdrop click, Escape, and the X; user must
//     click one of `buttons` to close.
//   accent - CSS color (e.g. "var(--accent)") applied to box border, title,
//     and header underline. Defaults to var(--line).
//   align - "top" pins the box near the top of the viewport instead of
//     vertically centering it.
function showModal({ title, body, buttons, nonDismissable, accent, align }) {
  return new Promise((resolve) => {
    const root = document.createElement("div");
    root.className = "modal-backdrop";
    if (align === "top") {
      root.style.alignItems = "flex-start";
      root.style.paddingTop = "8vh";
    }
    const box = document.createElement("div");
    box.className = "modal-box";
    box.style.maxWidth = "560px";
    if (accent) box.style.borderColor = accent;
    const close = (val) => {
      document.body.removeChild(root);
      document.documentElement.classList.remove("no-scroll");
      document.body.classList.remove("no-scroll");
      document.removeEventListener("keydown", onKey);
      resolve(val);
    };
    const onKey = (e) => { if (!nonDismissable && e.key === "Escape") close(null); };
    const btnHtml = (buttons || []).map((b, i) =>
      `<button data-i="${i}" class="${b.primary ? "go" : ""}" style="margin-left:8px">${escapeHtml(b.label)}</button>`
    ).join("");
    const headerStyle = accent ? ` style="border-bottom-color:${accent}"` : "";
    const titleStyle  = accent ? ` style="color:${accent}"` : "";
    const closeBtn    = nonDismissable ? "" : `<button data-close="1">×</button>`;
    box.innerHTML = `
      <div class="modal-header"${headerStyle}><h3${titleStyle}>${escapeHtml(title || "")}</h3>${closeBtn}</div>
      <div style="font-size:12.5px;line-height:1.6;color:var(--text)">${body || ""}</div>
      <div style="display:flex;justify-content:flex-end;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)">${btnHtml}</div>
    `;
    root.appendChild(box);
    document.body.appendChild(root);
    document.documentElement.classList.add("no-scroll");
    document.body.classList.add("no-scroll");
    root.addEventListener("click", (e) => {
      if (!nonDismissable && e.target === root) return close(null);
      if (!nonDismissable && e.target.dataset && e.target.dataset.close === "1") return close(null);
      const idx = e.target.dataset && e.target.dataset.i;
      if (idx != null) close(buttons[parseInt(idx, 10)].value);
    });
    document.addEventListener("keydown", onKey);
  });
}

$("modalClose").addEventListener("click", closeModal);
$("neuronModal").addEventListener("click", (e) => { if (e.target.id === "neuronModal") closeModal(); });
$("conceptModalClose").addEventListener("click", closeConceptModal);
$("conceptModal").addEventListener("click", (e) => { if (e.target.id === "conceptModal") closeConceptModal(); });
$("qatHelpModalClose").addEventListener("click", closeQatHelpModal);
$("qatHelpModal").addEventListener("click", (e) => { if (e.target.id === "qatHelpModal") closeQatHelpModal(); });
$("cQatHelpLink").addEventListener("click", (e) => { e.preventDefault(); openQatHelpModal(); });
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  closeModal();
  closeConceptModal();
  closeQatHelpModal();
});

cFfn.style.cursor = "pointer";
cTop.style.cursor = "pointer";
attachFfnHover(cFfn, ctxFfn, () => (currentFrame >= 0 ? frames[currentFrame] : null));
attachFfnHover(cFfnL, ctxFfnL, () => {
  if (!learningState.meta) return null;
  const ck = learningState.meta.checkpoints[learningState.ckptIdx];
  const data = learningState.framesByStep[ckptKey(ck)];
  if (!data || !data.frames) return null;
  return data.frames[learningState.currentFrame] || null;
});
attachTopNeuronsHover(cTop, ctxTop, () => (currentFrame >= 0 ? frames[currentFrame] : null));
attachTopNeuronsHover(cTopL, ctxTopL, () => {
  if (!learningState.meta) return null;
  const ck = learningState.meta.checkpoints[learningState.ckptIdx];
  const data = learningState.framesByStep[ckptKey(ck)];
  if (!data || !data.frames) return null;
  return data.frames[learningState.currentFrame] || null;
});
cFfn.addEventListener("click", (e) => {
  if (currentFrame < 0) return;
  const frame = frames[currentFrame];
  if (!frame || !frame.ffn_argmax) return;
  const rect = cFfn.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const padL = 24;
  if (x < padL) return;
  const L = frame.ffn_full.length, B = frame.ffn_full[0].length;
  const cellW = (rect.width - padL) / B, cellH = rect.height / L;
  const layer = Math.floor(y / cellH);
  const bucket = Math.floor((x - padL) / cellW);
  if (layer < 0 || layer >= L || bucket < 0 || bucket >= B) return;
  const ds = frame.ffn_downsample || 12;
  const argmaxInBucket = frame.ffn_argmax[layer][bucket];
  const neuronId = bucket * ds + argmaxInBucket;
  showNeuronModal(layer, neuronId);
});
cTop.addEventListener("click", (e) => {
  if (currentFrame < 0) return;
  const frame = frames[currentFrame];
  if (!frame || !frame.ffn_top) return;
  const rect = cTop.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const padL = 28;
  if (x < padL) return;
  const L = frame.ffn_top.length, K = frame.ffn_top[0].length;
  const cellW = (rect.width - padL) / K, cellH = rect.height / L;
  const layer = Math.floor(y / cellH);
  const k = Math.floor((x - padL) / cellW);
  if (layer < 0 || layer >= L || k < 0 || k >= K) return;
  const n = frame.ffn_top[layer][k];
  showNeuronModal(layer, n.id, n.v);
});

// learning tab click handlers — same pattern but read from learningState
function currentLearningFrame() {
  if (!learningState.meta) return null;
  const ck = learningState.meta.checkpoints[learningState.ckptIdx];
  const data = learningState.framesByStep[ckptKey(ck)];
  if (!data) return null;
  return data.frames[learningState.currentFrame] || null;
}
cFfnL.style.cursor = "pointer";
cTopL.style.cursor = "pointer";
cFfnL.addEventListener("click", (e) => {
  const frame = currentLearningFrame();
  if (!frame || !frame.ffn_argmax) return;
  const rect = cFfnL.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const padL = 24;
  if (x < padL) return;
  const L = frame.ffn_full.length, B = frame.ffn_full[0].length;
  const cellW = (rect.width - padL) / B, cellH = rect.height / L;
  const layer = Math.floor(y / cellH);
  const bucket = Math.floor((x - padL) / cellW);
  if (layer < 0 || layer >= L || bucket < 0 || bucket >= B) return;
  const ds = frame.ffn_downsample || 12;
  const argmaxInBucket = frame.ffn_argmax[layer][bucket];
  const neuronId = bucket * ds + argmaxInBucket;
  showNeuronModal(layer, neuronId);
});
cTopL.addEventListener("click", (e) => {
  const frame = currentLearningFrame();
  if (!frame || !frame.ffn_top) return;
  const rect = cTopL.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const padL = 28;
  if (x < padL) return;
  const L = frame.ffn_top.length, K = frame.ffn_top[0].length;
  const cellW = (rect.width - padL) / K, cellH = rect.height / L;
  const layer = Math.floor(y / cellH);
  const k = Math.floor((x - padL) / cellW);
  if (layer < 0 || layer >= L || k < 0 || k >= K) return;
  const n = frame.ffn_top[layer][k];
  showNeuronModal(layer, n.id, n.v);
});

// ============================================================
// LIVE TRAINING TAB
// ============================================================
const cLossT = $("cLossT"), ctxLossT = cLossT.getContext("2d");
const cLrT   = $("cLrT"),   ctxLrT   = cLrT.getContext("2d");
const cTpsT  = $("cTpsT"),  ctxTpsT  = cTpsT.getContext("2d");
const cGnT   = $("cGnT"),   ctxGnT   = cGnT.getContext("2d");

var trainPollTimer = null;
var trainLastText = null;
var trainRuns = [];
var trainSelectedRun = null;

async function loadRunsList() {
  const sel = $("runPicker");
  try {
    const r = await fetch("/runs?" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    trainRuns = data.runs || [];
    const prev = sel.value;
    if (!trainRuns.length) {
      sel.innerHTML = '<option value="">- no runs in models/ -</option>';
      trainSelectedRun = null;
      $("runCsvStatus").textContent = "no train.csv files found under models/";
      return;
    }
    sel.innerHTML = trainRuns.map(run => {
      const ago = ((Date.now() / 1000) - run.mtime);
      const agoTxt = ago < 60 ? Math.round(ago) + "s ago"
                   : ago < 3600 ? Math.round(ago / 60) + "m ago"
                   : ago < 86400 ? Math.round(ago / 3600) + "h ago"
                   : Math.round(ago / 86400) + "d ago";
      const dormant = (run.n_rows || 0) < 10 ? "  ·  (empty?)" : "";
      return `<option value="${run.name}">${run.name}  ·  ${run.n_rows} rows  ·  ${agoTxt}${dormant}</option>`;
    }).join("");
    if (prev && trainRuns.some(r => r.name === prev)) {
      sel.value = prev;
    } else {
      // pick the most-recently-modified run that has at least one logged step.
      // server already sorts trainRuns newest-first by mtime; just take the
      // first one with rows. falls back to the absolute newest if nothing logs.
      const withRows = trainRuns.find(r => (r.n_rows || 0) > 0);
      sel.value = (withRows ? withRows.name : trainRuns[0].name);
    }
    trainSelectedRun = sel.value || null;
  } catch (e) {
    sel.innerHTML = `<option value="">- load failed -</option>`;
    $("runCsvStatus").textContent = "runs error: " + e.message;
  }
}

async function loadTrainCsv() {
  if (!trainSelectedRun) return;
  try {
    const url = `/run/${encodeURIComponent(trainSelectedRun)}/csv?` + Date.now();
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const text = await r.text();
    if (text === trainLastText) {
      $("runCsvStatus").textContent = `(no change at ${new Date().toLocaleTimeString()} — run: ${trainSelectedRun})`;
      return;
    }
    trainLastText = text;
    parseAndRenderTrain(text);
    $("runCsvStatus").textContent = `loaded ${new Date().toLocaleTimeString()} — run: ${trainSelectedRun}`;
  } catch (e) {
    $("runCsvStatus").textContent = "error: " + e.message;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const sel = document.getElementById("runPicker");
  if (sel) sel.addEventListener("change", () => {
    trainSelectedRun = sel.value;
    trainLastText = null;
    loadTrainCsv();
    loadClassroomForRun(trainSelectedRun);
  });
  const btn = document.getElementById("runRefresh");
  if (btn) btn.addEventListener("click", async () => {
    await loadRunsList();
    trainLastText = null;
    loadTrainCsv();
    classroomState.run = null;
    loadClassroomForRun(trainSelectedRun);
  });
  const tlSel = document.getElementById("timelinePicker");
  if (tlSel) tlSel.addEventListener("change", () => {
    if (tlSel.value) setTimelineActive(tlSel.value);
  });
  const tlBtn = document.getElementById("timelineRefresh");
  if (tlBtn) tlBtn.addEventListener("click", async () => {
    const picked = await loadTimelinesList();
    if (!picked) return;
    // force refresh even when picked is unchanged: dump cached frames + classroom
    // state so newly-written probe/lens/step files surface.
    if (picked === learningTimelineName) {
      learningState.loaded = false;
      learningState.framesByStep = {};
      classroomStateL.run = null;
      ensureLearningLoaded();
      loadClassroomForLearning(picked);
    } else {
      setTimelineActive(picked);
    }
  });
});

function parseAndRenderTrain(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines.shift().split(",");
  const rows = lines.map(line => {
    const parts = line.split(",");
    const obj = {};
    header.forEach((h, i) => obj[h] = parts[i]);
    obj.step = parseInt(obj.step);
    obj.loss = parseFloat(obj.loss);
    obj.lr = parseFloat(obj.lr);
    obj.grad_norm = parseFloat(obj.grad_norm);
    obj.tok_per_s = parseFloat(obj.tok_per_s);
    obj.wall_s = parseFloat(obj.wall_s);
    return obj;
  });
  renderTrain(rows);
}

// the csv may contain several concatenated training runs (each starts at step 0).
// segment by detecting where step drops by > 100 between consecutive rows.
function lastRunRows(rows) {
  if (rows.length < 2) return { current: rows, totalRuns: 1, totalRows: rows.length };
  let resetIdx = 0, runs = 1;
  for (let i = 1; i < rows.length; i++) {
    if (rows[i].step + 100 < rows[i-1].step) { resetIdx = i; runs++; }
  }
  return { current: rows.slice(resetIdx), totalRuns: runs, totalRows: rows.length };
}

function plotTrainSeries(canvas, ctx, series, opts) {
  opts = opts || {};
  canvas.__series = series;
  canvas.__opts = opts;
  const { w: W, h: H } = fitCanvas(canvas);
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, W, H);
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const s of series) for (const p of s.points) {
    if (p.x < xMin) xMin = p.x;
    if (p.x > xMax) xMax = p.x;
    if (p.y < yMin) yMin = p.y;
    if (p.y > yMax) yMax = p.y;
  }
  if (!isFinite(xMin)) return;
  // optional fixed y bounds. opts.yMinFloor lowers yMin to at most this value
  // (useful for "anchor the chart at ppl 1.0"), opts.yMaxCeil raises yMax.
  if (typeof opts.yMinFloor === "number") yMin = Math.min(yMin, opts.yMinFloor);
  if (typeof opts.yMaxCeil  === "number") yMax = Math.max(yMax, opts.yMaxCeil);
  // optional x bounds — used to reserve x-axis room for future checkpoints
  // so the chart layout doesn't shift each time a probe lands.
  if (typeof opts.xMinFloor === "number") xMin = Math.min(xMin, opts.xMinFloor);
  if (typeof opts.xMaxCeil  === "number") xMax = Math.max(xMax, opts.xMaxCeil);
  if (opts.logY) {
    yMin = Math.max(yMin, 0.01);
    yMin = Math.log10(yMin);
    yMax = Math.log10(yMax + 0.01);
  }
  const xR = xMax - xMin || 1, yR = yMax - yMin || 1;
  const padL = opts.yTitle ? 64 : 50, padR = 12, padT = 8, padB = 22;
  const tickX = opts.yTitle ? 18 : 4;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const xS = x => padL + ((x - xMin) / xR) * plotW;
  const yS = y => {
    const v = opts.logY ? Math.log10(Math.max(y, 0.01)) : y;
    return padT + plotH - ((v - yMin) / yR) * plotH;
  };
  ctx.strokeStyle = "#1e2330"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  ctx.fillStyle = "#6f7480"; ctx.font = "10px ui-monospace,monospace";
  for (let i = 0; i <= 4; i++) {
    const y = padT + (plotH * i / 4);
    const v = yMax - (yR * i / 4);
    const lbl = opts.logY ? Math.pow(10, v).toFixed(2) : v.toFixed(2);
    ctx.fillText(lbl, tickX, y + 4);
    ctx.strokeStyle = "#171b24"; ctx.beginPath();
    ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
  }
  for (let i = 0; i <= 5; i++) {
    const x = padL + (plotW * i / 5);
    const v = xMin + (xR * i / 5);
    ctx.fillStyle = "#6f7480";
    ctx.fillText(Math.round(v).toLocaleString(), x - 16, H - 6);
  }
  // optional horizontal threshold lines, e.g. ppl 3.0 = fluent.
  // shape: opts.thresholdLines = [{ y, color, label, lineDash }]
  if (Array.isArray(opts.thresholdLines)) {
    ctx.save();
    for (const t of opts.thresholdLines) {
      const y = yS(t.y);
      ctx.strokeStyle = t.color || "#5dff9b";
      ctx.lineWidth = 1;
      ctx.setLineDash(t.lineDash || [4, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
      if (t.label) {
        ctx.setLineDash([]);
        ctx.fillStyle = t.color || "#5dff9b";
        ctx.font = "10px ui-monospace,monospace";
        ctx.fillText(t.label, W - padR - 6 - ctx.measureText(t.label).width, y - 3);
      }
    }
    ctx.restore();
  }
  // optional rotated y-axis title in the left margin.
  if (opts.yTitle) {
    ctx.save();
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = "#9aa0ad";
    ctx.font = "10px ui-monospace,monospace";
    const w = ctx.measureText(opts.yTitle).width;
    ctx.fillText(opts.yTitle, -w / 2, 0);
    ctx.restore();
  }
  for (const s of series) {
    if (s.points.length === 0) continue;
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.lineWidth = s.lw || 1.5;
    if (s.points.length > 1) {
      ctx.beginPath();
      ctx.moveTo(xS(s.points[0].x), yS(s.points[0].y));
      for (let i = 1; i < s.points.length; i++) ctx.lineTo(xS(s.points[i].x), yS(s.points[i].y));
      ctx.stroke();
    }
    if (s.dots) {
      for (const p of s.points) {
        ctx.beginPath(); ctx.arc(xS(p.x), yS(p.y), 3, 0, 6.3); ctx.fill();
      }
    }
  }
}

function detectPlateauT(valPoints) {
  // window: 8 val points (was 4). longer window averages out per-eval noise;
  // a single uptick at the end no longer flips the verdict from "improving" to
  // "plateau". slope is least-squares regression of val_loss vs eval_index,
  // normalized by mean(val_loss). that's the per-eval relative trend.
  const minPoints = 4, windowMax = 8;
  const flatPct = 0.005, slowPct = 0.02, regressPct = 0.003;
  if (valPoints.length < minPoints) {
    return { state: "warming", note: `need <span style="color:var(--accent)">${minPoints}</span> val points, have <span style="color:var(--warm)">${valPoints.length}</span>` };
  }
  const recent = valPoints.slice(-Math.min(windowMax, valPoints.length));
  const n = recent.length;
  const meanY = recent.reduce((s, p) => s + p.y, 0) / n;
  // ordinary least squares slope on y = a*i + b; i is sample index.
  let sx = 0, sy = 0, sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) {
    sx += i; sy += recent[i].y; sxy += i * recent[i].y; sxx += i * i;
  }
  const denom = (n * sxx - sx * sx) || 1e-9;
  const slope = (n * sxy - sx * sy) / denom;
  const slopePct = slope / Math.max(meanY, 1e-6);
  // also keep deltas for the displayed table (last few changes).
  const deltas = [];
  for (let i = 1; i < recent.length; i++) {
    const prev = recent[i-1].y, cur = recent[i].y;
    deltas.push((cur - prev) / Math.max(prev, 1e-6));
  }
  const meanAbs = deltas.reduce((s, d) => s + Math.abs(d), 0) / deltas.length;

  // priority: regressing > improving > slowing > plateau > bouncing
  // slope-based: positive trend is regressing, big negative trend is improving,
  // small negative trend is slowing, near-zero with low variance is plateau,
  // near-zero with high variance is bouncing.
  let state;
  if (slopePct >  regressPct)            state = "regressing";
  else if (slopePct < -slowPct)          state = "improving";
  else if (slopePct < -flatPct)          state = "slowing";
  else if (meanAbs   < 2 * flatPct)      state = "plateau";
  else                                    state = "bouncing";
  return { state, recent, deltas, avgPct: slopePct * 100, slopePct };
}

function deltaSpanT(p) {
  if (Math.abs(p) < 0.5) return `<span style="color:#aaa">${p >= 0 ? "+" : ""}${p.toFixed(2)}%</span>`;
  if (p < 0) return `<span style="color:#5dff9b">${p.toFixed(2)}%</span>`;
  return `<span style="color:#ff7d7d">+${p.toFixed(2)}%</span>`;
}

function renderTrainPlateau(valPoints) {
  const r = detectPlateauT(valPoints);
  const colors = {
    warming:    { bg: "#1a1d24", fg: "#9aa4b5", label: "WARMING UP" },
    improving:  { bg: "#103025", fg: "#5dff9b", label: "IMPROVING" },
    bouncing:   { bg: "#102538", fg: "#5dc8ff", label: "BOUNCING" },
    slowing:    { bg: "#2a2010", fg: "#ffae5d", label: "SLOWING" },
    plateau:    { bg: "#2a1010", fg: "#ff5d5d", label: "PLATEAU" },
    regressing: { bg: "#330a0a", fg: "#ff7d5d", label: "REGRESSING" },
  };
  const notes = {
    warming:    { what: "Not enough val measurements yet to judge a trend.",                                          act: "" },
    improving:  { what: "Val loss is dropping consistently across the last 4 evals.",                                  act: "Keep training." },
    bouncing:   { what: "Val is moving up and down with no clear direction.",                                          act: "Often a local minimum. Lower LR a bit, or keep going to see if average drifts down." },
    slowing:    { what: "Val is trending down at a small rate (slope between 0.5% and 2% per eval). Normal late-training behavior — the bulk of learning is done; remaining gains come from polish.", act: "Keep training. Cosine LR decay is doing its job. Stop early only if val starts curving back up." },
    plateau:    { what: "Val has flatlined across 4 evals. Diminishing returns.",                                      act: "Stop training, or sharply cool LR for one more pass to extract a final 1-3%." },
    regressing: { what: "Val loss is rising on average.",                                                              act: "Lower LR. If it persists, restart from the last good checkpoint. Check for overfitting." },
  };
  const c = colors[r.state] || colors.warming;
  const n = notes[r.state] || notes.warming;
  let html = `<div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:10px">
    <span style="font-size:18px;font-weight:700;letter-spacing:.04em;padding:4px 12px;border-radius:4px;background:${c.bg};color:${c.fg}">${c.label}</span>`;
  if (r.state !== "warming") {
    html += `<span class="meta">slope per eval over last ${r.recent.length} points: ${deltaSpanT(r.avgPct)}</span>`;
  }
  html += `</div>
    <p style="margin:4px 0 6px;color:var(--text)">${n.what}</p>
    ${n.act ? `<p style="margin:4px 0 12px;color:${c.fg}"><b>→ what to do:</b> ${n.act}</p>` : ""}`;

  if (r.state !== "warming") {
    html += `<table><tr><th>step</th><th>val loss</th><th>delta from prev</th></tr>`;
    for (let i = 0; i < r.recent.length; i++) {
      const p = r.recent[i];
      const dPct = i > 0 ? r.deltas[i-1] * 100 : null;
      html += `<tr>
        <td style="text-align:right">${p.x.toLocaleString()}</td>
        <td style="text-align:right">${p.y.toFixed(4)}</td>
        <td style="text-align:right">${dPct === null ? "-" : deltaSpanT(dPct)}</td>
      </tr>`;
    }
    html += `</table>`;
  } else {
    html += `<div class="meta">${r.note}</div>`;
  }
  $("trainPlateau").innerHTML = html;
}

function renderTrain(allRows) {
  const seg = lastRunRows(allRows);
  const rows = seg.current;
  // generic suffix match: any split ending in "_train" or equal to "train" is a train row;
  // any "_val" or equal to "val" is a val row. lets new training modes (qat2_train, mamba2_train,
  // future modes) light up the dashboard with no frontend changes.
  const isTrainSplit = s => s === "train" || (typeof s === "string" && s.endsWith("_train"));
  const isValSplit   = s => s === "val"   || (typeof s === "string" && (s.endsWith("_val") || s.startsWith("val_")));
  const trainAll = rows.filter(r => isTrainSplit(r.split)).map(r => ({ x: r.step, y: r.loss }));
  const valAll   = rows.filter(r => isValSplit(r.split)).map(r => ({ x: r.step, y: r.loss }));
  // legacy series names kept for back-compat with existing plot calls below; all map to the
  // same union now (parser doesn't care which qat phase emitted the row).
  const train    = trainAll;
  const val      = valAll;
  const qatTrain = [];
  const qatVal   = [];
  const valQat   = [];
  const lr       = rows.filter(r => isTrainSplit(r.split)).map(r => ({ x: r.step, y: r.lr }));
  const tps      = rows.filter(r => isTrainSplit(r.split)).map(r => ({ x: r.step, y: r.tok_per_s }));
  const gn       = rows.filter(r => isTrainSplit(r.split)).map(r => ({ x: r.step, y: r.grad_norm }));

  plotTrainSeries(cLossT, ctxLossT, [
    { points: train,    color: "#5dc8ff", lw: 1 },
    { points: val,      color: "#a899d4", lw: 2, dots: true },
    { points: qatTrain, color: "#ff9a3c", lw: 1 },
    { points: qatVal,   color: "#ff5d9b", lw: 2, dots: true },
    { points: valQat,   color: "#ff5d9b", lw: 1, dots: true },
  ], { logY: true });
  plotTrainSeries(cLrT,   ctxLrT,   [{ points: lr,  color: "#5dc8ff", lw: 1.5 }]);
  plotTrainSeries(cTpsT,  ctxTpsT,  [{ points: tps, color: "#5dff9b", lw: 1.5 }]);
  plotTrainSeries(cGnT,   ctxGnT,   [{ points: gn,  color: "#ff9a5d", lw: 1   }], { logY: true });

  const valSeries = (qatVal.length > 0 ? qatVal : (val.length > 0 ? val : valQat));
  renderTrainPlateau(valSeries);
  const lastTrainRow = (() => {
    for (let i = rows.length - 1; i >= 0; i--) {
      if (isTrainSplit(rows[i].split)) return rows[i];
    }
    return null;
  })();
  const lastValRow = (() => {
    for (let i = rows.length - 1; i >= 0; i--) {
      if (isValSplit(rows[i].split)) return rows[i];
    }
    return null;
  })();
  window._aiTrainHealth = {
    model: trainSelectedRun || "",
    series: valSeries.slice(-8),
    verdict: detectPlateauT(valSeries),
    latest: lastTrainRow ? {
      step:       lastTrainRow.step,
      train_loss: lastTrainRow.loss,
      lr:         lastTrainRow.lr,
      grad_norm:  lastTrainRow.grad_norm,
      tok_per_s:  lastTrainRow.tok_per_s,
      val_loss:   lastValRow ? lastValRow.loss : null,
    } : null,
  };
  const downsample = (pts, n) => {
    if (!pts || pts.length <= n) return pts || [];
    const stride = pts.length / n;
    const out = [];
    for (let i = 0; i < n; i++) out.push(pts[Math.floor(i * stride)]);
    if (out[out.length - 1] !== pts[pts.length - 1]) out.push(pts[pts.length - 1]);
    return out;
  };
  window._aiLossCurve = {
    model: trainSelectedRun || "",
    train: downsample(train, 12),
    val:   valSeries.slice(),
    verdict: detectPlateauT(valSeries),
  };

  const latest = rows[rows.length - 1];
  if (latest) {
    const trainCount = train.length + qatTrain.length;
    const valCount = val.length + qatVal.length + valQat.length;
    const wallH = (latest.wall_s / 3600).toFixed(2);
    $("trainLatest").innerHTML = `<div class="stat-cards">
      <div class="stat-card"><div class="k">step</div><div class="v">${latest.step.toLocaleString()}</div><div class="s">split: ${latest.split}</div></div>
      <div class="stat-card"><div class="k">loss</div><div class="v">${latest.loss.toFixed(4)}</div><div class="s">${trainCount} train · ${valCount} val rows</div></div>
      <div class="stat-card"><div class="k">learning rate</div><div class="v">${latest.lr.toExponential(2)}</div><div class="s">cosine schedule</div></div>
      <div class="stat-card"><div class="k">throughput</div><div class="v">${latest.tok_per_s.toFixed(0)} tok/s</div><div class="s">grad norm: ${latest.grad_norm.toFixed(2)}</div></div>
      <div class="stat-card"><div class="k">wall time</div><div class="v">${wallH} h</div><div class="s">${latest.wall_s.toFixed(0)} s elapsed</div></div>
      <div class="stat-card"><div class="k">csv history</div><div class="v">${seg.totalRuns} runs</div><div class="s">${seg.totalRows.toLocaleString()} total rows</div></div>
    </div>`;
  } else {
    $("trainLatest").innerHTML = `<div class="meta">no rows yet</div>`;
  }

  const recent = allRows.slice(-30).reverse();
  let html = "<table><tr><th>step</th><th style='text-align:left'>split</th><th>loss</th><th>lr</th><th>gn</th><th>tok/s</th><th>wall s</th></tr>";
  const splitColor = s => {
    if (s === "train") return "#5dc8ff";
    if (s === "val")   return "#a899d4";
    if (typeof s === "string" && s.endsWith("_train")) return "#ff9a3c";
    if (typeof s === "string" && (s.endsWith("_val") || s.startsWith("val_"))) return "#ff5d9b";
    return "#ddd";
  };
  for (const r of recent) {
    const c = splitColor(r.split);
    html += `<tr>
      <td style="text-align:right">${r.step.toLocaleString()}</td>
      <td style="text-align:left;color:${c}">${r.split}</td>
      <td style="text-align:right">${r.loss.toFixed(4)}</td>
      <td style="text-align:right">${r.lr.toExponential(2)}</td>
      <td style="text-align:right">${r.grad_norm.toFixed(2)}</td>
      <td style="text-align:right">${r.tok_per_s.toFixed(0)}</td>
      <td style="text-align:right">${r.wall_s.toFixed(1)}</td>
    </tr>`;
  }
  html += "</table>";
  $("trainRecent").innerHTML = html;
}

// ============================================================
// CLASSROOM DASHBOARD — tier 1 panels
// ============================================================
//
// state cached per-run. probe + lens fetched once per checkpoint, never re-polled.
// no extra requests added to the 5s csv loop.

const cConfEvoT = $("cConfEvoT"), ctxConfEvoT = cConfEvoT.getContext("2d");
const cConfEvoL = $("cConfEvoL"), ctxConfEvoL = cConfEvoL.getContext("2d");
const cReadGradeT = $("cReadGradeT"), ctxReadGradeT = cReadGradeT.getContext("2d");
const cReadGradeL = $("cReadGradeL"), ctxReadGradeL = cReadGradeL.getContext("2d");

// classroom panel refs — one set per tab. render functions key off these.
const classroomRefsT = {
  sizeMeterId: "trainSizeMeter",
  lensDriftId: "trainLensDrift",
  confCanvas:  cConfEvoT,
  confCtx:     ctxConfEvoT,
  readLevelId: "trainReadLevel",
  readLegendId:"trainReadLegend",
  readGradeCanvas: cReadGradeT,
  readGradeCtx:    ctxReadGradeT,
  mathLevelId:     "trainMathLevel",
  grammarLevelId:  "trainGrammarLevel",
  reasoningLevelId: "trainReasoningLevel",
  conceptsId:      "trainConcepts",
  conceptsHoverId: "trainConceptsHover",
  writingHealthId: "trainWritingHealth",
};
const classroomRefsL = {
  sizeMeterId: "learnSizeMeter",
  lensDriftId: "learnLensDrift",
  confCanvas:  cConfEvoL,
  confCtx:     ctxConfEvoL,
  readLevelId: "learnReadLevel",
  readLegendId:"learnReadLegend",
  readGradeCanvas: cReadGradeL,
  readGradeCtx:    ctxReadGradeL,
  mathLevelId:     "learnMathLevel",
  grammarLevelId:  "learnGrammarLevel",
  reasoningLevelId: "learnReasoningLevel",
  conceptsId:      "learnConcepts",
  conceptsHoverId: "learnConceptsHover",
  pruneBodyId:     "pruneBody",
  pruneTitleId:    "pruneTitle",
  pruneSubtitleId: "pruneSubtitle",
  writingHealthId: "learnWritingHealth",
};

function makeClassroomState() {
  return {
    run:          null,            // currently bound run name
    config:       null,            // config.json
    steps:        [],              // [{step, probe, lens}]
    probesByStep: {},              // step -> probe json
    lensByStep:   {},              // step -> {lens_logits[L][V], residual_norms[L]}
    confByStep:   {},              // step -> {margin, entropy, lens_consistency, residual_stab}
    gradesSteps:    [],            // sorted ints, steps with grades_step_*.json
    gradesByStep:   {},            // step -> {grades:{...}, estimated_reading_grade}
    mathSteps:      [],            // sorted ints, steps with math_step_*.json
    mathByStep:     {},            // step -> {tiers:{...}}
    grammarSteps:   [],            // sorted ints, steps with grammar_step_*.json
    grammarByStep:  {},            // step -> {types:{...}}
    reasoningSteps: [],            // sorted ints, steps with reasoning_step_*.json
    reasoningByStep:{},            // step -> {tiers:{...}}
    conceptsSteps:  [],            // sorted ints, steps with concepts_step_*.json
    conceptsByStep: {},            // step -> {concepts:{name:{surprise_bits}}}
    writingSteps:   [],            // sorted ints, steps with writing_health_step_*.json
    writingByStep:  {},            // step -> writing_health record
    loaded:       false,
  };
}

const classroomState  = makeClassroomState();   // live training tab
const classroomStateL = makeClassroomState();   // learning tab

function _u32le(b, o) { return b[o] | (b[o+1]<<8) | (b[o+2]<<16) | (b[o+3]*0x1000000); }
function _u16le(b, o) { return b[o] | (b[o+1]<<8); }

// minimal zip+npy reader. accepts the bytes of a .npz produced by np.savez_compressed.
// returns { name -> typed_array }. only handles deflate-raw (method 8) and stored (0).
async function parse_npz(buf) {
  const u8 = new Uint8Array(buf);
  const out = {};
  let i = 0;
  while (i + 4 <= u8.length) {
    const sig = _u32le(u8, i);
    if (sig !== 0x04034b50) break;             // local file header magic
    const method     = _u16le(u8, i + 8);
    const compSize   = _u32le(u8, i + 18);
    const uncompSize = _u32le(u8, i + 22);
    const nameLen    = _u16le(u8, i + 26);
    const extraLen   = _u16le(u8, i + 28);
    const name = new TextDecoder().decode(u8.subarray(i + 30, i + 30 + nameLen));
    const dataStart = i + 30 + nameLen + extraLen;
    const dataEnd = dataStart + compSize;
    let payload = u8.subarray(dataStart, dataEnd);
    if (method === 8) {
      const ds = new DecompressionStream("deflate-raw");
      const blob = new Blob([payload]);
      payload = new Uint8Array(await new Response(blob.stream().pipeThrough(ds)).arrayBuffer());
    } else if (method !== 0) {
      throw new Error("npz: unsupported compression " + method);
    }
    if (payload.length !== uncompSize) {
      // some compressors omit the uncompressed size in local header; trust payload.length
    }
    out[name] = parse_npy(payload);
    i = dataEnd;
  }
  return out;
}

// minimal NPY reader. supports 1D + 2D arrays of int32, float32.
function parse_npy(buf) {
  if (buf[0] !== 0x93 || String.fromCharCode(buf[1], buf[2], buf[3], buf[4], buf[5]) !== "NUMPY") {
    throw new Error("npy: bad magic");
  }
  const ver = buf[6];
  const headerLen = (ver === 1) ? _u16le(buf, 8) : _u32le(buf, 8);
  const headerStart = (ver === 1) ? 10 : 12;
  const header = new TextDecoder().decode(buf.subarray(headerStart, headerStart + headerLen));
  // header is a python dict literal: {'descr': '<i4', 'fortran_order': False, 'shape': (12, 256), }
  const dm = header.match(/'descr'\s*:\s*'([^']+)'/);
  const sm = header.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (!dm || !sm) throw new Error("npy: header parse: " + header);
  const descr = dm[1];
  const shape = sm[1].split(",").map(s => s.trim()).filter(s => s.length).map(s => parseInt(s, 10));
  const dataStart = headerStart + headerLen;
  const data = buf.subarray(dataStart);
  let arr;
  if (descr === "<i4")      arr = new Int32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  else if (descr === "<f4") arr = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  else if (descr === "<f8") arr = new Float64Array(data.buffer, data.byteOffset, data.byteLength / 8);
  else throw new Error("npy: dtype " + descr + " not handled");
  // copy to detach from the underlying buffer (fixes alignment + ownership across reuse)
  const flat = arr.slice();
  if (shape.length === 1) return { shape, data: flat };
  if (shape.length === 2) {
    const [r, c] = shape;
    const rows = new Array(r);
    for (let k = 0; k < r; k++) rows[k] = flat.subarray(k * c, (k + 1) * c);
    return { shape, data: flat, rows };
  }
  return { shape, data: flat };
}

// compute the four CONFIDENCE_MATH components from per-layer lens logits + norms.
// last layer's argmax is the sampled byte (we have no temperature sampling here).
// note: residual_stab approximated from residual_norms (no embedding access in npz);
// document this in the panel description.
function compute_confidence_components(lensRows, residualNorms, vocabSize) {
  const L = lensRows.length;
  if (L === 0) return null;
  const last = lensRows[L - 1];
  const V = vocabSize || last.length;
  // softmax-ish: lens_logits are int32 scaled ×1000. divide and softmax with 8.0 cap (matches app.py path).
  function softmax_top2(row) {
    let mx = -Infinity, mxIdx = 0;
    for (let v = 0; v < V; v++) { if (row[v] > mx) { mx = row[v]; mxIdx = v; } }
    let absMax = 1;
    for (let v = 0; v < V; v++) { const a = Math.abs(row[v]); if (a > absMax) absMax = a; }
    const probs = new Float64Array(V);
    let sum = 0;
    for (let v = 0; v < V; v++) {
      const s = row[v] / absMax * 8.0;
      const e = Math.exp(s - 8.0);  // numerical guard
      probs[v] = e; sum += e;
    }
    for (let v = 0; v < V; v++) probs[v] /= sum;
    let p1 = -Infinity, p2 = -Infinity, i1 = 0;
    for (let v = 0; v < V; v++) {
      if (probs[v] > p1) { p2 = p1; p1 = probs[v]; i1 = v; }
      else if (probs[v] > p2) { p2 = probs[v]; }
    }
    return { probs, top1: i1, p1, p2 };
  }
  const sm = softmax_top2(last);
  const sampled = sm.top1;
  // margin: (logit_top - logit_second) / sigma_logit. sigma = std of last layer's logits.
  let mean = 0;
  for (let v = 0; v < V; v++) mean += last[v];
  mean /= V;
  let varSum = 0;
  for (let v = 0; v < V; v++) { const d = last[v] - mean; varSum += d * d; }
  const sigma = Math.sqrt(varSum / V) || 1;
  // top1/top2 raw logits
  let r1 = -Infinity, r2 = -Infinity;
  for (let v = 0; v < V; v++) {
    if (last[v] > r1) { r2 = r1; r1 = last[v]; }
    else if (last[v] > r2) { r2 = last[v]; }
  }
  const margin = (r1 - r2) / sigma;
  // entropy: 1 - H(p)/log2(V)
  let H = 0;
  for (let v = 0; v < V; v++) {
    const p = sm.probs[v];
    if (p > 1e-12) H -= p * Math.log2(p);
  }
  const entropy = 1 - H / Math.log2(V);
  // lens consistency: fraction of layers whose argmax matches sampled byte.
  let agree = 0;
  for (let l = 0; l < L; l++) {
    let mxv = -Infinity, mxi = 0;
    const row = lensRows[l];
    for (let v = 0; v < V; v++) { if (row[v] > mxv) { mxv = row[v]; mxi = v; } }
    if (mxi === sampled) agree++;
  }
  const lens_consistency = agree / L;
  // residual stab proxy: 1 - normalized std of residual_norm deltas. smooth norm climb = stable commit.
  let stab = 0;
  if (residualNorms && residualNorms.length >= 2) {
    const n = residualNorms.length;
    let m = 0;
    for (let l = 0; l < n; l++) m += residualNorms[l];
    m /= n;
    const deltas = [];
    for (let l = 1; l < n; l++) deltas.push(residualNorms[l] - residualNorms[l - 1]);
    let dm = 0;
    for (const d of deltas) dm += d;
    dm /= deltas.length;
    let dv = 0;
    for (const d of deltas) dv += (d - dm) * (d - dm);
    const dstd = Math.sqrt(dv / deltas.length);
    stab = Math.max(0, Math.min(1, 1 - dstd / Math.max(m, 1e-6)));
  }
  return { sampled, margin, entropy, lens_consistency, residual_stab: stab };
}

async function classroomFetchLens(run, fname) {
  const url = `/timeline/${encodeURIComponent(run)}/${encodeURIComponent(fname)}?` + Date.now();
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error("HTTP " + r.status);
  const buf = await r.arrayBuffer();
  const obj = await parse_npz(buf);
  const ll = obj["lens_logits.npy"];
  const rn = obj["residual_norms.npy"];
  if (!ll || !rn) throw new Error("npz missing lens_logits/residual_norms");
  return { lens_rows: ll.rows, residual_norms: rn.data };
}

async function classroomFetchProbe(run, fname) {
  const url = `/timeline/${encodeURIComponent(run)}/${encodeURIComponent(fname)}?` + Date.now();
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return await r.json();
}

// shape -> total params. matches PyTorch Veritate (embeddings + per-layer Q/K/V/O + ffn up/down + lm_head tied).
// returns { total, parts: {embed, attn, ffn, head} }.
function compute_param_count(shape) {
  if (!shape) return null;
  const V = shape.vocab|0, H = shape.hidden|0, F = shape.ffn|0, L = shape.layers|0;
  if (!V || !H || !F || !L) return null;
  const embed = V * H;                       // tied lm head shares this
  const attn  = L * (4 * H * H);             // Q,K,V,O
  const ffn   = L * (2 * H * F);             // up + down
  const ln    = L * (2 * 2 * H) + 2 * H;     // 2 layernorms per block + final ln (gain+bias)
  const total = embed + attn + ffn + ln;
  return { total, parts: { embed, attn, ffn, ln } };
}

function fmt_bytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + " MB";
  return (n / (1024 * 1024 * 1024)).toFixed(2) + " GB";
}

const L3_BUDGET_BYTES = 96 * 1024 * 1024;  // 9800X3D L3

function render_size_meter(refs, run, cfg) {
  const root = $(refs.sizeMeterId);
  if (!cfg) {
    root.innerHTML = `<span class="meta">no config.json for run <b>${run}</b></span>`;
    return;
  }
  const pc = compute_param_count(cfg.shape);
  if (!pc) {
    root.innerHTML = `<span class="meta">config.json missing <code>shape</code> for run <b>${run}</b></span>`;
    return;
  }
  const int8 = pc.total;
  const int4 = Math.ceil(pc.total / 2);
  const fits = int8 <= L3_BUDGET_BYTES;
  const fitColor = fits ? "#5dff9b" : "#ff7d5d";
  const fitLabel = fits ? "FITS L3" : "SPILLS TO DRAM";
  const pct = (int8 / L3_BUDGET_BYTES * 100).toFixed(1);
  root.innerHTML = `
    <table>
      <tr><td class="l" style="text-align:left">run</td><td style="text-align:right">${run}</td></tr>
      <tr><td class="l" style="text-align:left">shape</td><td style="text-align:right">V=${cfg.shape.vocab}  H=${cfg.shape.hidden}  L=${cfg.shape.layers}  F=${cfg.shape.ffn}</td></tr>
      <tr><td class="l" style="text-align:left">total params</td><td style="text-align:right"><b>${pc.total.toLocaleString()}</b></td></tr>
      <tr><td class="l" style="text-align:left">  embed (tied)</td><td style="text-align:right">${pc.parts.embed.toLocaleString()}</td></tr>
      <tr><td class="l" style="text-align:left">  attention</td><td style="text-align:right">${pc.parts.attn.toLocaleString()}</td></tr>
      <tr><td class="l" style="text-align:left">  ffn</td><td style="text-align:right">${pc.parts.ffn.toLocaleString()}</td></tr>
      <tr><td class="l" style="text-align:left">INT8 weight bytes</td><td style="text-align:right">${fmt_bytes(int8)}  (${pct}% of 96 MB L3)</td></tr>
      <tr><td class="l" style="text-align:left">INT4 weight bytes</td><td style="text-align:right">${fmt_bytes(int4)}</td></tr>
      <tr><td class="l" style="text-align:left">L3 fit (9800X3D)</td><td style="text-align:right"><span style="background:${fits ? "#103025" : "#330a0a"};color:${fitColor};padding:2px 8px;border-radius:3px;font-weight:700">${fitLabel}</span></td></tr>
    </table>`;
}

function _prune_region_class(layer, totalLayers) {
  const t = totalLayers || 1;
  if (layer < t / 3)       return "region-sense";
  if (layer < (2 * t) / 3) return "region-assoc";
  return "region-output";
}

function _prune_region_chip(layer, totalLayers) {
  const c = _prune_region_class(layer, totalLayers);
  if (c === "region-sense")  return "cool";
  if (c === "region-assoc")  return "warm";
  return "hot";
}

function _prune_verdict(keep) {
  if (keep <= 0.30) return { cls: "aggressive", label: "prune hard" };
  if (keep <= 0.55) return { cls: "moderate",   label: "prune" };
  if (keep <= 0.85) return { cls: "trim",       label: "trim" };
  return { cls: "keep", label: "keep" };
}

async function load_pruning_report(refs, run) {
  const body = $(refs.pruneBodyId);
  const sub  = $(refs.pruneSubtitleId);
  if (!body) return;
  body.innerHTML = `<span class="meta">measuring activity (one forward pass over a corpus sample)…</span>`;
  if (sub) sub.textContent = `model: ${run} · scanning…`;
  let r;
  try {
    const resp = await fetch(`/pruning/report?model=${encodeURIComponent(run)}&samples=24`,
                             { cache: "no-store" });
    r = await resp.json();
  } catch (e) {
    body.innerHTML = `<span style="color:var(--hot)">${_backendErrMsg(e)}</span>`;
    return;
  }
  if (!r.ok) {
    body.innerHTML = `<span style="color:var(--warm)">${r.error || "no data"}</span>`;
    if (sub) sub.textContent = `model: ${run}`;
    return;
  }
  render_pruning_report(refs, r);
}

function render_pruning_report(refs, r) {
  const body = $(refs.pruneBodyId);
  const sub  = $(refs.pruneSubtitleId);
  if (sub) {
    sub.innerHTML = `model: <span style="color:var(--text)">${r.model}</span> · step ${r.step.toLocaleString()} · `
                  + `<b style="color:var(--warm)">${r.dead_pct}% dead weight</b> · `
                  + `${(r.n_params/1e6).toFixed(0)}M → <b style="color:var(--data-pos)">${(r.n_params_after/1e6).toFixed(0)}M</b> params · `
                  + `<b style="color:var(--data-pos)">${r.size_mb_after} MB</b> after`;
  }
  const layers = r.per_layer.length;
  const planByKeep = {};
  for (const e of r.per_layer) {
    const kk = e.keep.toFixed(2);
    if (!planByKeep[kk]) planByKeep[kk] = [];
    planByKeep[kk].push(e);
  }
  const tiers = Object.keys(planByKeep).sort((a, b) => parseFloat(a) - parseFloat(b));

  let rowsHtml = "";
  for (const e of r.per_layer) {
    const region = _prune_region_class(e.layer, layers);
    const regionLabel = region === "region-sense" ? "sense" :
                        region === "region-assoc" ? "association" : "output";
    const v = _prune_verdict(e.keep);
    const pct = (e.alive_frac * 100).toFixed(0);
    rowsHtml += `<div class="row ${region}">
      <span class="layer-tag">layer ${e.layer}</span>
      <span class="region">${regionLabel}</span>
      <div class="meter"><div class="fill" style="width:${pct}%"></div></div>
      <span class="pct">${pct}%</span>
      <span class="verdict-tag ${v.cls}">${v.label}</span>
    </div>`;
  }

  let planTableHtml = "";
  for (const kk of tiers) {
    const keep = parseFloat(kk);
    const v = _prune_verdict(keep);
    const widthLbl = `${Math.round(keep * 100)}% width`;
    const chips = planByKeep[kk].map(e => {
      const chipCls = _prune_region_chip(e.layer, layers);
      return `<span class="chip ${chipCls}">layer ${e.layer}</span>`;
    }).join("");
    planTableHtml += `<tr><td class="action ${v.cls}">${widthLbl}</td><td>${chips}</td></tr>`;
  }

  body.innerHTML = `
    <div class="prune-layers">
      <div class="header-row">
        <span>layer</span><span>region</span><span>active fraction</span><span>active</span><span>recommend</span>
      </div>
      ${rowsHtml}
    </div>

    <div class="prune-actions">
      <button class="btn primary" id="pruneGenBtn">generate pruning plugin</button>
      <button class="btn" id="pruneRefreshBtn">re-measure</button>
      <span class="status" id="pruneStatus"></span>
    </div>

    <details class="more"><summary>view proposed plan</summary><div class="more-content">
      <table class="plan-table">${planTableHtml}</table>
    </div></details>

    <details class="more"><summary>how this chains with training</summary><div class="more-content">
      <b>Generate pruning plugin</b> writes <code>plugins/prune_${r.model}_step${r.step}/</code>
      with a manifest, the plan as JSON, and a one-shot script. Run it from the Training tab;
      it produces a new pruned model alongside the original.
    </div></details>
  `;

  $("pruneGenBtn").addEventListener("click", async () => {
    const btn = $("pruneGenBtn");
    const st  = $("pruneStatus");
    btn.disabled = true;
    st.className = "status";
    st.textContent = "generating…";
    try {
      const resp = await fetch("/pruning/generate_plugin", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ model: r.model, step: r.step, plan: r.plan, samples: 8 })
      });
      const d = await resp.json();
      if (d.ok) {
        st.className = "status ok";
        st.textContent = `wrote plugins/${d.plugin_id}/ — open the Training tab to run it`;
      } else {
        st.className = "status err";
        st.textContent = `failed: ${d.error || "unknown"}`;
      }
    } catch (e) {
      st.className = "status err";
      st.textContent = _backendErrMsg(e);
    } finally {
      btn.disabled = false;
    }
  });

  $("pruneRefreshBtn").addEventListener("click", () => load_pruning_report(refs, r.model));
}

function render_confidence_evo(refs, run, steps, confByStep) {
  const have = steps.filter(s => s.lens && confByStep[s.step]);
  if (have.length === 0) {
    plotTrainSeries(refs.confCanvas, refs.confCtx, []);
    return;
  }
  const margin       = have.map(s => ({ x: s.step, y: confByStep[s.step].margin }));
  const entropy      = have.map(s => ({ x: s.step, y: confByStep[s.step].entropy }));
  const consistency  = have.map(s => ({ x: s.step, y: confByStep[s.step].lens_consistency }));
  const stab         = have.map(s => ({ x: s.step, y: confByStep[s.step].residual_stab }));
  // normalize margin to roughly [0,1] for co-plotting (margin can range wide). divide by max abs.
  const maxAbs = margin.reduce((m, p) => Math.max(m, Math.abs(p.y)), 1);
  const marginN = margin.map(p => ({ x: p.x, y: p.y / maxAbs }));
  plotTrainSeries(refs.confCanvas, refs.confCtx, [
    { points: marginN,     color: "#5dc8ff", lw: 1.5 },
    { points: entropy,     color: "#ffae5d", lw: 1.5 },
    { points: consistency, color: "#5dff9b", lw: 2   },
    { points: stab,        color: "#ff5d9b", lw: 1.5 },
  ]);
}

function render_lens_drift(refs, run, steps, lensByStep) {
  const root = $(refs.lensDriftId);
  const have = steps.filter(s => s.lens && lensByStep[s.step]);
  if (have.length === 0) {
    root.innerHTML = `<span style="color:var(--hot)">Lens capture is not configured for this run.</span>`;
    return;
  }
  const layers = lensByStep[have[0].step].lens_rows.length;
  function byte_glyph(b) {
    if (b === 32) return "<span style='color:var(--dim)'>␣</span>";
    if (b === 10) return "<span style='color:var(--dim)'>↵</span>";
    if (b === 9)  return "<span style='color:var(--dim)'>→</span>";
    if (b < 32 || b === 127 || b > 126) return "<span style='color:var(--dim)'>·" + b + "</span>";
    const c = String.fromCharCode(b);
    const safe = c === "<" ? "&lt;" : c === ">" ? "&gt;" : c === "&" ? "&amp;" : c;
    return safe;
  }
  let html = `<table style="font-size:10.5px"><tr><th style="text-align:left">layer</th>`;
  for (const s of have) html += `<th>step ${s.step.toLocaleString()}</th>`;
  html += `</tr>`;
  for (let L = 0; L < layers; L++) {
    html += `<tr><td style="text-align:left;color:var(--dim)">L${L}</td>`;
    for (const s of have) {
      const row = lensByStep[s.step].lens_rows[L];
      const V = row.length;
      // top-3 raw logit picks (raw int32 ordering — same as argmax).
      const idx = Array.from({length: V}, (_, k) => k);
      idx.sort((a, b) => row[b] - row[a]);
      const top = idx.slice(0, 3);
      // softmax on the row to get probs (mirror parse_npz path).
      let absMax = 1;
      for (let v = 0; v < V; v++) { const a = Math.abs(row[v]); if (a > absMax) absMax = a; }
      let sum = 0;
      const probs = new Float64Array(V);
      for (let v = 0; v < V; v++) {
        const e = Math.exp(row[v] / absMax * 8.0 - 8.0);
        probs[v] = e; sum += e;
      }
      for (let v = 0; v < V; v++) probs[v] /= sum;
      const cells = top.map(b => `<span style='display:inline-block;min-width:34px;text-align:center'><b>${byte_glyph(b)}</b><span style='color:var(--dim)'>·${(probs[b] * 100).toFixed(0)}%</span></span>`).join(" ");
      html += `<td style="text-align:left">${cells}</td>`;
    }
    html += `</tr>`;
  }
  html += `</table>`;
  root.innerHTML = html;
}

// shared progress-bar renderer used by every smartness-meter axis (reading,
// math, grammar, reasoning) so all ladders look identical. ticks are vertical
// hairlines that mark threshold boundaries (e.g. emerging / fluent), passed
// in as { pct: 0..100, opacity: 0..1 } pairs.
function renderProgressBar(pct, color, ticks) {
  const safe = Math.max(0, Math.min(100, pct));
  let tickHtml = "";
  if (ticks && ticks.length) {
    for (const t of ticks) {
      tickHtml += `<span style="position:absolute;left:${t.pct}%;top:0;height:100%;width:1px;background:rgba(255,255,255,${t.opacity})"></span>`;
    }
  }
  return `<div style="background:#1f242e;border-radius:2px;height:9px;overflow:hidden;position:relative"><span style="display:block;height:100%;width:${safe}%;background:${color}"></span>${tickHtml}</div>`;
}

// reading-level grade ladder (Pre-K -> PhD). matches docs/notes/GRADING_SCALE.md
// and training/checkpoint_probe.py::GRADE_LEVELS / GRADE_PPL_PASS.
const GRADE_ORDER       = ["prek", "k", "elem", "middle", "hs", "college", "phd"];
const GRADE_LABEL       = { prek: "Pre-K", k: "K", elem: "Elem", middle: "Middle", hs: "HS", college: "College", phd: "PhD" };
const GRADE_AGE         = { prek: "ages 3-4", k: "ages 5-6", elem: "ages 7-9", middle: "ages 10-13", hs: "ages 14-17", college: "ages 18-22", phd: "ages 23+" };
const GRADE_CORPUS      = { prek: "early reader", k: "primary narrative", elem: "chapter book", middle: "middle grade", hs: "literary novel", college: "academic essay", phd: "research abstract" };
const GRADE_CORPUS_FULL = { prek: "early-reader narrative, ~5-word sentences", k: "primary narrative, ~8-word sentences", elem: "chapter-book narrative, ~12-word sentences", middle: "middle-grade narrative, ~16-word sentences", hs: "literary novel prose, ~20-word sentences", college: "academic essay, ~24-word sentences", phd: "research abstract, ~28-word sentences" };
const GRADE_SENT_LEN    = { prek: "~5w sent", k: "~8w sent", elem: "~12w sent", middle: "~16w sent", hs: "~20w sent", college: "~24w sent", phd: "~28w sent" };
// Default thresholds used when a checkpoint lacks ppl_threshold (older grades_*.json
// files written before the relative-threshold change). The probe now writes a
// per-checkpoint floor + threshold derived from the model's best band, so each
// checkpoint can supply its own number; these constants are only the fallback.
const GRADE_FLUENT_PPL_DEFAULT  = 3.0;
const GRADE_EMERGING_FACTOR     = 1.5;  // emerging band = within this factor above the fluent threshold
const GRADE_PPL_CEILING         = 32.0; // matches checkpoint_probe.GRADE_PPL_CEILING

// Effective fluent threshold for a given grades_*.json record. Falls back to
// the constant when the per-checkpoint field is missing.
function gradeFluentPpl(record) {
  if (record && typeof record.ppl_threshold === "number") return record.ppl_threshold;
  return GRADE_FLUENT_PPL_DEFAULT;
}
function gradeEmergingPpl(record) {
  return gradeFluentPpl(record) * GRADE_EMERGING_FACTOR;
}

function gradeIndex(name) {
  const i = GRADE_ORDER.indexOf((name || "").toLowerCase());
  return i < 0 ? 0 : i;
}

// highest band the model is fluent at (ppl < threshold and below sanity ceiling). null if no band passes.
function highestPassingGrade(grades, record) {
  const fluent = gradeFluentPpl(record);
  let best = null;
  for (const g of GRADE_ORDER) {
    const e = grades && grades[g];
    if (e && typeof e.ppl === "number" && e.ppl < fluent && e.ppl < GRADE_PPL_CEILING) best = g;
  }
  return best;
}

function render_reading_level(refs, run, gradesSteps, gradesByStep, haveCheckpoints, config) {
  const root = $(refs.readLevelId);
  const canvas = refs.readGradeCanvas;
  const ctx = refs.readGradeCtx;
  if (!gradesSteps || gradesSteps.length === 0) {
    root.innerHTML = haveCheckpoints
      ? `<span style="color:var(--hot)">Reading-level grading is not configured for this run.</span>`
      : `<span style="color:var(--warm)">No checkpoint yet.</span>`;
    plotTrainSeries(canvas, ctx, []);
    return;
  }
  const latestStep = gradesSteps[gradesSteps.length - 1];
  const latest = gradesByStep[latestStep];
  if (!latest || !latest.grades) {
    root.innerHTML = `<span class="meta">latest grades_step_${latestStep}.json missing grades field</span>`;
    plotTrainSeries(canvas, ctx, []);
    return;
  }

  // legend: chips matching the trajectory line colors. lives above the chart
  // so the user can match band names to lines at a glance.
  if (refs.readLegendId) {
    const legendRoot = $(refs.readLegendId);
    if (legendRoot) {
      let lhtml = `<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;font-size:11px">`;
      for (const g of GRADE_ORDER) {
        lhtml += `<span style="display:inline-flex;align-items:center;gap:5px"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${GRADE_BAND_COLOR[g]}"></span>${GRADE_LABEL[g]}</span>`;
      }
      lhtml += `</div>`;
      legendRoot.innerHTML = lhtml;
    }
  }

  let html = "";
  // disclaimer: this metric measures vocabulary/prose-style familiarity, NOT
  // comprehension or generation. byte-ppl over a passage tells you whether the
  // model recognizes the words and local syntactic patterns of that register;
  // it doesn't say the model can answer questions about the text, write at
  // that level, or maintain long-range coherence. surface this up front so the
  // label "reading level" isn't read as "intelligence level".
  html += `<div class="desc" style="margin:0 0 10px;padding:8px 10px;border-left:3px solid var(--warm);background:rgba(255,170,80,0.06);font-size:11.5px;line-height:1.45">
    <b style="color:var(--warm)">What this measures &mdash; and what it doesn't.</b>
    This score reflects <b>which words and prose patterns the model knows</b>, not what it understands.
    It's the model being <i>read to</i>: at each byte we measure how predictable the next byte is. Low ppl on a college passage means the model has seen text like that before, not that it grasps college-level concepts.
    It does <b>not</b> measure: comprehension (the model is never asked a question about the passage), long-range coherence over paragraphs, or generation flow (writing at that level). It only measures local familiarity with the prose register.
  </div>`;
  // ladder rows: every band, with state badge + ppl + gap-to-fluent. corpus
  // source is folded into the leftmost cell so the eval-text origin sits next
  // to its score — band labels alone don't explain ppl, the corpus does. A
  // colored swatch matches the band's trajectory line for quick eye linkage.
  // Per-checkpoint fluent threshold: probe writes ppl_threshold = floor * 1.5
  // where floor = best band ppl (clamped). Older checkpoints lacking the field
  // fall back to the absolute 3.0 default.
  const fluentPpl = gradeFluentPpl(latest);
  const emergingPpl = gradeEmergingPpl(latest);
  if (typeof latest.ppl_threshold === "number" && typeof latest.ppl_floor === "number") {
    html += `<div class="desc" style="margin:0 0 10px;font-size:11px;color:var(--dim)">Threshold for this checkpoint: <b style="color:var(--text)">ppl &lt; ${fluentPpl.toFixed(2)}</b> (model floor ${latest.ppl_floor.toFixed(2)} &times; ${(latest.ppl_threshold_factor || GRADE_EMERGING_FACTOR).toFixed(2)}). Bands above ppl ${GRADE_PPL_CEILING} are blocked regardless &mdash; that's the random-output sanity ceiling.</div>`;
  }
  html += `<div style="display:grid;grid-template-columns:150px minmax(120px, 280px) 110px 150px;gap:6px 12px;font-size:11.5px;align-items:center">`;
  for (const g of GRADE_ORDER) {
    const e = latest.grades[g];
    const lbl = GRADE_LABEL[g];
    const age = GRADE_AGE[g];
    const corpus = GRADE_CORPUS[g];
    const corpusFull = GRADE_CORPUS_FULL[g];
    const sentLen = GRADE_SENT_LEN[g];
    const swatch = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${GRADE_BAND_COLOR[g]};margin-right:6px;vertical-align:middle"></span>`;
    const labelCell = `<div style="text-align:right" title="${escapeHtml(corpusFull)}">
      <b>${swatch}${lbl}</b>
      <div class="meta" style="font-size:10px">${age}</div>
      <div class="meta" style="font-size:10px;font-style:italic;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(corpus)} &middot; ${sentLen}</div>
    </div>`;
    if (!e || typeof e.ppl !== "number") {
      html += `${labelCell}<div style="grid-column:span 3"><span class="meta">no data</span></div>`;
      continue;
    }
    const ppl = e.ppl;
    const passed = ppl < fluentPpl && ppl < GRADE_PPL_CEILING;
    const emerging = !passed && ppl < emergingPpl && ppl < GRADE_PPL_CEILING;
    // bar: log-scale progress from random (ppl 256) to fluent (per-checkpoint).
    const LOG_RANDOM = Math.log(256), LOG_FLUENT = Math.log(fluentPpl);
    const fluencyPct = Math.max(2, Math.min(100, ((LOG_RANDOM - Math.log(Math.max(ppl, 1))) / (LOG_RANDOM - LOG_FLUENT)) * 100));
    const color = passed ? "#5dff9b" : (emerging ? "var(--warm)" : "var(--hot)");
    const fluentTip   = `FLUENT: ppl < ${fluentPpl.toFixed(2)} (under ${(Math.log2(fluentPpl)).toFixed(1)} bits/byte uncertainty). ppl 1.0 = perfect, ppl 256 = random.`;
    const emergingTip = `EMERGING: ppl ${fluentPpl.toFixed(2)} - ${emergingPpl.toFixed(2)}. Lower is better.`;
    const notYetTip   = `NOT YET: ppl > ${emergingPpl.toFixed(2)}. Red bands = prose styles the model rarely sees in training, not harder content.`;
    const badge = passed
      ? `<span class="case" style="background:#103025;color:#5dff9b" title="${fluentTip}">FLUENT</span>`
      : (emerging
        ? `<span class="case b-mid" title="${emergingTip}">EMERGING</span>`
        : `<span class="case" style="background:#330a0a;color:#ff7d5d" title="${notYetTip}">NOT YET</span>`);
    const gap = passed
      ? `ppl <b style="color:#5dff9b">${ppl.toFixed(2)}</b> <span class="meta">&lt; ${fluentPpl.toFixed(2)}</span>`
      : `ppl <b style="color:var(--hot)">${ppl.toFixed(2)}</b> <span class="meta">+${(ppl - fluentPpl).toFixed(2)} over</span>`;
    html += `${labelCell}
      ${renderProgressBar(fluencyPct, color, [{ pct: 91, opacity: 0.28 }])}
      <div style="text-align:left">${badge}</div>
      <div style="text-align:right">${gap}</div>`;
  }
  html += `</div>`;

  root.innerHTML = html;

  // trajectory: per-band fluency % over checkpoints. UP = better. converts
  // each ppl into the same log-scale 0..100% the ladder bars use, so the
  // bar pct and the trajectory share an identical y-axis scale. every band
  // is anchored at (step 0, fluency 0) — random init = ppl ~256 = 0% — so
  // all lines depart from the same origin and the chart shows a journey
  // toward fluent rather than auto-fitting around the first measurement.
  // x-axis is reserved out to total_steps from the manifest, so future
  // checkpoints land at predictable positions rather than reshuffling the
  // axis each time. threshold lines at 91 (emerging) and 100 (fluent).
  const LOG_RANDOM_T = Math.log(256);
  // Trajectory uses each checkpoint's own fluent threshold so the y-axis
  // 100%-line tracks the model's evolving "fluent" definition.
  const fluencyOfRecord = (ppl, record) => {
    const fluent = gradeFluentPpl(record);
    const denom = LOG_RANDOM_T - Math.log(fluent);
    return Math.max(0, Math.min(100,
      ((LOG_RANDOM_T - Math.log(Math.max(ppl, 1))) / denom) * 100
    ));
  };
  const bandSeries = GRADE_ORDER.map(g => ({ g, points: [{ x: 0, y: 0 }] }));
  for (const step of gradesSteps) {
    const r = gradesByStep[step];
    if (!r || !r.grades) continue;
    for (const s of bandSeries) {
      const e = r.grades[s.g];
      if (e && typeof e.ppl === "number" && e.ppl > 0) {
        s.points.push({ x: step, y: fluencyOfRecord(e.ppl, r) });
      }
    }
  }
  const series = bandSeries
    .filter(s => s.points.length > 1)  // need real measurement, not just origin
    .map(s => ({ points: s.points, color: GRADE_BAND_COLOR[s.g], lw: 1.5, dots: true }));
  const totalSteps = (config && config.training_args && config.training_args.total_steps) || null;
  plotTrainSeries(canvas, ctx, series, {
    yMinFloor: 0,
    yMaxCeil:  108,                       // headroom so the FLUENT label at y=100 isn't clipped
    xMinFloor: 0,
    xMaxCeil:  totalSteps || undefined,
    yTitle: "fluency % (higher = better)",
    thresholdLines: [
      { y: 100, color: "#5dff9b", label: "FLUENT (ppl 3.0)", lineDash: [4, 3] },
      { y: 91,  color: "#ffae5d", label: "EMERGING (ppl 4.5)", lineDash: [3, 4] },
    ],
  });
}

// per-band trajectory colors. red->violet rainbow tracks reading-difficulty
// order (Pre-K low / red, PhD high / violet). matches the ladder ordering.
const GRADE_BAND_COLOR = {
  prek:    "#ff5d5d",
  k:       "#ff9a5d",
  elem:    "#ffe45d",
  middle:  "#5dff9b",
  hs:      "#5dd6ff",
  college: "#5d9bff",
  phd:     "#b58aff",
};

// ----------------------------------------------------------------------------
// Score-axis rendering: shared by math, grammar, reasoning panels. Each axis
// is a list of tiers/types where the model scores 0..1 accuracy. Pass mark is
// SCORE_PASS = 0.80; "emerging" sits between SCORE_EMERGING (0.50) and pass.
// JSON shape:
//   math      -> { tiers: { tier: { correct, total, accuracy } } }
//   reasoning -> { tiers: { tier: { ... } } }
//   grammar   -> { types: { type: { ... } } }
const SCORE_PASS_PCT     = 0.80;
const SCORE_EMERGING_PCT = 0.50;

const AXIS_META = {
  math: {
    rootKey:  "mathLevelId",
    fieldKey: "tiers",
    order:    ["t1_arith1", "t2_arith2", "t3_algebra", "t4_word", "t5_multi"],
    label:    { t1_arith1: "Arithmetic 1",  t2_arith2: "Arithmetic 2", t3_algebra: "Algebra 1",
                t4_word:   "Word Problems", t5_multi:  "Multi-Step" },
    sub:      { t1_arith1: "single-digit + / -", t2_arith2: "two-digit + / - / *",
                t3_algebra: "x + b = c, solve",  t4_word:   "one-op story problems",
                t5_multi:   "(a + b) * c style" },
    colors:   { t1_arith1: "#ff5d5d", t2_arith2: "#ff9a5d", t3_algebra: "#ffe45d",
                t4_word:   "#5dff9b", t5_multi:  "#5d9bff" },
    title:    "math fluency",
    blurb:    "% correct via argmax-decode and string match. Floor on a TinyStories-trained model is expected — math curriculum has to be in training data for this to move.",
  },
  grammar: {
    rootKey:  "grammarLevelId",
    fieldKey: "types",
    order:    ["sv_agreement", "articles", "tense", "word_order"],
    label:    { sv_agreement: "Subject-Verb", articles: "Articles", tense: "Tense", word_order: "Word Order" },
    sub:      { sv_agreement: "agreement (cats sleep / sleeps)", articles: "a / an / the",
                tense: "past / present consistency", word_order: "constituent order" },
    colors:   { sv_agreement: "#ff5d5d", articles: "#ffae5d", tense: "#5dff9b", word_order: "#5d9bff" },
    title:    "grammar preference",
    blurb:    "Pairwise: lower mean per-byte NLL on the correct sentence vs. the mutated one counts as a preference. Raw fluency, no decoding.",
  },
  reasoning: {
    rootKey:  "reasoningLevelId",
    fieldKey: "tiers",
    order:    ["recall", "pattern", "deduction1", "deduction_n"],
    label:    { recall: "Recall", pattern: "Analogy", deduction1: "1-step Deduction", deduction_n: "Multi-step" },
    sub:      { recall: "fact completion (capitals, basic facts)", pattern: "cat:kitten :: dog:?",
                deduction1: "All A are B; X is A; so X is ?", deduction_n: "transitive ordering chains" },
    colors:   { recall: "#ff5d5d", pattern: "#ffae5d", deduction1: "#5dff9b", deduction_n: "#5d9bff" },
    title:    "reasoning",
    blurb:    "% correct via argmax-decode and string match. Recall and analogy will move first; deductions only after the model sees similar chains in training.",
  },
};

function _scoreColor(acc) {
  if (acc >= SCORE_PASS_PCT)     return "#5dff9b";
  if (acc >= SCORE_EMERGING_PCT) return "var(--warm)";
  return "var(--hot)";
}

function _scoreBadge(acc) {
  if (acc >= SCORE_PASS_PCT)
    return `<span class="case" style="background:#103025;color:#5dff9b">FLUENT</span>`;
  if (acc >= SCORE_EMERGING_PCT)
    return `<span class="case b-mid">EMERGING</span>`;
  return `<span class="case" style="background:#330a0a;color:#ff7d5d">NOT YET</span>`;
}

// Highest passing tier index (>= SCORE_PASS_PCT). null if none.
function _highestPassingTier(entries, order) {
  let best = null;
  for (const k of order) {
    const e = entries && entries[k];
    if (e && typeof e.accuracy === "number" && e.accuracy >= SCORE_PASS_PCT) best = k;
  }
  return best;
}

function render_score_axis(axisName, refs, axisSteps, axisByStep, haveCheckpoints) {
  const meta = AXIS_META[axisName];
  if (!meta) return;
  const root = $(refs[meta.rootKey]);
  if (!root) return;
  if (!axisSteps || axisSteps.length === 0) {
    root.innerHTML = haveCheckpoints
      ? `<span class="meta">no ${axisName} probe yet for this run</span>`
      : `<span style="color:var(--warm)">No checkpoint yet.</span>`;
    return;
  }
  const latestStep = axisSteps[axisSteps.length - 1];
  const latest = axisByStep[latestStep];
  const entries = latest && latest[meta.fieldKey];
  if (!entries) {
    root.innerHTML = `<span class="meta">latest ${axisName}_step_${latestStep}.json missing ${meta.fieldKey} field</span>`;
    return;
  }

  let html = `<div style="display:grid;grid-template-columns:160px minmax(120px, 260px) 110px 130px;gap:6px 12px;font-size:11.5px;align-items:center">`;
  for (const k of meta.order) {
    const e = entries[k];
    const lbl = meta.label[k];
    const sub = meta.sub[k] || "";
    const tierColor = (meta.colors && meta.colors[k]) || "var(--accent)";
    const swatch = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${tierColor};margin-right:6px;vertical-align:middle"></span>`;
    const labelCell = `<div style="text-align:right">
      <b>${swatch}${lbl}</b>
      <div class="meta" style="font-size:10px;font-style:italic;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(sub)}</div>
    </div>`;
    if (!e || typeof e.accuracy !== "number") {
      html += `${labelCell}<div style="grid-column:span 3"><span class="meta">no data</span></div>`;
      continue;
    }
    const acc = e.accuracy;
    const pct = Math.max(2, Math.min(100, acc * 100));
    const color = _scoreColor(acc);
    const badge = _scoreBadge(acc);
    const accColor = acc >= SCORE_PASS_PCT ? "#5dff9b" : (acc >= SCORE_EMERGING_PCT ? "var(--warm)" : "var(--hot)");
    const right = `<b style="color:${accColor}">${(acc * 100).toFixed(0)}%</b> <span class="meta">${e.correct}/${e.total}</span>`;
    html += `${labelCell}
      ${renderProgressBar(pct, color, [{ pct: 50, opacity: 0.18 }, { pct: 80, opacity: 0.28 }])}
      <div style="text-align:left">${badge}</div>
      <div style="text-align:right">${right}</div>`;
  }
  html += `</div>`;

  html += `<p class="desc" style="margin-top:10px">${escapeHtml(meta.blurb)}</p>`;

  root.innerHTML = html;
}

// concept categories. derived from training/checkpoint_probe.py::CONCEPTS — any
// concept added there but missing from a group will still show in the heatmap
// detail at the bottom; only the grouped strips are categorised.
const CONCEPT_GROUPS = [
  { name: "objects",    blurb: "everyday nouns the model sees constantly in TinyStories",  words: ["cat","dog","bird","fish","tree","house","car","ball","water","food","baby"] },
  { name: "emotions",   blurb: "internal states; common in story phrasing",                 words: ["happy","sad","angry","scared","kind","love"] },
  { name: "family",     blurb: "relationship words",                                        words: ["friend","mother","father"] },
  { name: "colors",     blurb: "modifiers tied to common nouns",                            words: ["red","blue","green","yellow"] },
  { name: "attributes", blurb: "size / temperature / speed adjectives",                     words: ["big","small","hot","cold","fast","slow"] },
  { name: "actions",    blurb: "verbs the model has to predict in motion contexts",         words: ["run","jump","eat","sleep","walk","read","write","play","sing","laugh"] },
  { name: "math",       blurb: "arithmetic vocabulary; gated on Stage D Q&A corpus",        words: ["number","one","two","three","plus","equals"] },
  { name: "meta",       blurb: "story structure & question / answer framing",               words: ["question","answer","story","end"] },
];
const CONCEPT_MASTER_BITS   = 1.0;
const CONCEPT_STRUGGLE_BITS = 2.5;

function conceptTier(bits) {
  if (typeof bits !== "number" || !isFinite(bits)) return null;
  if (bits < CONCEPT_MASTER_BITS)   return "mastered";
  if (bits < CONCEPT_STRUGGLE_BITS) return "learning";
  return "struggling";
}

function conceptDotColor(bits) {
  const t = conceptTier(bits);
  if (t === "mastered")   return "#5dff9b";
  if (t === "learning")   return "#ffae5d";
  if (t === "struggling") return "#ff5d5d";
  return "#1a2030";
}

// continuous gradient over absolute 0-4 bit scale: bright green at 0 bits,
// through yellow at 2 bits, to bright red at 4+. used by the per-concept
// heatmap row so within-tier variation reads visually.
function conceptHeatColor(bits) {
  if (typeof bits !== "number" || !isFinite(bits)) return "#0a0c12";
  const t = Math.max(0, Math.min(1, bits / 4));
  const hue = Math.round(120 * (1 - t));
  return `hsl(${hue}, 75%, 50%)`;
}

function drawConceptLine(canvas, series) {
  fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const W = canvas.clientWidth, H = canvas.clientHeight;
  ctx.fillStyle = "#06070a"; ctx.fillRect(0, 0, W, H);
  if (!series.length) return;
  const padL = 36, padR = 12, padT = 8, padB = 22;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxV = Math.max(8, ...series.map(s => s.v + 0.5));
  const xS = i => padL + (series.length === 1 ? plotW / 2 : (i / (series.length - 1)) * plotW);
  const yS = v => padT + plotH - (v / maxV) * plotH;
  ctx.fillStyle = "rgba(93, 255, 155, 0.10)";
  ctx.fillRect(padL, yS(CONCEPT_MASTER_BITS), plotW, plotH - (yS(CONCEPT_MASTER_BITS) - padT));
  ctx.fillStyle = "rgba(255, 174, 93, 0.07)";
  ctx.fillRect(padL, yS(CONCEPT_STRUGGLE_BITS), plotW, yS(CONCEPT_MASTER_BITS) - yS(CONCEPT_STRUGGLE_BITS));
  ctx.fillStyle = "rgba(255, 93, 93, 0.06)";
  ctx.fillRect(padL, padT, plotW, yS(CONCEPT_STRUGGLE_BITS) - padT);
  ctx.strokeStyle = "rgba(93, 255, 155, 0.4)"; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(padL, yS(CONCEPT_MASTER_BITS)); ctx.lineTo(padL + plotW, yS(CONCEPT_MASTER_BITS)); ctx.stroke();
  ctx.strokeStyle = "rgba(255, 93, 93, 0.4)";
  ctx.beginPath(); ctx.moveTo(padL, yS(CONCEPT_STRUGGLE_BITS)); ctx.lineTo(padL + plotW, yS(CONCEPT_STRUGGLE_BITS)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#6f7480";
  ctx.font = "10px ui-monospace,monospace";
  ctx.fillText("bits", 4, padT + 8);
  ctx.fillText("0", 4, padT + plotH - 2);
  ctx.lineWidth = 2;
  for (let i = 0; i < series.length - 1; i++) {
    ctx.strokeStyle = conceptDotColor(series[i].v);
    ctx.beginPath();
    ctx.moveTo(xS(i), yS(series[i].v));
    ctx.lineTo(xS(i + 1), yS(series[i + 1].v));
    ctx.stroke();
  }
  for (let i = 0; i < series.length; i++) {
    ctx.fillStyle = conceptDotColor(series[i].v);
    ctx.beginPath();
    ctx.arc(xS(i), yS(series[i].v), 3, 0, 6.3);
    ctx.fill();
  }
  ctx.fillStyle = "#6f7480";
  ctx.fillText(series[0].step.toLocaleString(), padL, H - 6);
  if (series.length > 1) {
    const lastTxt = series[series.length - 1].step.toLocaleString();
    const mw = ctx.measureText(lastTxt).width;
    ctx.fillText(lastTxt, padL + plotW - mw, H - 6);
  }
}

function openConceptModal(name, conceptsSteps, conceptsByStep, latest) {
  const series = [];
  for (const step of conceptsSteps) {
    const cs = conceptsByStep[step];
    const v = cs && cs.concepts && cs.concepts[name] && cs.concepts[name].surprise_bits;
    if (typeof v === "number" && isFinite(v)) series.push({ step, v });
  }
  if (!series.length) return;
  const latestV = series[series.length - 1].v;
  const tier = conceptTier(latestV);
  const tierColor = conceptDotColor(latestV);
  const tierName = tier === "mastered" ? "MASTERED" : tier === "learning" ? "LEARNING" : "STRUGGLING";
  const tierBg = tier === "mastered" ? "#103025" : tier === "learning" ? "#2a2010" : "#330a0a";
  const guesses = Math.max(1, Math.round(Math.pow(2, latestV)));
  $("conceptModalTitle").innerHTML = `${escapeHtml(name)}
    <span class="case" style="background:${tierBg};color:${tierColor};margin-left:10px">${tierName}</span>
    <span class="meta" style="margin-left:8px">${latestV.toFixed(2)} bits</span>`;
  const explainText = tier === "mastered"
    ? `When the model sees the preamble for "${escapeHtml(name)}", the right next byte is its first guess. It needed about <b>${guesses}</b> guess(es). The association is locked in &mdash; more training won't help much here.`
    : tier === "learning"
    ? `The right answer is on the model's shortlist but isn't its top pick. It would need about <b>${guesses}</b> guesses. More exposure to this domain usually closes the gap.`
    : `The model would need about <b>${guesses}</b> guesses to land on the right byte. The association isn't there yet &mdash; either the corpus hasn't covered it enough, or the preamble is genuinely ambiguous.`;
  const first = series[0].v, last = latestV, delta = first - last;
  const trendHtml = (Math.abs(delta) < 0.05)
    ? `<span class="meta">flat trajectory across ${series.length} probed checkpoints</span>`
    : (delta > 0
       ? `<b style="color:#5dff9b">&minus;${delta.toFixed(2)} bits</b> <span class="meta">improving since first probed (${first.toFixed(2)} &rarr; ${last.toFixed(2)})</span>`
       : `<b style="color:#ff7d5d">+${(-delta).toFixed(2)} bits</b> <span class="meta">regressing since first probed (${first.toFixed(2)} &rarr; ${last.toFixed(2)})</span>`);
  const lineCanvasId = `conceptLine_${Date.now()}`;
  const heatRowId    = `conceptHeat_${Date.now()}`;
  let body = `
    <p style="font-size:13px;line-height:1.55;color:var(--text);margin:0 0 14px">${explainText}</p>
    <div style="margin:0 0 6px">
      <div style="font-size:12px;color:var(--text);font-weight:600;margin-bottom:3px">Surprise (bits) over training</div>
      <div class="meta" style="font-size:11px;line-height:1.55">
        <span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">
          <span style="display:inline-block;width:10px;height:10px;background:#5dff9b;border-radius:2px"></span>
          <b style="color:#5dff9b">green</b>&nbsp;= mastered
        </span>
        <span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">
          <span style="display:inline-block;width:10px;height:10px;background:#ffae5d;border-radius:2px"></span>
          <b style="color:var(--warm)">orange</b>&nbsp;= learning
        </span>
        <span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">
          <span style="display:inline-block;width:10px;height:10px;background:#ff5d5d;border-radius:2px"></span>
          <b style="color:var(--hot)">red</b>&nbsp;= struggling
        </span>
      </div>
      <div class="meta" style="font-size:10.5px;font-style:italic;margin-top:2px">Lower is better.</div>
    </div>
    <canvas id="${lineCanvasId}" style="height:170px;width:100%"></canvas>
    <div style="margin:8px 0 14px;font-size:12px">${trendHtml}</div>
    <div style="margin:14px 0 6px">
      <div style="font-size:12px;color:var(--text);font-weight:600;margin-bottom:3px">Heatmap row &mdash; same data, color shows the bit value per checkpoint</div>
      <div class="meta" style="font-size:10.5px">Brighter green = closer to 0 bits (sure). Brighter red = closer to 4+ bits (lost). Continuous gradient so you can see change <i>within</i> a tier.</div>
    </div>
    <div id="${heatRowId}" style="display:flex;gap:2px"></div>`;
  const latestC = (latest && latest.concepts && latest.concepts[name]) || null;
  const topN = (latestC && Array.isArray(latestC.top_neurons)) ? latestC.top_neurons : [];
  if (topN.length) {
    const perLayer = {};
    for (const n of topN) {
      const cur = perLayer[n.layer];
      if (!cur || Math.abs(n.v) > Math.abs(cur.v)) perLayer[n.layer] = n;
    }
    const maxAbsV = Math.max(...topN.map(n => Math.abs(n.v))) || 1;
    const maxLayer = Math.max(...topN.map(n => n.layer));
    const layerColor = L => (L <= 3) ? "#5dc8ff" : (L <= 8) ? "var(--warm)" : "var(--hot)";
    const layerRegion = L => (L <= 3) ? "sense" : (L <= 8) ? "association" : "output";
    let bars = "";
    for (let L = 0; L <= maxLayer; L++) {
      const n = perLayer[L];
      if (!n) {
        bars += `<div title="L${L} ${layerRegion(L)}: no firing top-K neuron at this position" style="width:20px;height:46px;display:flex;flex-direction:column;justify-content:flex-end;align-items:center"><div style="width:14px;height:2px;background:#1a2030"></div><div class="meta" style="font-size:9px;margin-top:2px">L${L}</div></div>`;
        continue;
      }
      const h = Math.max(2, Math.round((Math.abs(n.v) / maxAbsV) * 32));
      bars += `<div title="L${L} ${layerRegion(L)} &middot; neuron n=${n.id} act=${n.v}" style="width:20px;height:46px;display:flex;flex-direction:column;justify-content:flex-end;align-items:center"><div style="width:14px;height:${h}px;background:${layerColor(L)};border-radius:1px"></div><div class="meta" style="font-size:9px;margin-top:2px">L${L}</div></div>`;
    }
    body += `
      <div style="margin-top:16px;padding-top:12px;border-top:1px dashed var(--line)">
        <div class="meta" style="font-size:11px;margin-bottom:6px"><b style="color:var(--text)">where this concept lives</b> &mdash; strongest FFN neuron per layer at the commit position, latest checkpoint. Bars show activation magnitude. Color is the brain region: <span style="color:#5dc8ff">L0&ndash;3 sense</span>, <span style="color:var(--warm)">L4&ndash;8 association</span>, <span style="color:var(--hot)">L9&ndash;11 output</span>.</div>
        <div style="display:flex;gap:3px;align-items:flex-end">${bars}</div>
      </div>`;
  } else {
    body += `<div class="meta" style="margin-top:14px;font-size:10.5px;padding-top:10px;border-top:1px dashed var(--line)">Layer signature unavailable for this checkpoint &mdash; needs the post-2026-04-29 <code>top_neurons</code> probe field.</div>`;
  }
  $("conceptModalBody").innerHTML = body;
  $("conceptModal").classList.remove("hidden");
  document.body.classList.add("no-scroll");
  requestAnimationFrame(() => {
    const c = $(lineCanvasId);
    if (c) drawConceptLine(c, series);
    const hr = $(heatRowId);
    if (hr) {
      hr.innerHTML = series.map(s =>
        `<div title="step ${s.step.toLocaleString()}: ${s.v.toFixed(2)} bits" style="flex:1;height:26px;background:${conceptHeatColor(s.v)};border-radius:2px"></div>`
      ).join("");
    }
  });
}

function render_concepts(refs, run, conceptsSteps, conceptsByStep, haveCheckpoints) {
  const root  = $(refs.conceptsId);
  const hover = $(refs.conceptsHoverId);
  if (!conceptsSteps || conceptsSteps.length === 0) {
    if (hover) hover.style.display = "none";
    root.innerHTML = haveCheckpoints
      ? `<span style="color:var(--hot)">Concept tracking is not configured for this run.</span>`
      : `<span style="color:var(--warm)">No checkpoint yet.</span>`;
    return;
  }
  if (hover) {
    hover.style.display = "";
    hover.innerHTML = `<span class="meta">click any concept below for its trajectory chart and the neurons that fire for it.</span>`;
  }
  const latestStep = conceptsSteps[conceptsSteps.length - 1];
  const latest = conceptsByStep[latestStep];
  if (!latest || !latest.concepts) {
    root.innerHTML = `<span class="meta">latest concepts probe is missing the concepts field</span>`;
    return;
  }
  const C = latest.concepts;

  let mastered = 0, learning = 0, struggling = 0, probed = 0;
  for (const name of Object.keys(C)) {
    const t = conceptTier(C[name] && C[name].surprise_bits);
    if (t === "mastered")   mastered++;
    else if (t === "learning")   learning++;
    else if (t === "struggling") struggling++;
    if (t) probed++;
  }

  const ranked = Object.keys(C)
    .filter(n => typeof C[n].surprise_bits === "number" && isFinite(C[n].surprise_bits))
    .sort((a, b) => C[b].surprise_bits - C[a].surprise_bits);
  const worst = ranked.slice(0, 5);

  let html = "";

  html += `<div style="font-size:12.5px;line-height:1.55;color:var(--text);margin-bottom:10px">
    We give the model a short preamble like <code>"the small "</code> and ask: how surprised is it by the next byte if the answer is <code>"cat"</code>? Lower surprise (in bits) = the model has the association locked in. Click any concept below for its training trajectory.
  </div>`;

  html += `<div style="display:flex;gap:18px;flex-wrap:wrap;align-items:baseline;margin-bottom:14px;font-size:12px">
    <span class="meta">latest probe at step <b>${latestStep.toLocaleString()}</b> &middot; <b>${probed}</b> of 50 concepts measured</span>
    <span><span class="case" style="background:#103025;color:#5dff9b">MASTERED</span><b>${mastered}</b> <span class="meta">&lt; ${CONCEPT_MASTER_BITS} bit</span></span>
    <span><span class="case b-mid">LEARNING</span><b>${learning}</b> <span class="meta">${CONCEPT_MASTER_BITS}&ndash;${CONCEPT_STRUGGLE_BITS} bits</span></span>
    <span><span class="case" style="background:#330a0a;color:#ff7d5d">STRUGGLING</span><b>${struggling}</b> <span class="meta">&gt; ${CONCEPT_STRUGGLE_BITS} bits</span></span>
  </div>`;

  if (worst.length) {
    html += `<div style="margin-bottom:14px">
      <div class="meta" style="margin-bottom:5px"><b style="color:var(--text)">biggest gaps right now</b> &mdash; the concepts the model is most surprised by. Click for details.</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">`;
    for (const name of worst) {
      const v = C[name].surprise_bits;
      html += `<span class="concept-row" data-name="${escapeHtml(name)}" style="cursor:pointer;display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border:1px solid var(--line);border-radius:3px;background:#0a0c12;font-size:11.5px">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${conceptDotColor(v)}"></span>
        <b>${escapeHtml(name)}</b>
        <span class="meta">${v.toFixed(2)} bits</span>
      </span>`;
    }
    html += `</div></div>`;
  }

  html += `<div style="display:grid;grid-template-columns:130px 1fr 100px;gap:9px 14px;font-size:11.5px;align-items:center;border-top:1px solid var(--line);padding-top:10px">`;
  for (const grp of CONCEPT_GROUPS) {
    let m = 0, total = 0, sumBits = 0;
    const rows = [];
    for (const name of grp.words) {
      const v = C[name] && C[name].surprise_bits;
      if (typeof v === "number" && isFinite(v)) {
        total++;
        if (conceptTier(v) === "mastered") m++;
        sumBits += v;
        rows.push({ name, v });
      }
    }
    const avg = total ? (sumBits / total) : null;
    const headColor = total ? conceptDotColor(avg) : "var(--dim)";
    const summary = total ? `<b>${m}/${total}</b> mastered` : `<span class="meta">no data</span>`;

    html += `<div style="text-align:right">
      <b style="color:${headColor}">${grp.name}</b>
      <div class="meta" style="font-size:10px;margin-top:2px">${escapeHtml(grp.blurb)}</div>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:5px">`;
    for (const r of rows) {
      html += `<span class="concept-row" data-name="${escapeHtml(r.name)}" title="${escapeHtml(r.name)}: ${r.v.toFixed(2)} bits &mdash; click for details" style="cursor:pointer;display:inline-flex;align-items:center;gap:5px;padding:2px 8px;border-radius:10px;background:rgba(255,255,255,.04);font-size:10.5px">
        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${conceptDotColor(r.v)}"></span>
        ${escapeHtml(r.name)}
      </span>`;
    }
    html += `</div>
    <div style="text-align:right">${summary}</div>`;
  }
  html += `</div>`;

  root.innerHTML = html;

  root.querySelectorAll(".concept-row").forEach(tr => {
    tr.addEventListener("click", () => {
      const name = tr.dataset.name;
      openConceptModal(name, conceptsSteps, conceptsByStep, latest);
    });
  });
}

// ---------- writing health -------------------------------------------------
//
// Card surfaces the five mathematical proxies for writing structure produced
// by checkpoint_probe.dump_writing_health: distinct-n, lexical chain density,
// pronoun-unbacked rate, repeat rate, self-perplexity. We deliberately surface
// the caveat *before* the numbers: these proxies catch mode collapse,
// repetition, broken anaphora, and off-distribution drift, but cannot detect
// narrative nonsense without an external judge with world knowledge. A user
// reading "the volcano married a sandwich" should see good scores here, and
// the caveat tells them why.

// Each metric: id, label, blurb, "good" direction, soft target, formatter,
// and `score(v)` -> [0..1] where 1 = ideal, 0 = baseline / untrained. The
// score function is what the trajectory chart plots, so all lines share the
// same y-axis and each line starts at (step 0, score 0) to mirror how the
// reading-level chart anchors every band at fluency=0% at the random-init.
const WRITING_METRICS = [
  { id: "distinct_3",        label: "vocabulary variety",
    blurb: "fraction of unique 3-word groups in the generation. low = the model is repeating phrases (mode collapse).",
    higher: true,  target: 0.92, fmt: v => v.toFixed(3),
    score: v => v == null ? null : Math.max(0, Math.min(1, v)) },
  { id: "lex_chain_density", label: "entity tracking",
    blurb: "fraction of meaningful words (length ≥ 3, not stop-words) that recur. higher = the model keeps referring back to the same characters/objects, suggesting it's tracking entities through the story.",
    higher: true,  target: 0.25, fmt: v => v.toFixed(3),
    score: v => v == null ? null : Math.max(0, Math.min(1, v / 0.40)) },
  { id: "pronoun_unbacked",  label: "broken pronouns",
    blurb: "fraction of pronouns (he/she/it/they) without a candidate referent in the previous 50 bytes. high = the model uses pronouns without setting up who they refer to.",
    higher: false, target: 0.15, fmt: v => v.toFixed(3),
    score: v => v == null ? null : Math.max(0, Math.min(1, 1 - v)) },
  { id: "repeat_rate",       label: "duplicate words",
    blurb: "fraction of consecutive duplicate words (the the). high = mode collapse into a single token.",
    higher: false, target: 0.03, fmt: v => v.toFixed(3),
    score: v => v == null ? null : Math.max(0, Math.min(1, 1 - v * 4)) },
  { id: "pmi",               label: "word co-occurrence",
    blurb: "Normalized PMI (NPMI, range -1..+1) of adjacent word pairs against the training corpus. +1 = pairs always co-occur in training, 0 = independent, -1 = unseen pairs. Catches diverse-but-gibberish output that vocabulary variety misses. Requires <stem>_bigrams.npz next to the corpus -- build with veritate_mri/tools/build_bigram_index.py.",
    higher: true,  target: 0.20,  fmt: v => v == null ? "(no corpus index)" : v.toFixed(3),
    score: v => v == null ? null : Math.max(0, Math.min(1, (v + 1) / 2)) },
  { id: "self_ppl",          label: "self-perplexity",
    blurb: "model's perplexity on its own generation. low = the model 'stands behind' what it just wrote; rising over training would mean it's drifting off-distribution from itself.",
    higher: false, target: 2.0,  fmt: v => v == null ? "—" : v.toFixed(2),
    // log-scale: ppl 256 (random) -> 0, ppl 1.5 -> 1.
    score: v => {
      if (v == null) return null;
      const LOG_RANDOM = Math.log(256), LOG_IDEAL = Math.log(1.5);
      const s = (LOG_RANDOM - Math.log(Math.max(v, 1))) / (LOG_RANDOM - LOG_IDEAL);
      return Math.max(0, Math.min(1, s));
    }},
];

function _whTone(metric, v, prev) {
  if (v == null || isNaN(v)) return "var(--dim)";
  let tone;
  if (metric.higher) {
    if (v >= metric.target) tone = "green";
    else if (v >= (metric.id === "pmi" ? -0.1 : metric.target * 0.7)) tone = "warm";
    else tone = "hot";
  } else {
    if (v <= metric.target) tone = "green";
    else if (v <= metric.target * 1.5) tone = "warm";
    else tone = "hot";
  }
  // Trend penalty: if the metric is moving the wrong way by more than a small
  // tolerance vs. the previous checkpoint, downgrade one step (green -> warm,
  // warm -> hot). This surfaces deterioration before a metric crosses the
  // target line. Tolerance is 5% of target for higher-better, 10% for
  // lower-better (lower-better has tighter targets so absolute swings matter
  // less).
  if (prev != null && !isNaN(prev) && tone !== "hot") {
    const tol = (metric.higher ? 0.05 : 0.10) * Math.max(metric.target, 0.01);
    const movingWrong = metric.higher ? (v < prev - tol) : (v > prev + tol);
    if (movingWrong) tone = (tone === "green") ? "warm" : "hot";
  }
  return tone === "green" ? "#5dff9b" : tone === "warm" ? "var(--warm)" : "var(--hot)";
}

function render_writing_health(refs, run, writingSteps, writingByStep, haveCheckpoints, config) {
  const root = $(refs.writingHealthId);
  if (!root) return;
  if (!writingSteps || writingSteps.length === 0) {
    root.innerHTML = haveCheckpoints
      ? `<span style="color:var(--hot)">No writing-health dumps yet for this run.</span>`
      : `<span style="color:var(--warm)">No checkpoint yet.</span>`;
    return;
  }
  const latestStep = writingSteps[writingSteps.length - 1];
  const latest = writingByStep[latestStep];
  if (!latest || !latest.aggregate) {
    root.innerHTML = `<span class="meta">writing_health_step_${latestStep}.json missing aggregate field</span>`;
    return;
  }

  let html = "";
  // PMI no-index hint (one short line).
  const cfg = latest.config || {};
  if (cfg.corpus_path && !cfg.pmi_index_path) {
    const stem = (cfg.corpus_path.match(/([^\\\/]+)_train\.bin$/) || [null, "<corpus>"])[1];
    html += `<div class="meta" style="margin:0 0 10px;font-size:11px">PMI disabled (no bigram index). Build with: <code>python veritate_mri/tools/build_bigram_index.py --corpus ${escapeHtml(stem)}</code></div>`;
  }

  // Legend colors avoid green/yellow/orange/red (those mean status: good/close/bad).
  // Picked for high mutual contrast: hot pink, electric cyan, deep violet, magenta,
  // teal, slate. Each is well-separated in hue and lightness.
  const WH_COLORS = {
    distinct_3:        "#ff3db8",   // hot magenta-pink
    lex_chain_density: "#3dd6ff",   // electric cyan
    pronoun_unbacked:  "#9b3dff",   // deep violet
    repeat_rate:       "#ffb3ff",   // light pink
    pmi:               "#3dffd6",   // teal
    self_ppl:          "#5d7dff",   // royal blue
  };
  // Prev-step record for trend-aware tinting. If a metric is moving the
  // wrong way vs. the previous checkpoint, _whTone downgrades the color.
  const prevStep = writingSteps.length >= 2 ? writingSteps[writingSteps.length - 2] : null;
  const prevAgg = prevStep != null ? (writingByStep[prevStep] || {}).aggregate : null;
  html += `<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(170px, 1fr));gap:8px 12px;margin-bottom:10px">`;
  for (const m of WRITING_METRICS) {
    const v = latest.aggregate[m.id];
    const prevV = prevAgg ? prevAgg[m.id] : null;
    const tone = _whTone(m, v, prevV);
    const arrow = m.higher ? "≥" : "≤";
    const swatch = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${WH_COLORS[m.id] || "#fff"};margin-right:5px;vertical-align:middle"></span>`;
    // Trend indicator: ↑/↓/→ next to the value. Color it by whether the
    // direction is good or bad for this metric.
    let trend = "";
    if (prevV != null && v != null) {
      const tol = (m.higher ? 0.02 : 0.005) * Math.max(m.target, 0.01);
      const dv = v - prevV;
      if (Math.abs(dv) <= tol) trend = `<span class="meta" style="font-size:10px;margin-left:4px;color:var(--dim)">→</span>`;
      else {
        const goodDir = m.higher ? (dv > 0) : (dv < 0);
        const arrowChar = dv > 0 ? "↑" : "↓";
        const trendColor = goodDir ? "#5dff9b" : "var(--hot)";
        trend = `<span style="font-size:11px;margin-left:4px;color:${trendColor}" title="vs step ${prevStep}: ${dv > 0 ? "+" : ""}${dv.toFixed(3)}">${arrowChar}</span>`;
      }
    }
    html += `<div title="${escapeHtml(m.blurb)}" style="background:rgba(255,255,255,.02);padding:6px 8px;border-left:3px solid ${tone};border-radius:3px">
      <div class="meta" style="font-size:10.5px">${swatch}${m.label}</div>
      <b style="color:${tone};font-size:14px">${v == null ? "—" : m.fmt(v)}</b>${trend}
      <span class="meta" style="font-size:10px;margin-left:6px">${arrow} ${m.target}</span>
    </div>`;
  }
  html += `</div>`;

  // Trajectory: HTML chip-legend (matches reading-level style), then canvas.
  const trajMetrics = ["distinct_3", "lex_chain_density", "pronoun_unbacked", "repeat_rate", "pmi"];
  if (writingSteps.length >= 2) {
    let lhtml = `<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;font-size:11px;margin:6px 0 4px">`;
    for (const id of trajMetrics) {
      const lbl = (WRITING_METRICS.find(m => m.id === id) || {}).label || id;
      lhtml += `<span style="display:inline-flex;align-items:center;gap:5px"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${WH_COLORS[id] || "#fff"}"></span>${lbl}</span>`;
    }
    lhtml += `</div>`;
    html += lhtml;
    html += `<canvas id="${refs.writingHealthId}_traj" height="200" style="width:100%;height:200px;background:rgba(255,255,255,.02);border-radius:4px"></canvas>`;
    html += `<div class="meta" style="margin:4px 0 8px;font-size:10px">trajectory: each line is one metric, score 0 (untrained) → 1 (ideal). UP = better. All lines start at 0 at step 0.</div>`;
  }

  // Latest-step sample preview.
  if (Array.isArray(latest.samples) && latest.samples.length > 0) {
    html += `<div class="meta" style="margin:10px 0 4px;font-size:10.5px">latest generations (step ${latestStep})</div>`;
    html += `<div style="display:flex;flex-direction:column;gap:6px">`;
    for (const s of latest.samples) {
      const promptHtml = `<span class="meta" style="font-style:italic">${escapeHtml(s.prompt)}</span>`;
      const genHtml = escapeHtml(s.generation || "");
      html += `<div style="background:rgba(255,255,255,.03);padding:6px 8px;border-radius:4px;font-size:11px;line-height:1.4;white-space:pre-wrap">${promptHtml}<span style="color:var(--text)">${genHtml}</span></div>`;
    }
    html += `</div>`;
  }

  root.innerHTML = html;

  // Trajectory: each line is a per-metric SCORE in [0..1] where 1 = ideal,
  // 0 = untrained baseline. All lines anchor at (step 0, score 0), mirroring
  // how the reading-level chart anchors every band at fluency=0% at random
  // init. Up = better. This is the same pattern as the reading-level chart.
  const canvas = document.getElementById(`${refs.writingHealthId}_traj`);
  if (canvas && writingSteps.length >= 1) {
    const ctx = canvas.getContext("2d");
    const metricById = {};
    for (const m of WRITING_METRICS) metricById[m.id] = m;
    const series = trajMetrics.map(id => {
      const m = metricById[id];
      if (!m || typeof m.score !== "function") return null;
      const points = [{ x: 0, y: 0 }];   // anchor every line at the random-init baseline
      for (const step of writingSteps) {
        const r = writingByStep[step];
        if (!r || !r.aggregate) continue;
        const v = r.aggregate[id];
        const s = m.score(v);
        if (s == null) continue;
        points.push({ x: step, y: s });
      }
      return { color: WH_COLORS[id] || "#fff", points, lw: 1.5, dots: true };
    }).filter(s => s && s.points.length > 1);   // need a real measurement, not just origin
    const totalSteps = (config && config.training_args && config.training_args.total_steps) || null;
    plotTrainSeries(canvas, ctx, series, {
      yMinFloor: 0,
      yMaxCeil:  1.08,
      xMinFloor: 0,
      xMaxCeil:  totalSteps || undefined,
      yTitle:    "score (1.0 = ideal, 0 = untrained)",
      thresholdLines: [
        { y: 1.0, color: "#5dff9b", label: "IDEAL", lineDash: [4, 3] },
      ],
    });
  }
}

// load classroom panels for a model name into the supplied (state, refs) pair.
// state holds the cached config/steps/probes/lens; refs hold the dom IDs / canvas to render into.
// shared by the live-training tab (trainSelectedRun) and the learning tab (timeline picker).
async function loadClassroomFor(state, refs, run) {
  if (!run) return;
  if (state.run === run && state.loaded && state.steps.length > 0) return;
  state.run = run;
  state.loaded = false;
  state.config = null;
  state.steps = [];
  state.probesByStep = {};
  state.lensByStep = {};
  state.confByStep = {};
  state.gradesSteps = [];
  state.gradesByStep = {};
  state.mathSteps = [];
  state.mathByStep = {};
  state.grammarSteps = [];
  state.grammarByStep = {};
  state.reasoningSteps = [];
  state.reasoningByStep = {};
  state.conceptsSteps = [];
  state.conceptsByStep = {};
  $(refs.sizeMeterId).innerHTML = `<span class="meta">loading…</span>`;
  $(refs.lensDriftId).innerHTML = `<span class="meta">loading…</span>`;
  if (refs.readLevelId) $(refs.readLevelId).innerHTML = `<span class="meta">loading…</span>`;
  if (refs.conceptsId)  $(refs.conceptsId).innerHTML  = `<span class="meta">loading…</span>`;
  // config.json — best-effort
  try {
    const r = await fetch(`/run/${encodeURIComponent(run)}/config?` + Date.now(), { cache: "no-store" });
    if (r.ok) state.config = await r.json();
  } catch (e) {}
  render_size_meter(refs, run, state.config);
  if (refs.pruneBodyId) load_pruning_report(refs, run);
  // probes index
  let steps = [];
  try {
    const r = await fetch(`/run/${encodeURIComponent(run)}/probes?` + Date.now(), { cache: "no-store" });
    if (r.ok) { const d = await r.json(); steps = d.steps || []; }
  } catch (e) {}
  state.steps = steps;
  // classroom index — grades + concepts. independent of probes.
  let classroomItems = [];
  try {
    const r = await fetch(`/run/${encodeURIComponent(run)}/classroom?` + Date.now(), { cache: "no-store" });
    if (r.ok) { const d = await r.json(); classroomItems = d.items || []; }
  } catch (e) {}
  const gradesFiles    = classroomItems.filter(it => it.kind === "grades");
  const mathFiles      = classroomItems.filter(it => it.kind === "math");
  const grammarFiles   = classroomItems.filter(it => it.kind === "grammar");
  const reasoningFiles = classroomItems.filter(it => it.kind === "reasoning");
  const conceptsFiles  = classroomItems.filter(it => it.kind === "concepts");
  const writingFiles   = classroomItems.filter(it => it.kind === "writing_health");
  if (steps.length === 0) {
    $(refs.lensDriftId).innerHTML = `<span style="color:var(--warm)">No checkpoint yet.</span>`;
    plotTrainSeries(refs.confCanvas, refs.confCtx, []);
  } else {
    // probes (small json files), one per step
    await Promise.all(steps.map(async s => {
      if (!s.probe) return;
      try { state.probesByStep[s.step] = await classroomFetchProbe(run, s.probe); }
      catch (e) { console.warn("probe fetch", s.step, e); }
    }));
    // lens npz files, one per step
    const V = (state.config && state.config.shape && state.config.shape.vocab) || 256;
    await Promise.all(steps.map(async s => {
      if (!s.lens) return;
      try {
        const lens = await classroomFetchLens(run, s.lens);
        state.lensByStep[s.step] = lens;
        const c = compute_confidence_components(lens.lens_rows, lens.residual_norms, V);
        if (c) state.confByStep[s.step] = c;
      } catch (e) { console.warn("lens fetch", s.step, e); }
    }));
    render_confidence_evo(refs, run, steps, state.confByStep);
    render_lens_drift(refs, run, steps, state.lensByStep);
  }
  // grades — small json each. fetch in parallel.
  await Promise.all(gradesFiles.map(async it => {
    try {
      const r = await fetch(`/timeline/${encodeURIComponent(run)}/${encodeURIComponent(it.file)}?` + Date.now(), { cache: "no-store" });
      if (r.ok) state.gradesByStep[it.step] = await r.json();
    } catch (e) { console.warn("grades fetch", it.step, e); }
  }));
  state.gradesSteps = Object.keys(state.gradesByStep).map(s => parseInt(s, 10)).sort((a, b) => a - b);
  if (refs.readLevelId) render_reading_level(refs, run, state.gradesSteps, state.gradesByStep, steps.length > 0, state.config);
  // math / grammar / reasoning — same shape, fetched in parallel.
  await Promise.all(mathFiles.map(async it => {
    try {
      const r = await fetch(`/timeline/${encodeURIComponent(run)}/${encodeURIComponent(it.file)}?` + Date.now(), { cache: "no-store" });
      if (r.ok) state.mathByStep[it.step] = await r.json();
    } catch (e) { console.warn("math fetch", it.step, e); }
  }));
  state.mathSteps = Object.keys(state.mathByStep).map(s => parseInt(s, 10)).sort((a, b) => a - b);
  if (refs.mathLevelId) render_score_axis("math", refs, state.mathSteps, state.mathByStep, steps.length > 0);
  await Promise.all(grammarFiles.map(async it => {
    try {
      const r = await fetch(`/timeline/${encodeURIComponent(run)}/${encodeURIComponent(it.file)}?` + Date.now(), { cache: "no-store" });
      if (r.ok) state.grammarByStep[it.step] = await r.json();
    } catch (e) { console.warn("grammar fetch", it.step, e); }
  }));
  state.grammarSteps = Object.keys(state.grammarByStep).map(s => parseInt(s, 10)).sort((a, b) => a - b);
  if (refs.grammarLevelId) render_score_axis("grammar", refs, state.grammarSteps, state.grammarByStep, steps.length > 0);
  await Promise.all(reasoningFiles.map(async it => {
    try {
      const r = await fetch(`/timeline/${encodeURIComponent(run)}/${encodeURIComponent(it.file)}?` + Date.now(), { cache: "no-store" });
      if (r.ok) state.reasoningByStep[it.step] = await r.json();
    } catch (e) { console.warn("reasoning fetch", it.step, e); }
  }));
  state.reasoningSteps = Object.keys(state.reasoningByStep).map(s => parseInt(s, 10)).sort((a, b) => a - b);
  if (refs.reasoningLevelId) render_score_axis("reasoning", refs, state.reasoningSteps, state.reasoningByStep, steps.length > 0);
  // concepts
  await Promise.all(conceptsFiles.map(async it => {
    try {
      const r = await fetch(`/timeline/${encodeURIComponent(run)}/${encodeURIComponent(it.file)}?` + Date.now(), { cache: "no-store" });
      if (r.ok) state.conceptsByStep[it.step] = await r.json();
    } catch (e) { console.warn("concepts fetch", it.step, e); }
  }));
  state.conceptsSteps = Object.keys(state.conceptsByStep).map(s => parseInt(s, 10)).sort((a, b) => a - b);
  if (refs.conceptsId) render_concepts(refs, run, state.conceptsSteps, state.conceptsByStep, steps.length > 0);
  // writing health
  await Promise.all(writingFiles.map(async it => {
    try {
      const r = await fetch(`/timeline/${encodeURIComponent(run)}/${encodeURIComponent(it.file)}?` + Date.now(), { cache: "no-store" });
      if (r.ok) state.writingByStep[it.step] = await r.json();
    } catch (e) { console.warn("writing_health fetch", it.step, e); }
  }));
  state.writingSteps = Object.keys(state.writingByStep).map(s => parseInt(s, 10)).sort((a, b) => a - b);
  if (refs.writingHealthId) render_writing_health(refs, run, state.writingSteps, state.writingByStep, steps.length > 0, state.config);
  state.loaded = true;
}

// thin wrappers — the existing call sites stay unchanged.
function loadClassroomForRun(run)      { return loadClassroomFor(classroomState,  classroomRefsT, run); }
function loadClassroomForLearning(run) { return loadClassroomFor(classroomStateL, classroomRefsL, run); }

// poll: re-check probe step count for the active live-training run. cheap json fetch.
// only triggers a full classroom reload when new lens/probe dumps have appeared.
async function maybeRefreshClassroomForRun(run) {
  if (!run || classroomState.run !== run) return;
  try {
    const r = await fetch(`/run/${encodeURIComponent(run)}/probes?` + Date.now(), { cache: "no-store" });
    if (!r.ok) return;
    const d = await r.json();
    const serverSteps = (d.steps || []).length;
    if (serverSteps !== classroomState.steps.length) {
      classroomState.loaded = false;
      await loadClassroomForRun(run);
    }
  } catch (e) {}
}

attachHoverInspect(cConfEvoT, "cConfEvoTHover", ["margin (norm)", "entropy", "lens-consistency", "residual-stab"], v => v.toFixed(3));
attachHoverInspect(cConfEvoL, "cConfEvoLHover", ["margin (norm)", "entropy", "lens-consistency", "residual-stab"], v => v.toFixed(3));

async function startTrainPolling() {
  if (trainPollTimer) return;
  await loadRunsList();
  loadTrainCsv();
  loadClassroomForRun(trainSelectedRun);
  trainPollTimer = setInterval(loadTrainCsv, 5000);
  // refresh runs list every 30s so newly-created runs show up
  setInterval(loadRunsList, 30000);
  // re-check classroom for new lens/probe dumps every 30s; reload only if step count grew
  setInterval(() => maybeRefreshClassroomForRun(trainSelectedRun), 30000);
}
function stopTrainPolling() {
  if (trainPollTimer) { clearInterval(trainPollTimer); trainPollTimer = null; }
}

// hover inspector — show value at cursor x
function attachHoverInspect(canvas, infoElId, seriesLabels, fmt) {
  const info = $(infoElId);
  fmt = fmt || (v => v.toFixed(4));
  canvas.style.cursor = "crosshair";
  canvas.addEventListener("mouseleave", () => { info.textContent = "hover to inspect"; });
  canvas.addEventListener("mousemove", e => {
    const series = canvas.__series;
    if (!series) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const padL = 50, padR = 12;
    if (x < padL || x > rect.width - padR) { info.textContent = "hover to inspect"; return; }
    const ref = series.find(s => s.points.length > 0);
    if (!ref) return;
    const pts = ref.points;
    const xMin = pts[0].x, xMax = pts[pts.length-1].x;
    const plotW = rect.width - padL - padR;
    const targetX = xMin + ((x - padL) / plotW) * (xMax - xMin);
    let html = `<b>step ${Math.round(targetX).toLocaleString()}</b>`;
    for (let i = 0; i < series.length; i++) {
      const s = series[i];
      if (s.points.length === 0) continue;
      let closest = s.points[0], minDx = Math.abs(s.points[0].x - targetX);
      for (const p of s.points) {
        const dx = Math.abs(p.x - targetX);
        if (dx < minDx) { minDx = dx; closest = p; }
      }
      const lbl = (seriesLabels && seriesLabels[i]) || ("series" + i);
      html += `   <span style="color:${s.color}">${lbl} <b>${fmt(closest.y)}</b> @ ${closest.x.toLocaleString()}</span>`;
    }
    info.innerHTML = html;
  });
}

attachHoverInspect(cLossT, "cLossTHover", ["train", "val", "qat_train", "qat_val", "val_qat"], v => v.toFixed(4));
attachHoverInspect(cLrT,   "cLrTHover",   ["lr"],          v => v.toExponential(2));
attachHoverInspect(cTpsT,  "cTpsTHover",  ["tok/s"],       v => Math.round(v).toLocaleString());
attachHoverInspect(cGnT,   "cGnTHover",   ["grad_norm"],   v => v.toFixed(2));

// ---- shared dropdown component ----
function injectDetails() {
  document.querySelectorAll("[data-detail]").forEach(el => {
    if (el.dataset.injected) return;
    const tpl = document.getElementById("detail-" + el.dataset.detail);
    if (!tpl) return;
    el.appendChild(tpl.content.cloneNode(true));
    el.dataset.injected = "1";
  });
}

// ---- ascii reference ----
function buildAscii() {
  const el = $("asciiRef");
  let html = "";
  for (let b = 0; b < 128; b++) {
    let cls = "ascii-cell";
    let glyph;
    if (b === 9)  { glyph = "→";   cls += " wh"; }
    else if (b === 10) { glyph = "↵"; cls += " wh"; }
    else if (b === 13) { glyph = "↩"; cls += " wh"; }
    else if (b === 32) { glyph = "␣"; cls += " wh"; }
    else if (b < 32 || b === 127) { glyph = "·"; cls += " np"; }
    else { glyph = String.fromCharCode(b); }
    if (b >= 48 && b <= 57) cls += " dig";
    else if ((b >= 65 && b <= 90) || (b >= 97 && b <= 122)) cls += " let";
    const safe = (glyph === "<") ? "&lt;" : (glyph === ">") ? "&gt;" : (glyph === "&") ? "&amp;" : glyph;
    html += `<div class="${cls}" data-byte="${b}" title="byte ${b} = 0x${b.toString(16).padStart(2,"0")}"><span class="c">${safe}</span><span class="b">${b}</span></div>`;
  }
  el.innerHTML = html;
}
buildAscii();
_buildFfnLegend(_layerCount);

let _asciiCurrentByte = -1;
function highlightAsciiByte(b) {
  const el = $("asciiRef");
  if (!el || b == null) return;
  if (_asciiCurrentByte === b) return;
  if (_asciiCurrentByte >= 0) {
    const prev = el.querySelector(`.ascii-cell[data-byte="${_asciiCurrentByte}"]`);
    if (prev) prev.classList.remove("current");
  }
  const next = el.querySelector(`.ascii-cell[data-byte="${b}"]`);
  if (next) next.classList.add("current");
  _asciiCurrentByte = b;
}

// ---- init ----
injectDetails();

// add expand toggle to every panel containing a canvas
function attachExpandButtons() {
  document.querySelectorAll(".panel").forEach(panel => {
    if (!panel.querySelector("canvas") && !panel.dataset.expandable) return;
    const h2 = panel.querySelector("h2");
    if (!h2 || h2.querySelector(".expand-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "expand-btn";
    btn.textContent = "expand";
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const isExpanded = panel.classList.toggle("expanded");
      btn.textContent = isExpanded ? "collapse" : "expand";
      panel.querySelectorAll("canvas").forEach(fitCanvas);
      window.dispatchEvent(new Event("resize"));
    });
    h2.appendChild(btn);
  });
}
attachExpandButtons();

function attachCollapseButtons() {
  document.querySelectorAll(".panel.collapsible").forEach(panel => {
    const h2 = panel.querySelector("h2");
    if (!h2 || h2.querySelector(".collapse-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "expand-btn collapse-btn";
    btn.textContent = panel.classList.contains("collapsed") ? "expand" : "collapse";
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const isCollapsed = panel.classList.toggle("collapsed");
      btn.textContent = isCollapsed ? "expand" : "collapse";
    });
    h2.appendChild(btn);
  });
}
attachCollapseButtons();

[cFfn, cTop, cTel, cFlow, cDecisive, cConfBar, cConfTrend].forEach(fitCanvas);
window.addEventListener("resize", () => {
  [cFfn, cTop, cTel, cFlow, cDecisive, cConfBar, cConfTrend].forEach(fitCanvas);
  if (currentFrame >= 0) render(frames[currentFrame]);
  if (learningState.loaded && document.querySelector('.tab-body[data-tab="learning"]').classList.contains("active")) {
    [cFfnL, cTopL, cTelL, cSatL, cQuantKlL, cDecisiveL, cConfEvoL, cReadGradeL,
     cCoactL, cSurpriseL].forEach(fitCanvas);
    drawQuantKl(cQuantKlL, ctxQuantKlL, learningState.meta.checkpoints, learningState.ckptIdx);
    renderLearning();
    if (typeof renderTier2ForLearning === "function") renderTier2ForLearning();
    if (classroomStateL.loaded) {
      render_confidence_evo(classroomRefsL, classroomStateL.run, classroomStateL.steps, classroomStateL.confByStep);
      render_reading_level(classroomRefsL, classroomStateL.run, classroomStateL.gradesSteps, classroomStateL.gradesByStep, true, classroomStateL.config);
    }
  }
  if (document.querySelector('.tab-body[data-tab="training"]').classList.contains("active") && trainLastText) {
    [cLossT, cLrT, cTpsT, cGnT, cConfEvoT, cReadGradeT].forEach(fitCanvas);
    parseAndRenderTrain(trainLastText);
    if (classroomState.loaded) {
      render_confidence_evo(classroomRefsT, classroomState.run, classroomState.steps, classroomState.confByStep);
      render_reading_level(classroomRefsT, classroomState.run, classroomState.gradesSteps, classroomState.gradesByStep, true, classroomState.config);
    }
  }
});

// ============================================================
// CLASSROOM DASHBOARD — tier 2 panels (Learning tab only)
// ============================================================

const cCoactL    = $("cCoactL"),    ctxCoactL    = cCoactL.getContext("2d");
const cSurpriseL = $("cSurpriseL"), ctxSurpriseL = cSurpriseL.getContext("2d");

const tier2State = { run: null, coact: {}, surprise: null };

function regionColorL(layer) {
  const t = 12;
  if (layer < t / 3)       return "#5dc8ff";
  if (layer < (2 * t) / 3) return "#ffae5d";
  return "#ff5d5d";
}

async function fetchTier2Coact(run, step) {
  if (tier2State.coact[step] !== undefined) return tier2State.coact[step];
  const r = await fetch(`/run/${encodeURIComponent(run)}/coactivation/${step}`);
  const d = r.ok ? await r.json() : null;
  if (tier2State.run === run) tier2State.coact[step] = d;
  return d;
}

async function fetchTier2Surprise(run) {
  if (tier2State.surprise !== null) return tier2State.surprise;
  const r = await fetch(`/run/${encodeURIComponent(run)}/surprise`);
  const d = r.ok ? await r.json() : null;
  if (tier2State.run === run) tier2State.surprise = d;
  return d;
}

function drawCoactL(d) {
  const info = $("learnCoactInfo");
  const hover = $("cCoactLHover");
  cCoactL.__edges = null;
  if (!d || !d.pairs || d.pairs.length === 0) {
    cCoactL.style.display = "none";
    if (hover) hover.style.display = "none";
    const n = (learningState.meta && learningState.meta.checkpoints) ? learningState.meta.checkpoints.length : 0;
    info.innerHTML = n > 0
      ? `<span style="color:var(--hot)">Co-activation graph not available for this checkpoint.</span>`
      : `<span style="color:var(--warm)">No checkpoint yet.</span>`;
    return;
  }
  cCoactL.style.display = "";
  if (hover) hover.style.display = "";
  requestAnimationFrame(() => {
    const { w: W, h: H } = fitCanvas(cCoactL);
    ctxCoactL.fillStyle = "#06070a"; ctxCoactL.fillRect(0, 0, W, H);
    const padL = 36, padR = 12, padT = 14, padB = 22;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    let layerMax = 0, neuronMax = 0;
    for (const n of d.nodes) {
      if (n.layer > layerMax) layerMax = n.layer;
      if (n.neuron > neuronMax) neuronMax = n.neuron;
    }
    if (neuronMax < 1) neuronMax = 3072;
    const xS = nid => padL + (nid / neuronMax) * plotW;
    const yS = L => padT + (layerMax > 0 ? (L / layerMax) * plotH : plotH * 0.5);
    ctxCoactL.strokeStyle = "#171b24"; ctxCoactL.lineWidth = 1;
    for (let L = 0; L <= layerMax; L++) {
      const y = yS(L);
      ctxCoactL.beginPath(); ctxCoactL.moveTo(padL, y); ctxCoactL.lineTo(W - padR, y); ctxCoactL.stroke();
      ctxCoactL.fillStyle = "#6f7480"; ctxCoactL.font = "10px ui-monospace,monospace";
      ctxCoactL.fillText("L" + L, 4, y + 3);
    }
    const maxC = d.pairs.reduce((m, p) => Math.max(m, p.c), 1);
    const maxLift = d.pairs.reduce((m, p) => Math.max(m, p.lift), 1);
    const edgePixels = [];
    for (const p of d.pairs) {
      const x1 = xS(p.i[1]), y1 = yS(p.i[0]);
      const x2 = xS(p.j[1]), y2 = yS(p.j[0]);
      const a = 0.18 + 0.62 * (p.c / maxC);
      const liftN = Math.min(1, p.lift / maxLift);
      const r = Math.floor(40 + 200 * liftN);
      const g = Math.floor(180 + 75 * liftN);
      const b = Math.floor(255 - 130 * liftN);
      ctxCoactL.strokeStyle = `rgba(${r},${g},${b},${a.toFixed(2)})`;
      ctxCoactL.lineWidth = 1;
      ctxCoactL.beginPath(); ctxCoactL.moveTo(x1, y1); ctxCoactL.lineTo(x2, y2); ctxCoactL.stroke();
      edgePixels.push({ x1, y1, x2, y2, p });
    }
    const nodePixels = [];
    for (const n of d.nodes) {
      const cx = xS(n.neuron), cy = yS(n.layer);
      ctxCoactL.fillStyle = regionColorL(n.layer);
      ctxCoactL.beginPath(); ctxCoactL.arc(cx, cy, 2.4, 0, 6.3); ctxCoactL.fill();
      nodePixels.push({ x: cx, y: cy, layer: n.layer, neuron: n.neuron });
    }
    cCoactL.__edges = edgePixels;
    cCoactL.__nodes = nodePixels;
    info.innerHTML = `<span class="meta">${d.pairs.length} pairs · ${d.nodes.length} active neurons · ${d.n_tokens} probe tokens · |act|>${d.threshold} · click a node to inspect</span>`;
  });
}

function _surpriseClass(bits) {
  if (bits == null)    return "easy";
  if (bits < 1.0)      return "easy";
  if (bits < 2.5)      return "normal";
  if (bits < 4.5)      return "warm";
  return "hot";
}

function drawSurpriseL(d) {
  const info     = $("learnSurpriseInfo");
  const promptEl = $("learnSurprisePrompt");
  if (!d || !d.steps || d.steps.length === 0) {
    cSurpriseL.style.display = "none";
    if (promptEl) promptEl.innerHTML = "";
    const n = (learningState.meta && learningState.meta.checkpoints) ? learningState.meta.checkpoints.length : 0;
    info.innerHTML = n > 0
      ? `<span style="color:var(--hot)">Difficulty data not available for this run.</span>`
      : `<span style="color:var(--warm)">No checkpoint yet.</span>`;
    return;
  }
  cSurpriseL.style.display = "";
  requestAnimationFrame(() => {
    const { w: W, h: H } = fitCanvas(cSurpriseL);
    ctxSurpriseL.fillStyle = "#06070a"; ctxSurpriseL.fillRect(0, 0, W, H);
    const nCkpt = d.steps.length, nTok = d.tokens.length;
    if (nTok === 0) {
      cSurpriseL.style.display = "none";
      info.innerHTML = `<span style="color:var(--hot)">Test prompt is empty for this run.</span>`;
      if (promptEl) promptEl.innerHTML = "";
      return;
    }

    const avg = new Array(nCkpt).fill(0);
    const cnt = new Array(nCkpt).fill(0);
    for (let i = 0; i < nCkpt; i++) {
      for (let j = 0; j < nTok; j++) {
        const v = d.surprise[i][j];
        if (v == null) continue;
        avg[i] += v; cnt[i]++;
      }
      avg[i] = cnt[i] > 0 ? avg[i] / cnt[i] : null;
    }
    const validAvgs = avg.filter(v => v != null);
    const yMin = Math.min(...validAvgs);
    const yMax = Math.max(...validAvgs);
    const yPad = Math.max(0.05, (yMax - yMin) * 0.15);
    const y0 = yMin - yPad, y1 = yMax + yPad;
    const padL = 44, padR = 12, padT = 8, padB = 18;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const xS = i => padL + (nCkpt > 1 ? (i / (nCkpt - 1)) * plotW : plotW / 2);
    const yS = v => padT + plotH - ((v - y0) / Math.max(1e-9, y1 - y0)) * plotH;

    ctxSurpriseL.strokeStyle = "#1e2330"; ctxSurpriseL.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const yy = padT + plotH * g / 4;
      ctxSurpriseL.beginPath();
      ctxSurpriseL.moveTo(padL, yy); ctxSurpriseL.lineTo(W - padR, yy);
      ctxSurpriseL.stroke();
    }
    ctxSurpriseL.fillStyle = "#6f7480"; ctxSurpriseL.font = "10px ui-monospace,monospace";
    for (let g = 0; g <= 4; g++) {
      const yv = y1 - (y1 - y0) * g / 4;
      ctxSurpriseL.fillText(yv.toFixed(2), 4, padT + plotH * g / 4 + 4);
    }
    ctxSurpriseL.fillText("bits/byte", 4, padT - 1);
    ctxSurpriseL.fillText(d.steps[0].toLocaleString(), padL, H - 4);
    if (nCkpt > 1) {
      ctxSurpriseL.fillText(d.steps[nCkpt - 1].toLocaleString(), W - padR - 40, H - 4);
    }

    ctxSurpriseL.strokeStyle = "#5dc8ff"; ctxSurpriseL.lineWidth = 2;
    ctxSurpriseL.beginPath();
    let started = false;
    for (let i = 0; i < nCkpt; i++) {
      if (avg[i] == null) continue;
      const x = xS(i), y = yS(avg[i]);
      if (!started) { ctxSurpriseL.moveTo(x, y); started = true; }
      else          { ctxSurpriseL.lineTo(x, y); }
    }
    ctxSurpriseL.stroke();

    ctxSurpriseL.fillStyle = "#5dc8ff";
    for (let i = 0; i < nCkpt; i++) {
      if (avg[i] == null) continue;
      ctxSurpriseL.beginPath();
      ctxSurpriseL.arc(xS(i), yS(avg[i]), 2.5, 0, Math.PI * 2);
      ctxSurpriseL.fill();
    }

    let trendVerdict;
    if (validAvgs.length < 2) {
      trendVerdict = `<b style="color:var(--soft)">single checkpoint</b>: need at least two to show a trend.`;
    } else {
      const first = validAvgs[0], last = validAvgs[validAvgs.length - 1];
      const drop = first - last;
      const dropPct = (drop / Math.max(1e-9, first)) * 100;
      if (dropPct >= 5) {
        trendVerdict = `<b style="color:var(--data-pos)">improving</b>: average difficulty fell ${dropPct.toFixed(0)}% (${first.toFixed(2)} → ${last.toFixed(2)} bits/byte).`;
      } else if (dropPct <= -5) {
        trendVerdict = `<b style="color:var(--hot)">regressing</b>: average difficulty rose ${(-dropPct).toFixed(0)}% (${first.toFixed(2)} → ${last.toFixed(2)} bits/byte).`;
      } else {
        trendVerdict = `<b style="color:var(--warm)">flat</b>: average difficulty barely moved (${first.toFixed(2)} → ${last.toFixed(2)} bits/byte). The model has likely converged.`;
      }
    }
    info.innerHTML = `<span class="meta">${trendVerdict}</span>`;

    if (promptEl) {
      const lastIdx = nCkpt - 1;
      const row = d.surprise[lastIdx] || [];
      const safe = c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c] || c);
      let html = "";
      for (let j = 0; j < nTok; j++) {
        const b = d.tokens[j];
        const v = row[j];
        const cls = _surpriseClass(v);
        const ch = (b === 10) ? "<br/>"
                 : (b === 32) ? "&nbsp;"
                 : (b < 32 || b > 126) ? "·"
                 : safe(String.fromCharCode(b));
        const title = (v != null)
          ? `'${b === 10 ? "\\n" : b === 32 ? "space" : String.fromCharCode(b)}' at position ${j}: ${v.toFixed(2)} bits to predict`
          : `position ${j}: no data`;
        html += `<span class="ch ${cls}" title="${title}">${ch}</span>`;
      }
      promptEl.innerHTML = html;
    }
  });
}

cCoactL.style.cursor = "crosshair";
cCoactL.addEventListener("mouseleave", () => { $("cCoactLHover").textContent = "hover to inspect"; });
cCoactL.addEventListener("click", e => {
  const nodes = cCoactL.__nodes;
  if (!nodes || !nodes.length) return;
  const rect = cCoactL.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = null, bestD = 8;
  for (const n of nodes) {
    const d = Math.hypot(mx - n.x, my - n.y);
    if (d < bestD) { bestD = d; best = n; }
  }
  if (best) showNeuronModal(best.layer, best.neuron);
});
cCoactL.addEventListener("mousemove", e => {
  const edges = cCoactL.__edges;
  const info = $("cCoactLHover");
  if (!edges) return;
  const rect = cCoactL.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = null, bestD = 6;
  for (const ed of edges) {
    const dx = ed.x2 - ed.x1, dy = ed.y2 - ed.y1;
    const L2 = dx*dx + dy*dy;
    let t = ((mx - ed.x1) * dx + (my - ed.y1) * dy) / Math.max(1e-6, L2);
    t = Math.max(0, Math.min(1, t));
    const px = ed.x1 + t * dx, py = ed.y1 + t * dy;
    const d = Math.hypot(mx - px, my - py);
    if (d < bestD) { bestD = d; best = ed.p; }
  }
  if (!best) { info.textContent = "hover to inspect"; return; }
  info.innerHTML = `<b>L${best.i[0]}/n${best.i[1]} ↔ L${best.j[0]}/n${best.j[1]}</b>   co-fires: <b>${best.c}</b>   lift: <b>${best.lift.toFixed(2)}×</b>`;
});

async function renderTier2ForLearning() {
  const run = learningTimelineName;
  if (!run) return;
  if (tier2State.run !== run) {
    tier2State.run = run;
    tier2State.coact = {};
    tier2State.surprise = null;
  }
  const meta = learningState.meta;
  const ck = meta && meta.checkpoints[learningState.ckptIdx];
  if (ck) {
    const guardStep = ck.step;
    const isStale = () => (
      tier2State.run !== run ||
      !learningState.meta.checkpoints[learningState.ckptIdx] ||
      learningState.meta.checkpoints[learningState.ckptIdx].step !== guardStep
    );
    fetchTier2Coact(run, ck.step).then(d => { if (!isStale()) drawCoactL(d); });
    if (atlasConceptArmed) atlasGoConcept();
  } else {
    drawCoactL(null);
  }
  fetchTier2Surprise(run).then(d => { if (tier2State.run === run) drawSurpriseL(d); });
}

let atlasConceptArmed = false;

// fetch model meta on load (without generating)
fetch("/meta").then(r => r.json()).then(setMeta).catch(() => {});

const backendsState = { pytorch: { loaded: false, pending: false }, c: { loaded: false, pending: false }, busy: false, lastError: "" };

function _renderBackendState() {
  const sel = $("backend"), lbl = $("backendState");
  if (!sel || !lbl) return;
  const which = sel.value;
  const s = backendsState[which] || { loaded: false, pending: false };
  const buildBusy = which === "c" && s.build && s.build.status === "building";
  if (backendsState.busy || s.pending || buildBusy) {
    let detail = "initializing";
    if (which === "c" && s.build) {
      if (s.build.status === "building") detail = "building engine";
      else if (s.build.status === "ok" && !s.loaded) detail = "starting subprocess";
    }
    lbl.innerHTML = `<span class="spinner"></span><b style="color:var(--warm)">${detail}</b>`;
    return;
  }
  // C backend with zero exported .bin files — surface the real cause clearly.
  // "not loaded" with no exported model is just confusing; the user has to know
  // to go to Training → export to .bin before anything will work.
  if (which === "c" && !s.loaded && s.bins_available === 0) {
    lbl.innerHTML = `<b style="color:var(--warm)">no exported model</b> &mdash; <a href="#training" style="color:var(--accent)">Training &rarr; export to .bin</a>`;
    _applyGenerateGate();
    return;
  }
  // C backend selected a non-QAT bin: engine refuses to load (it would
  // produce gibberish). Show a clear "QAT required" state instead of a
  // generic "not loaded", and disable the Generate button.
  if (which === "c" && !s.loaded && s.blocked_reason === "qat_required") {
    const bm = s.blocked_model || s.model_dir || "model";
    lbl.innerHTML = `<b style="color:var(--warm)">model not QAT-trained</b> &mdash; generate disabled. switch to PyTorch backend, or retrain <code>${bm}</code> with <code>qat_enabled=true</code>.`;
    _applyGenerateGate();
    return;
  }
  const err = backendsState.lastError ? ` <span style="color:var(--hot)">${backendsState.lastError}</span>` : "";
  lbl.innerHTML = (s.loaded
    ? `<b style="color:var(--data-pos)">ready</b>`
    : `<b style="color:var(--dim)">not loaded</b>`) + err;
  _applyGenerateGate();
}

function _applyGenerateGate() {
  // Disable Generate when the currently-selected backend is unusable.
  // Use case: c backend has act_boost>1 (QAT required), or no model.
  const sel = $("backend"), goBtn = $("go");
  if (!sel || !goBtn) return;
  const which = sel.value;
  const s = backendsState[which] || {};
  const generating = goBtn.dataset.generating === "1";
  let block = false;
  let title = "";
  if (which === "c") {
    if (s.blocked_reason === "qat_required") {
      block = true;
      title = `model '${s.blocked_model || ""}' is not QAT-trained — switch to PyTorch backend or retrain with qat_enabled=true`;
    } else if (!s.loaded && s.bins_available === 0) {
      block = true;
      title = "no exported .bin — Training → export to .bin first";
    }
  }
  if (block && !generating) {
    goBtn.disabled = true;
    goBtn.title = title;
  } else if (!generating) {
    goBtn.disabled = false;
    goBtn.title = "";
  }
}

function _pollBackends() {
  return fetch("/backends").then(r => r.json()).then(d => {
    backendsState.pytorch = d.pytorch || { loaded: false, pending: false };
    backendsState.c       = d.c       || { loaded: false, pending: false };
    _renderBackendState();
    return d;
  }).catch(() => null);
}

function _waitUntilSettled(which) {
  return _pollBackends().then(d => {
    const s = (d && d[which]) || {};
    const stillPending = s.pending || (which === "c" && s.build && s.build.status === "building");
    if (!stillPending) {
      backendsState.busy = false;
      if (which === "c" && s.build && s.build.status === "failed" && !s.loaded) {
        backendsState.lastError = (s.build && s.build.error) || "build failed";
      } else if (which === "c" && s.bins_available === 0) {
        // suppress "did not load"; _renderBackendState shows the no-export hint
        backendsState.lastError = "";
      } else if (!s.loaded && !backendsState.lastError) {
        backendsState.lastError = "did not load";
      }
      _renderBackendState();
      return;
    }
    setTimeout(() => _waitUntilSettled(which), 1000);
  });
}

function _toggleBackend(which, action) {
  backendsState.busy = true;
  backendsState.lastError = "";
  _renderBackendState();
  return fetch(`/backends/${which}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ action }),
  }).then(r => r.json().catch(() => ({ error: `http ${r.status}` })))
    .then(d => {
      if (d && d.error) backendsState.lastError = String(d.error);
      if (action === "unload") { backendsState.busy = false; return _pollBackends(); }
      return _waitUntilSettled(which);
    })
    .catch(err => {
      backendsState.busy = false;
      backendsState.lastError = String(err && err.message || err);
      _renderBackendState();
    });
}

document.addEventListener("DOMContentLoaded", () => {
  const sel = $("backend");
  if (!sel) return;
  sel.addEventListener("change", () => {
    backendsState.lastError = "";
    const which = sel.value;
    const s = backendsState[which] || {};
    if (which === "c" && s.bins_available === 0) { _renderBackendState(); return; }
    if (!s.loaded) _toggleBackend(which, "load");
    else _renderBackendState();
  });
  _pollBackends().then(d => {
    if (!d) return;
    const which = sel.value;
    const s = d[which];
    // skip the load round-trip when c is selected but nothing has been
    // exported yet — there is nothing the server can load, and the
    // backend-state line is already showing the "no exported model" hint.
    if (which === "c" && s && s.bins_available === 0) return;
    if (s && !s.loaded && !s.pending) _toggleBackend(which, "load");
  });
});

// ============================================================
// TRAINING (Training tab) - flow-driven trainer plugin form
// ============================================================
// surfaces only manifest.kind === "trainer" plugins, grouped by manifest.flow.
// renders the form from MANIFEST.args. supports str/int/float/bool/text/path,
// plus dashboard-resolved types: corpus, model_name, model_step (cascades
// from a sibling model_name via depends_on). live composes the model name
// from corpus+size+precision+variant per glass_model_roe rule 1.

const trainState = {
  list: [],
  flow: null,
  selected: null,
  running: { status: "idle" },
  discovery: { corpora: [], models: [] },
};

function _trEl(id) { return document.getElementById(id); }
function _trEsc(s) { return String(s).replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c])); }

function _trModelByName(n) {
  return trainState.discovery.models.find(m => m.name === n) || null;
}

function _trBuildInput(a) {
  const name = a.name;
  const def  = a.default !== undefined ? a.default : "";
  const t    = a.type;
  // Inputs sized by their grid cell (see .trArgsGrid CSS in _trRenderForm).
  // No min-widths here — they would prevent the responsive layout from
  // collapsing on narrow viewports.
  const inputBase = "background:#0a0c12;border:1px solid var(--line);color:var(--text);padding:3px 6px;font:inherit;border-radius:2px;width:100%;box-sizing:border-box;min-width:0";
  if (t === "bool") {
    return `<input type="checkbox" data-arg="${_trEsc(name)}" ${def ? "checked" : ""}>`;
  }
  if (t === "text") {
    return `<textarea data-arg="${_trEsc(name)}" rows="2" style="${inputBase};padding:4px 6px;resize:vertical">${_trEsc(def)}</textarea>`;
  }
  if (t === "str" && Array.isArray(a.choices)) {
    const opts = ['<option value="">- pick -</option>']
      .concat(a.choices.map(c => `<option value="${_trEsc(c)}" ${c === def ? "selected" : ""}>${_trEsc(c)}</option>`))
      .join("");
    return `<select data-arg="${_trEsc(name)}" style="${inputBase}">${opts}</select>`;
  }
  if (t === "corpus") {
    const all = trainState.discovery.corpora || [];
    const source = a.source || "any";
    const selectedPluginId = trainState.selected ? trainState.selected.id : null;
    const filtered = all.filter(c => {
      if (source === "shared")  return c.source === "shared";
      if (source === "bundled") return c.source === "bundle" && c.plugin_id === selectedPluginId;
      return true;
    });
    if (filtered.length === 0) {
      const where = source === "bundled"
        ? `<code>plugins/${_trEsc(selectedPluginId || "<bundle>")}/corpus/&lt;name&gt;_train.bin</code>`
        : `<code>plugins/corpus/&lt;name&gt;_train.bin</code>`;
      const what = source === "bundled"
        ? "this bundle's own corpus folder"
        : (source === "shared" ? "the shared corpus folder (plugins/corpus/)" : "any corpus folder");
      return `<p class="train-no-corpus-error">No training data file found in ${what}. Drop a <code>&lt;name&gt;_train.bin</code> file at ${where}, then click <i>refresh</i>.</p><input type="hidden" data-arg="${_trEsc(name)}" value="">`;
    }
    const opts = ['<option value="">- pick a training data file -</option>']
      .concat(filtered.map(c => `<option value="${_trEsc(c.stem)}" ${c.stem === def ? "selected" : ""}>${_trEsc(c.label)}</option>`))
      .join("");
    return `<select data-arg="${_trEsc(name)}" style="${inputBase}">${opts}</select>`;
  }
  if (t === "model_name") {
    const opts = ['<option value="">- pick a model -</option>']
      .concat(trainState.discovery.models.map(m => `<option value="${_trEsc(m.name)}" ${m.name === def ? "selected" : ""}>${_trEsc(m.name)}</option>`))
      .join("");
    return `<select data-arg="${_trEsc(name)}" data-arg-kind="model_name" style="${inputBase}">${opts}</select>`;
  }
  if (t === "model_step") {
    return `<select data-arg="${_trEsc(name)}" data-arg-kind="model_step" data-depends-on="${_trEsc(a.depends_on || "")}" style="${inputBase}"><option value="">- pick model first -</option></select>`;
  }
  const itype = (t === "int" || t === "float") ? "number" : "text";
  const step  = t === "float" ? ' step="any"' : (t === "int" ? ' step="1"' : "");
  return `<input type="${itype}" data-arg="${_trEsc(name)}" value="${_trEsc(def)}"${step} style="${inputBase}">`;
}

function _trArgEl(name) {
  return document.querySelector(`#trainArgs [data-arg="${name}"]`);
}

function _trArgVal(name) {
  const el = _trArgEl(name);
  if (!el) return "";
  if (el.type === "checkbox") return !!el.checked;
  return el.value;
}

function _trUpdateComposedName() {
  const out = _trEl("trainComposedName");
  if (!out) return;
  const corpus    = _trArgVal("corpus");
  const size      = _trArgVal("size");
  const precision = _trArgVal("precision");
  const version   = _trArgVal("version");
  if (!corpus && !size && !precision && !version) { out.textContent = ""; return; }
  const corpusLeaf = corpus.includes(":") ? corpus.split(":").pop() : corpus;
  const parts = [corpusLeaf || "?", size || "?", precision || "?", version || "?"];
  out.textContent = parts.join("_");
}

// Memory estimator for the train form. Computed from form values + the
// detected hw budget when available.
//
//   static = params × bytes-per-param
//   acts   = batch × seq × hidden × layers × 13 × bpv × bptt_window
//   total  = (static + acts) × 1.15  (PyTorch overhead pad)
//
// Bytes-per-param is always 16 for AdamW (fp32 weights + fp32 grads + fp32 m
// + fp32 v) — autocast keeps weights/grads/optim in fp32 even when the
// forward path runs in bf16. Earlier versions used 14 for bf16, which was
// wrong and made the estimate read low. With 8-bit AdamW (MEGA only),
// optimizer state drops to 2 bytes/param, total becomes 10.
//
// bpv (bytes per activation element) follows the autocast dtype: bf16 = 2,
// fp32 = 4. Activations are saved in the autocast dtype.
//
// Activation multiplier 13 = per-layer count of stored residual, qkv, and
// ffn intermediate tensors for the multimind transformer block. SDPA uses
// FlashAttention so there is no quadratic attention term. n_chunks sets the
// runtime memory horizon but not VRAM; bptt_window controls how many chunks
// of activation graph stay live for backward (1 = lowest VRAM but write-path
// frozen, 4 = balanced, n_chunks = full BPTT and max VRAM). Falls back to 1
// when bptt_window unset. Activation checkpointing halves activations.
//
// Budget for the red/yellow warning comes from saved system_specs.json:
// discrete GPU -> sum of vram_total. Integrated / unified memory (Apple
// Silicon, integrated Intel) -> ram_total × 0.7 (leaves room for OS + other
// processes). The estimate caps as a *warning*, not a guarantee — real
// usage varies with sequence packing, KV cache, M3 adapter slot table,
// MoE routing, QAT fake-quant overhead, and PyTorch fragmentation.
const _TR_SIZE_PRESETS = {
  // multimind_m3 / m1 sizes (dense)
  "30m":  { layers: 10, hidden: 512,  ffn: 2048, heads: 8,  params: 31e6  },
  "80m":  { layers: 12, hidden: 768,  ffn: 3072, heads: 12, params: 85e6  },
  "120m": { layers: 12, hidden: 896,  ffn: 3584, heads: 14, params: 115e6 },
  "200m": { layers: 16, hidden: 1024, ffn: 4096, heads: 16, params: 202e6 },
  "400m": { layers: 24, hidden: 1280, ffn: 5120, heads: 20, params: 472e6 },
  "800m": { layers: 28, hidden: 1536, ffn: 6144, heads: 24, params: 793e6 },
  // multimind_mega sizes (MoE total params; per-byte active << total)
  "850m": { layers: 12, hidden: 1024, ffn: 4096, heads: 16, params: 860e6  },
  "1b":   { layers: 12, hidden: 1280, ffn: 3840, heads: 16, params: 1023e6 },
  "1b5":  { layers: 14, hidden: 1408, ffn: 4224, heads: 16, params: 1500e6 },
  // scratch_transformer-style sizes (subset)
  "5m":   { layers: 6,  hidden: 256,  ffn: 1024, heads: 4,  params: 5e6   },
  "7m":   { layers: 8,  hidden: 256,  ffn: 1024, heads: 4,  params: 7e6   },
  "20m":  { layers: 8,  hidden: 512,  ffn: 2048, heads: 8,  params: 20e6  },
};

// ---------------------------------------------------------------
// Trainer form schema. The dashboard owns labels, helps, types,
// choices, and required-flags. Plugin manifests only carry preset
// values via manifest.defaults — see plugins/<id>/manifest.json.
// Keyed by manifest.flow. Field order = render order.
// ---------------------------------------------------------------
const TRAINER_SCHEMA = {
  scratch: [
    // ---- required (every trainer) ----
    { name: "corpus",       type: "corpus", required: true,  label: "training data file",      help: "the file the model reads." },
    { name: "size",         type: "str",    required: true,  label: "model size",              help: "bigger = smarter, slower, more VRAM. choices depend on plugin: Multimind M1/M3 use 30m..800m dense, MEGA uses 200m/850m/1b/1b5 MoE.", choices: ["30m","80m","120m","200m","400m","800m","850m","1b","1b5"] },
    { name: "precision",    type: "str",    required: true,  label: "number precision",        help: "bf16 = half memory. fp32 = double memory.", choices: ["bf16","fp32"] },
    { name: "version",      type: "str",    required: true,  label: "version tag",             help: "label for this run; v1, v1b, v2a..." },
    { name: "description",  type: "text",   required: true,  label: "what this model is for",  help: "saved into the model's config.json." },
    // ---- standard training loop ----
    { name: "total_steps",  type: "int",                     label: "total training steps",    help: "more = longer training, more learned." },
    { name: "batch_size",   type: "int",                     label: "batch size",              help: "higher = faster, more VRAM." },
    { name: "seq",          type: "int",                     label: "sequence length",         help: "higher = wider per-step view, more VRAM." },
    { name: "n_chunks",     type: "int",                     label: "TBPTT chunks per step",   help: "higher = more bytes seen per step (and on Multimind M3, a longer adapter memory horizon at runtime). Activation VRAM scales with bptt_window, not n_chunks. More chunks = more compute per step." },
    { name: "base_lr",      type: "float",                   label: "peak learning rate",      help: "higher = bigger weight updates (risk: divergence). type 0.0001 or 1e-4." },
    { name: "min_lr",       type: "float",                   label: "minimum learning rate",   help: "where the LR settles at the end. type 0.00001 or 1e-5." },
    { name: "warmup_steps", type: "int",                     label: "warmup steps",            help: "more = slower ramp from 0 to peak LR." },
    { name: "lr_schedule",  type: "str",                     label: "lr schedule",             help: "shape of the LR curve after warmup. cosine = smooth decay, linear = straight decay, constant = stays at peak.", choices: ["cosine","linear","constant"] },
    { name: "weight_decay", type: "float",                   label: "weight decay",            help: "higher = stronger weight shrink (regularization)." },
    { name: "beta1",        type: "float",                   label: "adamw beta1",             help: "adamw first-moment decay. 0.9 is the default. lower = optimizer reacts faster to recent gradients." },
    { name: "beta2",        type: "float",                   label: "adamw beta2",             help: "adamw second-moment decay. 0.95 for LMs. 0.999 for general. lower = quicker variance estimate." },
    { name: "label_smoothing", type: "float",                label: "label smoothing",         help: "0.0 = off. small values (0.05 to 0.1) discourage overconfidence on the target byte." },
    { name: "grad_clip",    type: "float",                   label: "gradient clip",           help: "lower = stricter cap on per-step gradient norm." },
    { name: "ckpt_every",   type: "int",                     label: "checkpoint every",        help: "lower = more frequent saves to disk." },
    { name: "log_every",    type: "int",                     label: "log every",               help: "lower = denser train.csv (busier chart)." },
    { name: "eval_every",   type: "int",                     label: "eval every",              help: "lower = more frequent validation runs." },
    { name: "eval_iters",   type: "int",                     label: "eval batches",            help: "higher = more accurate validation loss." },
    { name: "seed",         type: "int",                     label: "random seed",             help: "different seed = different random run." },
    // ---- experimental / test-model clusters (all ADVANCED) ----
    // Adapter cluster (M1 + M3)
    { name: "rank",            type: "int",        advanced: true, label: "adapter rank",            help: "Multimind M3 trainer only (Hebbian-adapter architecture, not Apple Silicon). higher = more memory capacity in the adapter." },
    { name: "n_slots",         type: "int",        advanced: true, label: "schema slots",            help: "Multimind M1 trainer only (slot-memory architecture, not Apple Silicon). number of named slot vectors in the working-memory table. 256 is the canonical doc value." },
    { name: "alpha",           type: "float",      advanced: true, label: "adapter write alpha",     help: "Multimind M1/M3 trainers only (memory-adapter architectures, not Apple Silicon). higher = each token writes harder to memory." },
    { name: "inject_layer",    type: "int",        advanced: true, label: "inject layer (-1=auto)",  help: "Multimind M1/M3 trainers only (memory-adapter architectures, not Apple Silicon). which layer the adapter attaches to. -1 = mid-stack." },
    { name: "init_from",       type: "model_name", advanced: true, label: "init base from model",    help: "Multimind M1 trainer only (slot-memory architecture, not Apple Silicon). load base weights from another model's latest checkpoint and train M1 adapter on top. New model named <init_from>_m1." },
    { name: "bptt_window",     type: "int",        advanced: true, label: "BPTT window (chunks)",    help: "TBPTT-using trainers (Veritate base + Multimind M1/M3 — this is a training-loop knob, not Apple Silicon). BPTT = how many chunks of past activations the gradient looks back through. 1 = write-path frozen (will not train), 4 = balanced, n_chunks = full backpropagation through time (max VRAM, slowest)." },
    // MEGA cluster (ternary + MoE moonshot)
    { name: "quant_mode",      type: "str",        advanced: true, label: "weight quant mode",       help: "MEGA only. int8 (1 byte/param), int4 (0.5 bytes, 2x density), or ternary (BitNet b1.58, ~0.2 bytes, 5x density — fits 1B+ in 96 MB L3).", choices: ["int8","int4","ternary"] },
    { name: "n_experts",       type: "int",        advanced: true, label: "MoE experts",             help: "MEGA only. number of FFN experts per block. Total params scale linearly. Only top-k of these fire per byte at decode time." },
    { name: "router_topk",     type: "int",        advanced: true, label: "router top-k",            help: "MEGA only. experts that fire per token. 1 = sticky single expert (cheapest, L3-fittest). 2+ = soft mixture (better quality, more compute, more L3 pressure)." },
    { name: "router_aux_loss", type: "float",      advanced: true, label: "router balance weight",   help: "MEGA only. Switch-Transformer load-balance auxiliary loss coefficient. Prevents router collapse onto one expert. 0.01 is the standard." },
    // ---- checkboxes (all together at the end) ----
    { name: "use_act_ckpt",  type: "bool", featured: true,                  label: "activation checkpointing",   help: "trades ~30% slower compute for ~50% less activation VRAM. Recommended on tight VRAM budgets." },
    { name: "qat_enabled",   type: "bool", featured: true,                  label: "QAT enabled",                help: "train with quantization-aware fake-quant in the forward pass. The quant scheme is INT8 by default; MEGA additionally honors quant_mode (int8/int4/ternary). Result exports to a v9 binary that runs on the C engine." },
    { name: "freeze_base",   type: "bool", featured: true, advanced: true,  label: "freeze base",                help: "Multimind M1 trainer only (slot-memory architecture, not Apple Silicon). if checked, only the adapter trains; base stays exactly as init_from. Cleanest when starting from a converged base." },
    { name: "use_8bit_adam", type: "bool", featured: true, advanced: true,  label: "8-bit AdamW (bitsandbytes)", help: "MEGA only. INT8 optimizer state via bitsandbytes. Cuts m+v memory ~75%. Required to fit 1B-class MoE on 12 GB VRAM." },
  ],
  continue: [
    // ---- required ----
    { name: "resume",       type: "model_name", required: true, label: "model to continue", help: "pick a previously-trained model. its config decides shape, corpus, and adapter settings." },
    // ---- standard training loop ----
    { name: "total_steps",  type: "int",                        label: "total training steps",  help: "extend or shorten the run. existing value comes from the model's config." },
    { name: "batch_size",   type: "int",                        label: "batch size",            help: "higher = faster, more VRAM." },
    { name: "n_chunks",     type: "int",                        label: "TBPTT chunks per step", help: "higher = more bytes seen per step (and on Multimind M3, a longer adapter memory horizon at runtime). Activation VRAM scales with bptt_window, not n_chunks. More chunks = more compute per step." },
    { name: "base_lr",      type: "float",                      label: "peak learning rate",    help: "higher = bigger weight updates. lower this for fine-tuning. type 0.0001 or 1e-4." },
    { name: "min_lr",       type: "float",                      label: "minimum learning rate", help: "where the LR settles at the end. type 0.00001 or 1e-5." },
    { name: "warmup_steps", type: "int",                        label: "warmup steps",          help: "more = slower ramp from 0 to peak LR. usually irrelevant on resume since you start above 0." },
    { name: "lr_schedule",  type: "str",                        label: "lr schedule",           help: "shape of the LR curve after warmup. cosine = smooth decay, linear = straight decay, constant = stays at peak.", choices: ["cosine","linear","constant"] },
    { name: "weight_decay", type: "float",                      label: "weight decay",          help: "higher = stronger weight shrink (regularization)." },
    { name: "beta1",        type: "float",                      label: "adamw beta1",           help: "adamw first-moment decay. 0.9 is the default. lower = optimizer reacts faster to recent gradients." },
    { name: "beta2",        type: "float",                      label: "adamw beta2",           help: "adamw second-moment decay. 0.95 for LMs. 0.999 for general. lower = quicker variance estimate." },
    { name: "label_smoothing", type: "float",                   label: "label smoothing",       help: "0.0 = off. small values (0.05 to 0.1) discourage overconfidence on the target byte." },
    { name: "grad_clip",    type: "float",                      label: "gradient clip",         help: "lower = stricter cap on per-step gradient norm. raise if you see clipping kill the signal." },
    { name: "ckpt_every",   type: "int",                        label: "checkpoint every",      help: "lower = more frequent saves to disk." },
    { name: "log_every",    type: "int",                        label: "log every",             help: "lower = denser train.csv (busier chart)." },
    { name: "eval_every",   type: "int",                        label: "eval every",            help: "lower = more frequent validation runs." },
    { name: "eval_iters",   type: "int",                        label: "eval batches",          help: "higher = more accurate validation loss, slower eval." },
    // ---- experimental / test-model knobs (ADVANCED) ----
    { name: "bptt_window",  type: "int", advanced: true,        label: "BPTT window (chunks)",  help: "TBPTT-using trainers (Veritate base + Multimind M1/M3 — this is a training-loop knob, not Apple Silicon). BPTT = how many chunks of past activations the gradient looks back through. 1 = write-path frozen (will not train), 4 = balanced, n_chunks = full backpropagation through time (max VRAM, slowest)." },
    // ---- checkboxes (all together at the end) ----
    { name: "use_act_ckpt",  type: "bool", featured: true,                  label: "activation checkpointing",   help: "trades ~30% slower compute for ~50% less activation VRAM. Recommended on tight VRAM budgets." },
    { name: "qat_enabled",   type: "bool", featured: true,                  label: "QAT enabled",                help: "fine-tune the source model into a new <source>_qat model with quantization-aware training. Use a low lr (1e-5). Result exports cleanly to the C engine." },
    { name: "use_8bit_adam", type: "bool", featured: true, advanced: true,  label: "8-bit AdamW (bitsandbytes)", help: "MEGA only. INT8 optimizer state via bitsandbytes. Required to fit 1B-class MoE on 12 GB VRAM. Set the same value as the original run." },
  ],
};

// Build the per-render arg list for a plugin: schema fields, filtered to only
// those the plugin's manifest declares (or that are required), with manifest
// values overlaid as `default`. The filter keeps each plugin's form focused —
// MEGA doesn't see M3's `rank` field, M3 doesn't see MEGA's `n_experts`, etc.
// The schema is the catalog of known knobs; the manifest opts each plugin in.
function _trArgsForPlugin(p) {
  if (!p || !p.manifest) return [];
  const sch  = TRAINER_SCHEMA[trainState.flow] || [];
  const defs = p.manifest.defaults || {};
  return sch
    .filter(a => a.required || Object.prototype.hasOwnProperty.call(defs, a.name))
    .map(a => {
      const out = Object.assign({}, a);
      if (defs[a.name] !== undefined) out.default = defs[a.name];
      return out;
    });
}

const _TR_CONTINUE_CFG_CACHE = {};

function _trEnsureContinueCfg(name) {
  if (!name) return null;
  if (_TR_CONTINUE_CFG_CACHE[name] !== undefined) return _TR_CONTINUE_CFG_CACHE[name];
  _TR_CONTINUE_CFG_CACHE[name] = null;
  fetch(`/run/${encodeURIComponent(name)}/config`).then(r => r.ok ? r.json() : null).then(cfg => {
    _TR_CONTINUE_CFG_CACHE[name] = cfg || false;
    _trUpdateVramEstimate();
  }).catch(() => { _TR_CONTINUE_CFG_CACHE[name] = false; });
  return null;
}

// Cached system specs for the estimator and auto-optimize. Populated lazily
// on first call; refreshed by /sys/detect handler.
let _sysSpecsCache = null;
function _trMemoryBudget() {
  const s = _sysSpecsCache;
  if (!s || !s.platform) return null;
  const gpus = s.gpus || [];
  const discrete = gpus.find(g => g && !g.integrated && g.vram_total);
  if (discrete) return { bytes: discrete.vram_total, label: discrete.name || "GPU", kind: "vram" };
  const ramTotal = (s.memory && s.memory.total_bytes) || null;
  if (ramTotal) return { bytes: Math.round(ramTotal * 0.7), label: "unified memory", kind: "unified" };
  return null;
}

function _trEstimateMemory() {
  let size        = _trArgVal("size");
  let precision   = _trArgVal("precision");
  let seq         = parseInt(_trArgVal("seq"), 10);
  const batch     = parseInt(_trArgVal("batch_size"), 10);
  const bpttRaw   = parseInt(_trArgVal("bptt_window"), 10);
  const ckpt      = _trArgEl("use_act_ckpt");
  const ckptOn    = ckpt && ckpt.type === "checkbox" && ckpt.checked;
  const a8        = _trArgEl("use_8bit_adam");
  const adam8On   = a8 && a8.type === "checkbox" && a8.checked;

  if (!size || !precision || !seq) {
    const resumeName = _trArgVal("resume");
    if (resumeName) {
      const cfg = _trEnsureContinueCfg(resumeName);
      if (cfg) {
        const ta = cfg.training_args || {};
        const sh = cfg.shape || {};
        size      = size      || ta.size;
        precision = precision || ta.precision;
        seq       = seq       || sh.seq || ta.seq;
      }
    }
  }

  const sz = _TR_SIZE_PRESETS[size];
  if (!sz || !batch || !seq) return null;

  const bpv = precision === "bf16" ? 2 : 4;
  const bytesPerParam = adam8On ? 10 : 16;
  const bpttWindow = Math.max(1, isNaN(bpttRaw) ? 1 : bpttRaw);

  const staticBytes = sz.params * bytesPerParam;
  let actBytes = batch * seq * sz.hidden * sz.layers * 13 * bpv * bpttWindow;
  if (ckptOn) actBytes *= 0.5;
  const totalRaw = staticBytes + actBytes;
  const total = Math.round(totalRaw * 1.15);
  return { staticBytes, actBytes, total, ckptOn, adam8On, size, precision, seq, batch, bpttWindow };
}

function _trUpdateVramEstimate() {
  const out = _trEl("trainVramEst");
  if (!out) return;
  const est = _trEstimateMemory();
  if (!est) { out.textContent = ""; return; }
  const fmt = (b) => b >= 1e9 ? (b / 1e9).toFixed(1) + " GB" : (b / 1e6).toFixed(0) + " MB";

  const budget = _trMemoryBudget();
  let badgeColor = "var(--warm)", budgetSuffix = "";
  if (budget) {
    const ratio = est.total / budget.bytes;
    if (ratio >= 0.95)      badgeColor = "var(--hot)";
    else if (ratio >= 0.7)  badgeColor = "var(--warm)";
    else                    badgeColor = "var(--data-pos)";
    const pct = (ratio * 100).toFixed(0);
    budgetSuffix = ` <span style="color:${badgeColor}">of ${fmt(budget.bytes)} ${budget.label} (${pct}%)</span>`;
  }

  out.innerHTML =
    `<b style="color:${badgeColor}">estimated</b>` +
    ` <b style="color:var(--text)">≈ ${fmt(est.total)}</b>${budgetSuffix}` +
    ` <span style="color:var(--dim)">(static ${fmt(est.staticBytes)} + acts ${fmt(est.actBytes)}` +
    `${est.ckptOn ? ", act-ckpt" : ""}${est.adam8On ? ", 8bit-adam" : ""}, +15% overhead)</span>`;
}

// Auto-pick training settings. Gated by Advanced telemetry consent + a
// detected sys_specs file. Fills batch_size, base_lr, use_act_ckpt, and
// (when total_steps is set) warmup_steps + log/eval/ckpt cadence. Leaves
// total_steps and architecture-specific knobs to the manifest defaults
// or the user. The Veritate trainers do not implement gradient
// accumulation, so batch_size is the effective batch.
function _trUpdateAutoOptimizeVisibility() {
  const row = $("trainAutoOptimizeRow");
  if (!row) return;
  const consent = !!(settingsState.current && settingsState.current.analytics_advanced_enabled);
  const haveSpecs = !!(_sysSpecsCache && _sysSpecsCache.platform);
  row.style.display = (consent && haveSpecs) ? "flex" : "none";
}

function _trSetArgVal(name, val) {
  const el = _trArgEl(name);
  if (!el) return false;
  if (el.type === "checkbox") { el.checked = !!val; }
  else { el.value = val; }
  el.dispatchEvent(new Event("change", { bubbles: true }));
  el.dispatchEvent(new Event("input",  { bubbles: true }));
  return true;
}

function _trAutoOptimize() {
  const status = $("trainAutoOptimizeStatus");
  const setStatus = (msg, color) => { if (status) { status.textContent = msg; status.style.color = color || "var(--dim)"; } };

  const budget = _trMemoryBudget();
  if (!budget) { setStatus("no system specs detected — run Detect my system first.", "var(--hot)"); return; }

  const plugin  = trainState.selected;
  const defs    = (plugin && plugin.manifest && plugin.manifest.defaults) || {};
  const manifestBatch = parseInt(defs.batch_size, 10) || 8;
  const manifestLR    = parseFloat(defs.base_lr) || 6e-4;

  // Memory-fit search. Start from manifest batch (already tuned per
  // architecture) and only adjust to fit hardware: shrink if over budget,
  // grow if there is comfortable headroom. Trainers do not implement
  // gradient accumulation so batch_size is the effective batch.
  const tryEstimate = (batch, ckpt) => {
    const bEl = _trArgEl("batch_size");
    const cEl = _trArgEl("use_act_ckpt");
    const prevB = bEl ? bEl.value : null;
    const prevC = cEl && cEl.type === "checkbox" ? cEl.checked : null;
    if (bEl) bEl.value = batch;
    if (cEl) cEl.checked = !!ckpt;
    const e = _trEstimateMemory();
    if (bEl && prevB !== null) bEl.value = prevB;
    if (cEl && prevC !== null) cEl.checked = prevC;
    return e;
  };

  const TARGET_RATIO = 0.75;     // aim for ~75% of budget
  const TIGHT_RATIO  = 0.92;     // anything above this is "too tight"
  const HEADROOM_RATIO = 0.45;   // anything below means there is room to grow

  let targetBatch = manifestBatch;
  let actCkpt = !!defs.use_act_ckpt;

  // Step 1: shrink if manifest at current ckpt setting overshoots.
  let est = tryEstimate(targetBatch, actCkpt);
  while (est && est.total > budget.bytes * TIGHT_RATIO && (targetBatch > 1 || !actCkpt)) {
    if (!actCkpt) {
      actCkpt = true;            // try act_ckpt before halving
    } else if (targetBatch > 1) {
      targetBatch = Math.max(1, Math.floor(targetBatch / 2));
    } else {
      break;
    }
    est = tryEstimate(targetBatch, actCkpt);
  }

  // Step 2: if still way under budget at current settings, try doubling
  // batch (cap at 4× manifest to avoid extreme lr scaling).
  const batchCap = manifestBatch * 4;
  let growEst = est;
  while (growEst && growEst.total < budget.bytes * HEADROOM_RATIO && targetBatch * 2 <= batchCap) {
    const next = targetBatch * 2;
    const probe = tryEstimate(next, actCkpt);
    if (!probe || probe.total > budget.bytes * TARGET_RATIO) break;
    targetBatch = next;
    growEst = probe;
  }

  // sqrt scaling rule for lr.
  const lrScale = Math.sqrt(targetBatch / manifestBatch);
  const newLR = +(manifestLR * lrScale).toPrecision(2);

  _trSetArgVal("batch_size",    targetBatch);
  _trSetArgVal("base_lr",       newLR);
  _trSetArgVal("use_act_ckpt",  actCkpt);

  // Cadence + warmup as % of total_steps. total_steps is user-owned;
  // if it's empty we don't touch the time-based knobs.
  const totalSteps = parseInt(_trArgVal("total_steps"), 10);
  if (totalSteps > 0) {
    _trSetArgVal("warmup_steps", Math.max(50,  Math.round(totalSteps * 0.03)));
    _trSetArgVal("log_every",    Math.max(10,  Math.round(totalSteps * 0.001)));
    _trSetArgVal("eval_every",   Math.max(100, Math.round(totalSteps * 0.05)));
    _trSetArgVal("ckpt_every",   Math.max(200, Math.round(totalSteps * 0.10)));
  }

  _trUpdateVramEstimate();
  setStatus(`set: batch=${targetBatch}, lr=${newLR}, act_ckpt=${actCkpt ? "on" : "off"}${totalSteps > 0 ? ", warmup/eval/ckpt scaled to total_steps" : " (set total_steps then re-run for warmup/eval/ckpt)"}.`, "var(--data-pos)");
}

function _trUpdateStepCascades() {
  document.querySelectorAll('#trainArgs [data-arg-kind="model_step"]').forEach(sel => {
    const dep = sel.dataset.dependsOn;
    if (!dep) return;
    const depEl = _trArgEl(dep);
    const modelName = depEl ? depEl.value : "";
    const m = _trModelByName(modelName);
    const cur = sel.value;
    if (!m) { sel.innerHTML = '<option value="">- pick model first -</option>'; return; }
    sel.innerHTML = '<option value="">- pick a step -</option>' +
      m.steps.map(s => `<option value="${s}" ${String(s) === String(cur) ? "selected" : ""}>${s}</option>`).join("");
  });
}

function _trFmtBytes(n) {
  if (n == null) return "?";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + " MB";
  return (n / (1024 * 1024 * 1024)).toFixed(2) + " GB";
}

function _trFmtAge(mtime) {
  if (mtime == null) return "?";
  const dt = new Date(mtime * 1000);
  return dt.toISOString().substring(0, 19).replace("T", " ") + " UTC";
}

const trainCorpusMetaState = { stem: null, inflight: false };

function _trUpdateCorpusMeta() {
  const box = _trEl("trainCorpusMeta");
  if (!box) return;
  const stem = _trArgVal("corpus");
  if (!stem) { box.style.display = "none"; box.innerHTML = ""; trainCorpusMetaState.stem = null; return; }
  if (trainCorpusMetaState.stem === stem && box.dataset.loaded === "1") return;
  trainCorpusMetaState.stem = stem;
  if (trainCorpusMetaState.inflight) return;
  trainCorpusMetaState.inflight = true;
  box.dataset.loaded = "0";
  box.style.display = "block";
  box.innerHTML = `<span style="color:var(--dim)">hashing corpus <b>${_trEsc(stem)}</b> ...</span>`;
  fetch(`/corpus/${encodeURIComponent(stem)}/usage`).then(r => r.json()).then(d => {
    trainCorpusMetaState.inflight = false;
    if (trainCorpusMetaState.stem !== stem) return;
    if (d.error) { box.innerHTML = `<span style="color:var(--hot)">${_trEsc(d.error)}</span>`; return; }
    const tr = d.train, va = d.val;
    const nMatches = (d.models || []).length;
    const matchLabel = nMatches === 0
      ? '<span style="color:var(--dim)">no models trained on this corpus yet</span>'
      : `<span style="color:var(--data-pos)">shared with ${nMatches} model${nMatches === 1 ? "" : "s"}</span>: ${(d.models || []).map(m => _trEsc(m.name)).join(", ")}`;
    let html = `<div style="color:var(--text);margin-bottom:4px">about this training data file</div>`;
    html += `<div><b>training file</b> &mdash; ${_trEsc(stem)}_train.bin &middot; ${_trFmtBytes(tr.bytes)} &middot; fingerprint <span style="color:var(--accent)">${_trEsc(tr.sha256.substring(0, 12))}</span> &middot; saved ${_trFmtAge(tr.mtime)}</div>`;
    if (va) {
      html += `<div><b>validation file</b> &mdash; ${_trEsc(stem)}_val.bin &middot; ${_trFmtBytes(va.bytes)} &middot; fingerprint <span style="color:var(--accent)">${_trEsc(va.sha256.substring(0, 12))}</span> &middot; saved ${_trFmtAge(va.mtime)}</div>`;
    } else {
      html += `<div style="color:var(--warm)"><b>validation file</b> &mdash; none on disk (the trainer will skip val loss)</div>`;
    }
    html += `<div style="margin-top:4px">${matchLabel}</div>`;
    html += `<div style="margin-top:4px;font-family:inherit;color:var(--dim)">the fingerprint (sha256) is recorded in the new model's config so two models can be compared honestly: same fingerprint = same training data, byte-for-byte.</div>`;
    box.innerHTML = html;
    box.dataset.loaded = "1";
  }).catch(e => {
    trainCorpusMetaState.inflight = false;
    box.innerHTML = `<span style="color:var(--hot)">corpus meta failed: ${_trEsc(String(e))}</span>`;
  });
}

function _trWireArgListeners() {
  document.querySelectorAll('#trainArgs [data-arg]').forEach(el => {
    const fn = () => { _trUpdateComposedName(); _trUpdateStepCascades(); _trUpdateCorpusMeta(); _trUpdateVramEstimate(); };
    el.addEventListener("change", fn);
    el.addEventListener("input",  fn);
  });
}

function _trApplyDefaults() {
  const p = trainState.selected;
  if (!p || !p.manifest) return;
  const defs = p.manifest.defaults || {};
  for (const k of Object.keys(defs)) {
    const el = _trArgEl(k);
    if (!el) continue;
    if (el.dataset.argKind === "model_step") continue;
    if (el.type === "checkbox") el.checked = !!defs[k];
    else el.value = String(defs[k]);
  }
  _trUpdateStepCascades();
  for (const k of Object.keys(defs)) {
    const el = _trArgEl(k);
    if (el && el.dataset.argKind === "model_step") {
      el.value = String(defs[k]);
    }
  }
  _trAutoPickBundledCorpus();
}

function _trAutoPickBundledCorpus() {
  const p = trainState.selected;
  if (!p) return;
  for (const a of _trArgsForPlugin(p)) {
    if (a.type !== "corpus") continue;
    const el = _trArgEl(a.name);
    if (!el || el.value) continue;
    const matches = (trainState.discovery.corpora || []).filter(
      c => c.source === "bundle" && c.plugin_id === p.id
    );
    if (matches.length === 1) {
      el.value = matches[0].stem;
    } else if (matches.length > 1) {
      const exact = matches.find(c => c.stem.split(":").pop() === p.id.split("/").pop());
      if (exact) el.value = exact.stem;
    }
  }
}

function _trRenderForm() {
  const p      = trainState.selected;
  const wrap   = _trEl("trainFormWrap");
  const argsEl = _trEl("trainArgs");
  const descEl = _trEl("trainDesc");
  const runRow = _trEl("trainRunRow");
  if (!p) {
    if (wrap)   wrap.style.display = "none";
    if (runRow) runRow.style.display = "none";
    if (descEl) descEl.textContent = "";
    if (argsEl) argsEl.innerHTML = "";
    return;
  }
  if (descEl) descEl.textContent = p.manifest.description || "";
  const introEl = _trEl("trainFormIntro");
  if (introEl) {
    introEl.innerHTML = `<b>${_trEsc(p.manifest.name || p.id)}</b> &middot; <span style="color:var(--hot)">*</span> required.`;
  }
  // Responsive grid: small fields auto-pack into columns; wide types (text/path)
  // and bool span the full row. Inline `style` keeps it self-contained — no
  // class additions to the platform stylesheet.
  let html = `<style>
    .trArgsGrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px 14px; font-size: 11.5px; }
    .trArgsGrid .cell { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
    .trArgsGrid .cell.wide { grid-column: 1 / -1; }
    .trArgsGrid .cell label { color: var(--text); font-size: 11px; }
    .trArgsGrid .cell .help { color: var(--dim); font-size: 10px; line-height: 1.35; }
    .trArgsGrid .cell input[type="text"],
    .trArgsGrid .cell input[type="number"],
    .trArgsGrid .cell select,
    .trArgsGrid .cell textarea { width: 100%; min-width: 0; box-sizing: border-box; }
    .trArgsGrid .cell.bool { flex-direction: row; align-items: center; gap: 6px; }
    .trArgsGrid .cell.bool label { font-size: 11px; }
    .trArgsGrid .cell.featured { border-top: 1px solid var(--line); padding-top: 8px; margin-top: 4px; }
    .trArgsGrid .cell.featured label { color: var(--text); font-weight: 600; }
    .trArgsGrid .cell.featured .help { color: var(--dim); }
  </style>
  <div class="trArgsGrid">`;
  for (const a of _trArgsForPlugin(p)) {
    const req   = a.required ? '<span style="color:var(--hot)"> *</span>' : "";
    const label = a.label ? _trEsc(a.label) : _trEsc(a.name);
    const advBadge = a.advanced ? `<span style="color:var(--warm);font-weight:600">ADVANCED</span> · ` : "";
    const help  = a.help ? `<div class="help">${advBadge}${_trEsc(a.help)}</div>` : "";
    const wide  = (a.type === "text" || a.type === "path") ? " wide" : "";
    const featured = a.featured ? " featured" : "";
    if (a.type === "bool") {
      html += `<div class="cell bool wide${featured}">${_trBuildInput(a)}<label>${label}${req}</label>${help && ` <span class="help" style="margin-left:6px">${_trEsc(a.help || "")}</span>` || ""}</div>`;
    } else {
      html += `<div class="cell${wide}${featured}"><label>${label}${req}</label>${_trBuildInput(a)}${help}</div>`;
    }
  }
  html += `</div>`;
  if (argsEl) argsEl.innerHTML = html;
  if (wrap)   wrap.style.display = "block";
  if (runRow) runRow.style.display = "flex";
  _trWireArgListeners();
  _trApplyDefaults();
  _trUpdateComposedName();
  _trUpdateStepCascades();
  _trUpdateCorpusMeta();
  _trUpdateVramEstimate();
}

function _trCollectArgs() {
  const p = trainState.selected;
  if (!p) return null;
  const out = {};
  for (const a of _trArgsForPlugin(p)) {
    const el = _trArgEl(a.name);
    if (!el) continue;
    if (a.type === "bool") { out[a.name] = !!el.checked; continue; }
    if (el.value === "") continue;
    if      (a.type === "int")        out[a.name] = parseInt(el.value, 10);
    else if (a.type === "float")      out[a.name] = parseFloat(el.value);
    else if (a.type === "model_step") out[a.name] = parseInt(el.value, 10);
    else                              out[a.name] = el.value;
  }
  return out;
}

function _trValidateArgs() {
  const p = trainState.selected;
  if (!p) return "no trainer selected";
  for (const a of _trArgsForPlugin(p)) {
    if (!a.required) continue;
    const el = _trArgEl(a.name);
    if (!el) return `missing field: ${a.name}`;
    const v = (el.type === "checkbox") ? el.checked : el.value;
    if (v === "" || v === null || v === undefined) return `required: ${a.name}`;
  }
  return null;
}

function _trRenderStatus() {
  const s    = trainState.running || { status: "idle" };
  const el   = _trEl("trainStatus");
  const stop = _trEl("trainStop");
  const run  = _trEl("trainRun");
  const runRow = _trEl("trainRunRow");
  if (!el) return;
  const colors = { idle: "var(--dim)", running: "var(--warm)", ok: "var(--data-pos)", failed: "var(--hot)", stopped: "var(--dim)" };
  const c  = colors[s.status] || "var(--dim)";
  const id = s.plugin_id ? ` <b>${_trEsc(s.plugin_id)}</b>` : "";
  el.innerHTML = `<b style="color:${c}">${s.status}</b>${id}`;
  if (stop) stop.disabled = s.status !== "running";
  if (run)  run.disabled  = s.status === "running" || !trainState.selected;
  // Whenever a plugin is in flight, the stop button MUST be reachable. Some
  // earlier states could collapse trainRunRow (e.g. a transient empty plugin
  // list nulling trainState.selected). Force-show it here so the user can
  // always cancel a running run.
  if (runRow && s.status === "running") runRow.style.display = "flex";
}

function _trMatchesFlow(manifest, flow) {
  if (!manifest) return false;
  const f = manifest.flow;
  return Array.isArray(f) ? f.includes(flow) : f === flow;
}

function _trFiltered() {
  if (!trainState.flow) return [];
  return trainState.list.filter(p => p.manifest && p.manifest.kind === "trainer" && _trMatchesFlow(p.manifest, trainState.flow));
}

function _trRenderPicker() {
  const sel  = _trEl("trainPicker");
  const row  = _trEl("trainPickerRow");
  const hint = _trEl("trainEmptyHint");
  if (!sel || !row) return;
  if (!trainState.flow || trainState.flow === "export") { row.style.display = "none"; return; }
  row.style.display = "flex";
  const list = _trFiltered();
  const cur  = sel.value;
  if (list.length === 0) {
    sel.innerHTML = `<option value="">- no ${_trEsc(trainState.flow)} trainers -</option>`;
    if (hint) {
      hint.style.display = "inline";
      hint.textContent = `drop a plugin into plugins/ (single .py or bundle folder with plugin.py) with kind:"trainer", flow:"${trainState.flow}"`;
    }
  } else {
    sel.innerHTML = '<option value="">- pick a trainer -</option>' +
      list.map(p => `<option value="${_trEsc(p.id)}">${_trEsc(p.manifest.name || p.id)}</option>`).join("");
    if (hint) hint.style.display = "none";
    if (cur && list.some(p => p.id === cur)) sel.value = cur;
  }
  document.querySelectorAll(".trainFlowBtn").forEach(b => {
    const on = b.dataset.flow === trainState.flow;
    b.style.fontWeight  = on ? "700" : "400";
    b.style.borderColor = on ? "var(--accent)" : "";
  });
}

function _trPoll() {
  return Promise.all([
    fetch("/plugins").then(r => r.json()),
    fetch("/train/discovery").then(r => r.json()),
  ]).then(([plug, disc]) => {
    trainState.list      = plug.plugins || [];
    trainState.running   = plug.running || { status: "idle" };
    trainState.discovery = disc || { corpora: [], models: [] };
    // Only treat the selected trainer as "gone" if the scan returned a non-empty
    // list. A transient empty response would otherwise null trainState.selected
    // and collapse the form + run row, locking the user out of stopping a run.
    const selectedGone = trainState.selected
                         && trainState.list.length > 0
                         && !trainState.list.some(p => p.id === trainState.selected.id);
    if (selectedGone) trainState.selected = null;
    _trRenderPicker();
    if (selectedGone) _trRenderForm();
    _trRenderStatus();
    _exRender();
  }).catch(() => { _trRenderStatus(); });
}

function _exRender() {
  const row = _trEl("exportPickerRow");
  const sel = _trEl("exportModel");
  if (!row || !sel) return;
  if (trainState.flow !== "export") { row.style.display = "none"; return; }
  row.style.display = "flex";
  const models = (trainState.discovery && trainState.discovery.models) || [];
  const cur = sel.value;
  sel.innerHTML = '<option value="">- pick a model -</option>' +
    models.map(m => `<option value="${_trEsc(m.name)}">${_trEsc(m.name)}</option>`).join("");
  if (cur && models.some(m => m.name === cur)) sel.value = cur;
  _exPopulateSteps();
}

function _exPopulateSteps() {
  const sel     = _trEl("exportModel");
  const stepSel = _trEl("exportStep");
  const runBtn  = _trEl("exportRun");
  const status  = _trEl("exportStatus");
  if (!sel || !stepSel) return;
  const models = (trainState.discovery && trainState.discovery.models) || [];
  const m = models.find(x => x.name === sel.value);
  if (!m) {
    stepSel.innerHTML = '<option value="">- pick a step -</option>';
    stepSel.disabled = true;
    if (runBtn) runBtn.disabled = true;
    if (status) { status.textContent = ""; status.style.color = "var(--dim)"; }
    return;
  }
  const steps = (m.steps || []).slice().sort((a, b) => b - a);
  stepSel.innerHTML = steps.map((s, i) =>
    `<option value="${s}">${s}${i === 0 ? " (latest)" : ""}</option>`).join("");
  stepSel.disabled = false;
  if (runBtn) runBtn.disabled = false;
  if (status) { status.textContent = ""; status.style.color = "var(--dim)"; }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".trainFlowBtn").forEach(b => {
    b.addEventListener("click", () => {
      trainState.flow = b.dataset.flow;
      trainState.selected = null;
      _trRenderPicker();
      _trRenderForm();
      _trRenderStatus();
      _exRender();
    });
  });
  const exModel = _trEl("exportModel");
  if (exModel) exModel.addEventListener("change", _exPopulateSteps);
  const exRun = _trEl("exportRun");
  if (exRun) exRun.addEventListener("click", async () => {
    const sel    = _trEl("exportModel");
    const stepEl = _trEl("exportStep");
    const status = _trEl("exportStatus");
    if (!sel.value) { status.style.color = "var(--hot)"; status.textContent = "pick a model"; return; }
    const step = parseInt(stepEl.value, 10);
    status.style.color = "var(--dim)";
    status.textContent = "exporting...";
    exRun.disabled = true;
    try {
      const r = await fetch(`/export/${encodeURIComponent(sel.value)}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(isFinite(step) ? { step } : {}),
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok || !d.ok) {
        status.style.color = "var(--hot)";
        status.textContent = "failed: " + (d.error || `http ${r.status}`);
      } else {
        const mb = (d.bytes / 1048576).toFixed(1);
        status.style.color = "var(--data-pos)";
        status.textContent = `wrote ${mb} MB at step ${d.step}`;
      }
    } catch (e) {
      status.style.color = "var(--hot)";
      status.textContent = "error: " + e.message;
    } finally {
      exRun.disabled = false;
    }
  });
  const sel = _trEl("trainPicker");
  if (sel) sel.addEventListener("change", () => {
    trainState.selected = trainState.list.find(p => p.id === sel.value) || null;
    _trRenderForm();
    _trRenderStatus();
  });
  const refresh = _trEl("trainRefresh");
  if (refresh) refresh.addEventListener("click", _trPoll);
  const run = _trEl("trainRun");
  if (run) run.addEventListener("click", () => {
    const err = _trValidateArgs();
    if (err) { alert(err); return; }
    const p = trainState.selected;
    fetch("/plugins/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: p.id, args: _trCollectArgs() }),
    }).then(r => r.json()).then(_trPoll).catch(_trPoll);
    // collapse the settings form once a run is launched; user can re-open via the
    // "edit settings" button if they need to change anything for the next run.
    const wrap = _trEl("trainFormWrap");
    const tog  = _trEl("trainFormToggle");
    if (wrap) wrap.style.display = "none";
    if (tog)  tog.style.display  = "inline-block";
  });
  const stop = _trEl("trainStop");
  if (stop) stop.addEventListener("click", () => {
    fetch("/plugins/stop", { method: "POST" }).then(r => r.json()).then(_trPoll).catch(_trPoll);
  });
  const tog = _trEl("trainFormToggle");
  if (tog) tog.addEventListener("click", () => {
    const wrap = _trEl("trainFormWrap");
    if (!wrap) return;
    const opening = wrap.style.display === "none";
    wrap.style.display = opening ? "block" : "none";
    tog.textContent    = opening ? "hide settings" : "edit settings";
  });
  const openLink = _trEl("trainOpenFolder");
  if (openLink) openLink.addEventListener("click", e => {
    e.preventDefault();
    fetch("/plugins/open_folder", { method: "POST" }).catch(() => {});
  });
  _trPoll();
  setInterval(_trPoll, 3000);
});

// ============================================================
// LOGS TAB
// ============================================================
// streams /logs/stream (server-sent events) into the panel. polls /engine/status
// for the build state and reflects it on both the Logs tab and the C-backend
// toggle in the Generation tab.

const LOG_LEVEL_COLOR = { info: "#a6b0bf", ok: "#5dff9b", warn: "#ffae5d", error: "#ff5d5d" };
const logState = { entries: [], filter: "all", evt: null };

function _renderLogs() {
  const el = $("logStream");
  if (!el) return;
  const filt = logState.filter;
  const rows = filt === "all" ? logState.entries : logState.entries.filter(e => e.level === filt);
  const html = rows.slice(-1000).map(e => {
    const ts = new Date(e.ts * 1000).toISOString().substring(11, 19);
    const lv = String(e.level || "info").toUpperCase().padEnd(5);
    const sr = String(e.source || "?").padEnd(8);
    const color = LOG_LEVEL_COLOR[e.level] || "#a6b0bf";
    const msg = (e.msg || "").replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));
    return `<div><span style="color:#6f7480">${ts}</span> <span style="color:${color}">${lv}</span> <span style="color:#7a8294">${sr}</span> ${msg}</div>`;
  }).join("");
  el.innerHTML = html;
  el.scrollTop = el.scrollHeight;
  $("logCount").textContent = `${rows.length} of ${logState.entries.length}`;
}

function _logEntryPush(e) {
  logState.entries.push(e);
  if (logState.entries.length > 2000) logState.entries.splice(0, 500);
  _renderLogs();
}

function _logsSubscribe() {
  if (logState.evt) return;
  fetch("/logs/snapshot")
    .then(r => r.json())
    .then(d => {
      logState.entries = d.entries || [];
      _renderLogs();
      logState.evt = new EventSource("/logs/stream");
      logState.evt.onmessage = ev => {
        try {
          const e = JSON.parse(ev.data);
          if (e && e.seq) _logEntryPush(e);
        } catch (err) {}
      };
      logState.evt.onerror = () => {};
    })
    .catch(() => {});
}

let _backendOptSynced = false;
function _syncBackendOption(binaryPresent) {
  const opt = $("backend") && $("backend").querySelector('option[value="c"]');
  if (!opt) return;
  if (binaryPresent) {
    if (opt.disabled || opt.textContent !== "Veritate") {
      opt.disabled = false;
      opt.textContent = "Veritate";
      if (!_backendOptSynced) {
        _backendOptSynced = true;
        if (typeof refreshCModels === "function") refreshCModels().then(applyBackendUI);
      }
    }
  } else {
    opt.textContent = "Veritate (not built — run build.bat)";
    _backendOptSynced = false;
  }
}

function _renderEngineStatus(s) {
  _syncBackendOption(!!s.binary_present);
  const el = $("engineStatus");
  if (!el) return;
  const color = {
    ok:       "var(--data-pos)",
    building: "var(--warm)",
    failed:   "var(--hot)",
    idle:     "var(--dim)",
    skipped:  "var(--dim)",
  }[s.status] || "var(--dim)";
  const present = s.binary_present ? "present" : "missing";
  el.innerHTML = `<span class="stat">os <b>${s.os}</b></span>
                  <span class="stat" style="margin-left:18px">arch <b>${s.arch}</b></span>
                  <span class="stat" style="margin-left:18px">status <b style="color:${color}">${s.status.toUpperCase()}</b></span>
                  <span class="stat" style="margin-left:18px">binary <b style="color:${s.binary_present ? 'var(--data-pos)' : 'var(--hot)'}">${present}</b></span>
                  <span class="stat" style="margin-left:18px">subprocess <b>${s.c_subprocess_running ? 'running' : 'stopped'}</b></span>
                  ${s.error ? `<div class="meta" style="margin-top:6px;color:var(--hot)">${s.error.replace(/[<>&]/g, c => ({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]))}</div>` : ""}`;
}

function _pollEngineStatus() {
  fetch("/engine/status").then(r => r.json()).then(_renderEngineStatus).catch(() => {});
}

document.addEventListener("DOMContentLoaded", () => {
  const lf = $("logLevelFilter");
  if (lf) lf.addEventListener("change", () => { logState.filter = lf.value; _renderLogs(); });
  const lc = $("logClear");
  if (lc) lc.addEventListener("click", () => { logState.entries = []; _renderLogs(); });
  _logsSubscribe();
  _pollEngineStatus();
  setInterval(_pollEngineStatus, 5000);
});

// ============================================================================
// settings + system metrics + hud
// ============================================================================
const settingsState = { loaded: false, current: null, saving: false };

function _fmtBytes(n) {
  if (n == null) return "—";
  const u = ["B","KB","MB","GB","TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i >= 2 ? n.toFixed(1) : Math.round(n)) + " " + u[i];
}

function _applyHudVisibility(on, detailed) {
  const hud = $("hud");
  if (!hud) return;
  hud.classList.toggle("on", !!on);
  hud.classList.toggle("detailed", !!detailed);
  document.body.classList.toggle("hud-on", !!on);
}

function _renderHud(snap) {
  if (!snap || !snap.available) {
    $("hudCpuVal").textContent = "n/a";
    $("hudMemVal").textContent = "n/a";
    $("hudGpus").innerHTML = "";
    return;
  }
  const cpuNorm = snap.cpu_count ? Math.min(100, snap.cpu_pct / snap.cpu_count) : snap.cpu_pct;
  $("hudCpuFill").style.width = cpuNorm + "%";
  $("hudCpuVal").textContent  = (snap.cpu_pct || 0).toFixed(0) + "%";
  $("hudCpuDetail").textContent = `${cpuNorm.toFixed(1)}% norm · ${snap.cpu_count} cores`;
  const sysMemPct = snap.sys_mem_total ? (snap.sys_mem_used / snap.sys_mem_total * 100) : 0;
  const procMemPct = snap.sys_mem_total ? (snap.rss_bytes / snap.sys_mem_total * 100) : 0;
  $("hudMemFill").style.width = sysMemPct + "%";
  $("hudMemVal").textContent  = sysMemPct.toFixed(0) + "%";
  $("hudMemDetail").textContent = `${_fmtBytes(snap.sys_mem_used)} of ${_fmtBytes(snap.sys_mem_total)} · this proc ${_fmtBytes(snap.rss_bytes)} (${procMemPct.toFixed(1)}%)`;
  const gpuHost = $("hudGpus");
  const gpus = snap.gpus || [];
  if (!gpus.length) { gpuHost.innerHTML = ""; return; }
  gpuHost.innerHTML = gpus.map((g, i) => {
    const load = g.load_pct;
    const w = load == null ? 0 : Math.min(100, load);
    const val = load == null ? "—" : Math.round(load) + "%";
    const tag = g.integrated ? "iGPU" : "GPU";
    let detail = g.name;
    if (g.vram_used != null && g.vram_total != null) {
      detail += ` · ${_fmtBytes(g.vram_used)}/${_fmtBytes(g.vram_total)}`;
    } else if (g.vram_total != null) {
      detail += ` · ${_fmtBytes(g.vram_total)} VRAM`;
    }
    if (load == null) detail += ` · no load telemetry`;
    return `<div class="hud-bar gpu" title="${g.name}"><span class="lbl">${tag}${i > 0 ? (i+1) : ""}</span><div class="track"><div class="fill" style="width:${w}%"></div></div><span class="val">${val}</span><span class="detail">${detail}</span></div>`;
  }).join("");
}

function _renderSysmetrics(snap) {
  const host = $("sysmetrics");
  if (!host) return;
  if (!snap || !snap.available) {
    host.innerHTML = `<div class="meta">${snap && snap.reason ? snap.reason : "metrics unavailable"}</div>`;
    return;
  }
  const cpuNorm = snap.cpu_count ? (snap.cpu_pct / snap.cpu_count).toFixed(1) : snap.cpu_pct.toFixed(1);
  const memUsedSys = _fmtBytes(snap.sys_mem_used);
  const memTotalSys = _fmtBytes(snap.sys_mem_total);
  const memPct = snap.sys_mem_total ? (snap.rss_bytes / snap.sys_mem_total * 100).toFixed(1) : "—";
  let html = "";
  html += `<div class="group cpu"><h4>CPU — this process</h4>
    <div class="row"><span class="k">total</span><span class="v">${snap.cpu_pct.toFixed(1)}%</span></div>
    <div class="row"><span class="k">normalized</span><span class="v">${cpuNorm}% of ${snap.cpu_count} cores</span></div></div>`;
  html += `<div class="group mem"><h4>Memory — this process</h4>
    <div class="row"><span class="k">RSS</span><span class="v">${_fmtBytes(snap.rss_bytes)}</span></div>
    <div class="row"><span class="k">% of system</span><span class="v">${memPct}%</span></div>
    <div class="row"><span class="k">system used</span><span class="v">${memUsedSys} / ${memTotalSys}</span></div></div>`;
  const gpus = snap.gpus || [];
  if (gpus.length) {
    for (const g of gpus) {
      const load = g.load_pct == null ? "—" : g.load_pct.toFixed(0) + "%";
      const vram = (g.vram_used != null && g.vram_total != null)
        ? _fmtBytes(g.vram_used) + " / " + _fmtBytes(g.vram_total)
        : (g.vram_total != null ? "— / " + _fmtBytes(g.vram_total) : "—");
      html += `<div class="group gpu"><h4>${g.integrated ? "iGPU" : "GPU"} — ${g.vendor || "?"}</h4>
        <div class="row"><span class="k">name</span><span class="v" style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${g.name}">${g.name}</span></div>
        <div class="row"><span class="k">load</span><span class="v">${load}</span></div>
        <div class="row"><span class="k">VRAM</span><span class="v">${vram}</span></div></div>`;
    }
  } else {
    html += `<div class="group gpu"><h4>GPU</h4><div class="row"><span class="k">adapters</span><span class="v">none detected</span></div></div>`;
  }
  host.innerHTML = html;
}

let _sysPollTimer = null;
function _sysPollTick() {
  fetch("/sys_metrics").then(r => r.json()).then(snap => {
    if (settingsState.current && settingsState.current.hud_enabled) _renderHud(snap);
    if (document.querySelector('.tab-body[data-tab="logs"]').classList.contains("active")) _renderSysmetrics(snap);
  }).catch(() => {});
}
function _sysPollEnsure() {
  const wantHud  = !!(settingsState.current && settingsState.current.hud_enabled);
  const wantLogs = document.querySelector('.tab-body[data-tab="logs"]').classList.contains("active");
  const want = wantHud || wantLogs;
  if (want && !_sysPollTimer) {
    _sysPollTick();
    _sysPollTimer = setInterval(_sysPollTick, 1000);
  } else if (!want && _sysPollTimer) {
    clearInterval(_sysPollTimer);
    _sysPollTimer = null;
  }
}

function _applySettingsToUI(s) {
  document.querySelectorAll('input[name="pytorchMode"]').forEach(r => {
    r.checked = (r.value === s.pytorch_load_mode);
    const wrap = r.closest("label.opt");
    if (wrap) wrap.classList.toggle("checked", r.checked);
  });
  $("idleSecs").value = s.pytorch_idle_unload_secs;
  $("idleTimeoutWrap").style.display = (s.pytorch_load_mode === "on_demand") ? "" : "none";
  $("hudEnable").checked = !!s.hud_enabled;
  $("hudDetailed").checked = !!s.hud_detailed;
  $("hudDetailed").disabled = !s.hud_enabled;
  _applyHudVisibility(!!s.hud_enabled, !!s.hud_detailed);
  const adv = $("analyticsAdvancedEnable");
  if (adv) adv.checked = !!s.analytics_advanced_enabled;
  const ch = s.update_channel || "stable";
  document.querySelectorAll('input[name="updateChannel"]').forEach(r => {
    r.checked = (r.value === ch);
    const wrap = r.closest("label.opt");
    if (wrap) wrap.classList.toggle("checked", r.checked);
  });
  const ar = $("updateAutoReload");
  if (ar) ar.checked = !!s.auto_reload_on_update;
  const aiEn = $("aiEnable");
  if (aiEn) aiEn.checked = !!s.ai_enabled;
  const aiEp = $("aiEndpointUser");
  if (aiEp) aiEp.value = s.ai_endpoint_user || "";
  const aiKy = $("aiApiKeyUser");
  if (aiKy) aiKy.value = s.ai_api_key_user || "";
  if (typeof _AI !== "undefined" && _AI && _AI.applyEnabled) _AI.applyEnabled(!!s.ai_enabled);
}

function _fmtRuntime(secs) {
  secs = Math.max(0, Math.floor(secs || 0));
  const d = Math.floor(secs / 86400);
  const h = Math.floor((secs % 86400) / 3600);
  const m = Math.floor((secs % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}
function _fmtAgo(ts) {
  if (!ts) return "never";
  const sec = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (sec < 60)    return `${sec}s ago`;
  if (sec < 3600)  return `${Math.floor(sec/60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec/3600)}h ago`;
  return `${Math.floor(sec/86400)}d ago`;
}
function _renderHeartbeatStatus(s) {
  if (!s) return;
  const mid = $("heartbeatMachineId"); if (mid) mid.textContent = s.machine_id || ".";
  const ls  = $("heartbeatLastSend");
  if (ls) {
    if (!s.last_send_ts) { ls.textContent = "never"; ls.style.color = "var(--dim)"; ls.title = ""; }
    else {
      const okSend = (s.last_send_status >= 200 && s.last_send_status < 300);
      ls.textContent = `${_fmtAgo(s.last_send_ts)} (${okSend ? "ok" : "fail"})`;
      ls.style.color = okSend ? "var(--data-pos)" : "var(--hot)";
      ls.title = okSend ? "" : (s.last_send_error || "no error reason recorded");
      if (!okSend) ls.style.cursor = "help";
    }
  }
  const rs = $("heartbeatRestarts"); if (rs) rs.textContent = String(s.restarts || 0);
  const rt = $("heartbeatRuntime");  if (rt) rt.textContent = _fmtRuntime(s.total_runtime_secs);
}
function _refreshHeartbeatStatus() {
  fetch("/heartbeat/status").then(r => r.json()).then(_renderHeartbeatStatus).catch(() => {});
}

let updateState = { current: null };

function _renderUpdateStatus(st) {
  if (!st) return;
  updateState.current = st;
  if (st.channel) {
    document.querySelectorAll('input[name="updateChannel"]').forEach(r => {
      r.checked = (r.value === st.channel);
      const wrap = r.closest("label.opt");
      if (wrap) wrap.classList.toggle("checked", r.checked);
    });
  }
  const br = $("updateBranch"); if (br) br.textContent = st.branch || "(none)";
  const hd = $("updateHead");   if (hd) hd.textContent = st.head_short || ".";
  const bh = $("updateBehind");
  if (bh) {
    if (st.behind == null) { bh.textContent = "unknown"; bh.style.color = "var(--dim)"; }
    else if (st.behind === 0) { bh.textContent = "up to date"; bh.style.color = "var(--data-pos)"; }
    else { bh.textContent = `${st.behind} behind`; bh.style.color = "var(--warm)"; }
  }
  const lc = $("updateLastCheck");
  if (lc) {
    const ts = (st.last && st.last.last_check_ts) || 0;
    const msg = (st.last && st.last.last_check_msg) || "";
    lc.textContent = ts ? _fmtAgo(ts) : "never";
    if (msg) {
      lc.textContent += ` · ${msg}`;
      lc.style.color = "var(--warm)";
    } else {
      lc.style.color = "";
    }
  }
  const pull = $("updatePullBtn");
  if (pull) pull.disabled = !(st.update_available && !st.dirty);
  const banner = $("updateBanner");
  if (banner) {
    if (st.update_available) {
      banner.style.display = "";
      banner.textContent = `update available · ${st.behind} new commit${st.behind === 1 ? "" : "s"} on ${st.channel}`;
    } else {
      banner.style.display = "none";
    }
  }
  const dot = document.querySelector('.tab[data-tab="settings"] .notify-dot');
  if (dot) dot.style.display = st.update_available ? "" : "none";
}

function _refreshUpdateStatus() {
  fetch("/app/update_status").then(r => r.json()).then(_renderUpdateStatus).catch(() => {});
}

// Stale-bin banner. /models/bin_health reports any .bin in models/<name>/
// that the current engine refuses to load (e.g. retired format versions left
// over from a merge). Hovering the banner shows which models are affected;
// users re-export from the most recent .pt checkpoint to clear it.
function _renderBinHealth(data) {
  const el = $("staleBinBanner");
  if (!el || !data) return;
  const stale = (data.models || []).filter(m => m.stale);
  if (stale.length === 0) {
    el.style.display = "none";
    return;
  }
  el.style.display = "";
  el.textContent = `${stale.length} stale .bin · re-export`;
  el.title = stale
    .map(m => `${m.name}  v${m.version} (${m.label}): ${m.reason || "stale"}`)
    .join("\n");
}

function _refreshBinHealth() {
  fetch("/models/bin_health").then(r => r.json()).then(_renderBinHealth).catch(() => {});
}

if (typeof window !== "undefined") {
  // Poll once on load and every 30s; cheap header-only reads.
  _refreshBinHealth();
  setInterval(_refreshBinHealth, 30000);
}

function _saveSettings(patch) {
  if (settingsState.saving) return;
  settingsState.saving = true;
  fetch("/settings", { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(patch) })
    .then(r => r.json())
    .then(s => { settingsState.current = s; _applySettingsToUI(s); _sysPollEnsure(); _trUpdateAutoOptimizeVisibility(); })
    .catch(() => {})
    .finally(() => { settingsState.saving = false; });
}

function _lifecycleSetButtonsDisabled(disabled) {
  const rsb = $("restartServerBtn");
  const ksb = $("killServerBtn");
  const srb = $("softReloadBtn");
  if (rsb) rsb.disabled = disabled;
  if (ksb) ksb.disabled = disabled;
  if (srb) srb.disabled = disabled;
}

function _lifecycleWaitForServer(label) {
  let attempts = 0;
  const maxAttempts = 60;
  const tick = () => {
    attempts++;
    fetch("/sys_metrics", { cache: "no-store" })
      .then(r => r.ok ? r.json() : Promise.reject("not ok"))
      .then(() => {
        if (label) { label.textContent = "server back up — reloading…"; label.style.color = "var(--data-pos)"; }
        setTimeout(() => location.reload(), 600);
      })
      .catch(() => {
        if (attempts >= maxAttempts) {
          if (label) { label.textContent = "server did not come back. relaunch manually."; label.style.color = "var(--hot)"; }
          _lifecycleSetButtonsDisabled(false);
          return;
        }
        setTimeout(tick, 1000);
      });
  };
  setTimeout(tick, 1500);
}

function _lifecycleSoftReload() {
  const label = $("lifecycleStatus");
  _lifecycleSetButtonsDisabled(true);
  if (label) { label.textContent = "soft reloading..."; label.style.color = "var(--data-pos)"; }
  fetch("/lifecycle/soft_reload", { method: "POST" })
    .then(r => {
      if (r.status === 404 || r.status === 405) {
        throw new Error("server does not have the soft_reload route. Do one full reload python (or kill+relaunch) to pick up the new endpoint, then soft reload will work.");
      }
      return r.json();
    })
    .then(res => {
      if (!res.ok) {
        if (label) { label.textContent = `failed: ${res.error || "unknown"}`; label.style.color = "var(--hot)"; }
        _lifecycleSetButtonsDisabled(false);
        return;
      }
      if (label) { label.textContent = "waiting for server to come back..."; label.style.color = "var(--data-pos)"; }
      _lifecycleWaitForServer(label);
    })
    .catch(e => {
      if (label) { label.textContent = `request failed: ${e.message || e}`; label.style.color = "var(--hot)"; }
      _lifecycleSetButtonsDisabled(false);
    });
}

function _lifecycleRestart() {
  const label = $("lifecycleStatus");
  _lifecycleSetButtonsDisabled(true);
  if (label) { label.textContent = "reloading python..."; label.style.color = "var(--warm)"; }
  fetch("/lifecycle/restart", { method: "POST" })
    .then(r => r.json())
    .then(res => {
      if (!res.ok) {
        if (label) { label.textContent = `failed: ${res.error || "unknown"}`; label.style.color = "var(--hot)"; }
        _lifecycleSetButtonsDisabled(false);
        return;
      }
      if (label) { label.textContent = "waiting for server to come back..."; label.style.color = "var(--warm)"; }
      _lifecycleWaitForServer(label);
    })
    .catch(e => {
      if (label) { label.textContent = _backendErrMsg(e); label.style.color = "var(--hot)"; }
      _lifecycleSetButtonsDisabled(false);
    });
}

function _lifecycleConfirmKill() {
  return new Promise(resolve => {
    const backdrop = document.createElement("div");
    backdrop.className = "modal-backdrop";
    backdrop.innerHTML = `
      <div class="modal-box" style="max-width:520px;border-color:var(--hot)">
        <div class="modal-header" style="border-bottom-color:var(--hot)">
          <h3 style="color:var(--hot)">Kill everything</h3>
        </div>
        <p style="color:var(--hot);font-size:13px;line-height:1.55;margin:6px 0 12px">
          This stops the training subprocess, the C engine, and the PyTorch backend, then exits the Python process. Uncheckpointed training progress will be lost. The launcher will not come back; you will have to re-run it manually from the terminal.
        </p>
        <p style="color:var(--soft);font-size:12px;margin:0 0 6px">Type <b style="color:var(--hot)">kill</b> to confirm:</p>
        <input id="killConfirmInput" type="text" autocomplete="off" spellcheck="false"
               style="width:100%;background:#0a0c12;border:1px solid var(--hot);color:var(--hot);
                      padding:8px 10px;border-radius:3px;font-family:inherit;font-size:13px;
                      box-sizing:border-box;margin-bottom:12px" />
        <div style="display:flex;gap:10px;justify-content:flex-end">
          <button id="killConfirmCancel" class="action" type="button">cancel</button>
          <button id="killConfirmGo" class="action" type="button" disabled
                  style="border-color:var(--hot);color:var(--hot);opacity:0.5">kill</button>
        </div>
      </div>
    `;
    document.body.appendChild(backdrop);
    const input  = backdrop.querySelector("#killConfirmInput");
    const goBtn  = backdrop.querySelector("#killConfirmGo");
    const cxBtn  = backdrop.querySelector("#killConfirmCancel");
    const finish = ok => { backdrop.remove(); resolve(ok); };
    input.addEventListener("input", () => {
      const armed = input.value.trim().toLowerCase() === "kill";
      goBtn.disabled = !armed;
      goBtn.style.opacity = armed ? "1" : "0.5";
    });
    input.addEventListener("keydown", e => {
      if (e.key === "Enter" && !goBtn.disabled) finish(true);
      else if (e.key === "Escape") finish(false);
    });
    goBtn.addEventListener("click", () => finish(true));
    cxBtn.addEventListener("click", () => finish(false));
    backdrop.addEventListener("click", e => { if (e.target === backdrop) finish(false); });
    setTimeout(() => input.focus(), 30);
  });
}

function _lifecycleKill() {
  _lifecycleConfirmKill().then(ok => {
    if (!ok) return;
    _lifecycleKillExecute();
  });
}

// Branch-switch confirm modal. Triggered when the user picks a different
// channel while a training run is active. Resolves to "cancel", "keep", or
// "full":
//   cancel — no-op, revert the radio selection
//   keep   — switch the branch on disk but leave the running training alone
//   full   — kill training, switch, fully reload the app
function _branchSwitchConfirm(targetChannel) {
  return new Promise(resolve => {
    const backdrop = document.createElement("div");
    backdrop.className = "modal-backdrop";
    backdrop.innerHTML = `
      <div class="modal-box" style="max-width:580px;border-color:var(--warm)">
        <div class="modal-header" style="border-bottom-color:var(--warm)">
          <h3 style="color:var(--warm)">Switch channel to ${targetChannel}?</h3>
        </div>
        <p style="color:var(--warm);font-size:13px;line-height:1.55;margin:6px 0 10px">
          A training run is active. Switching channels replaces the code on disk with whatever is on the target branch. Two options:
        </p>
        <ul style="font-size:12px;line-height:1.5;color:var(--soft);margin:0 0 14px;padding-left:18px">
          <li><b style="color:var(--hot)">FULLY reset</b> — kills the training subprocess, the C engine, and the PyTorch backend, switches the branch, then reloads the entire app. Uncheckpointed training progress is lost; saved checkpoints on disk survive.</li>
          <li><b style="color:var(--warm)">not kill training</b> — switches the branch on disk but leaves the running training process alone. Training keeps using the old code in memory; the dashboard sees the new branch on next reload. Risky if the new branch changes a checkpoint format your run depends on.</li>
        </ul>
        <p style="color:var(--soft);font-size:12px;margin:0 0 6px">Type <b style="color:var(--warm)">switch</b> to arm the buttons:</p>
        <input id="switchConfirmInput" type="text" autocomplete="off" spellcheck="false"
               style="width:100%;background:#0a0c12;border:1px solid var(--warm);color:var(--warm);
                      padding:8px 10px;border-radius:3px;font-family:inherit;font-size:13px;
                      box-sizing:border-box;margin-bottom:12px" />
        <div style="display:flex;gap:10px;justify-content:flex-end;flex-wrap:wrap">
          <button id="switchConfirmCancel" class="action" type="button">cancel</button>
          <button id="switchConfirmKeep" class="action" type="button" disabled
                  style="border-color:var(--warm);color:var(--warm);opacity:0.5">not kill training</button>
          <button id="switchConfirmFull" class="action" type="button" disabled
                  style="border-color:var(--hot);color:var(--hot);opacity:0.5">FULLY reset</button>
        </div>
      </div>
    `;
    document.body.appendChild(backdrop);
    const input   = backdrop.querySelector("#switchConfirmInput");
    const keepBtn = backdrop.querySelector("#switchConfirmKeep");
    const fullBtn = backdrop.querySelector("#switchConfirmFull");
    const cxBtn   = backdrop.querySelector("#switchConfirmCancel");
    const finish  = result => { backdrop.remove(); resolve(result); };
    input.addEventListener("input", () => {
      const armed = input.value.trim().toLowerCase() === "switch";
      keepBtn.disabled = !armed;
      fullBtn.disabled = !armed;
      keepBtn.style.opacity = armed ? "1" : "0.5";
      fullBtn.style.opacity = armed ? "1" : "0.5";
    });
    input.addEventListener("keydown", e => { if (e.key === "Escape") finish("cancel"); });
    keepBtn.addEventListener("click", () => finish("keep"));
    fullBtn.addEventListener("click", () => finish("full"));
    cxBtn.addEventListener("click",   () => finish("cancel"));
    backdrop.addEventListener("click", e => { if (e.target === backdrop) finish("cancel"); });
    setTimeout(() => input.focus(), 30);
  });
}

// Final "are you sure" gate before the FULLY reset path actually fires.
function _branchSwitchAreYouSure() {
  return new Promise(resolve => {
    const backdrop = document.createElement("div");
    backdrop.className = "modal-backdrop";
    backdrop.innerHTML = `
      <div class="modal-box" style="max-width:480px;border-color:var(--hot)">
        <div class="modal-header" style="border-bottom-color:var(--hot)">
          <h3 style="color:var(--hot)">Are you sure?</h3>
        </div>
        <p style="color:var(--hot);font-size:13px;line-height:1.55;margin:6px 0 12px">
          This kills the training process, switches the branch, and reloads the entire app. Uncheckpointed progress is lost. Saved checkpoints survive.
        </p>
        <div style="display:flex;gap:10px;justify-content:flex-end">
          <button id="confirmAreYouSureCancel" class="action" type="button">cancel</button>
          <button id="confirmAreYouSureGo" class="action" type="button"
                  style="border-color:var(--hot);color:var(--hot)">yes, fully reset</button>
        </div>
      </div>
    `;
    document.body.appendChild(backdrop);
    const goBtn = backdrop.querySelector("#confirmAreYouSureGo");
    const cxBtn = backdrop.querySelector("#confirmAreYouSureCancel");
    const finish = ok => { backdrop.remove(); resolve(ok); };
    goBtn.addEventListener("click", () => finish(true));
    cxBtn.addEventListener("click", () => finish(false));
    backdrop.addEventListener("click", e => { if (e.target === backdrop) finish(false); });
  });
}

function _lifecycleKillExecute() {
  const label = $("lifecycleStatus");
  _lifecycleSetButtonsDisabled(true);
  if (label) { label.textContent = "killing..."; label.style.color = "var(--hot)"; }
  fetch("/lifecycle/kill", { method: "POST" })
    .then(r => r.json())
    .then(res => {
      if (!res.ok) {
        if (label) { label.textContent = `failed: ${res.error || "unknown"}`; label.style.color = "var(--hot)"; }
        _lifecycleSetButtonsDisabled(false);
        return;
      }
      setTimeout(() => {
        if (label) { label.textContent = "server killed. relaunch with python run.py."; label.style.color = "var(--hot)"; }
      }, 800);
    })
    .catch(() => {
      if (label) { label.textContent = "server killed. relaunch with python run.py."; label.style.color = "var(--hot)"; }
    });
}

// Render the "remote behind" indicator + enable the update button only when
// the local branch is behind origin. Mirrors the main Veritate update row.
function _renderRepoBehind(behindEl, updateBtn, s) {
  if (!behindEl) return;
  if (!s || !s.is_repo) {
    behindEl.textContent = "unknown";
    behindEl.style.color = "var(--dim)";
    if (updateBtn) updateBtn.disabled = true;
    return;
  }
  if (s.behind == null) {
    behindEl.textContent = "unknown (run check)";
    behindEl.style.color = "var(--dim)";
    if (updateBtn) updateBtn.disabled = true;
    return;
  }
  if (s.behind === 0) {
    behindEl.textContent = "up to date";
    behindEl.style.color = "var(--data-pos)";
    if (updateBtn) updateBtn.disabled = true;
    return;
  }
  behindEl.textContent = `${s.behind} new commit${s.behind === 1 ? "" : "s"} on ${s.branch || "origin"}`;
  behindEl.style.color = "var(--warm)";
  if (updateBtn) updateBtn.disabled = false;
}


function _pluginsApplyStatus(s) {
  const remote   = $("pluginsRemote");
  const head     = $("pluginsHead");
  const branch   = $("pluginsBranch");
  const ahead    = $("pluginsAheadBehind");
  const aheadL   = $("pluginsAheadBehindLine");
  const behindEl = $("pluginsBehind");
  const upBtn    = $("pluginsUpdateBtn");
  if (!remote) return;
  if (!s.exists) {
    remote.textContent = s.default_remote_url || "—";
    head.textContent = "(plugins/ does not exist — update to clone)";
    branch.textContent = "";
    aheadL.style.display = "none";
    _renderRepoBehind(behindEl, upBtn, null);
    if (upBtn) upBtn.disabled = false;  // allow update to clone
    return;
  }
  if (!s.is_repo) {
    remote.textContent = s.default_remote_url || "—";
    head.textContent = "(plugins/ exists but is not a git repo)";
    branch.textContent = "";
    aheadL.style.display = "none";
    _renderRepoBehind(behindEl, upBtn, null);
    return;
  }
  remote.textContent = s.remote_url || "(no origin remote)";
  head.textContent = s.head_short ? s.head_short : "(no commits yet)";
  branch.textContent = s.branch ? `· ${s.branch}` : "";
  if (s.ahead != null && s.behind != null && (s.ahead || s.behind)) {
    aheadL.style.display = "";
    ahead.textContent = `${s.ahead} ahead, ${s.behind} behind origin/${s.branch || "?"}`;
  } else {
    aheadL.style.display = "none";
  }
  _renderRepoBehind(behindEl, upBtn, s);
  const last = s.last || {};
  const lab = $("pluginsSyncStatus");
  if (lab && last.action) {
    if (last.ok) {
      lab.textContent = `last ${last.action}: ok`;
      lab.style.color = "var(--data-pos)";
    } else if (last.ok === false) {
      lab.textContent = `last ${last.action} failed: ${last.message || "see logs"}`;
      lab.style.color = "var(--hot)";
    }
  }
}

function _pluginsRefreshStatus() {
  fetch("/plugins/git/status").then(r => r.json()).then(_pluginsApplyStatus).catch(() => {});
}

function _pluginsCheckTrigger() {
  const btn = $("pluginsCheckBtn");
  const lab = $("pluginsSyncStatus");
  if (btn) btn.disabled = true;
  if (lab) { lab.textContent = "checking…"; lab.style.color = "var(--warm)"; }
  fetch("/plugins/git/check", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
    .then(r => r.json())
    .then(res => {
      if (res.status) _pluginsApplyStatus(res.status);
      if (lab) {
        if (res.ok) { lab.textContent = "checked"; lab.style.color = "var(--data-pos)"; }
        else { lab.textContent = `check failed: ${res.error || "unknown error"}`; lab.style.color = "var(--hot)"; }
      }
      if (!res.status) _pluginsRefreshStatus();
    })
    .catch(e => { if (lab) { lab.textContent = _backendErrMsg(e); lab.style.color = "var(--hot)"; } })
    .finally(() => { if (btn) btn.disabled = false; });
}

function _pluginsUpdateTrigger() {
  const btn = $("pluginsUpdateBtn");
  const lab = $("pluginsSyncStatus");
  if (btn) btn.disabled = true;
  if (lab) { lab.textContent = "updating…"; lab.style.color = "var(--warm)"; }
  fetch("/plugins/git/sync", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
    .then(r => r.json())
    .then(res => {
      if (res.ok) {
        if (lab) { lab.textContent = `${res.action || "update"}: ok`; lab.style.color = "var(--data-pos)"; }
        if (res.status) _pluginsApplyStatus(res.status);
        else _pluginsRefreshStatus();
      } else {
        if (lab) { lab.textContent = `failed: ${res.error || "unknown error"}`; lab.style.color = "var(--hot)"; }
        _pluginsRefreshStatus();
      }
    })
    .catch(e => { if (lab) { lab.textContent = _backendErrMsg(e); lab.style.color = "var(--hot)"; } })
    .finally(() => { if (btn) btn.disabled = false; });
}

function _modelsApplyStatus(s) {
  const remote   = $("modelsRemote");
  const head     = $("modelsHead");
  const branch   = $("modelsBranch");
  const ahead    = $("modelsAheadBehind");
  const aheadL   = $("modelsAheadBehindLine");
  const behindEl = $("modelsBehind");
  const upBtn    = $("modelsUpdateBtn");
  if (!remote) return;
  if (!s.exists) {
    remote.textContent = s.default_remote_url || ".";
    head.textContent = "(models/ does not exist. update to clone)";
    branch.textContent = "";
    aheadL.style.display = "none";
    _renderRepoBehind(behindEl, upBtn, null);
    if (upBtn) upBtn.disabled = false;  // allow update to clone
    return;
  }
  if (!s.is_repo) {
    remote.textContent = s.default_remote_url || ".";
    head.textContent = "(models/ exists but is not a git repo)";
    branch.textContent = "";
    aheadL.style.display = "none";
    _renderRepoBehind(behindEl, upBtn, null);
    return;
  }
  remote.textContent = s.remote_url || "(no origin remote)";
  head.textContent = s.head_short ? s.head_short : "(no commits yet)";
  branch.textContent = s.branch ? `· ${s.branch}` : "";
  if (s.ahead != null && s.behind != null && (s.ahead || s.behind)) {
    aheadL.style.display = "";
    ahead.textContent = `${s.ahead} ahead, ${s.behind} behind origin/${s.branch || "?"}`;
  } else {
    aheadL.style.display = "none";
  }
  _renderRepoBehind(behindEl, upBtn, s);
  const last = s.last || {};
  const lab = $("modelsSyncStatus");
  if (lab && last.action) {
    if (last.ok) {
      lab.textContent = `last ${last.action}: ok`;
      lab.style.color = "var(--data-pos)";
    } else if (last.ok === false) {
      lab.textContent = `last ${last.action} failed: ${last.message || "see logs"}`;
      lab.style.color = "var(--hot)";
    }
  }
}

function _modelsRefreshStatus() {
  fetch("/models/git/status").then(r => r.json()).then(_modelsApplyStatus).catch(() => {});
}

function _modelsCheckTrigger() {
  const btn = $("modelsCheckBtn");
  const lab = $("modelsSyncStatus");
  if (btn) btn.disabled = true;
  if (lab) { lab.textContent = "checking…"; lab.style.color = "var(--warm)"; }
  fetch("/models/git/check", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
    .then(r => r.json())
    .then(res => {
      if (res.status) _modelsApplyStatus(res.status);
      if (lab) {
        if (res.ok) { lab.textContent = "checked"; lab.style.color = "var(--data-pos)"; }
        else { lab.textContent = `check failed: ${res.error || "unknown error"}`; lab.style.color = "var(--hot)"; }
      }
      if (!res.status) _modelsRefreshStatus();
    })
    .catch(e => { if (lab) { lab.textContent = _backendErrMsg(e); lab.style.color = "var(--hot)"; } })
    .finally(() => { if (btn) btn.disabled = false; });
}

function _modelsUpdateTrigger() {
  const btn = $("modelsUpdateBtn");
  const lab = $("modelsSyncStatus");
  if (btn) btn.disabled = true;
  if (lab) { lab.textContent = "updating…"; lab.style.color = "var(--warm)"; }
  fetch("/models/git/sync", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })
    .then(r => r.json())
    .then(res => {
      if (res.ok) {
        if (lab) { lab.textContent = `${res.action || "update"}: ok`; lab.style.color = "var(--data-pos)"; }
        if (res.status) _modelsApplyStatus(res.status);
        else _modelsRefreshStatus();
      } else {
        if (lab) { lab.textContent = `failed: ${res.error || "unknown error"}`; lab.style.color = "var(--hot)"; }
        _modelsRefreshStatus();
      }
    })
    .catch(e => { if (lab) { lab.textContent = _backendErrMsg(e); lab.style.color = "var(--hot)"; } })
    .finally(() => { if (btn) btn.disabled = false; });
}

function _pollBuildStatus() {
  fetch("/engine/status").then(r => r.json()).then(s => {
    const el = $("buildStatusLine");
    if (!el) return;
    const status = s.status || "idle";
    el.textContent = status === "building" ? "building…" : status;
    el.style.color = status === "failed"   ? "var(--hot)" :
                     status === "ok"       ? "var(--data-pos)" :
                     status === "building" ? "var(--warm)" : "var(--dim)";
    const btn = $("settingsBuildBtn");
    if (btn) btn.disabled = (status === "building");
  }).catch(() => {});
}

function _renderSysSpecs(s) {
  const view = $("sysSpecsView");
  if (!view) return;
  if (!s || s.detected === false || !s.platform) {
    view.textContent = "not detected yet";
    view.style.color = "var(--dim)";
    return;
  }
  const fmtBytes = (b) => {
    if (!b) return "?";
    const gb = b / (1024 ** 3);
    return gb >= 1 ? gb.toFixed(1) + " GB" : (b / (1024 ** 2)).toFixed(0) + " MB";
  };
  const p = s.platform || {}, cpu = s.cpu || {}, mem = s.memory || {};
  const gpuLines = (s.gpus || []).map(g => {
    const tag = g.integrated ? "integrated" : "discrete";
    const vram = g.vram_total ? " &middot; " + fmtBytes(g.vram_total) + " VRAM" : "";
    return `<div>${escapeHtml(g.name || g.vendor || "GPU")} <span style="color:var(--dim)">(${tag}${vram})</span></div>`;
  }).join("") || `<div style="color:var(--dim)">no GPU detected</div>`;
  view.innerHTML = `
    <div>${escapeHtml(p.system || "?")} ${escapeHtml(p.release || "")} (${escapeHtml(p.machine || "?")}) &middot; Python ${escapeHtml(p.python || "?")}</div>
    <div>${cpu.count_logical || "?"} logical cores${cpu.count_physical ? " (" + cpu.count_physical + " physical)" : ""} &middot; ${fmtBytes(mem.total_bytes)} RAM</div>
    ${gpuLines}
    <div style="color:var(--dim);margin-top:4px">captured ${new Date((s.captured_at || 0) * 1000).toLocaleString()}</div>
  `;
  view.style.color = "var(--text)";
}

async function showConsentModal({ allowDecline }) {
  const body = `
    <div style="display:grid;gap:10px">
      <div style="padding:10px 14px;background:#0f1218;border-radius:3px">
        <div style="font-size:13px;font-weight:600;color:var(--accent);margin-bottom:4px">
          Heartbeat <span style="font-size:10.5px;color:var(--dim);font-weight:400;margin-left:6px">required &middot; every 6h</span>
        </div>
        <div style="font-size:11.5px;color:var(--text);line-height:1.55">
          Hashed machine id, OS, uptime, restart and error counts, model count. Lets us see Veritate is alive in the wild.
        </div>
        <div style="font-size:11px;color:var(--dim);line-height:1.5;margin-top:4px">
          No prompts, checkpoints, or source.
        </div>
      </div>
      <div style="padding:10px 14px;background:#0f1218;border-radius:3px">
        <div style="font-size:13px;font-weight:600;color:var(--accent);margin-bottom:4px">
          Hardware &amp; training <span style="font-size:10.5px;color:var(--dim);font-weight:400;margin-left:6px">optional</span>
        </div>
        <div style="font-size:11.5px;color:var(--text);line-height:1.55">
          Once per machine: CPU, RAM, GPU specs &mdash; tells us what to support next. Per training run: model name, size, arch, precision, batch size, total steps &mdash; tells us what the community trains.
        </div>
        <div style="font-size:11px;color:var(--warm);line-height:1.5;margin-top:4px">
          The platform sends your hardware specs once and it's never repeated. No weights, datasets, or training metrics are sent. This helps us support more platforms.
        </div>
      </div>
      <div style="font-size:11px;color:var(--dim);text-align:center;margin-top:2px">change any time in Settings &rsaquo; Advanced telemetry</div>
    </div>
  `;
  const buttons = [
    { label: "Heartbeat only", value: "decline" },
    { label: "Share hardware", value: "accept", primary: true },
  ];
  const choice = await showModal({ title: "Data Consent", body, buttons });
  if (choice == null && !allowDecline) return;
  const advanced = (choice === "accept");
  _saveSettings({ analytics_advanced_enabled: advanced, consent_modal_seen: true });
}

async function showBuildNoticesIfAny() {
  let notices = [];
  try {
    const r = await fetch("/settings/notices");
    const j = await r.json();
    notices = Array.isArray(j.notices) ? j.notices : [];
  } catch (_) { return; }
  if (!notices.length) return;
  const items = notices.map(n =>
    `<p style="margin:0 0 10px">${escapeHtml(n.message)}</p>`
  ).join("");
  const body = `<div style="font-size:13px;line-height:1.55">${items}</div>`;
  const maxBuild = notices.reduce((m, n) => Math.max(m, n.build || 0), 0);
  const title = notices.length === 1
    ? `Build ${notices[0].build} notice`
    : `Build notices (through build ${maxBuild})`;
  await showModal({
    title,
    body,
    buttons: [{ label: "Got it", value: "ack", primary: true }],
    nonDismissable: true,
    accent: "var(--accent)",
    align: "top",
  });
  _saveSettings({ last_acknowledged_build: maxBuild });
}

document.addEventListener("DOMContentLoaded", () => {
  fetch("/settings").then(r => r.json()).then(s => {
    settingsState.current = s;
    settingsState.loaded = true;
    _applySettingsToUI(s);
    _sysPollEnsure();
    _trUpdateAutoOptimizeVisibility();
    if (!s.consent_modal_seen) showConsentModal({ allowDecline: false });
    else showBuildNoticesIfAny();
  }).catch(() => {});

  const autoBtn = $("trainAutoOptimizeBtn");
  if (autoBtn) autoBtn.addEventListener("click", _trAutoOptimize);

  document.querySelectorAll('input[name="pytorchMode"]').forEach(r => {
    r.addEventListener("change", () => {
      if (!r.checked) return;
      document.querySelectorAll('input[name="pytorchMode"]').forEach(other => {
        const w = other.closest("label.opt");
        if (w) w.classList.toggle("checked", other === r);
      });
      $("idleTimeoutWrap").style.display = (r.value === "on_demand") ? "" : "none";
      _saveSettings({ pytorch_load_mode: r.value });
    });
  });
  const idle = $("idleSecs");
  if (idle) idle.addEventListener("change", () => {
    const v = Math.max(60, parseInt(idle.value, 10) || 600);
    _saveSettings({ pytorch_idle_unload_secs: v });
  });
  const hud = $("hudEnable");
  if (hud) hud.addEventListener("change", () => {
    _saveSettings({ hud_enabled: hud.checked });
  });
  const hudDet = $("hudDetailed");
  if (hudDet) hudDet.addEventListener("change", () => {
    _saveSettings({ hud_detailed: hudDet.checked });
  });
  const adv = $("analyticsAdvancedEnable");
  if (adv) adv.addEventListener("change", () => {
    _saveSettings({ analytics_advanced_enabled: adv.checked });
  });
  const reviewBtn = $("reviewConsentBtn");
  if (reviewBtn) reviewBtn.addEventListener("click", () => { showConsentModal({ allowDecline: true }); });
  const detectBtn = $("sysDetectBtn");
  if (detectBtn) detectBtn.addEventListener("click", () => {
    detectBtn.disabled = true;
    const prev = detectBtn.textContent;
    detectBtn.textContent = "detecting…";
    fetch("/sys/detect", { method: "POST" })
      .then(r => r.json())
      .then(s => { _sysSpecsCache = s && s.platform ? s : null; _renderSysSpecs(s); _trUpdateVramEstimate(); _trUpdateAutoOptimizeVisibility(); })
      .catch(() => {})
      .finally(() => { detectBtn.disabled = false; detectBtn.textContent = prev; });
  });
  fetch("/sys/specs").then(r => r.json()).then(s => { _sysSpecsCache = s && s.platform ? s : null; _renderSysSpecs(s); _trUpdateVramEstimate(); _trUpdateAutoOptimizeVisibility(); }).catch(() => {});
  const hbBtn = $("heartbeatSendBtn");
  if (hbBtn) hbBtn.addEventListener("click", () => {
    const lab = $("heartbeatSendStatus");
    if (lab) { lab.textContent = "sending..."; lab.style.color = "var(--warm)"; }
    hbBtn.disabled = true;
    fetch("/heartbeat/send", { method: "POST" })
      .then(r => r.json())
      .then(s => {
        _renderHeartbeatStatus(s);
        if (lab) {
          lab.textContent = s.ok ? "sent" : "failed";
          lab.style.color = s.ok ? "var(--data-pos)" : "var(--hot)";
        }
      })
      .catch(e => { if (lab) { lab.textContent = _backendErrMsg(e); lab.style.color = "var(--hot)"; } })
      .finally(() => { hbBtn.disabled = false; });
  });
  _refreshHeartbeatStatus();
  setInterval(_refreshHeartbeatStatus, 30000);

  document.querySelectorAll('input[name="updateChannel"]').forEach(r => {
    r.addEventListener("change", async () => {
      if (!r.checked) return;
      // Mark the visual selection optimistically; we may revert on cancel.
      document.querySelectorAll('input[name="updateChannel"]').forEach(other => {
        const w = other.closest("label.opt");
        if (w) w.classList.toggle("checked", other === r);
      });
      const lab = $("updateActionStatus");

      // Detect active training before firing the switch.
      let trainingActive = false;
      try {
        const pl = await fetch("/plugins").then(x => x.json());
        trainingActive = !!(pl && pl.running && pl.running.status === "running");
      } catch (_) { /* if /plugins is down, fall through and let the switch try */ }

      const _doSwitch = (postAction) => {
        if (lab) { lab.textContent = postAction === "full" ? "switching + reloading…" : "switching channel…"; lab.style.color = "var(--warm)"; }
        return fetch("/app/update_channel", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ channel: r.value }),
        }).then(x => x.json()).then(async res => {
          if (res && res.ok) {
            if (res.status) _renderUpdateStatus(res.status);
            if (postAction === "full") {
              // Kill + restart the entire app. The /lifecycle/restart endpoint
              // returns once the relaunch is queued; the page will lose its
              // socket and reload itself.
              if (lab) { lab.textContent = "switched. fully resetting…"; lab.style.color = "var(--warm)"; }
              try { await fetch("/lifecycle/restart", { method: "POST" }); } catch (_) {}
              setTimeout(() => location.reload(), 1500);
              return;
            }
            if (lab) { lab.textContent = "channel switched"; lab.style.color = "var(--data-pos)"; }
            _refreshUpdateStatus();
          } else {
            if (lab) { lab.textContent = `failed: ${res && res.error || "unknown"}`; lab.style.color = "var(--hot)"; }
            _refreshUpdateStatus();
          }
        }).catch(e => { if (lab) { lab.textContent = _backendErrMsg(e); lab.style.color = "var(--hot)"; } });
      };

      if (!trainingActive) {
        _doSwitch("normal");
        return;
      }

      // Training is active. Modal flow.
      const decision = await _branchSwitchConfirm(r.value);
      if (decision === "cancel") {
        if (lab) { lab.textContent = "switch cancelled"; lab.style.color = "var(--dim)"; }
        _refreshUpdateStatus();   // resync radio to actual server-side channel
        return;
      }
      if (decision === "keep") {
        _doSwitch("keep");
        return;
      }
      if (decision === "full") {
        const yes = await _branchSwitchAreYouSure();
        if (!yes) {
          if (lab) { lab.textContent = "switch cancelled"; lab.style.color = "var(--dim)"; }
          _refreshUpdateStatus();
          return;
        }
        _doSwitch("full");
      }
    });
  });
  const ar = $("updateAutoReload");
  if (ar) ar.addEventListener("change", () => {
    _saveSettings({ auto_reload_on_update: ar.checked });
  });
  const aiEn = $("aiEnable");
  if (aiEn) aiEn.addEventListener("change", () => {
    _saveSettings({ ai_enabled: aiEn.checked });
    if (typeof _AI !== "undefined" && _AI.applyEnabled) _AI.applyEnabled(aiEn.checked);
  });
  const aiSave = $("aiSaveBtn");
  if (aiSave) aiSave.addEventListener("click", () => {
    const lab = $("aiSaveStatus");
    if (lab) { lab.textContent = "saving..."; lab.style.color = "var(--warm)"; }
    _saveSettings({
      ai_endpoint_user: ($("aiEndpointUser").value || "").trim(),
      ai_api_key_user:  ($("aiApiKeyUser").value  || "").trim(),
    });
    if (lab) { lab.textContent = "saved"; lab.style.color = "var(--data-pos)"; }
  });
  const askRT = $("askAiRecentTrain");
  if (askRT) askRT.addEventListener("click", () => {
    const sel = $("runPicker");
    const name = sel && sel.value ? sel.value : "";
    if (!name) { window.ai_ask("recent_train", {}, "explain recent training"); return; }
    window.ai_ask("recent_train", { model: name }, "explain recent training");
  });
  const askLC = $("askAiLossCurve");
  if (askLC) askLC.addEventListener("click", () => {
    const cached = window._aiLossCurve || null;
    const sel = $("runPicker");
    const name = (cached && cached.model) || (sel && sel.value) || "";
    const verdict = cached && cached.verdict ? cached.verdict : { state: "warming" };
    window.ai_ask("loss_curve", {
      model: name,
      state: verdict.state || "warming",
      slope_pct: typeof verdict.avgPct === "number" ? verdict.avgPct : null,
      train: cached && cached.train ? cached.train : [],
      val:   cached && cached.val   ? cached.val   : [],
    }, "explain loss curve");
  });
  const askTH = $("askAiTrainHealth");
  if (askTH) askTH.addEventListener("click", () => {
    const cached = window._aiTrainHealth || null;
    const sel = $("runPicker");
    const name = (cached && cached.model) || (sel && sel.value) || "";
    const verdict = cached && cached.verdict ? cached.verdict : { state: "warming" };
    const recent  = cached && cached.series ? cached.series : [];
    window.ai_ask("train_health", {
      model: name,
      state: verdict.state || "warming",
      slope_pct: typeof verdict.avgPct === "number" ? verdict.avgPct : null,
      recent: recent,
      latest: cached && cached.latest ? cached.latest : null,
    }, "explain training health");
  });
  const ucb = $("updateCheckBtn");
  if (ucb) ucb.addEventListener("click", () => {
    const lab = $("updateActionStatus");
    if (lab) { lab.textContent = "checking..."; lab.style.color = "var(--warm)"; }
    ucb.disabled = true;
    fetch("/app/update_check", { method: "POST" })
      .then(r => r.json())
      .then(res => {
        if (res && res.status) _renderUpdateStatus(res.status);
        if (lab) {
          if (res && res.ok) { lab.textContent = "checked"; lab.style.color = "var(--data-pos)"; }
          else { lab.textContent = `failed: ${res && res.error || "unknown"}`; lab.style.color = "var(--hot)"; }
        }
      })
      .catch(e => { if (lab) { lab.textContent = _backendErrMsg(e); lab.style.color = "var(--hot)"; } })
      .finally(() => { ucb.disabled = false; });
  });
  const upb = $("updatePullBtn");
  if (upb) upb.addEventListener("click", () => {
    const lab = $("updateActionStatus");
    const willReload = !!(updateState.current && updateState.current.last && false) ||
                       !!($("updateAutoReload") && $("updateAutoReload").checked);
    if (lab) { lab.textContent = willReload ? "pulling + reloading..." : "pulling..."; lab.style.color = "var(--warm)"; }
    upb.disabled = true;
    fetch("/app/update_pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reload: willReload }),
    })
      .then(r => r.json())
      .then(res => {
        if (res && res.status) _renderUpdateStatus(res.status);
        if (lab) {
          if (res && res.ok) {
            lab.textContent = willReload ? "pulled, reloading..." : "pulled";
            lab.style.color = "var(--data-pos)";
            if (willReload) _lifecycleWaitForServer(lab);
          } else {
            lab.textContent = `failed: ${res && res.error || "unknown"}`;
            lab.style.color = "var(--hot)";
          }
        }
      })
      .catch(e => { if (lab) { lab.textContent = _backendErrMsg(e); lab.style.color = "var(--hot)"; } })
      .finally(() => { upb.disabled = false; _refreshUpdateStatus(); });
  });
  const ub = $("updateBanner");
  if (ub) ub.addEventListener("click", () => {
    const tab = document.querySelector('.tab[data-tab="settings"]');
    if (tab) tab.click();
  });
  _refreshUpdateStatus();
  setInterval(_refreshUpdateStatus, 60000);
  const bb = $("settingsBuildBtn");
  if (bb) bb.addEventListener("click", () => {
    fetch("/engine/build", { method: "POST" }).catch(() => {});
    setTimeout(_pollBuildStatus, 200);
  });
  _pollBuildStatus();
  setInterval(_pollBuildStatus, 3000);

  const pcb = $("pluginsCheckBtn");
  if (pcb) pcb.addEventListener("click", _pluginsCheckTrigger);
  const pub = $("pluginsUpdateBtn");
  if (pub) pub.addEventListener("click", _pluginsUpdateTrigger);
  _pluginsRefreshStatus();
  setInterval(_pluginsRefreshStatus, 15000);

  const mcb = $("modelsCheckBtn");
  if (mcb) mcb.addEventListener("click", _modelsCheckTrigger);
  const mub = $("modelsUpdateBtn");
  if (mub) mub.addEventListener("click", _modelsUpdateTrigger);
  _modelsRefreshStatus();
  setInterval(_modelsRefreshStatus, 15000);

  const srb = $("softReloadBtn");
  if (srb) srb.addEventListener("click", _lifecycleSoftReload);
  const rsb = $("restartServerBtn");
  if (rsb) rsb.addEventListener("click", _lifecycleRestart);
  const ksb = $("killServerBtn");
  if (ksb) ksb.addEventListener("click", _lifecycleKill);

  document.querySelectorAll(".tab").forEach(t => {
    t.addEventListener("click", () => setTimeout(_sysPollEnsure, 50));
  });
});

// ---- atlas (v8 interpretability layer) ----
function atlasCurrentModelStep() {
  const picker = $("timelinePicker");
  const name = picker ? picker.value : "";
  let step = 0;
  if (typeof learningState !== "undefined" && learningState.meta && learningState.meta.checkpoints) {
    const c = learningState.meta.checkpoints[learningState.ckptIdx];
    if (c && c.step != null) step = c.step;
  }
  return { name, step };
}

function atlasRenderNeuronTable(elId, neurons) {
  const el = $(elId);
  if (!el) return;
  if (!neurons || !neurons.length) {
    el.innerHTML = `<div style="color:var(--dim);font-size:11px">no results</div>`;
    return;
  }
  let html = `<table class="dla-table"><thead><tr>
    <th style="text-align:left">layer</th><th style="text-align:left">neuron</th><th>score</th>
  </tr></thead><tbody>`;
  for (const r of neurons) {
    const score = (typeof r.score === "number") ? r.score.toFixed(4) : "";
    html += `<tr data-layer="${r.layer}" data-neuron="${r.neuron}">
      <td class="layer-cell">L${r.layer}</td>
      <td class="neuron-cell">#${r.neuron}</td>
      <td>${score}</td>
    </tr>`;
  }
  html += `</tbody></table>`;
  el.innerHTML = html;
  el.querySelectorAll("tbody tr").forEach(tr => {
    tr.addEventListener("click", () => {
      showNeuronModal(parseInt(tr.dataset.layer, 10), parseInt(tr.dataset.neuron, 10));
    });
    tr.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      const L = parseInt(tr.dataset.layer, 10);
      const N = parseInt(tr.dataset.neuron, 10);
      ablateAndRegen(L, N);
    });
  });
}

function atlasFetch(url, statusEl, onOk) {
  if (statusEl) statusEl.textContent = "loading...";
  fetch(url)
    .then(r => r.json())
    .then(d => {
      if (d && d.error) {
        if (statusEl) statusEl.textContent = "error: " + d.error;
        return;
      }
      if (statusEl) statusEl.textContent = "";
      onOk(d);
    })
    .catch(e => { if (statusEl) statusEl.textContent = "fetch failed: " + e; });
}

function atlasGoConcept() {
  const { name, step } = atlasCurrentModelStep();
  if (!name) { $("atlasConceptStatus").textContent = "pick a model in the timeline picker first"; return; }
  atlasConceptArmed = true;
  const sub = ($("atlasConceptInput") || {}).value || "";
  const url = `/atlas/concept?model=${encodeURIComponent(name)}&step=${step}&substring=${encodeURIComponent(sub)}`;
  atlasFetch(url, $("atlasConceptStatus"), d => {
    $("atlasConceptStatus").textContent = `${d.n_matched}/${d.n_frames} frames matched`;
    atlasRenderNeuronTable("atlasConceptResult", d.neurons);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const c = $("atlasConceptGo");          if (c) c.addEventListener("click", atlasGoConcept);
});

// ---- live training stream (v8 tier 4 receiver) ----
// Auto-subscribe to /train_stream when the training tab is active. The brain-scan
// feed and status pill reveal themselves only when frames actually arrive — silent
// when the trainer isn't publishing, no buttons, no extra panel.
// (_trainStreamEvt and _trainStreamCount are hoisted near activateTab.)

function trainStreamStart() {
  if (_trainStreamEvt) return;
  const status = $("trainStreamStatus");
  const feed = $("trainStreamFeed");
  _trainStreamEvt = new EventSource("/train_stream");
  _trainStreamEvt.onmessage = (e) => {
    if (!e.data) return;
    let payload;
    try { payload = JSON.parse(e.data); } catch (_) { return; }
    _trainStreamCount += 1;
    if (status) status.textContent = `brain stream · ${_trainStreamCount} frames`;
    if (feed) {
      if (feed.style.display === "none") feed.style.display = "block";
      const line = document.createElement("div");
      const stepStr = payload.step != null ? `step ${payload.step}` : "(no step)";
      const lossStr = payload.loss != null ? `loss=${payload.loss}` : "";
      line.textContent = `${stepStr} ${lossStr} ${JSON.stringify(payload).slice(0, 240)}`;
      feed.appendChild(line);
      while (feed.childNodes.length > 200) feed.removeChild(feed.firstChild);
      feed.scrollTop = feed.scrollHeight;
    }
  };
  _trainStreamEvt.onerror = () => {
    if (status && _trainStreamCount === 0) status.textContent = "";
  };
}

function trainStreamStop() {
  if (_trainStreamEvt) { _trainStreamEvt.close(); _trainStreamEvt = null; }
  const status = $("trainStreamStatus");
  if (status) status.textContent = "";
}

// ---- wiki ----
const wikiState = {
  loaded:   false,
  loading:  false,
  cats:     [],
  current:  null,
  entries:  {},
  selected: {},
};

async function ensureWikiLoaded() {
  if (wikiState.loaded || wikiState.loading) {
    if (wikiState.loaded) renderWikiSubtabs();
    return;
  }
  wikiState.loading = true;
  try {
    const r = await fetch("/wiki");
    const d = await r.json();
    wikiState.cats = d.categories || [];
    if (wikiState.cats.length && !wikiState.current) {
      wikiState.current = wikiState.cats[0].name;
    }
    wikiState.loaded = true;
    renderWikiSubtabs();
    if (wikiState.current) await loadWikiCategory(wikiState.current);
  } catch (e) {
    $("wikiList").innerHTML = `<div class="wiki-empty">failed to load wiki: ${escapeHtml(String(e))}</div>`;
  } finally {
    wikiState.loading = false;
  }
}

function renderWikiSubtabs() {
  const wrap = $("wikiSubtabs");
  if (!wrap) return;
  if (!wikiState.cats.length) {
    wrap.innerHTML = `<div class="wiki-empty">No categories yet. Add a folder under <code>veritate_mri/wiki/</code>.</div>`;
    return;
  }
  wrap.innerHTML = wikiState.cats.map(c => {
    const active = c.name === wikiState.current ? " active" : "";
    return `<div class="wiki-subtab${active}" data-cat="${escapeHtml(c.name)}">${escapeHtml(prettifyCat(c.name))}<span class="count">${c.n_entries}</span></div>`;
  }).join("");
  wrap.querySelectorAll(".wiki-subtab").forEach(el => {
    el.addEventListener("click", () => {
      const cat = el.dataset.cat;
      if (cat === wikiState.current) return;
      wikiState.current = cat;
      renderWikiSubtabs();
      loadWikiCategory(cat);
    });
  });
}

function prettifyCat(s) {
  return s.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

async function loadWikiCategory(category) {
  const list = $("wikiList");
  list.innerHTML = `<div class="wiki-empty">loading…</div>`;
  $("wikiEntry").innerHTML = `<div class="wiki-empty">Pick an entry on the left.</div>`;
  try {
    let entries = wikiState.entries[category];
    if (!entries) {
      const r = await fetch(`/wiki/${encodeURIComponent(category)}`);
      const d = await r.json();
      entries = d.entries || [];
      wikiState.entries[category] = entries;
    }
    if (!entries.length) {
      list.innerHTML = `<div class="wiki-empty">No entries yet in <code>${escapeHtml(category)}</code>.</div>`;
      return;
    }
    list.innerHTML = entries.map(e => {
      const tags = (e.tags || []).map(t =>
        `<span class="wiki-tag">${escapeHtml(t)}</span>`).join("");
      const summary = e.summary ? `<div class="s">${escapeHtml(e.summary)}</div>` : "";
      const date = e.date ? `<div class="d">${escapeHtml(e.date)}</div>` : "";
      return `<div class="wiki-list-item" data-slug="${escapeHtml(e.slug)}">
        <div class="t">${escapeHtml(e.title || e.slug)}</div>
        ${date}
        ${summary}
        ${tags ? `<div class="tags">${tags}</div>` : ""}
      </div>`;
    }).join("");
    list.querySelectorAll(".wiki-list-item").forEach(el => {
      el.addEventListener("click", () => loadWikiEntry(category, el.dataset.slug));
    });
    const selected = wikiState.selected[category];
    const target = selected || entries[0].slug;
    loadWikiEntry(category, target);
  } catch (e) {
    list.innerHTML = `<div class="wiki-empty">failed: ${escapeHtml(String(e))}</div>`;
  }
}

async function loadWikiEntry(category, slug) {
  wikiState.selected[category] = slug;
  $("wikiList").querySelectorAll(".wiki-list-item").forEach(el => {
    el.classList.toggle("active", el.dataset.slug === slug);
  });
  const view = $("wikiEntry");
  view.innerHTML = `<div class="wiki-empty">loading…</div>`;
  try {
    const r = await fetch(`/wiki/${encodeURIComponent(category)}/${encodeURIComponent(slug)}`);
    if (!r.ok) {
      view.innerHTML = `<div class="wiki-empty">not found.</div>`;
      return;
    }
    const d = await r.json();
    const meta = [];
    if (d.date) meta.push(escapeHtml(d.date));
    if (d.tags && d.tags.length) meta.push(d.tags.map(t => escapeHtml(t)).join(", "));
    view.innerHTML = `
      <div class="wiki-entry-meta">
        <span class="title">${escapeHtml(d.title || slug)}</span>
        ${meta.join(" · ")}
      </div>
      <div class="wiki-body">${d.body_html || ""}</div>`;
    view.scrollTop = 0;
  } catch (e) {
    view.innerHTML = `<div class="wiki-empty">failed: ${escapeHtml(String(e))}</div>`;
  }
}

async function loadVersions() {
  const el = document.getElementById("versionList");
  if (!el) return;
  const LABEL = {
    channel: "Channel",
    build:   "Platform Build",
    engine:  "Veritate Engine",
    mri:     "MRI",
    format:  "Model API",
    plugins: "Plugin Engine",
  };
  const EXPLAIN = {
    channel: {
      title: "Channel",
      body:  "Which fork this build is on. stable is the canonical mainline. experimental is a fork."
    },
    build: {
      title: "Platform Build",
      body:  "Build number for the dashboard, settings layout, JS, and Python glue around the engine. Bumps when the UI changes, when routes are added, or when packaging shifts. It does not imply the inference engine, model format, or plugin contract changed; those have their own versions."
    },
    engine: {
      title: "Veritate Engine",
      body:  "Version of the hand-written C inference engine that runs the byte-level model in INT8. Owns the kernels, the runtime shape, and the per-token decode budget. Bumps when kernels change, when the engine ABI changes, or when a new arch (AVX-512, NEON, etc.) ships. Same engine version across machines means bitwise-identical decode."
    },
    mri: {
      title: "MRI",
      body:  "Version of the live interpretability layer: hooks, dump artifacts (TFRM frames), and the on-disk format for per-step probes. Bumps when hook payloads gain or lose fields, or when frame field indices shift. Trainers, the engine, and the dashboard must agree on this number to render frames correctly."
    },
    format: {
      title: "Model API",
      body:  "Version of the on-disk model contract: the .bin layout the engine consumes, the checkpoint shape PyTorch saves, and the config.json keys readers expect. Bumps when a new field is added or an existing one changes meaning. Older models may still load, but only when the engine knows how to interpret the older revision."
    },
    plugins: {
      title: "Plugin Engine",
      body:  "Version of the plugin contract surfaced by veritate.plugin: manifest schema, lifecycle hooks, and the platform calls a plugin is allowed to make. Bumps when a hook is added, renamed, or removed. Plugins compiled against an older contract version may refuse to load."
    },
  };
  const ORDER = ["channel", "build", "engine", "mri", "format", "plugins"];
  const SEP = '<span style="color:var(--text);font-size:14px;font-weight:300;margin:0 4px">|</span>';
  function channelMeta(raw) {
    const s = (raw || "").toString().toLowerCase();
    if (!s || s === "stable") return { hide: true, display: "", color: null };
    if (s === "development" || s === "dev") return { hide: false, display: "dev", color: "var(--warm)" };
    if (s === "experimental") return { hide: false, display: "experimental", color: "var(--cool)" };
    return { hide: false, display: s, color: "var(--cool)" };
  }
  function chip(k, val) {
    const label = LABEL[k] || k;
    const isChannel = (k === "channel");
    let displayVal = String(val);
    let valStyle = "color:var(--text)";
    if (isChannel) {
      const meta = channelMeta(val);
      if (!meta.hide) {
        displayVal = meta.display;
        valStyle = `color:${meta.color};font-weight:600`;
      }
    }
    return `<a href="#" data-vkey="${escapeHtml(k)}" style="color:var(--soft);text-decoration:none;border-bottom:1px dotted var(--line);cursor:pointer">${escapeHtml(label)} <span style="${valStyle}">${escapeHtml(displayVal)}</span></a>`;
  }
  try {
    const r = await fetch("/versions");
    if (!r.ok) throw new Error(r.status);
    const v = await r.json();
    const meta = channelMeta(v.channel);
    for (const id of ["channelBadge"]) {
      const node = document.getElementById(id);
      if (!node) continue;
      if (!meta.hide) {
        node.textContent = meta.display;
        node.style.color = meta.color;
        node.style.display = "";
      } else {
        node.textContent = "";
        node.style.display = "none";
      }
    }
    const parts = [];
    for (const k of ORDER) {
      if (v[k] === undefined) continue;
      if (k === "channel" && channelMeta(v[k]).hide) continue;
      parts.push(chip(k, v[k]));
    }
    for (const [k, val] of Object.entries(v)) {
      if (ORDER.includes(k)) continue;
      parts.push(chip(k, val));
    }
    el.innerHTML = parts.join(SEP);
    el.querySelectorAll("a[data-vkey]").forEach(a => {
      a.addEventListener("click", (e) => {
        e.preventDefault();
        const k = a.getAttribute("data-vkey");
        const info = EXPLAIN[k] || { title: LABEL[k] || k, body: "No description available." };
        const ver = v[k];
        _showVersionModal(info.title, ver, info.body);
      });
    });
  } catch (e) {
    el.textContent = "versions unavailable";
  }
}

function _showVersionModal(title, version, body) {
  let backdrop = document.getElementById("versionInfoModal");
  if (!backdrop) {
    backdrop = document.createElement("div");
    backdrop.id = "versionInfoModal";
    backdrop.className = "modal-backdrop hidden";
    backdrop.innerHTML = `
      <div class="modal-box" style="max-width:520px">
        <div class="modal-header">
          <h3 id="versionInfoTitle"></h3>
          <button type="button" id="versionInfoClose" style="background:transparent;border:none;color:var(--dim);font-size:18px;cursor:pointer;padding:0 4px">.</button>
        </div>
        <div id="versionInfoBody" style="color:var(--soft);font-size:12px;line-height:1.6"></div>
      </div>`;
    document.body.appendChild(backdrop);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) backdrop.classList.add("hidden"); });
    document.getElementById("versionInfoClose").addEventListener("click", () => backdrop.classList.add("hidden"));
  }
  const titleEl = document.getElementById("versionInfoTitle");
  titleEl.innerHTML = `${escapeHtml(title)} <span style="color:var(--accent);font-weight:500;margin-left:8px">${escapeHtml(String(version))}</span>`;
  document.getElementById("versionInfoBody").textContent = body;
  backdrop.classList.remove("hidden");
}
loadVersions();

/* ============================ ai assist ============================ */
const _AI = (() => {
  const PHRASES = [
    "thinking really hard",
    "counting on fingers",
    "squinting at the numbers",
    "flipping through the manual",
    "checking my notes",
    "yelling at the intern",
    "cracking knuckles",
    "doing the long division",
    "asking the smart one",
    "consulting the napkin",
    "pouring more coffee",
    "rifling through the drawer",
    "rereading the question",
    "sounding it out",
    "chewing on it",
    "looking under the desk",
    "finding my glasses",
    "flipping a coin",
    "tapping the desk",
    "checking the back of the napkin",
    "asking around the office",
    "opening a fresh notebook",
    "running the numbers again",
    "looking it up the long way",
    "consulting the binder",
    "squaring the math",
  ];
  const SLOW_PHRASES = [
    "ok this one's actually hard hold on",
    "hm",
    "wait let me reread the question",
    "one sec im cooking",
    "ok ok ok give me a minute",
    "ok new plan",
    "i swear i know this",
    "ok yes. wait no",
    "give me one more sec sorry",
    "the answer is in here i can feel it",
    "this is embarrassing how long this is taking",
    "pls dont close the tab",
    "why is this question harder than it looked",
    "i was right the first time. probably. nope go back",
    "one more pass and then im sending it",
    "promise im still here",
    "don't worry im not asleep",
    'would it help if i said "almost done"?',
    "ok genuinely not far now",
    "unless. nope yes. ok",
    "alright drafting the real answer",
    "ok ok ok i see it now",
    "ok now i'm just being thorough",
    "polishing it a little, sorry",
    "retyping that whole part",
    "final lap promise",
    "40% there. now 38%. wait",
  ];
  const SLOW_THRESHOLD_MS = 35000;
  const CHAR_BASE_MS  = 55;
  const CHAR_JITTER   = 70;
  const PAUSE_CHANCE  = 0.07;
  const PAUSE_MIN_MS  = 140;
  const PAUSE_MAX_MS  = 360;
  const HOLD_MIN_MS   = 2200;
  const HOLD_MAX_MS   = 3600;

  let backdrop = null;
  let modal = null;
  let bodyEl = null;
  let tagEl = null;
  let typerTimer = null;
  let phraseOrder = [];
  let phraseCursor = 0;
  let inflight = null;
  let lastAnswerText = "";
  let openedAt = 0;
  let slowMode = false;

  function _rand(min, max) { return min + Math.random() * (max - min); }

  function _shuffleCopy(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function _esc(s) { return s.replace(/[<>&]/g, c => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c])); }

  function _ensure() {
    if (backdrop) return;
    backdrop = document.getElementById("aiBackdrop");
    modal    = document.getElementById("aiModal");
    bodyEl   = document.getElementById("aiBody");
    tagEl    = document.getElementById("aiStateTag");
    document.getElementById("aiCloseBtn").addEventListener("click", _close);
    document.getElementById("aiDoneBtn").addEventListener("click", _close);
    document.getElementById("aiSaveTxtBtn").addEventListener("click", _saveTxt);
  }

  function _setState(name, tagText) {
    modal.classList.remove("state-loading", "state-ok", "state-error");
    bodyEl.classList.remove("loading", "error");
    if (name) {
      modal.classList.add("state-" + name);
      if (name === "loading") bodyEl.classList.add("loading");
      if (name === "error")   bodyEl.classList.add("error");
    }
    tagEl.textContent = tagText || "";
  }

  function _stopTyper() {
    if (typerTimer) { clearTimeout(typerTimer); typerTimer = null; }
  }

  function _typeChar(text, idx, after) {
    if (idx > text.length) {
      typerTimer = setTimeout(after, _rand(HOLD_MIN_MS, HOLD_MAX_MS));
      return;
    }
    bodyEl.innerHTML = _esc(text.slice(0, idx)) + '<span class="ai-caret"></span>';
    let delay = CHAR_BASE_MS + Math.random() * CHAR_JITTER;
    if (Math.random() < PAUSE_CHANCE) delay += _rand(PAUSE_MIN_MS, PAUSE_MAX_MS);
    typerTimer = setTimeout(() => _typeChar(text, idx + 1, after), delay);
  }

  function _nextPhrase() {
    const elapsed = Date.now() - openedAt;
    if (!slowMode && elapsed >= SLOW_THRESHOLD_MS) {
      slowMode = true;
      phraseOrder = _shuffleCopy(SLOW_PHRASES);
      phraseCursor = 0;
    }
    if (phraseCursor >= phraseOrder.length) {
      phraseOrder = _shuffleCopy(slowMode ? SLOW_PHRASES : PHRASES);
      phraseCursor = 0;
    }
    const p = phraseOrder[phraseCursor++];
    bodyEl.innerHTML = '<span class="ai-caret"></span>';
    _typeChar(p, 0, _nextPhrase);
  }

  function _open(titleText) {
    _ensure();
    document.getElementById("aiTitle").textContent = titleText || "ask";
    backdrop.classList.remove("hidden");
    document.body.classList.add("ai-modal-open");
    bodyEl.innerHTML = "";
    lastAnswerText = "";
    document.getElementById("aiSaveTxtBtn").disabled = true;
    _setState("loading", "loading");
    openedAt = Date.now();
    slowMode = false;
    phraseOrder = _shuffleCopy(PHRASES);
    phraseCursor = 0;
    _nextPhrase();
  }

  function _close() {
    _stopTyper();
    if (inflight) { try { inflight.abort(); } catch (e) {} inflight = null; }
    if (backdrop) backdrop.classList.add("hidden");
    document.body.classList.remove("ai-modal-open");
  }

  function _renderAnswer(text) {
    _stopTyper();
    lastAnswerText = text || "";
    bodyEl.textContent = lastAnswerText;
    _setState("ok", "answer");
    document.getElementById("aiSaveTxtBtn").disabled = !lastAnswerText;
  }

  function _renderError(msg) {
    _stopTyper();
    lastAnswerText = "";
    bodyEl.textContent = msg;
    _setState("error", "error");
    document.getElementById("aiSaveTxtBtn").disabled = true;
  }

  function _saveTxt() {
    if (!lastAnswerText) return;
    const ts = new Date().toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, 19);
    const blob = new Blob([lastAnswerText + "\n"], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ai_answer_${ts}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function ask(kind, payload, titleText) {
    _open(titleText);
    const ctl = ("AbortController" in window) ? new AbortController() : null;
    inflight = ctl;
    try {
      const r = await fetch("/ai/ask", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ kind, payload }),
        signal:  ctl ? ctl.signal : undefined,
      });
      const data = await r.json();
      if (data && data.ok) _renderAnswer(data.answer || "");
      else                 _renderError("error: " + ((data && data.error) || "unknown"));
    } catch (e) {
      if (e && e.name === "AbortError") return;
      _renderError("error: " + (e && e.message || e));
    } finally {
      inflight = null;
    }
  }

  function applyEnabled(enabled) {
    document.body.classList.toggle("ai-disabled", !enabled);
  }

  return { ask, applyEnabled };
})();
window.ai_ask = _AI.ask;

