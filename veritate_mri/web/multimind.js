/* Developed by Carpathian, LLC. Distribution Not Authorized. */
/* veritate_mri/web/multimind.js */
/* Multi-Mind dashboard frontend. Consumes /multimind/* endpoints, renders
   status chips, region firing strip, affect trace, gate bars, regions table,
   and conversation ledger. */

const $ = id => document.getElementById(id);

// ---- constants ----
const POLL_STATUS_MS = 5000;
const POLL_REGIONS_MS = 15000;
const POLL_CONVS_MS = 30000;
const EVENT_BUFFER_CAP = 4000;
const COUNTDOWN_TICK_MS = 1000;
const MAX_BYTES_DEFAULT = 256;

const COLOR_VALENCE = "#5dc8ff";
const COLOR_AROUSAL = "#ffae5d";
const COLOR_STRIPE_BG = "rgba(255,255,255,0.025)";
const COLOR_PLACEHOLDER_BG = "#06070a";

// ---- state ----
const events = [];
let regions = [];
let countdownTarget = null;

// ---- helpers ----
function fmtPctRaw(x) {
  if (x == null || !isFinite(x)) return "--";
  return x.toFixed(1) + "%";
}

function fmtParams(n) {
  if (n == null || !isFinite(n)) return "--";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + " K";
  return String(n);
}

function fmtElapsed(secs) {
  if (secs == null || !isFinite(secs) || secs < 0) return "--";
  const s = Math.floor(secs);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(sec).padStart(2, "0");
  return h + ":" + mm + ":" + ss;
}

function relTimeAgo(epochSecs) {
  if (epochSecs == null || !isFinite(epochSecs)) return "never";
  const now = Date.now() / 1000;
  const diff = Math.max(0, now - epochSecs);
  if (diff < 60) return Math.floor(diff) + "s ago";
  if (diff < 3600) return Math.floor(diff / 60) + "m ago";
  if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
  return Math.floor(diff / 86400) + "d ago";
}

function fmtNum(x, digits) {
  if (x == null || !isFinite(x)) return "--";
  return Number(x).toFixed(digits);
}

async function safeFetch(url, opts) {
  try {
    const res = await fetch(url, opts || {});
    if (!res.ok) return null;
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    if (ct.indexOf("application/json") >= 0) return await res.json();
    return await res.text();
  } catch (e) {
    return null;
  }
}

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function clearChildren(el) {
  if (!el) return;
  while (el.firstChild) el.removeChild(el.firstChild);
}

// ---- status ----
async function loadStatus() {
  const data = await safeFetch("/multimind/status");
  if (!data) {
    setText("mtmStatusModel", "model not available");
    setText("mtmStatusRole", "--");
    setText("mtmStatusUptime", "--");
    setText("mtmStatusLastSleep", "--");
    return;
  }
  const isStub = data.mode === "stub";
  const prefix = isStub ? "[stub] " : "";
  let modelText;
  if (data.model_loaded && data.model_name) {
    const step = (data.model_step != null) ? " @ step " + data.model_step : "";
    modelText = prefix + data.model_name + step;
  } else {
    modelText = prefix + "no model loaded";
  }
  setText("mtmStatusModel", modelText);
  setText("mtmStatusRole", data.role || "--");
  setText("mtmStatusUptime", fmtElapsed(data.uptime_secs));
  setText("mtmStatusLastSleep", relTimeAgo(data.last_sleep_at));

  if (data.next_sleep_eta_secs != null && isFinite(data.next_sleep_eta_secs)) {
    countdownTarget = Date.now() / 1000 + data.next_sleep_eta_secs;
  } else {
    countdownTarget = null;
    setText("mtmSleepCountdown", "--");
  }
}

function tickCountdown() {
  if (countdownTarget == null) {
    setText("mtmSleepCountdown", "--");
    return;
  }
  const remaining = countdownTarget - (Date.now() / 1000);
  if (remaining <= 0) {
    setText("mtmSleepCountdown", "due now");
    return;
  }
  setText("mtmSleepCountdown", fmtElapsed(remaining));
}

// ---- regions ----
async function loadRegions() {
  const data = await safeFetch("/multimind/regions");
  const legendEl = $("mtmFiringLegend");
  const barsEl = $("mtmGateBars");
  const tbl = $("mtmRegionsTable");
  const tbody = tbl ? tbl.querySelector("tbody") || tbl.appendChild(document.createElement("tbody")) : null;

  if (!data || !Array.isArray(data.regions) || data.regions.length === 0) {
    regions = [];
    if (legendEl) legendEl.innerHTML = '<span class="mtm-empty">no regions reported</span>';
    if (barsEl) clearChildren(barsEl);
    if (tbody) {
      clearChildren(tbody);
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 6;
      td.className = "mtm-empty";
      td.textContent = "no regions reported";
      tr.appendChild(td);
      tbody.appendChild(tr);
    }
    return;
  }

  regions = data.regions;
  renderLegend(legendEl, regions);
  renderGateBars(barsEl, regions);
  renderRegionsTable(tbody, regions, data.mode);
}

function renderLegend(el, regs) {
  if (!el) return;
  clearChildren(el);
  for (const r of regs) {
    const item = document.createElement("span");
    item.className = "mtm-swatch-item";
    const swatch = document.createElement("span");
    swatch.className = "mtm-swatch";
    swatch.style.background = r.color || "#888";
    const label = document.createElement("span");
    label.className = "mtm-legend-label";
    label.textContent = r.name + " (" + r.slug + ")";
    item.appendChild(swatch);
    item.appendChild(label);
    el.appendChild(item);
  }
}

function renderGateBars(el, regs) {
  if (!el) return;
  clearChildren(el);
  for (const r of regs) {
    const row = document.createElement("div");
    row.className = "mtm-gate-bar";
    row.dataset.slug = r.slug;
    const label = document.createElement("span");
    label.className = "mtm-gate-bar-label";
    label.textContent = r.name;
    const track = document.createElement("div");
    track.className = "mtm-gate-bar-track";
    const fill = document.createElement("div");
    fill.className = "mtm-gate-bar-fill";
    fill.style.width = "0%";
    fill.style.background = r.color || "#888";
    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    el.appendChild(row);
  }
}

function renderRegionsTable(tbody, regs, mode) {
  if (!tbody) return;
  clearChildren(tbody);
  for (const r of regs) {
    const tr = document.createElement("tr");
    const cells = [
      r.name || r.slug,
      r.specialty || "--",
      fmtParams(r.params),
      fmtNum(r.last_drift_l2, 4),
      fmtNum(r.specialty_ppl, 2),
      fmtPctRaw(r.fired_pct),
    ];
    for (let i = 0; i < cells.length; i++) {
      const td = document.createElement("td");
      td.textContent = cells[i];
      if (i === 0) {
        const sw = document.createElement("span");
        sw.className = "mtm-row-swatch";
        sw.style.background = r.color || "#888";
        td.insertBefore(sw, td.firstChild);
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

// ---- firing strip ----
function drawFiringStrip(evs) {
  const c = $("mtmFiringCanvas");
  const placeholder = $("mtmFiringPlaceholder");
  if (!c) return;
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.fillStyle = COLOR_PLACEHOLDER_BG;
  ctx.fillRect(0, 0, w, h);

  if (!evs || evs.length === 0 || regions.length === 0) {
    if (placeholder) placeholder.style.display = "";
    return;
  }
  if (placeholder) placeholder.style.display = "none";

  const rowH = h / regions.length;
  for (let i = 0; i < regions.length; i++) {
    if (i % 2 === 1) {
      ctx.fillStyle = COLOR_STRIPE_BG;
      ctx.fillRect(0, i * rowH, w, rowH);
    }
  }

  const maxCols = Math.max(1, w);
  const visible = evs.slice(-maxCols);
  const pxPerByte = Math.max(1, Math.floor(w / Math.max(1, visible.length)));

  const slugToRow = new Map();
  for (let i = 0; i < regions.length; i++) slugToRow.set(regions[i].slug, i);
  const slugToColor = new Map();
  for (const r of regions) slugToColor.set(r.slug, r.color || "#888");

  for (let i = 0; i < visible.length; i++) {
    const ev = visible[i];
    const x = i * pxPerByte;
    const regs = ev && ev.regions ? ev.regions : [];
    for (const rg of regs) {
      const rowIdx = slugToRow.get(rg.slug);
      if (rowIdx == null) continue;
      const color = slugToColor.get(rg.slug) || "#888";
      const alpha = Math.max(0, Math.min(1, rg.gate_weight || 0));
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.fillRect(x, rowIdx * rowH + 1, Math.max(1, pxPerByte - 0.3), Math.max(1, rowH - 2));
      ctx.globalAlpha = 1;
    }
  }
}

// ---- affect trace ----
function drawAffectTrace(evs) {
  const c = $("mtmAffectCanvas");
  if (!c) return;
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.fillStyle = COLOR_PLACEHOLDER_BG;
  ctx.fillRect(0, 0, w, h);

  if (!evs || evs.length === 0) return;

  const maxCols = Math.max(1, w);
  const visible = evs.slice(-maxCols);
  const stepX = w / Math.max(1, visible.length);

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  // valence: -1..1 -> h..0
  ctx.strokeStyle = COLOR_VALENCE;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < visible.length; i++) {
    const a = visible[i].affect;
    if (!a || a.valence == null || !isFinite(a.valence)) continue;
    const v = Math.max(-1, Math.min(1, a.valence));
    const x = i * stepX;
    const y = h * (1 - (v + 1) / 2);
    if (!started) { ctx.moveTo(x, y); started = true; }
    else ctx.lineTo(x, y);
  }
  if (started) ctx.stroke();

  // arousal: 0..1 -> h..0
  ctx.strokeStyle = COLOR_AROUSAL;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  started = false;
  for (let i = 0; i < visible.length; i++) {
    const a = visible[i].affect;
    if (!a || a.arousal == null || !isFinite(a.arousal)) continue;
    const ar = Math.max(0, Math.min(1, a.arousal));
    const x = i * stepX;
    const y = h * (1 - ar);
    if (!started) { ctx.moveTo(x, y); started = true; }
    else ctx.lineTo(x, y);
  }
  if (started) ctx.stroke();
}

// ---- gate bars update ----
function updateGateBars(latestEvent) {
  if (!latestEvent || !latestEvent.regions) return;
  const barsEl = $("mtmGateBars");
  if (!barsEl) return;
  for (const r of latestEvent.regions) {
    const w = Math.max(0, Math.min(1, r.gate_weight || 0));
    const row = barsEl.querySelector('[data-slug="' + r.slug + '"]');
    if (!row) continue;
    const fill = row.querySelector(".mtm-gate-bar-fill");
    if (fill) fill.style.width = (w * 100).toFixed(1) + "%";
    if (fill) fill.classList.toggle("mtm-bar-refractory", !!r.refractory);
  }
}

// ---- render full event stream ----
function renderEvents() {
  drawFiringStrip(events);
  drawAffectTrace(events);
  if (events.length > 0) updateGateBars(events[events.length - 1]);
}

function pushEvent(ev) {
  if (!ev) return;
  events.push(ev);
  if (events.length > EVENT_BUFFER_CAP) events.splice(0, events.length - EVENT_BUFFER_CAP);
}

function resetEvents() {
  events.length = 0;
}

// ---- conversations ----
async function loadConversations() {
  const data = await safeFetch("/multimind/conversations");
  const ul = $("mtmConvList");
  if (!ul) return;
  clearChildren(ul);
  if (!data || !Array.isArray(data.conversations) || data.conversations.length === 0) {
    const li = document.createElement("li");
    li.className = "mtm-empty";
    li.textContent = "no conversations yet";
    ul.appendChild(li);
    return;
  }
  for (const c of data.conversations) {
    const li = document.createElement("li");
    li.className = "mtm-conv-item";
    li.dataset.id = c.id;
    const when = document.createElement("span");
    when.className = "mtm-conv-when";
    when.textContent = relTimeAgo(c.ts);
    const preview = document.createElement("span");
    preview.className = "mtm-conv-preview";
    preview.textContent = c.prompt_preview || "(empty)";
    const count = document.createElement("span");
    count.className = "mtm-conv-count";
    count.textContent = (c.byte_count != null ? c.byte_count : "--") + " bytes";
    li.appendChild(when);
    li.appendChild(preview);
    li.appendChild(count);
    li.addEventListener("click", () => replayConversation(c.id));
    ul.appendChild(li);
  }
}

async function replayConversation(id) {
  const data = await safeFetch("/multimind/conversations/" + encodeURIComponent(id));
  if (!data) return;
  resetEvents();
  const evs = Array.isArray(data.events) ? data.events : [];
  for (const e of evs) pushEvent(e);
  renderEvents();
  console.info("multimind: replayed conversation", id, "events=" + evs.length);
}

// ---- send / sample ----
async function onSendClick() {
  const promptEl = $("mtmPrompt");
  const maxEl = $("mtmMaxBytes");
  const btn = $("mtmSendBtn");
  if (!promptEl) return;
  const prompt = promptEl.value || "";
  const maxBytes = maxEl ? (parseInt(maxEl.value, 10) || MAX_BYTES_DEFAULT) : MAX_BYTES_DEFAULT;

  if (btn) btn.disabled = true;
  resetEvents();
  renderEvents();

  try {
    const res = await fetch("/multimind/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: prompt, max_bytes: maxBytes }),
    });
    if (!res.ok) {
      showInlineMessage("sample request failed");
      return;
    }
    const data = await res.json();
    await handleJsonSample(data);
  } catch (e) {
    showInlineMessage("sample request failed");
  } finally {
    if (btn) btn.disabled = false;
    loadConversations();
  }
}

async function handleJsonSample(data) {
  if (!data) return;
  const evs = Array.isArray(data.events) ? data.events : [];
  for (const e of evs) pushEvent(e);
  renderEvents();
}

function showInlineMessage(msg) {
  const panel = $("mtmSleepPanel");
  if (!panel) return;
  let note = panel.querySelector(".mtm-inline-note");
  if (!note) {
    note = document.createElement("div");
    note.className = "mtm-inline-note";
    panel.appendChild(note);
  }
  note.textContent = msg;
  setTimeout(() => { if (note) note.textContent = ""; }, 4000);
}

// ---- force sleep ----
async function onForceSleepClick() {
  const btn = $("mtmForceSleepBtn");
  if (btn) btn.disabled = true;
  try {
    const res = await fetch("/multimind/sleep/trigger", { method: "POST" });
    if (!res.ok) {
      showInlineMessage("could not start sleep cycle");
      return;
    }
    const data = await res.json();
    if (data && data.ok) {
      setText("mtmStatusRole", "sleeping");
    } else {
      showInlineMessage("sleep cycle already running");
    }
  } catch (e) {
    showInlineMessage("could not start sleep cycle");
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ---- polling ----
function startPolling() {
  setInterval(loadStatus, POLL_STATUS_MS);
  setInterval(loadRegions, POLL_REGIONS_MS);
  setInterval(loadConversations, POLL_CONVS_MS);
  setInterval(tickCountdown, COUNTDOWN_TICK_MS);
}

// ---- init ----
function wireHandlers() {
  const send = $("mtmSendBtn");
  if (send) send.addEventListener("click", onSendClick);
  const sleep = $("mtmForceSleepBtn");
  if (sleep) sleep.addEventListener("click", onForceSleepClick);
  const prompt = $("mtmPrompt");
  if (prompt) {
    prompt.addEventListener("keydown", e => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        onSendClick();
      }
    });
  }
  const maxEl = $("mtmMaxBytes");
  if (maxEl && !maxEl.value) maxEl.value = String(MAX_BYTES_DEFAULT);
}

async function init() {
  wireHandlers();
  await Promise.all([loadStatus(), loadRegions(), loadConversations()]);
  renderEvents();
  startPolling();
  console.info("multimind: initialized");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
