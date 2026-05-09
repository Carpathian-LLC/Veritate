# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - standalone in-dashboard explainer. one entry point: ask(kind, payload) -> dict.
# - each kind owns its own minimal system prompt and payload builder. no shared
#   global prompt; each section ships only the facts it needs.
# - new sections plug in by adding one entry to KINDS.
# veritate_mri/ai_assist.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import ssl
import urllib.error
import urllib.request

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

import logs as logmod
import settings as settings_mod
from readers import train_csv as train_csv_reader
from readers import config as cfg_reader

# ------------------------------------------------------------------------------------
# Constants

REQUEST_TIMEOUT_SECS = 60
MODEL_FIELD          = "default"
RECENT_TAIL_ROWS     = 12
LOG_SOURCE           = "ai_assist"
RAW_LOG_CHARS        = 600

OUTPUT_RULES = (
    "Reply in 5 to 7 short lines total. Exactly: one takeaway line, 3 or 4 bullets, one 'next:' line.\n"
    "No intro, transition, or recap lines. Do not write 'Here are', 'Based on', 'In summary', or similar. Go straight from takeaway to bullets.\n"
    "Every bullet MUST cite at least one concrete number from the data: a step, a loss value, a percent delta, an lr, a grad norm, a tok/s. Keep each bullet under 22 words.\n"
    "Do not restate the same metric in two bullets. Pick the most informative angle.\n"
    "Skip bullets about anything that is in normal range and not actionable. Fewer bullets is better than padding.\n"
    "Final line starts with 'next:' and gives one concrete action (keep training, lower LR by Nx, raise val frequency, restart from step N, etc.).\n"
    "No emoji. No emdash, use a colon, period, or comma. No markdown headers. Plain English."
)

PROMPT_RECENT_TRAIN = (
    "You explain the most recent training metrics for one Veritate run.\n"
    "Honor these facts:\n"
    "- train.csv loss is TRAIN loss only. Never claim overfitting from train loss alone, that needs val loss.\n"
    "- 5-10% step-to-step jitter is normal noise, not a plateau or regression. Do not flag it.\n"
    "- For MoE models, n_params_active is one expert plus shared layers, not pruning.\n"
    "- vocab=256 means byte-level tokens. seq is in bytes.\n"
    "- Compare loss across the supplied rows by step delta, not by row order alone.\n"
    "- If grad_norm or tok_per_s look unusual, say in what direction and by how much vs the other rows.\n"
    f"\n{OUTPUT_RULES}"
)

PROMPT_LOSS_CURVE = (
    "You explain the shape of one Veritate run's loss curve.\n"
    "You see a downsampled train series, the recent val series, a gap series (val minus train at matching steps), and the dashboard plateau verdict.\n"
    "Honor these facts:\n"
    "- The y axis is cross-entropy in nats. Random byte model is about 5.55. Healthy byte LM lands 0.7 to 1.0.\n"
    "- 5-10% step-to-step jitter on train loss is normal noise.\n"
    "- vocab=256 byte-level. seq is in bytes.\n"
    "- Only call overfitting if val is rising AND train keeps falling, with the gap widening.\n"
    "Hard rules, no exceptions:\n"
    "- Only cite (step, loss) pairs that appear verbatim in the data block. Never invent steps or values not listed.\n"
    "- The train series is DOWNSAMPLED. A delta on a train line is the delta to the prior LISTED train step in this block, not the prior actual training step. When you cite a train delta, name both endpoints (e.g., 'down 22% from step 360 to step 680'). Never write 'previous step' for a train delta.\n"
    "- Bullets must be in ascending step order.\n"
    "- Treat absolute changes under 1% as noise. Do not flag them.\n"
    "- Discuss gap trends only by citing entries from the gap series. Never invent a gap value for a step that is not in the gap series.\n"
    "- The 'next:' action must respect the schedule. If lr is already decaying via the schedule and the verdict is IMPROVING, SLOWING, or WARMING, do NOT propose lowering lr by hand. Recommend continuing to a concrete milestone step from the schedule. Manual lr changes are only valid when the verdict is REGRESSING or BOUNCING.\n"
    "- The takeaway line must include at least one number (a step, a loss, a percent, a slope). No vague openers like 'oscillations around a stable minimum'.\n"
    f"\n{OUTPUT_RULES}"
)

PROMPT_TRAIN_HEALTH = (
    "You explain a training health verdict for a Veritate run. The dashboard has already\n"
    "labeled the current val loss trend; your job is to translate that label and the\n"
    "recent val points plus the latest live training metrics into specific, numeric advice.\n"
    "Honor these facts:\n"
    "- The verdict is derived from the slope of the last up-to-8 val loss points.\n"
    "- States and what they mean: IMPROVING (slope clearly negative, keep going), SLOWING (still dropping but gently, normal late-training), PLATEAU (flat across recent evals, diminishing returns), BOUNCING (no clear direction, often a local minimum), REGRESSING (val rising on average, lower LR or restart from last good checkpoint), WARMING UP (under 4 val points, wait).\n"
    "- Trust the dashboard verdict. Do not relabel it.\n"
    "- Don't invent overfitting calls beyond what the verdict implies.\n"
    "- Cite the actual val loss numbers and percent deltas from the recent val measurements.\n"
    "- Cite the live latest-row metrics (step, train_loss, val_loss, lr, grad_norm, tok_per_s) when relevant.\n"
    "- Train vs val gap: if train_loss is well below latest val_loss, say the gap in absolute terms and as a percent.\n"
    f"\n{OUTPUT_RULES}"
)

# ------------------------------------------------------------------------------------
# Functions

def _resolve_credentials():
    s = settings_mod.get()
    endpoint = (s.get("ai_endpoint_user") or "").strip() or s.get("ai_endpoint", "")
    key      = (s.get("ai_api_key_user")  or "").strip() or s.get("ai_api_key", "")
    return endpoint, key


def _snip(s):
    s = s or ""
    return s if len(s) <= RAW_LOG_CHARS else s[:RAW_LOG_CHARS] + f"... [+{len(s)-RAW_LOG_CHARS} chars]"


def _post_chat(endpoint, key, system_prompt, user_message):
    body = {
        "model":    MODEL_FIELD,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }
    req_body = json.dumps(body).encode("utf-8")
    logmod.info(LOG_SOURCE, f"POST {endpoint} bytes={len(req_body)} sys_chars={len(system_prompt)} user_chars={len(user_message)}")
    req = urllib.request.Request(
        endpoint,
        data    = req_body,
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        },
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECS, context=_SSL_CTX) as resp:
        status = resp.status
        raw = resp.read().decode("utf-8", errors="replace")
    logmod.info(LOG_SOURCE, f"resp status={status} bytes={len(raw)}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logmod.error(LOG_SOURCE, f"json parse fail: {e}; raw={_snip(raw)}")
        raise
    choices = data.get("choices") or []
    if not choices:
        finish = data.get("error") or data.get("message") or ""
        logmod.warn(LOG_SOURCE, f"no choices in response. keys={list(data.keys())} detail={_snip(str(finish))} raw={_snip(raw)}")
        return ""
    msg = choices[0].get("message") or {}
    finish_reason = choices[0].get("finish_reason")
    content = (msg.get("content") or "").strip()
    if not content:
        logmod.warn(LOG_SOURCE, f"empty content. finish_reason={finish_reason} message_keys={list(msg.keys())} raw={_snip(raw)}")
    else:
        logmod.ok(LOG_SOURCE, f"answer chars={len(content)} finish_reason={finish_reason}")
    return content


def _format_row(row, fields):
    parts = []
    for k in fields:
        v = row.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _build_recent_train(payload):
    name = (payload or {}).get("model") or ""
    if not name:
        raise ValueError("missing model name")
    cfg = cfg_reader.load(name) or {}
    rows = train_csv_reader.load(name) or []
    tail = rows[-RECENT_TAIL_ROWS:]
    if not tail:
        raise ValueError(f"no train.csv rows for {name}")

    shape   = cfg.get("shape", {}) or {}
    mega    = cfg.get("mega", {}) or {}
    targs   = cfg.get("training_args", {}) or {}

    spec_lines = [f"name: {cfg.get('name', name)}"]
    if shape:
        spec_lines.append(
            "shape: layers={layers} hidden={hidden} ffn={ffn} heads={heads} seq={seq} vocab={vocab}".format(
                layers=shape.get("layers"), hidden=shape.get("hidden"), ffn=shape.get("ffn"),
                heads=shape.get("heads"), seq=shape.get("seq"), vocab=shape.get("vocab"),
            )
        )
    if cfg.get("training") or cfg.get("quant_mode"):
        spec_lines.append(f"training: {cfg.get('training','?')} quant_mode: {cfg.get('quant_mode','?')}")
    if mega:
        spec_lines.append(
            f"moe: experts={mega.get('n_experts')} topk={mega.get('router_topk')} aux={mega.get('router_aux_loss')}"
        )
    if cfg.get("n_params_total") or cfg.get("n_params_active"):
        spec_lines.append(
            f"params: total={cfg.get('n_params_total')} active={cfg.get('n_params_active')}"
        )
    if targs.get("total_steps") or targs.get("base_lr"):
        spec_lines.append(
            "schedule: total_steps={ts} base_lr={blr} min_lr={mlr} warmup={wu} schedule={sch}".format(
                ts=targs.get("total_steps"), blr=targs.get("base_lr"),
                mlr=targs.get("min_lr"),     wu=targs.get("warmup_steps"),
                sch=targs.get("lr_schedule"),
            )
        )

    fields = ["step", "split", "loss", "lr", "grad_norm", "tok_per_s"]
    row_lines = [_format_row(r, fields) for r in tail]

    user_message = (
        "Specs:\n" + "\n".join(f"- {ln}" for ln in spec_lines) +
        f"\n\nLast {len(tail)} train.csv rows:\n" + "\n".join(f"- {ln}" for ln in row_lines) +
        "\n\nQuestion: in plain English, what do these last rows say about progress, "
        "and is anything worth flagging given the schedule and model shape?"
    )
    return PROMPT_RECENT_TRAIN, user_message


def _build_train_health(payload):
    p = payload or {}
    name   = p.get("model") or ""
    state  = (p.get("state") or "").upper()
    slope  = p.get("slope_pct")
    pts    = p.get("recent") or []
    latest = p.get("latest") or {}
    if not name:
        raise ValueError("missing model name")
    if not state:
        raise ValueError("missing state")

    cfg = cfg_reader.load(name) or {}
    targs = cfg.get("training_args", {}) or {}
    spec_lines = [f"name: {cfg.get('name', name)}"]
    if targs.get("total_steps"):
        spec_lines.append(
            "schedule: total_steps={ts} base_lr={blr} min_lr={mlr} warmup={wu} schedule={sch}".format(
                ts=targs.get("total_steps"), blr=targs.get("base_lr"),
                mlr=targs.get("min_lr"),     wu=targs.get("warmup_steps"),
                sch=targs.get("lr_schedule"),
            )
        )

    pt_lines = []
    prev = None
    for pt in pts[-8:]:
        step = pt.get("x")
        y    = pt.get("y")
        if y is None: continue
        if prev is None:
            pt_lines.append(f"step={step} val_loss={y:.4g}")
        else:
            d = (y - prev) / max(prev, 1e-6) * 100.0
            sign = "+" if d >= 0 else ""
            pt_lines.append(f"step={step} val_loss={y:.4g} delta={sign}{d:.2f}%")
        prev = y

    latest_lines = []
    if latest:
        order = ["step", "train_loss", "val_loss", "lr", "grad_norm", "tok_per_s"]
        latest_lines = [_format_row(latest, order)]
        tl = latest.get("train_loss")
        vl = latest.get("val_loss")
        if isinstance(tl, (int, float)) and isinstance(vl, (int, float)) and tl > 0:
            gap = vl - tl
            gap_pct = gap / max(tl, 1e-6) * 100.0
            sign = "+" if gap_pct >= 0 else ""
            latest_lines.append(f"val_minus_train_gap={gap:.4g} ({sign}{gap_pct:.2f}% of train_loss)")

    slope_text = f"{slope:.3f}%" if isinstance(slope, (int, float)) else "n/a"
    user_message = (
        f"Dashboard verdict: {state}\n"
        f"Slope per eval over recent val points: {slope_text}\n"
        + ("Run specs:\n" + "\n".join(f"- {ln}" for ln in spec_lines) + "\n" if spec_lines else "")
        + ("Latest live training metrics:\n" + "\n".join(f"- {ln}" for ln in latest_lines) + "\n" if latest_lines else "")
        + (f"Recent val measurements:\n" + "\n".join(f"- {ln}" for ln in pt_lines) if pt_lines else "Recent val measurements: none provided")
        + "\n\nQuestion: in plain English with concrete numbers, what does this verdict mean for this run, "
        "and what should the user do next?"
    )
    return PROMPT_TRAIN_HEALTH, user_message


def _series_lines(pts, label, max_n):
    out = []
    prev = None
    for pt in pts[-max_n:]:
        x = pt.get("x"); y = pt.get("y")
        if y is None: continue
        if prev is None:
            out.append(f"{label} step={x} loss={y:.4g}")
        else:
            d = (y - prev) / max(prev, 1e-6) * 100.0
            sign = "+" if d >= 0 else ""
            out.append(f"{label} step={x} loss={y:.4g} delta={sign}{d:.2f}%")
        prev = y
    return out


def _gap_series(name, val_pts, max_n, step_tol):
    rows = train_csv_reader.load(name) or []
    train_by_step = {}
    for r in rows:
        if (r.get("split") or "") != "train": continue
        s = r.get("step"); l = r.get("loss")
        if s is None or l is None: continue
        try: train_by_step[int(s)] = float(l)
        except (TypeError, ValueError): continue
    out = []
    prev_gap = None
    for pt in (val_pts or [])[-max_n:]:
        s = pt.get("x"); v = pt.get("y")
        if s is None or v is None: continue
        try: s_int = int(s); v_f = float(v)
        except (TypeError, ValueError): continue
        tl = train_by_step.get(s_int)
        if tl is None:
            cands = [k for k in train_by_step.keys() if abs(k - s_int) <= step_tol]
            if cands:
                tl = train_by_step[min(cands, key=lambda k: abs(k - s_int))]
        if tl is None: continue
        gap = v_f - tl
        if prev_gap is None:
            out.append(f"step={s_int} val={v_f:.4g} train={tl:.4g} gap={gap:+.4g}")
        else:
            d = gap - prev_gap
            out.append(f"step={s_int} val={v_f:.4g} train={tl:.4g} gap={gap:+.4g} gap_delta={d:+.4g}")
        prev_gap = gap
    return out


def _build_loss_curve(payload):
    p = payload or {}
    name  = p.get("model") or ""
    state = (p.get("state") or "").upper()
    slope = p.get("slope_pct")
    train = p.get("train") or []
    val   = p.get("val")   or []
    if not name:
        raise ValueError("missing model name")
    if not train and not val:
        raise ValueError("no points in loss curve payload")

    cfg = cfg_reader.load(name) or {}
    targs = cfg.get("training_args", {}) or {}
    spec_lines = [f"name: {cfg.get('name', name)}"]
    if targs.get("total_steps"):
        spec_lines.append(
            "schedule: total_steps={ts} base_lr={blr} min_lr={mlr} warmup={wu} schedule={sch}".format(
                ts=targs.get("total_steps"), blr=targs.get("base_lr"),
                mlr=targs.get("min_lr"),     wu=targs.get("warmup_steps"),
                sch=targs.get("lr_schedule"),
            )
        )

    train_lines = _series_lines(train, "train", 12)
    val_lines   = _series_lines(val,   "val",   12)
    gap_lines   = _gap_series(name, val, 8, 20)

    slope_text = f"{slope:.3f}%" if isinstance(slope, (int, float)) else "n/a"
    user_message = (
        f"Dashboard val plateau verdict: {state or 'n/a'}\n"
        f"Slope per eval over recent val points: {slope_text}\n"
        + ("Run specs:\n" + "\n".join(f"- {ln}" for ln in spec_lines) + "\n" if spec_lines else "")
        + ("Train loss samples (downsampled, deltas are between listed rows):\n" + "\n".join(f"- {ln}" for ln in train_lines) + "\n" if train_lines else "")
        + ("Val loss points:\n" + "\n".join(f"- {ln}" for ln in val_lines) + "\n" if val_lines else "Val loss points: none provided\n")
        + ("Gap series (val minus train at matching steps):\n" + "\n".join(f"- {ln}" for ln in gap_lines) if gap_lines else "Gap series: none provided")
        + "\n\nQuestion: in plain English with concrete numbers, describe the curve shape, "
        "the train vs val gap trend from the gap series, and what to do next."
    )
    return PROMPT_LOSS_CURVE, user_message


KINDS = {
    "recent_train": _build_recent_train,
    "train_health": _build_train_health,
    "loss_curve":   _build_loss_curve,
}


def ask(kind, payload):
    s = settings_mod.get()
    if not s.get("ai_enabled"):
        logmod.warn(LOG_SOURCE, f"ask kind={kind} blocked: ai_disabled")
        return {"ok": False, "error": "ai_disabled"}
    builder = KINDS.get(kind)
    if builder is None:
        logmod.error(LOG_SOURCE, f"ask kind={kind} unknown")
        return {"ok": False, "error": f"unknown kind: {kind}"}
    endpoint, key = _resolve_credentials()
    if not endpoint or not key:
        logmod.error(LOG_SOURCE, f"ask kind={kind} missing endpoint or api key")
        return {"ok": False, "error": "missing endpoint or api key"}
    try:
        system_prompt, user_message = builder(payload)
    except (ValueError, KeyError, TypeError) as e:
        logmod.error(LOG_SOURCE, f"ask kind={kind} payload error: {e}")
        return {"ok": False, "error": f"payload error: {e}"}
    try:
        text = _post_chat(endpoint, key, system_prompt, user_message)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        logmod.error(LOG_SOURCE, f"http {e.code} {e.reason}: {_snip(err_body)}")
        detail = _snip(err_body) if err_body else e.reason
        return {"ok": False, "error": f"http {e.code}: {detail}"}
    except urllib.error.URLError as e:
        logmod.error(LOG_SOURCE, f"network: {e.reason}")
        return {"ok": False, "error": f"network: {e.reason}"}
    except (TimeoutError, json.JSONDecodeError) as e:
        logmod.error(LOG_SOURCE, f"{type(e).__name__}: {e}")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not text:
        logmod.error(LOG_SOURCE, f"ask kind={kind} empty response (see prior warn for raw)")
        return {"ok": False, "error": "empty response (see Logs tab for raw response)"}
    return {"ok": True, "kind": kind, "answer": text}
