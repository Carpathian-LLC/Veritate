# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - discovers checks, runs them, dumps logs and a summary.json per run.
# - imports done lazily so importing the runner never fails on missing deps.
# tests/selftest/_runner.py
# ------------------------------------------------------------------------------------
# Imports

import importlib
import json
import os
import time

from . import _ctx
from . import _log
from . import _status

# ------------------------------------------------------------------------------------
# Constants

PACKAGE_PREFIX  = "tests.selftest.checks."
RUN_ENTRY       = "run"
ATTR_SLOW       = "SLOW"
ATTR_REQ_MODEL  = "REQUIRES_MODEL"
ATTR_REQ_ENGINE = "REQUIRES_ENGINE_BUILT"
ATTR_REQ_TORCH  = "REQUIRES_TORCH"
ATTR_REQ_NET    = "REQUIRES_NETWORK"
ATTR_AREA       = "AREA"

# ------------------------------------------------------------------------------------
# Functions

def discover():
    """return sorted list of check module short names found under checks/."""
    out = []
    if not os.path.isdir(_ctx.CHECKS_DIR):
        return out
    for fname in sorted(os.listdir(_ctx.CHECKS_DIR)):
        if not fname.startswith(_ctx.CHECK_PREFIX):
            continue
        if not fname.endswith(_ctx.CHECK_SUFFIX):
            continue
        out.append(fname[:-len(_ctx.CHECK_SUFFIX)])
    return out


def _load_module(short_name):
    return importlib.import_module(PACKAGE_PREFIX + short_name)


def _skip_for_flags(mod, options):
    if options.get("skip_slow") and getattr(mod, ATTR_SLOW, False):
        return "slow"
    if options.get("no_network") and getattr(mod, ATTR_REQ_NET, False):
        return "network"
    return None


def _filter(names, options):
    only = options.get("only") or []
    skip = options.get("skip") or []
    out  = []
    for n in names:
        if only and n not in only and _strip_check(n) not in only:
            continue
        if skip and (n in skip or _strip_check(n) in skip):
            continue
        out.append(n)
    return out


def _strip_check(n):
    return n[len(_ctx.CHECK_PREFIX):] if n.startswith(_ctx.CHECK_PREFIX) else n


def run_one(short_name, ctx):
    """import and execute a single check. returns _status.Result."""
    log_path = _ctx.check_log_path(ctx.run_id, short_name)
    _log.write_line(log_path, f"START {short_name}")
    t0 = time.time()
    try:
        mod = _load_module(short_name)
    except Exception as exc:
        _log.dump_exception(log_path, exc)
        return _status.Result(short_name, _status.STATUS_FAIL,
                              f"import failed: {type(exc).__name__}",
                              log_path=log_path, elapsed=time.time() - t0)

    why = _skip_for_flags(mod, ctx.options)
    if why is not None:
        _log.write_line(log_path, f"SKIP (flag: {why})")
        return _status.Result(short_name, _status.STATUS_SKIP,
                              f"skipped by flag: {why}",
                              log_path=log_path, elapsed=time.time() - t0)

    fn = getattr(mod, RUN_ENTRY, None)
    if fn is None:
        _log.write_line(log_path, f"FAIL no run() entry")
        return _status.Result(short_name, _status.STATUS_FAIL,
                              "module has no run(ctx) function",
                              log_path=log_path, elapsed=time.time() - t0)

    try:
        with _log.capture(log_path):
            result = fn(ctx)
    except Exception as exc:
        _log.dump_exception(log_path, exc)
        return _status.Result(short_name, _status.STATUS_FAIL,
                              f"raised {type(exc).__name__}: {exc}",
                              log_path=log_path, elapsed=time.time() - t0)

    if not isinstance(result, _status.Result):
        _log.write_line(log_path, "FAIL run() did not return a Result")
        return _status.Result(short_name, _status.STATUS_FAIL,
                              "run(ctx) did not return Result",
                              log_path=log_path, elapsed=time.time() - t0)

    result.log_path = log_path
    result.elapsed  = time.time() - t0
    _log.write_line(log_path, f"END {result.status} ({result.elapsed:.2f}s): {result.summary}")
    return result


def run_all(options):
    """run every discovered check (filtered by options). write summary.json."""
    _ctx.install_paths()
    run_id  = _ctx.new_run_id()
    log_dir = _ctx.run_log_dir(run_id)
    os.makedirs(log_dir, exist_ok=True)
    ctx = _ctx.Ctx(run_id, log_dir, options)

    run_log = _ctx.run_log_path(run_id)
    _log.tee(run_log, f"selftest run {run_id}")
    _log.tee(run_log, f"log dir: {log_dir}")

    names = _filter(discover(), options)
    if not names:
        _log.tee(run_log, "no checks selected.")
        _write_summary(run_id, [], 0.0)
        return 0, []

    _log.tee(run_log, f"running {len(names)} checks")
    results = []
    t_total = time.time()
    for n in names:
        t0 = time.time()
        res = run_one(n, ctx)
        results.append(res)
        glyph = _status.GLYPH.get(res.status, "[??]")
        _log.tee(run_log, f"{glyph} {n}  ({time.time() - t0:.2f}s)  {res.summary}")

    total = time.time() - t_total
    _write_summary(run_id, results, total)
    _log.tee(run_log, _aggregate_line(results, total))
    exit_code = 1 if any(r.status == _status.STATUS_FAIL for r in results) else 0
    return exit_code, results


def _aggregate_line(results, total):
    counts = {s: 0 for s in _status.STATUS_ORDER}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return (
        f"done in {total:.2f}s  "
        f"ok={counts[_status.STATUS_OK]}  "
        f"skip={counts[_status.STATUS_SKIP]}  "
        f"fail={counts[_status.STATUS_FAIL]}"
    )


def _write_summary(run_id, results, total):
    payload = {
        "run_id":  run_id,
        "elapsed": round(total, 3),
        "counts":  _count_dict(results),
        "results": [r.to_dict() for r in results],
    }
    path = _ctx.summary_path(run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _count_dict(results):
    out = {s: 0 for s in _status.STATUS_ORDER}
    for r in results:
        out[r.status] = out.get(r.status, 0) + 1
    return out
