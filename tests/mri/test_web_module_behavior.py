# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the dashboard's javascript has structural tests (test_web_modules.py) and a browser
#   smoke, but no test of what its logic COMPUTES. This runs a module's pure functions
#   inside node with a stub window/document and asserts on the values, so a silent
#   mis-parse is caught here rather than seen as a flat chart. No test framework and no
#   dependency: node evaluates the file, prints JSON, pytest asserts on it. Skips without
#   node, like the other javascript tests.
# tests/mri/test_web_module_behavior.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import shutil
import subprocess

import pytest
from conftest import REPO_ROOT

# ------------------------------------------------------------------------------------
# Constants

WEB = os.path.join(REPO_ROOT, "veritate_mri", "web")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

HEADER = "step,split,loss,lr,grad_norm,tok_per_s"

# ------------------------------------------------------------------------------------
# Functions


def _run(module, expression):
    """Evaluate one browser module in node behind a stub window, return the JSON it makes."""
    script = f"""
      const fs = require("fs");
      const vm = require("vm");
      const sandbox = {{ window: {{}}, document: {{ addEventListener() {{}} }}, console }};
      sandbox.globalThis = sandbox;
      vm.createContext(sandbox);
      vm.runInContext(fs.readFileSync({json.dumps(os.path.join(WEB, module))}, "utf8"), sandbox);
      const out = vm.runInContext({json.dumps(expression)}, sandbox);
      console.log(JSON.stringify(out));
    """
    p = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def _csv(rows):
    return HEADER + "\n" + "\n".join(rows) + "\n"


def test_the_module_loads_in_isolation():
    """It must not reach for anything index.js defines at load time."""
    assert _run("image_live.js", "Object.keys(window.ImageLive).sort()") == \
        ["TRAINER_ID", "create", "parseCsv"]


def test_train_and_val_rows_are_separated():
    body = _csv(["1,train,0.9,3e-05,0.4,100", "1,val,1.1,3e-05,,", "2,train,0.8,3e-05,0.3,100"])
    out = _run("image_live.js", f"window.ImageLive.parseCsv({json.dumps(body)})")
    assert [r["step"] for r in out["train"]] == [1, 2]
    assert [r["loss"] for r in out["val"]] == [1.1]


def test_a_row_with_an_unparseable_number_is_dropped():
    """A torn line from a run being written to must not become a point at zero."""
    body = _csv(["1,train,0.9,3e-05,0.4,100", "2,train,,3e-05,0.4,100", "3,train,0.7,3e-05,0.4,100"])
    out = _run("image_live.js", f"window.ImageLive.parseCsv({json.dumps(body)})")
    assert [r["step"] for r in out["train"]] == [1, 3]


def test_only_the_latest_run_in_the_file_is_charted():
    """A resumed model's train.csv holds the previous run's rows ahead of the new ones;
    charting both draws a line that jumps backwards in time."""
    body = _csv(["10,train,0.9,3e-05,0.4,100", "11,train,0.8,3e-05,0.4,100",
                 "1,train,0.7,3e-05,0.4,100", "2,train,0.6,3e-05,0.4,100"])
    out = _run("image_live.js", f"window.ImageLive.parseCsv({json.dumps(body)})")
    assert [r["step"] for r in out["train"]] == [1, 2]


def test_the_previous_runs_val_rows_go_with_it():
    """They carry HIGHER step numbers than the new run's, so filtering val by "at or after
    the first train step" kept them and drew a chart with two runs on it."""
    body = _csv(["10,train,0.9,3e-05,0.4,100", "10,val,1.0,3e-05,,",
                 "1,train,0.7,3e-05,0.4,100", "2,val,0.65,3e-05,,"])
    out = _run("image_live.js", f"window.ImageLive.parseCsv({json.dumps(body)})")
    assert [r["step"] for r in out["val"]] == [2]
    assert [r["step"] for r in out["train"]] == [1]


def test_the_starting_weights_val_row_survives_a_resume():
    """An armed run logs val at the resume step BEFORE its first train row, and that reading
    is the baseline its stop rule compares against. Truncating the ordered stream keeps it;
    filtering val by the first train step dropped it."""
    body = _csv(["70399,train,0.15,3e-05,0.2,100", "70400,val,0.38,3e-05,,",
                 "70200,val,0.42,3e-05,,", "70201,train,0.73,3e-05,0.8,100"])
    out = _run("image_live.js", f"window.ImageLive.parseCsv({json.dumps(body)})")
    assert [r["step"] for r in out["val"]] == [70200]
    assert [r["step"] for r in out["train"]] == [70201]


def test_an_empty_or_header_only_file_parses_to_nothing():
    for body in ("", HEADER + "\n"):
        out = _run("image_live.js", f"window.ImageLive.parseCsv({json.dumps(body)})")
        assert out == {"train": [], "val": []}


def test_the_columns_are_read_by_name_not_position():
    """A trainer that adds or reorders a column must not shift every value by one."""
    body = ("tok_per_s,loss,step,split,lr,grad_norm\n"
            "100,0.9,7,train,3e-05,0.4\n")
    out = _run("image_live.js", f"window.ImageLive.parseCsv({json.dumps(body)})")
    assert out["train"] == [{"step": 7, "loss": 0.9, "lr": 3e-05, "gn": 0.4, "tps": 100}]
