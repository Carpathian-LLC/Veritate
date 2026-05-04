# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - parse a model's train.csv into list of dicts. mtime-keyed cache.
# - schema: step,split,loss,lr,grad_norm,tok_per_s,wall_s,seed
#   split column suffix-matched to "train" / "val" by the dashboard parser.
# veritate_mri/readers/train_csv.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from . import paths

# ------------------------------------------------------------------------------------
# Constants

_CACHE = {}

NUMERIC_FIELDS = ("loss", "lr", "grad_norm", "tok_per_s", "wall_s")

# ------------------------------------------------------------------------------------
# Functions

def _parse_row(header, parts):
    if len(parts) != len(header):
        return None
    d = dict(zip(header, parts))
    try:
        d["step"] = int(float(d.get("step", 0)))
    except (TypeError, ValueError):
        return None
    for k in NUMERIC_FIELDS:
        v = d.get(k)
        if v in (None, ""):
            d[k] = None
            continue
        try:
            d[k] = float(v)
        except (TypeError, ValueError):
            d[k] = None
    return d


def load(name):
    p = paths.train_csv_path(name)
    try:
        st = os.stat(p)
    except OSError:
        return []
    hit = _CACHE.get(p)
    if hit and hit[0] == st.st_mtime:
        return hit[1]
    rows = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
            if not header_line:
                return []
            header = header_line.split(",")
            for line in f:
                row = _parse_row(header, line.rstrip("\n").split(","))
                if row is not None:
                    rows.append(row)
    except OSError:
        return []
    _CACHE[p] = (st.st_mtime, rows)
    return rows


def raw_text(name):
    p = paths.train_csv_path(name)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def is_present(name):
    return os.path.isfile(paths.train_csv_path(name))


def file_stat(name):
    p = paths.train_csv_path(name)
    try:
        return os.stat(p)
    except OSError:
        return None
