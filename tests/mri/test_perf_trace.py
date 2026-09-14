# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - perf_trace is the one-shot decode-timing harness. Its numbers land in the ledgers, so
#   what is pinned is the arithmetic and the resolution rules: percentiles over a known
#   list, empty input reporting zeros rather than raising, the first inter-frame gap
#   dropped because it carries prefill, and exe/model resolution falling back to what this
#   box actually built and most recently wrote. No engine is spawned.
# tests/mri/test_perf_trace.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from tools import perf_trace as pt

# ------------------------------------------------------------------------------------
# Functions


def _frame(read, parse, inter, size=64):
    return {"t_read_pipe_ms": read, "t_parse_ms": parse, "t_engine_inter_ms": inter,
            "frame_size_bytes": size}


def test_percentiles_over_a_known_list():
    values = [float(v) for v in range(1, 101)]
    assert pt._percentile(values, 50) == 51.0
    assert pt._percentile(values, 99) == 99.0
    assert pt._percentile(values, 0) == 1.0


def test_an_empty_sample_reports_zeros_rather_than_raising():
    """A run that produced no frames must still render a report saying so."""
    assert pt._percentile([], 50) == 0.0
    assert pt._stats([]) == {"avg": 0.0, "p50": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}


def test_stats_over_a_known_sample():
    s = pt._stats([1.0, 2.0, 3.0, 4.0])
    assert s["avg"] == 2.5 and s["min"] == 1.0 and s["max"] == 4.0


def test_the_first_inter_frame_gap_is_dropped():
    """The first gap is measured from the start of the stream, so it carries the prefill.
    Averaging it in would make a long prompt look like slow decoding."""
    agg = pt.aggregate({"trace": [_frame(1, 1, 500), _frame(1, 1, 10), _frame(1, 1, 20)],
                        "token_count": 3})
    assert agg["frames"] == 3
    assert agg["engine"]["max"] == 20
    assert agg["read"]["avg"] == 1


def test_an_empty_trace_aggregates_to_zeros():
    agg = pt.aggregate({"trace": [], "token_count": 0})
    assert agg["frames"] == 0 and agg["frame_size_bytes"] == 0


def test_an_explicit_exe_wins_and_a_missing_one_falls_back(tmp_path, monkeypatch):
    """--exe points at a build under test; anything else uses the binary this box built,
    so the harness behaves the same on macOS, Linux and Windows."""
    built = tmp_path / "veritate_engine"
    built.write_bytes(b"x")
    monkeypatch.setattr(pt.paths, "engine_binary_path", lambda: str(built))
    override = tmp_path / "other"
    override.write_bytes(b"y")
    assert pt._resolve_exe(str(override)) == str(override)
    assert pt._resolve_exe(None) == str(built)
    assert pt._resolve_exe(str(tmp_path / "nope")) == str(built)


def test_no_built_engine_resolves_to_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(pt.paths, "engine_binary_path", lambda: str(tmp_path / "absent"))
    assert pt._resolve_exe(None) is None


def test_the_newest_servable_model_is_picked(tmp_path, monkeypatch):
    """A model dir counts only when it holds both a bin and a config; of those, the most
    recently written bin is the one the box is serving."""
    monkeypatch.setattr(pt.paths, "MODELS_ROOT", str(tmp_path))
    for name, age in (("old", 100), ("new", 0), ("no_config", 0)):
        d = tmp_path / name
        d.mkdir()
        binp = d / pt.paths.BIN_NAME
        binp.write_bytes(b"w")
        if name != "no_config":
            (d / pt.paths.CONFIG_NAME).write_text("{}", encoding="utf-8")
        os.utime(binp, (1_700_000_000 - age, 1_700_000_000 - age))
    assert pt._resolve_model(None) == str(tmp_path / "new" / pt.paths.BIN_NAME)


def test_no_model_resolves_to_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(pt.paths, "MODELS_ROOT", str(tmp_path))
    assert pt._resolve_model(None) is None


def _result(trace, wall_ms=20.0, tokens=2):
    return {"trace": trace, "payload_bytes": 128, "total_wall_ms": wall_ms,
            "outer_wall_ms": wall_ms + 5, "total_bytes": 16, "token_count": tokens}


def test_the_report_names_what_was_measured():
    """The markdown is what gets pasted into a ledger entry; without the binary, the model
    and the prompt it is an unattributable number."""
    result = _result([_frame(1, 1, 500), _frame(1, 1, 10)])
    md = pt.render_markdown(result, pt.aggregate(result), "/e/exe", "/m/bin", "hi")
    assert "/e/exe" in md and "/m/bin" in md and "hi" in md


def test_the_conclusion_is_computed_from_this_run():
    """It used to end with a fixed paragraph quoting numbers from the run it was written
    for, which reads as current whatever the harness just measured."""
    slow = pt.render_markdown(_result([_frame(40, 1, 0), _frame(1, 1, 10)], wall_ms=100.0),
                              pt.aggregate(_result([_frame(40, 1, 0), _frame(1, 1, 10)])),
                              "e", "m", "p")
    fast = pt.render_markdown(_result([_frame(1, 1, 0), _frame(1, 1, 10)], wall_ms=100.0),
                              pt.aggregate(_result([_frame(1, 1, 0), _frame(1, 1, 10)])),
                              "e", "m", "p")
    assert "Conclusion: 50.00 ms per token over 2 token(s)" in slow
    assert "40% of the run" in slow
    assert "1% of the run" in fast
