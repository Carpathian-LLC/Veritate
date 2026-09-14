# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - run_suites_on_model is the dashboard's deep-eval entry: suite names are normalised,
#   unknown ones ignored, each runner gets its data path (override or default), its
#   limit, and a progress adapter that prefixes the suite name. The runners are stubbed.
# tests/mri/test_run_eval.py
# ------------------------------------------------------------------------------------
# Imports:

from eval import hellaswag, ifeval, mmlu, run_eval

# ------------------------------------------------------------------------------------
# Functions


def test_suites_are_dispatched_with_their_paths_limits_and_a_prefixed_progress(monkeypatch):
    calls = {}

    def fake(suite):
        def run(model, data_path, limit=None, verbose=False, progress_cb=None, **kw):
            calls[suite] = {"data": data_path, "limit": limit, "kw": kw}
            progress_cb(1, 2, {"gold": 0})
            return {"suite": suite, "n": 2}
        return run
    monkeypatch.setattr(mmlu, "run_mmlu", fake("mmlu"))
    monkeypatch.setattr(hellaswag, "run_hellaswag", fake("hellaswag"))
    monkeypatch.setattr(ifeval, "run_ifeval", fake("ifeval"))
    seen = []
    out = run_eval.run_suites_on_model("model", [" MMLU ", "hellaswag", "nope", ""], limit=5, mmlu_mode="both",
                                       hellaswag_data="/hs.json", progress_cb=lambda s, i, n: seen.append((s, i, n)))
    assert set(out["suites"]) == {"mmlu", "hellaswag"}
    assert calls["mmlu"]["data"] == mmlu.DEFAULT_DATA and calls["mmlu"]["kw"]["mode"] == "both"
    assert calls["hellaswag"]["data"] == "/hs.json" and calls["hellaswag"]["limit"] == 5
    assert seen == [("mmlu", 1, 2), ("hellaswag", 1, 2)]
    assert run_eval.run_suites_on_model("model", []) == {"suites": {}}


def test_ifeval_receives_its_generation_settings(monkeypatch):
    calls = {}
    monkeypatch.setattr(ifeval, "run_ifeval",
                        lambda model, data_path, max_new, limit, verbose, progress_cb, chat: calls.update(
                            data=data_path, max_new=max_new, chat=chat) or {"suite": "ifeval"})
    run_eval.run_suites_on_model("model", ["ifeval"], ifeval_max_new=64, ifeval_chat=True)
    assert calls == {"data": ifeval.DEFAULT_DATA, "max_new": 64, "chat": True}
