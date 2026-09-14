# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - PagedAdamW keeps its Adam moments in mmap-backed files. Pins: the update equals
#   torch's decoupled AdamW step for step, a resume from the same state_dir reads the
#   moments back (no warm restart), a throwaway run removes its scratch dir, and the
#   argument checks refuse a degenerate optimizer at construction.
# tests/training/test_paged_optimizer.py
# ------------------------------------------------------------------------------------
# Imports:

import gc
import os

import pytest
import torch

from veritate_core.plugin import paged_optimizer as po

# ------------------------------------------------------------------------------------
# Functions


def _pair(seed=0):
    torch.manual_seed(seed)
    w = torch.randn(6, 4)
    return torch.nn.Parameter(w.clone()), torch.nn.Parameter(w.clone())


def test_the_update_matches_torch_adamw_step_for_step(tmp_path):
    """Same weights, same grads: the paged step and torch.optim.AdamW agree within fp32 noise."""
    a, b = _pair()
    paged = po.PagedAdamW([a], lr=1e-2, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.1, state_dir=str(tmp_path))
    ref = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.1)
    for t in range(5):
        torch.manual_seed(100 + t)
        g = torch.randn(6, 4)
        a.grad, b.grad = g.clone(), g.clone()
        paged.step()
        ref.step()
        assert torch.allclose(a, b, atol=1e-6), t
    paged.close()


def test_moments_survive_a_reopen_of_the_same_state_dir(tmp_path):
    """A resume binds the files the first run wrote; a fresh state_dir starts from zero."""
    a, _ = _pair()
    first = po.PagedAdamW([a], lr=1e-2, state_dir=str(tmp_path))
    a.grad = torch.ones(6, 4)
    first.step()
    saved = first.state_dict()
    assert saved["steps"] == [1] and set(os.listdir(tmp_path)) == {"exp_avg_0.bin", "exp_avg_sq_0.bin"}
    again = po.PagedAdamW([a], lr=1e-2, state_dir=str(tmp_path))
    again.load_state_dict(saved)
    st = again.state[a]
    assert st["step"] == 1
    assert torch.allclose(st["exp_avg"], torch.full((24,), 0.1))
    assert torch.allclose(st["exp_avg_sq"], torch.full((24,), 0.05))
    fresh = po.PagedAdamW([a], lr=1e-2, state_dir=str(tmp_path / "other"))
    assert float(fresh.state[a]["exp_avg"].abs().sum()) == 0.0
    fresh.close()


def test_a_throwaway_run_removes_its_scratch_dir_and_a_kept_one_stays(tmp_path):
    a, _ = _pair()
    scratch = po.PagedAdamW([a], lr=1e-2)
    d = scratch._state_dir
    assert os.path.isdir(d) and os.path.basename(d).startswith(po.SCRATCH_PREFIX)
    scratch.close()
    assert not os.path.exists(d)
    kept = po.PagedAdamW([a], lr=1e-2, state_dir=str(tmp_path))
    kept.close()
    assert os.path.isdir(tmp_path)


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.parametrize("kw", [{"lr": 0.0}, {"lr": 1e-3, "betas": (1.0, 0.9)}, {"lr": 1e-3, "eps": 0.0}])
def test_degenerate_arguments_are_refused_without_a_noisy_destructor(kw):
    """The refused object is still finalized; close() must not raise on the half-built instance."""
    a, _ = _pair()
    with pytest.raises(ValueError):
        po.PagedAdamW([a], **kw)
    gc.collect()
