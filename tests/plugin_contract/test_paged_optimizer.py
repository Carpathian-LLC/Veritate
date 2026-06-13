# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - PagedAdamW must be numerically identical to torch.optim.AdamW (its state just
#   lives on disk). These pin update parity over many steps, the tiny state_dict
#   contract, resume-from-disk, and temp-dir cleanup.
# tests/plugin_contract/test_paged_optimizer.py
# ------------------------------------------------------------------------------------
# Imports

import os

import torch

from veritate_core.plugin import paged_optimizer as po

# ------------------------------------------------------------------------------------
# Constants

LR       = 3e-4
BETAS    = (0.9, 0.95)
EPS      = 1e-6
WD       = 0.1
STEPS    = 25
SHAPES   = ((64, 32), (128,), (16, 8, 4))
TOL      = 1e-6

# ------------------------------------------------------------------------------------
# Functions


def _params(seed):
    torch.manual_seed(seed)
    return [torch.randn(*s, requires_grad=True) for s in SHAPES]


def _set_grads(params, step):
    for i, p in enumerate(params):
        torch.manual_seed(1000 + step * 17 + i)
        p.grad = torch.randn_like(p)


def test_parity_with_torch_adamw(tmp_path):
    """PagedAdamW matches torch.optim.AdamW param-for-param over many steps."""
    ref_p = _params(0)
    pag_p = _params(0)
    ref = torch.optim.AdamW(ref_p, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    pag = po.PagedAdamW(pag_p, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD,
                        state_dir=str(tmp_path / "state"))
    for step in range(STEPS):
        _set_grads(ref_p, step)
        _set_grads(pag_p, step)
        ref.step()
        pag.step()
    for a, b in zip(ref_p, pag_p):
        assert torch.allclose(a, b, atol=TOL, rtol=TOL), (a - b).abs().max().item()


def test_state_lives_on_disk_not_in_state_dict(tmp_path):
    """state_dict carries step counts + dir, never the moment buffers."""
    p = _params(0)
    opt = po.PagedAdamW(p, lr=LR, state_dir=str(tmp_path / "s"))
    _set_grads(p, 0)
    opt.step()
    sd = opt.state_dict()
    assert sd["steps"] == [1, 1, 1]
    assert sd["state_dir"] == str(tmp_path / "s")
    assert "exp_avg" not in sd and "state" not in sd
    assert os.path.isfile(tmp_path / "s" / f"{po.M_PREFIX}0{po.STATE_SUFFIX}")


def test_resume_rebinds_disk_state(tmp_path):
    """Stopping after N steps and resuming from the saved dir matches a straight run."""
    ref_p = _params(0)
    ref = po.PagedAdamW(ref_p, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD,
                        state_dir=str(tmp_path / "ref"))
    for step in range(8):
        _set_grads(ref_p, step)
        ref.step()

    run_p = _params(0)
    first = po.PagedAdamW(run_p, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD,
                          state_dir=str(tmp_path / "run"))
    for step in range(5):
        _set_grads(run_p, step)
        first.step()
    sd = first.state_dict()
    del first

    second = po.PagedAdamW(run_p, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD,
                           state_dir=str(tmp_path / "run"))
    second.load_state_dict(sd)
    for step in range(5, 8):
        _set_grads(run_p, step)
        second.step()

    for a, b in zip(ref_p, run_p):
        assert torch.allclose(a, b, atol=TOL, rtol=TOL), (a - b).abs().max().item()


def test_temp_dir_cleaned_on_close():
    """An optimizer with no state_dir owns its temp dir and removes it on close."""
    p = _params(0)
    opt = po.PagedAdamW(p, lr=LR)
    d = opt._state_dir
    assert os.path.isdir(d)
    opt.close()
    assert not os.path.isdir(d)
