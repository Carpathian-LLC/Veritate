# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Forgetting --loss_mask on a ChatML corpus fails SILENTLY: the run looks healthy,
#   loss falls, and the model comes out fluent but unable to answer a question,
#   because loss was computed over the user's turns too. It cost a full SFT run.
# - The trainer therefore refuses to start on a ChatML-dense corpus until the caller
#   decides. These tests pin BOTH directions: it must block an SFT corpus, and it
#   must NOT block a pretrain corpus that merely contains some chat data.
# tests/training/test_loss_mask_guard.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import pytest

TRAINER_COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "trainers", "common")
if TRAINER_COMMON not in sys.path:
    sys.path.insert(0, TRAINER_COMMON)

import vanilla_trainer as vt

# ------------------------------------------------------------------------------------
# Constants

TURN = b"<|im_start|>user\nhi<|im_end|><|im_start|>assistant\nhello<|im_end|>"
PROSE = b"The orchard road floods every spring and nobody has fixed the culvert. " * 4

# ------------------------------------------------------------------------------------
# Functions


class _Args:
    def __init__(self, loss_mask="off", corpus="c"):
        self.loss_mask = loss_mask
        self.corpus = corpus


def _corpus(tmp_path, name, payload, size_bytes=400_000):
    p = os.path.join(tmp_path, f"{name}_train.bin")
    with open(p, "wb") as f:
        f.write((payload * (size_bytes // len(payload) + 1))[:size_bytes])
    return [(name, None, p, 1.0)]


def _mixed(tmp_path, name, chat_share, size_bytes=400_000):
    """A pretrain-shaped corpus: mostly prose with a chat dose folded in."""
    p = os.path.join(tmp_path, f"{name}_train.bin")
    chat_bytes = int(size_bytes * chat_share)
    body = ((TURN * (chat_bytes // len(TURN) + 1))[:chat_bytes]
            + (PROSE * (size_bytes // len(PROSE) + 1))[:size_bytes - chat_bytes])
    with open(p, "wb") as f:
        f.write(body)
    return [(name, None, p, 1.0)]


def test_sft_corpus_without_loss_mask_is_refused(tmp_path):
    mix = _corpus(tmp_path, "sft", TURN)
    with pytest.raises(SystemExit, match="loss_mask"):
        vt.require_loss_mask_decision(_Args(corpus="sft"), mix, ["trainer.py", "--corpus", "sft"])


def test_the_refusal_names_the_fix(tmp_path):
    """The message has to tell the caller what to pass, or it just blocks them."""
    mix = _corpus(tmp_path, "sft", TURN)
    with pytest.raises(SystemExit) as e:
        vt.require_loss_mask_decision(_Args(corpus="sft"), mix, ["trainer.py"])
    msg = str(e.value)
    assert "--loss_mask assistant" in msg
    assert "--loss_mask off" in msg


def test_explicit_off_is_allowed(tmp_path):
    """Training every byte on purpose is legitimate; it just has to be a decision."""
    mix = _corpus(tmp_path, "sft", TURN)
    vt.require_loss_mask_decision(_Args(corpus="sft"), mix, ["trainer.py", "--loss_mask", "off"])


def test_explicit_off_via_equals_form_is_allowed(tmp_path):
    mix = _corpus(tmp_path, "sft", TURN)
    vt.require_loss_mask_decision(_Args(corpus="sft"), mix, ["trainer.py", "--loss_mask=off"])


def test_assistant_masking_is_allowed(tmp_path):
    mix = _corpus(tmp_path, "sft", TURN)
    vt.require_loss_mask_decision(_Args(loss_mask="assistant", corpus="sft"), mix, ["trainer.py"])


def test_pure_prose_corpus_is_not_blocked(tmp_path):
    mix = _corpus(tmp_path, "web", PROSE)
    vt.require_loss_mask_decision(_Args(corpus="web"), mix, ["trainer.py"])


def test_pretrain_corpus_with_a_small_chat_dose_is_not_blocked(tmp_path):
    """veritate_v1 is 1.6% ChatML. Blocking it would have stopped a live pretrain."""
    mix = _mixed(tmp_path, "pretrain", chat_share=0.02)
    vt.require_loss_mask_decision(_Args(corpus="pretrain"), mix, ["trainer.py"])


def test_density_is_measured_not_assumed(tmp_path):
    chat = _corpus(tmp_path, "allchat", TURN)[0][2]
    prose = _corpus(tmp_path, "allprose", PROSE)[0][2]
    assert vt.chatml_density(chat) == 1.0
    assert vt.chatml_density(prose) == 0.0


def test_a_missing_corpus_file_does_not_crash_the_guard(tmp_path):
    """The guard must never be the thing that breaks a launch."""
    assert vt.chatml_density(os.path.join(tmp_path, "nope_train.bin")) == 0.0
