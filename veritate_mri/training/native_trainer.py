# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Native trainer for canonical Veritate. Runs scratch + continue flows without
#   a plugin: the dashboard's TRAINER_SCHEMA supplies every knob, this file
#   consumes the same CLI surface a plugin would. Refine = continue with a
#   different corpus stem; distill is opt-in via --teacher.
# - One model class only: veritate_core.model.Veritate (rule 11a). Tools / sizes
#   come from the size preset table in the dashboard; this file accepts whichever
#   hidden/layers/ffn/heads pair the form sent and trusts them.
# - Save discipline (rule 21): every checkpoint goes through save.save(),
#   every CSV row through save.append_train_row().
# veritate_mri/training/native_trainer.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# Bootstrap path: plugin runner spawns this as a subprocess with cwd=repo root.
# Insert <repo>/veritate_mri so `from readers/runtime/training/...` resolve, and
# <repo> so `from veritate_core import ...` resolves.
_HERE     = os.path.dirname(os.path.abspath(__file__))
_MRI_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_REPO     = os.path.normpath(os.path.join(_MRI_ROOT, ".."))
for _p in (_REPO, _MRI_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from readers import paths as paths_mod, models as models_mod   # noqa: E402
from training import save as save_mod                          # noqa: E402
from veritate_core import model as veritate_model              # noqa: E402
from veritate_core import qat as veritate_qat                  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

VOCAB_BYTE_LEVEL  = veritate_model.VOCAB_BYTE_LEVEL
PRECISIONS        = ("fp32", "bf16")
LR_SCHEDULES      = ("cosine", "linear", "constant", "wsd")
WSD_DECAY_KINDS   = ("sqrt", "linear", "cosine")
QAT_MODES         = ("int8", "int4", "ternary")

CKPT_PREFIX       = "step_"
CKPT_SUFFIX       = ".pt"

# Size catalog. Used as a fallback when the form / CLI passes only --size; the
# dashboard's per-trainer manifest.sizes blocks are the primary source. Native
# trainer (no plugin manifest loaded) needs a self-contained table.
SIZE_PRESETS = {
    "5m":   dict(hidden=256,  layers=6,  ffn=1024,  heads=4),
    "7m":   dict(hidden=256,  layers=8,  ffn=1024,  heads=4),
    "10m":  dict(hidden=320,  layers=8,  ffn=1280,  heads=8),
    "20m":  dict(hidden=512,  layers=8,  ffn=2048,  heads=8),
    "30m":  dict(hidden=512,  layers=10, ffn=2048,  heads=8),
    "50m":  dict(hidden=640,  layers=10, ffn=2560,  heads=10),
    "70m":  dict(hidden=640,  layers=12, ffn=2560,  heads=10),
    "80m":  dict(hidden=768,  layers=12, ffn=3072,  heads=12),
    "85m":  dict(hidden=768,  layers=12, ffn=3072,  heads=12),
    "120m": dict(hidden=896,  layers=12, ffn=3584,  heads=14),
    "160m": dict(hidden=1024, layers=12, ffn=4096,  heads=16),
    "200m": dict(hidden=1024, layers=16, ffn=4096,  heads=16),
    "350m": dict(hidden=1024, layers=24, ffn=4096,  heads=16),
    "400m": dict(hidden=1280, layers=24, ffn=5120,  heads=20),
    "800m": dict(hidden=1536, layers=28, ffn=6144,  heads=24),
    "1b3":  dict(hidden=2048, layers=24, ffn=8192,  heads=16),
    "2b":   dict(hidden=2560, layers=24, ffn=10240, heads=20),
    "3b":   dict(hidden=2560, layers=32, ffn=10240, heads=32),
}

# ------------------------------------------------------------------------------------
# Functions

def _pick_device(requested):
    # CLI --device wins. When --device=auto, consult VERITATE_DEVICE env var
    # (set by dashboard's Device preference) before falling through to auto-detect.
    if requested == "auto":
        forced = (os.environ.get("VERITATE_DEVICE") or "auto").strip().lower()
        if forced in ("cuda", "mps", "cpu"):
            requested = forced
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return "cuda"
    if requested == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and paths.current_arch() == paths.ARCH_ARM64):
            raise RuntimeError("MPS requested but unavailable")
        return "mps"
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and paths.current_arch() == paths.ARCH_ARM64:
        return "mps"
    return "cpu"


def _lr_at(step, total, warmup, base_lr, min_lr, schedule, wsd_decay_frac, wsd_decay_kind):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    if schedule == "constant":
        return base_lr
    if schedule == "linear":
        return base_lr + (min_lr - base_lr) * p
    if schedule == "wsd":
        decay_frac = max(1e-6, min(1.0, float(wsd_decay_frac)))
        stable_p = 1.0 - decay_frac
        if p <= stable_p:
            return base_lr
        q = (p - stable_p) / decay_frac
        if wsd_decay_kind == "linear":
            shape = 1.0 - q
        elif wsd_decay_kind == "cosine":
            shape = 0.5 * (1.0 + math.cos(math.pi * q))
        else:
            shape = 1.0 - math.sqrt(q)
        return min_lr + (base_lr - min_lr) * shape
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def _resolve_shape(args):
    """Fill hidden/layers/ffn/heads from the size preset if not supplied."""
    preset = SIZE_PRESETS.get(args.size)
    if preset is None and args.hidden and args.layers and args.ffn and args.heads:
        return  # form supplied a custom shape; trust it
    if preset is None:
        raise ValueError(
            f"unknown size '{args.size}' and no explicit hidden/layers/ffn/heads supplied. "
            f"Known sizes: {sorted(SIZE_PRESETS)}"
        )
    if not args.hidden: args.hidden = preset["hidden"]
    if not args.layers: args.layers = preset["layers"]
    if not args.ffn:    args.ffn    = preset["ffn"]
    if not args.heads:  args.heads  = preset["heads"]


def _make_loader(bin_path, seq_len, batch_size, seed):
    arr = np.memmap(bin_path, dtype=np.uint8, mode="r")
    n = len(arr)
    if n < seq_len + 2:
        raise ValueError(f"corpus too small: {n} bytes < {seq_len + 2}")
    rng = np.random.RandomState(seed)

    def draw():
        starts = rng.randint(0, n - seq_len - 1, size=batch_size, dtype=np.int64)
        toks = np.empty((batch_size, seq_len), dtype=np.int64)
        tgts = np.empty((batch_size, seq_len), dtype=np.int64)
        for b, s in enumerate(starts):
            toks[b] = arr[s:s + seq_len]
            tgts[b] = arr[s + 1:s + 1 + seq_len]
        return torch.from_numpy(toks), torch.from_numpy(tgts)

    return draw, n


def _latest_step(model_dir):
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"no checkpoints/ under {model_dir}")
    steps = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith(CKPT_PREFIX) and fn.endswith(CKPT_SUFFIX):
            try:
                steps.append(int(fn[len(CKPT_PREFIX):-len(CKPT_SUFFIX)]))
            except ValueError:
                continue
    if not steps:
        raise FileNotFoundError(f"no step_*.pt in {ckpt_dir}")
    return max(steps)


def _resolve_output_dir(args):
    """Path precedence: --resume wins, then --name + size composes a fresh slug,
    else error. Sets args.output_dir and returns the model dir basename."""
    if args.resume and args.resume.strip():
        args.output_dir = os.path.join(paths_mod.MODELS_ROOT, args.resume.strip())
    elif args.name and args.name.strip():
        composed = save_mod.compose_name(args.name, args.size)
        args.output_dir = os.path.join(paths_mod.MODELS_ROOT, composed)
    else:
        raise ValueError("native_trainer needs either --name (scratch) or --resume (continue)")
    os.makedirs(args.output_dir, exist_ok=True)
    return os.path.basename(os.path.normpath(args.output_dir))


def _resolve_corpus(args):
    """Map --corpus stem to (train_path, val_path). Continue flow may leave
    `corpus` blank, in that case the source model's `config.json` carries the
    original corpus stem and we restore it (matches the plugin contract:
    "leave blank to keep the original corpus this model was trained on")."""
    if args.resume and not args.corpus:
        from readers import config as cfg_reader
        cfg = cfg_reader.load(args.resume.strip()) or {}
        ta = cfg.get("training_args") or {}
        original = ta.get("corpus") or cfg.get("corpus")
        if original:
            args.corpus = str(original)
    if args.corpus and not args.corpus_bin:
        train, val = save_mod.resolve_corpus(args.corpus)
        args.corpus_bin = train
        args.val_bin    = val or ""
    if not args.corpus_bin or not os.path.isfile(args.corpus_bin):
        raise FileNotFoundError(
            f"corpus train .bin not found. Pick a corpus from the form, or"
            f" set --corpus_bin to a .bin path. Got corpus_bin={args.corpus_bin!r}"
        )


def _build_model(args):
    model = veritate_model.Veritate(
        vocab=args.vocab, hidden=args.hidden, layers=args.layers,
        ffn=args.ffn, heads=args.heads, seq=args.seq,
    )
    if args.qat_enabled:
        veritate_qat.set_qat(model, True)
        if args.quant_mode and args.quant_mode in QAT_MODES:
            for m in model.modules():
                if isinstance(m, veritate_model.QuantLinear):
                    m.quant_mode = args.quant_mode
    return model


def _maybe_resume(model, opt, args, device):
    if not args.resume:
        return 0
    last = _latest_step(args.output_dir)
    ckpt_path = os.path.join(args.output_dir, "checkpoints", f"step_{last}{CKPT_SUFFIX}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", last))


def _save_args_for_config(args, model):
    """Snapshot of args + shape that save.save() will record into config.json
    on first call. Subsequent calls keep the original record (it is sticky)."""
    out = vars(args).copy()
    out["vocab"]   = model.vocab
    out["hidden"]  = model.hidden
    out["layers"]  = model.layers
    out["ffn"]     = model.ffn
    out["heads"]   = model.heads
    out["seq"]     = model.seq
    if args.qat_enabled:
        out["training"] = "qat"
    if not args.description or not str(args.description).strip():
        parts = []
        for k in ("corpus", "size", "precision", "version", "variant"):
            v = getattr(args, k, "")
            if v: parts.append(f"{k}={v}")
        out["description"] = " ".join(parts) or "native veritate trainer"
    return out


# ------------------------------------------------------------------------------------
# Args

def _parse_args():
    ap = argparse.ArgumentParser(description="Native canonical-Veritate trainer (no plugin needed).")
    # path / identity
    ap.add_argument("--name",        type=str, default="")
    ap.add_argument("--resume",      type=str, default="")
    ap.add_argument("--description", type=str, default="")
    ap.add_argument("--version",     type=str, default="")
    ap.add_argument("--variant",     type=str, default="")
    ap.add_argument("--device",      type=str, default="auto", choices=("auto","cpu","cuda","mps"))
    # corpus
    ap.add_argument("--corpus",      type=str, default="")
    ap.add_argument("--corpus_bin",  type=str, default="")
    ap.add_argument("--val_bin",     type=str, default="")
    # shape
    ap.add_argument("--size",        type=str, default="85m")
    ap.add_argument("--vocab",       type=int, default=VOCAB_BYTE_LEVEL)
    ap.add_argument("--hidden",      type=int, default=0)
    ap.add_argument("--layers",      type=int, default=0)
    ap.add_argument("--ffn",         type=int, default=0)
    ap.add_argument("--heads",       type=int, default=0)
    ap.add_argument("--seq",         type=int, default=1024)
    # training loop
    ap.add_argument("--precision",   type=str, default="bf16", choices=PRECISIONS)
    ap.add_argument("--total_steps", type=int, default=20000)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--n_chunks",    type=int, default=1)
    ap.add_argument("--bptt_window", type=int, default=1)
    ap.add_argument("--base_lr",     type=float, default=3e-4)
    ap.add_argument("--min_lr",      type=float, default=3e-6)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--lr_schedule", type=str, default="wsd", choices=LR_SCHEDULES)
    ap.add_argument("--wsd_decay_frac", type=float, default=0.1)
    ap.add_argument("--wsd_decay_kind", type=str, default="sqrt", choices=WSD_DECAY_KINDS)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1",       type=float, default=0.9)
    ap.add_argument("--beta2",       type=float, default=0.95)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--grad_clip",   type=float, default=1.0)
    # cadence
    ap.add_argument("--ckpt_every",  type=int, default=500)
    ap.add_argument("--log_every",   type=int, default=50)
    ap.add_argument("--eval_every",  type=int, default=500)
    ap.add_argument("--eval_iters",  type=int, default=16)
    ap.add_argument("--seed",        type=int, default=0)
    # toggles
    ap.add_argument("--use_act_ckpt", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--qat_enabled",  action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--quant_mode", type=str, default="int8", choices=QAT_MODES)
    # The dashboard form renders the FULL TRAINER_SCHEMA for every trainer (the
    # schema is the source of truth, see trainers/readme.md). The native
    # trainer ignores knobs it does not implement (MTP heads, M3 adapter, MoE
    # router, freeze_base, etc.); parse_known_args() silently drops them so a
    # user toggling, say, `freeze_base` on the form does not crash this run.
    args, _unused = ap.parse_known_args()
    return args


# ------------------------------------------------------------------------------------
# Main

def main():
    args = _parse_args()
    _resolve_shape(args)
    _resolve_corpus(args)
    name = _resolve_output_dir(args)

    device_type = _pick_device(args.device)
    device = torch.device(device_type)
    amp_dtype = torch.bfloat16 if (args.precision == "bf16" and device_type == "cuda") else None

    print(f"[native] device={device_type} amp={amp_dtype}", flush=True)
    print(f"[native] output={args.output_dir}", flush=True)
    print(f"[native] shape h={args.hidden} L={args.layers} ffn={args.ffn} heads={args.heads} seq={args.seq}", flush=True)
    print(f"[native] corpus_bin={args.corpus_bin}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = _build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[native] params: {n_params:,} ({n_params/1e6:.1f}M)", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr, betas=(args.beta1, args.beta2),
        eps=1e-6, weight_decay=args.weight_decay, foreach=True,
    )
    start_step = _maybe_resume(model, opt, args, device)
    if start_step:
        print(f"[native] resumed from step {start_step}", flush=True)
    if start_step >= args.total_steps:
        print(f"[native] start_step ({start_step}) >= total_steps ({args.total_steps}). "
              f"Nothing to do, bump --total_steps to continue.", flush=True)
        return

    train_draw, n_train = _make_loader(args.corpus_bin, args.seq, args.batch_size, args.seed)
    val_draw, n_val = (None, 0)
    if args.val_bin and os.path.isfile(args.val_bin):
        val_draw, n_val = _make_loader(args.val_bin, args.seq, args.batch_size, args.seed + 1)
    print(f"[native] train_bytes={n_train:,} val_bytes={n_val:,}", flush=True)

    model.train()
    t0 = time.time()
    last_log = t0
    buf_loss, buf_n = 0.0, 0

    for step in range(start_step + 1, args.total_steps + 1):
        lr = _lr_at(step, args.total_steps, args.warmup_steps, args.base_lr, args.min_lr,
                    args.lr_schedule, args.wsd_decay_frac, args.wsd_decay_kind)
        for g in opt.param_groups:
            g["lr"] = lr

        toks, tgts = train_draw()
        toks, tgts = toks.to(device), tgts.to(device)
        opt.zero_grad(set_to_none=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                _, loss = model(toks, tgts)
        else:
            _, loss = model(toks, tgts)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[native] step {step}: NaN/Inf loss, skipping", flush=True)
            continue

        loss.backward()
        gnorm = None
        if args.grad_clip > 0:
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        opt.step()

        buf_loss += float(loss.detach())
        buf_n += 1

        if step % args.log_every == 0:
            mean_ce = buf_loss / max(1, buf_n)
            now = time.time()
            tok_s = (args.log_every * args.batch_size * args.seq) / max(1e-6, now - last_log)
            save_mod.append_train_row(name, step, "train", mean_ce, lr=lr,
                                      grad_norm=gnorm, tok_per_s=tok_s,
                                      wall_s=now - t0, seed=args.seed)
            print(f"[native] step {step:>6} ce={mean_ce:.4f} lr={lr:.2e} tok/s={tok_s:.0f} t={now-t0:.0f}s", flush=True)
            buf_loss, buf_n = 0.0, 0
            last_log = now

        if val_draw is not None and (step % args.eval_every == 0 or step == args.total_steps):
            model.eval()
            with torch.no_grad():
                vsum, vn = 0.0, 0
                for _ in range(args.eval_iters):
                    vtoks, vtgts = val_draw()
                    vtoks, vtgts = vtoks.to(device), vtgts.to(device)
                    if amp_dtype is not None:
                        with torch.autocast(device_type=device_type, dtype=amp_dtype):
                            _, vloss = model(vtoks, vtgts)
                    else:
                        _, vloss = model(vtoks, vtgts)
                    if not (torch.isnan(vloss) or torch.isinf(vloss)):
                        vsum += float(vloss); vn += 1
                vmean = vsum / max(1, vn)
            model.train()
            save_mod.append_train_row(name, step, "val", vmean, lr=lr,
                                      wall_s=time.time() - t0, seed=args.seed)
            print(f"[native] step {step:>6} val={vmean:.4f}", flush=True)

        if step % args.ckpt_every == 0 or step == args.total_steps:
            ckpt_args = _save_args_for_config(args, model)
            path = save_mod.save(model, name, step, optimizer=opt, args=ckpt_args)
            print(f"[native] checkpoint + hooks: {path}", flush=True)

    print(f"[native] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
