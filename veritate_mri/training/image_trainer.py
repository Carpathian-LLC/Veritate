# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The canonical image trainer (IDEA 24). Pictures in, a model out, one launch: it
#   owns the whole image pipeline the text trainer cannot -- decode a sample of the set
#   into the pixel cache, fit the codec if none is named, stream the whole set into a
#   corpus if it is missing or stale, then train the masked-grid model. The dashboard's
#   Images flow needs one field filled in: which set of pictures.
# - Every stage reports to models/<name>/progress.json (image_progress) from the first
#   second, so the Training tab shows what the run is doing long before train.csv has
#   a row. The device is in that file too: the codec fit, the corpus encode and the
#   training loop run on the GPU (mps/cuda); only JPEG decoding is CPU work, by nature.
# - Stop is a request, not a kill. SIGTERM (the dashboard's Stop button) sets a flag;
#   a stage in progress raises StopRequested and the training loop saves a checkpoint
#   before it returns. A stopped run exits 0 with its last step on disk.
# - What is shared with the text trainer is shared by import, not by copy: the flag
#   parser and its unknown-flag policy, size shapes, the lr schedule, the optimizer
#   builder, config.json, resume, evaluation, the checkpoint/save contract and the
#   train.csv rows the dashboard plots. What is different is owned here: geometry,
#   the codec, the corpus, and a loop with one objective and no chunking.
# - Fixed by what an image model is, not knobs: trunk=dense (the patched trunks gather
#   on text byte boundaries a code stream lacks), objective=masked_grid, causal=False,
#   hooks off by default (the checkpoint probes are text probes).
# - seq is derived, not typed: image_code_bytes + caption_bytes, rounded up. A seq
#   smaller than the image would train on half a picture and report a loss anyway.
#   A frame above MAX_EDGE is refused: 1920x1080 is 20,736 code bytes per picture and
#   an 869 GB cache over a phone library; the model works on a fixed small frame.
# - The manifest the dashboard renders and the flags this process parses are the same
#   dict (readers/trainers.IMAGE_TRAINER_MANIFEST), so the form and the parser cannot
#   drift. Launch is through POST /trainers/run, like every trainer (rule 13).
# veritate_mri/training/image_trainer.py
# ------------------------------------------------------------------------------------
# Imports

import json
import os
import signal
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_MRI_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _MRI_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import veritate_trainer as vt
from readers import paths
from readers import trainers as trainers_reader
from tools import build_image_corpus, fit_image_codec

from veritate_core.model import Veritate
from veritate_core.plugin import hardware, image_codec, image_grid, image_probe, image_progress, save
from veritate_core.plugin.image_progress import StopRequested

# ------------------------------------------------------------------------------------
# Constants

SEQ_MULTIPLE       = 64
MAX_EDGE           = 1024
CODEC_NAME_SUFFIX  = "_codec"
CORPUS_STEM_SUFFIX = "_img"
IMAGE_EXTS         = fit_image_codec.IMAGE_EXTS
STOP_SIGNALS       = ("SIGTERM", "SIGINT")

_STOP = {"requested": False}

# ------------------------------------------------------------------------------------
# Functions


def request_stop(*_a):
    """Set by the signal handlers; polled by every stage and the training loop."""
    _STOP["requested"] = True


def stop_requested():
    return _STOP["requested"]


def _install_stop_handlers():
    """Returns the previous handlers so run() can restore them (tests run in-process)."""
    _STOP["requested"] = False
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous = {}
    for sig_name in STOP_SIGNALS:
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            previous[sig] = signal.signal(sig, request_stop)
        except (ValueError, OSError):
            pass
    return previous


def _restore_handlers(previous):
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError, TypeError):
            pass


def round_up(n, multiple):
    return ((int(n) + multiple - 1) // multiple) * multiple


def check_geometry(height, width, patch, planes):
    """The number the whole run hangs on: bytes of encoded image per record."""
    if patch <= 0 or height <= 0 or width <= 0:
        raise ValueError("height, width and patch must be positive")
    if height > MAX_EDGE or width > MAX_EDGE:
        raise ValueError("picture size " + str(height) + "x" + str(width) + " is above the "
                         + str(MAX_EDGE) + " px limit; the model works on a fixed small frame "
                         "(320 is the default) and every photo is scaled to it")
    if height % patch or width % patch:
        raise ValueError("patch " + str(patch) + " does not divide " + str(height) + "x" + str(width))
    if planes < 1:
        raise ValueError("planes must be at least 1")
    return int(planes) * (height // patch) * (width // patch)


def resolve_seq(seq, code_bytes, caption_bytes):
    """0 means derive: the image plus a caption budget, rounded up. An explicit seq
    smaller than the image is refused, not clipped."""
    seq = int(seq or 0)
    if seq <= 0:
        seq = round_up(code_bytes + max(0, int(caption_bytes or 0)), SEQ_MULTIPLE)
    if seq < code_bytes:
        raise ValueError("seq " + str(seq) + " is smaller than image_code_bytes " + str(code_bytes)
                         + "; the objective cannot see a whole image. Use 0 to derive it.")
    return seq


def count_set_images(set_name):
    set_dir = paths.image_set_dir(set_name)
    if not os.path.isdir(set_dir):
        raise ValueError("no image set at " + set_dir + " -- add photos to a set first")
    n = sum(1 for f in os.listdir(set_dir)
            if not f.startswith(".") and f.lower().endswith(IMAGE_EXTS))
    if n == 0:
        raise ValueError("image set '" + set_name + "' holds no pictures")
    return n


def count_set_captions(set_name):
    """Captions are part of what a corpus holds: a captioning pass after a build must
    make the next launch rebuild, or the words never reach the model."""
    set_dir = paths.image_set_dir(set_name)
    return sum(1 for f in os.listdir(set_dir)
               if not f.startswith(".") and f.lower().endswith(".txt"))


def ensure_codec(args, codec_name, device, prog):
    """Stage 1. Load the named codec, or fit one on a sample of the set. Returns
    (codec, report); report is None when an existing codec was loaded."""
    path = paths.codec_path(codec_name)
    if os.path.isfile(path):
        codec = image_codec.load(path)
        if codec.patch != int(args.patch) or codec.planes != int(args.planes):
            raise ValueError("codec '" + codec_name + "' has patch " + str(codec.patch)
                             + " planes " + str(codec.planes) + " but the run asks for patch "
                             + str(args.patch) + " planes " + str(args.planes)
                             + "; a corpus is unreadable under a different codec")
        print("codec: " + codec_name + " (existing, " + path + ")", flush=True)
        prog.skip("decode", "codec " + codec_name + " already fitted")
        prog.skip("codec", "codec " + codec_name + " reused")
        return codec, None
    sample = int(args.codec_images or 0)
    print("codec: fitting '" + codec_name + "' on set '" + args.image_set + "' ("
          + str(args.codec_epochs) + " epochs" + (", " + str(sample) + " pictures" if sample else "")
          + ")", flush=True)

    def progress(stage, done, total, **facts):
        if stage == "decode":
            prog.stage("decode", done, total, "decoding pictures " + format(done, ",") + " / "
                       + format(total, ","))
            return
        if prog.state["stages"]["decode"]["state"] != image_progress.STAGE_DONE:
            if prog.state["stages"]["decode"]["state"] == image_progress.STAGE_RUNNING:
                prog.done("decode")
            else:
                prog.skip("decode", "pictures already in the cache")
        msg = ("fitting codec  epoch " + str(facts.get("epoch")) + " / " + str(facts.get("epochs"))
               + ("  loss " + format(facts["loss"], ".4f") if "loss" in facts else "")
               + ("  PSNR " + format(facts["psnr"], ".1f") + " dB" if "psnr" in facts else ""))
        prog.stage("codec", done, total, msg, **facts)

    report = fit_image_codec.fit(
        args.image_set, codec_name, height=int(args.height), width=int(args.width),
        planes=int(args.planes), patch=int(args.patch), epochs=int(args.codec_epochs),
        batch_size=int(args.codec_batch_size), lr=float(args.codec_lr), device=device,
        limit=sample, seed=int(args.seed), verbose=True, progress=progress,
        should_stop=stop_requested)
    best = min(report["history"], key=lambda h: h["l1"]) if report["history"] else {}
    print("codec: fitted in " + str(report["seconds"]) + "s, held-out L1 "
          + format(best.get("l1", float("nan")), ".4f") + "  PSNR "
          + format(best.get("psnr", float("nan")), ".2f") + " dB", flush=True)
    prog.done("codec", "codec fitted: PSNR " + format(best.get("psnr", float("nan")), ".1f") + " dB",
              psnr=round(float(best.get("psnr", 0.0)), 2), images=report["images"])
    return image_codec.load(path), report


def ensure_corpus(args, stem, codec_name, device, prog):
    """Stage 2. Build <stem>_{train,val}.bin from the whole set unless bins built from
    this exact set, codec and geometry already exist. The sidecar is the record of what
    a corpus holds; without it a stale corpus is indistinguishable from a current one."""
    want = {"set": args.image_set, "codec": codec_name, "height": int(args.height),
            "width": int(args.width), "images": count_set_images(args.image_set),
            "captions": count_set_captions(args.image_set)}
    meta_path = paths.image_corpus_meta_path(stem)
    train_path = paths.corpus_train_path(stem)
    if os.path.isfile(meta_path) and os.path.isfile(train_path):
        with open(meta_path, encoding="utf-8") as handle:
            have = json.load(handle)
        if all(have.get(k) == v for k, v in want.items()):
            print("corpus: " + stem + " is current (" + str(have["images"]) + " images, "
                  + str(have["image_code_bytes"]) + " code bytes)", flush=True)
            prog.skip("encode", "corpus " + stem + " is current")
            return have
        print("corpus: " + stem + " is stale (" + ", ".join(
            k + " " + str(have.get(k)) + " -> " + str(v) for k, v in want.items()
            if have.get(k) != v) + "); rebuilding", flush=True)
    else:
        print("corpus: building " + stem, flush=True)
    report = build_image_corpus.build_streaming(
        paths, args.image_set, codec_name, stem, int(args.height), int(args.width),
        device=device, verbose=True, should_stop=stop_requested,
        progress=lambda d, t: prog.stage("encode", d, t, "encoding pictures " + format(d, ",")
                                         + " / " + format(t, ",")))
    meta = dict(want)
    meta.update({k: report[k] for k in ("image_code_bytes", "planes", "patch",
                                        "train_records", "val_records",
                                        "train_bytes", "val_bytes")})
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1)
    os.replace(meta_path + ".tmp", meta_path)
    print("corpus: " + str(meta["train_records"]) + " train / " + str(meta["val_records"])
          + " val records, " + str(meta["train_bytes"]) + " B", flush=True)
    prog.done("encode", "corpus: " + format(meta["train_records"], ",") + " train / "
              + format(meta["val_records"], ",") + " val pictures",
              records=meta["train_records"] + meta["val_records"])
    return meta


def run(plugin_id):
    manifest = trainers_reader.IMAGE_TRAINER_MANIFEST
    size_presets = vt._size_presets(manifest)
    args = vt.parse_args(manifest)
    resume_mode = bool(args.resume)
    if resume_mode:
        vt.apply_resume_overrides(args, sys.argv)
    save.require_description(args.description)
    if args.size not in size_presets:
        raise ValueError("unknown size: " + str(args.size) + " (valid: " + ", ".join(size_presets) + ")")
    if args.precision not in vt.PRECISIONS:
        raise ValueError("unknown precision: " + str(args.precision))
    if args.lr_schedule not in vt.LR_SCHEDULES:
        raise ValueError("unknown lr_schedule: " + str(args.lr_schedule))
    if not resume_mode and not (args.image_set or "").strip():
        raise ValueError("image_set is required: which set of pictures to train on")
    # Not choices. An image model is this or it is not an image model.
    args.trunk = "dense"
    args.objective = vt.OBJECTIVE_IMAGE
    args.training_kind = vt.TRAINING_KIND_IMAGE
    args.image_set = (args.image_set or "").strip()

    name = args.resume if resume_mode else save.compose_name(args.name, args.size)
    print("model name: " + name, flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = hardware.pick_device()
    amp_dtype = hardware.resolve_precision(args.precision, device)
    print("device: " + device + "  precision: "
          + (args.precision if amp_dtype is not None else "fp32 (autocast off)"), flush=True)
    code_bytes = check_geometry(int(args.height), int(args.width), int(args.patch), int(args.planes))

    previous_handlers = _install_stop_handlers()
    prog = image_progress.Progress(paths.model_dir(name), device, total_steps=int(args.total_steps))
    try:
        _run_stages(args, name, device, amp_dtype, code_bytes, size_presets, resume_mode,
                    plugin_id, prog)
    except StopRequested as e:
        prog.end(image_progress.RUN_STOPPED, str(e))
        print("stopped: " + str(e), flush=True)
    except BaseException as e:
        prog.end(image_progress.RUN_FAILED, type(e).__name__ + ": " + str(e))
        raise
    finally:
        _restore_handlers(previous_handlers)


def _run_stages(args, name, device, amp_dtype, code_bytes, size_presets, resume_mode,
                plugin_id, prog):
    codec_name = (args.codec or "").strip() or (name + CODEC_NAME_SUFFIX)
    stem = (args.corpus or "").strip() or (name + CORPUS_STEM_SUFFIX)
    print("geometry: " + str(args.height) + "x" + str(args.width) + "  patch " + str(args.patch)
          + "  planes " + str(args.planes) + "  -> " + str(code_bytes) + " code bytes/image",
          flush=True)

    ensure_codec(args, codec_name, device, prog)
    meta = ensure_corpus(args, stem, codec_name, device, prog)
    if int(meta["image_code_bytes"]) != code_bytes:
        raise ValueError("corpus reports " + str(meta["image_code_bytes"])
                         + " code bytes but the geometry says " + str(code_bytes))
    train_path = paths.corpus_train_path(stem)
    val_path = paths.corpus_val_path(stem)
    if not os.path.isfile(val_path):
        val_path = None
        print("WARNING: no val bin for " + stem + "; this run reports no validation loss", flush=True)

    seq = resolve_seq(args.seq, code_bytes, args.caption_bytes)
    args.seq = seq
    args.image_code_bytes = code_bytes
    args.codec = codec_name
    args.corpus = stem
    print("seq: " + str(seq) + " (" + str(code_bytes) + " image + " + str(seq - code_bytes)
          + " caption bytes)", flush=True)

    shape = vt.shape_for_run(args, size_presets)
    model = Veritate(vocab=vt.VOCAB_BYTE_LEVEL, hidden=shape["hidden"], layers=shape["layers"],
                     ffn=shape["ffn"], heads=shape["heads"], seq=seq, causal=False)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("params: " + str(n_params) + "  shape: hidden=" + str(shape["hidden"]) + " layers="
          + str(shape["layers"]) + " ffn=" + str(shape["ffn"]) + " heads=" + str(shape["heads"])
          + "  bidirectional", flush=True)

    resume_step, resume_opt_state = 0, None
    if resume_mode:
        resume_step = vt.latest_checkpoint_step(name)
        print("resume: " + name + "  from step " + str(resume_step), flush=True)
        resume_opt_state = vt.load_resume_state(model, name, resume_step, device, require_complete=True)
        dropped = save.truncate_train_csv_at(name, resume_step)
        if dropped:
            print("train.csv: dropped " + str(dropped) + " stale rows past step " + str(resume_step), flush=True)
    else:
        corpus_hash = save.hash_corpus(stem)
        vt.write_config(name, args, shape, n_params, corpus_hash, plugin_id=plugin_id)
        print("wrote: " + paths.config_path(name), flush=True)

    train_draw, train_n = image_grid.make_record_loader(
        train_path, seq, args.batch_size, code_bytes, image_codec.MASK_BYTE, args.seed)
    val_draw = None
    if val_path:
        val_draw, _ = image_grid.make_record_loader(
            val_path, seq, args.batch_size, code_bytes, image_codec.MASK_BYTE, args.seed + 1)
    print("objective: masked_grid  records: " + str(train_n) + "  batch: " + str(args.batch_size),
          flush=True)
    prog.note(params=n_params, records=train_n, seq=seq, image_code_bytes=code_bytes)

    opt = vt.build_optimizer(vt.trainable_params(model), args, device, model=model)
    if resume_opt_state is not None:
        try:
            opt.load_state_dict(resume_opt_state)
            print("optimizer state restored", flush=True)
        except Exception as e:
            print("optimizer state restore skipped: " + str(e), flush=True)

    def checkpoint(step):
        ckpt_args = vars(args).copy()
        ckpt_args.update({"vocab": model.vocab, "hidden": model.hidden, "layers": model.layers,
                          "ffn": model.ffn, "heads": model.heads, "seq": model.seq})
        skip_dumps, hook_label = vt.hook_plan(args, step)
        ckpt_path = save.save(model, name, step, optimizer=opt, args=ckpt_args, dump_set=skip_dumps)
        print("checkpoint + hooks(" + hook_label + "): " + ckpt_path, flush=True)
        prog.note(last_checkpoint_step=step, last_checkpoint_at=time.time())
        # The picture model's own probe: samples, fill test, per-plane accuracy,
        # attention. What the Models tab shows instead of the text probes.
        try:
            probe_codec = image_codec.load(paths.codec_path(codec_name))
            pm = image_probe.dump(model, probe_codec, {"height": int(args.height), "width": int(args.width),
                                                      "seq": seq, "code_bytes": code_bytes},
                                  name, step, val_path, device)
            print("image probe: fill acc " + format(pm.get("fill_accuracy", 0.0), ".3f")
                  + "  codes used " + str(pm.get("codes_used")) + "/" + str(image_codec.CODEBOOK_ENTRIES)
                  + "  (" + str(pm.get("seconds")) + "s)", flush=True)
            prog.note(fill_accuracy=pm.get("fill_accuracy"), codes_used=pm.get("codes_used"),
                      probe_step=step)
        except Exception as e:
            print("image probe skipped: " + type(e).__name__ + ": " + str(e), flush=True)

    t0 = time.time()
    last_log, last_log_step = t0, resume_step
    skipped = 0
    last_saved = resume_step
    last_loss = float("nan")
    prog.stage("train", resume_step, args.total_steps, "training  step " + str(resume_step) + " / "
               + format(args.total_steps, ","))
    for step in range(resume_step + 1, args.total_steps + 1):
        lr = vt.lr_at(step, args.total_steps, args.warmup_steps, args.base_lr, args.min_lr,
                      schedule=args.lr_schedule, wsd_decay_frac=args.wsd_decay_frac,
                      wsd_decay_kind=args.wsd_decay_kind)
        for g in opt.param_groups:
            g["lr"] = lr
        toks, tgts = train_draw()
        toks = toks.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = image_grid.masked_step(model, toks, tgts, amp_dtype, device, backward=True)
        if loss is None:
            skipped += 1
            if skipped == 1 or skipped % args.log_every == 0:
                print("WARNING step " + str(step) + ": non-finite loss, step skipped ("
                      + str(skipped) + " so far)", flush=True)
            continue
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True)
            skipped += 1
            if skipped == 1 or skipped % args.log_every == 0:
                print("WARNING step " + str(step) + ": non-finite grad norm, step skipped ("
                      + str(skipped) + " so far)", flush=True)
            continue
        opt.step()
        last_loss = float(loss)

        if step % args.log_every == 0 or step == 1:
            now = time.time()
            window_steps = step - last_log_step
            window_s = max(1e-6, now - last_log)
            tok_per_s = window_steps * args.batch_size * seq / window_s
            img_per_s = window_steps * args.batch_size / window_s
            print("step " + str(step) + "  loss " + format(float(loss), ".4f") + "  lr " + format(lr, ".2e")
                  + "  gn " + format(float(gn), ".3f") + "  tok/s " + format(tok_per_s, ".0f")
                  + "  img/s " + format(img_per_s, ".1f") + "  elapsed " + format(now - t0, ".0f") + "s",
                  flush=True)
            save.append_train_row(name, step, "train", float(loss), lr=lr, grad_norm=float(gn),
                                  tok_per_s=tok_per_s, wall_s=now - t0, seed=args.seed)
            prog.stage("train", step, args.total_steps, "training  step " + format(step, ",") + " / "
                       + format(args.total_steps, ",") + "  loss " + format(float(loss), ".4f"),
                       loss=round(float(loss), 5), lr=lr, grad_norm=round(float(gn), 4),
                       img_per_s=round(img_per_s, 2), tok_per_s=round(tok_per_s), skipped=skipped)
            last_log, last_log_step = now, step

        if val_draw is not None and step % args.eval_every == 0:
            v = vt.evaluate(model, val_draw, args.eval_iters, seq, amp_dtype, 1,
                            device_type=device, objective=vt.OBJECTIVE_IMAGE)
            if v is not None:
                print("step " + str(step) + "  val_loss " + format(v, ".4f"), flush=True)
                save.append_train_row(name, step, "val", v, lr=lr, wall_s=time.time() - t0,
                                      seed=args.seed)
                prog.note(val_loss=round(float(v), 5), val_step=step)

        if step % args.ckpt_every == 0 or step == args.total_steps:
            checkpoint(step)
            last_saved = step

        if stop_requested():
            if last_saved != step:
                checkpoint(step)
            prog.stage("train", step, args.total_steps, "stopped at step " + format(step, ","))
            raise StopRequested("stopped at step " + str(step) + "; checkpoint saved")

    prog.done("train", "trained " + format(args.total_steps, ",") + " steps  loss "
              + format(last_loss, ".4f"))
    prog.end(image_progress.RUN_DONE, "done")
    print("done.", flush=True)


# ------------------------------------------------------------------------------------
# Entry point

if __name__ == "__main__":
    run(plugin_id=trainers_reader.IMAGE_TRAINER_ID)
