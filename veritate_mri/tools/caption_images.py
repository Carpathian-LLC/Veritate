# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The captioning stage of image training: a vision teacher describes every picture in
#   a set and the description is written beside it as <image>.txt, the sidecar the
#   corpus builder already reads. Words steer generation only as far as the captions
#   the model trained on; folder names give it a vocabulary of folder names, this gives
#   it a vocabulary of what is in the pictures.
# - A separate stage on purpose: it costs teacher calls, it can be previewed on one
#   picture before spending them, it is resumable (already-captioned pictures are
#   skipped unless asked otherwise), and the trainer notices new captions and rebuilds
#   the corpus on the next launch.
# - Pictures are downscaled to --max-edge and sent as JPEG: a vision model does not
#   need 12 megapixels to say "a dog on a beach", and tokens are what a caption costs.
# - Provider adapter: the teacher client passes messages through as JSON, so an image
#   part rides along. OpenAI-style parts for every provider except Anthropic, whose
#   image part has its own shape.
# - usage: .veritate_venv/bin/python -m tools.caption_images <set> [--provider ollama]
#          [--model llava:13b] [--style sentence|tags|detailed|custom] [--prompt ...]
#          [--max-words 40] [--max-edge 768] [--concurrency 4] [--overwrite] [--limit 0]
# veritate_mri/tools/caption_images.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import base64
import concurrent.futures as futures
import io
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from readers import paths  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

IMAGE_EXTS     = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CAPTION_SUFFIX = ".txt"
STYLES = {
    "sentence": {
        "label": "one sentence",
        "prompt": "Describe this photo in one plain sentence of at most {max_words} words: the main "
                  "subject, where it is, and what is happening. No preamble, no opinions.",
    },
    "tags": {
        "label": "tags",
        "prompt": "List the main subjects, setting, colors, lighting and mood of this photo as 5 to 12 "
                  "short comma-separated tags. No sentences, no preamble.",
    },
    "detailed": {
        "label": "detailed",
        "prompt": "Describe this photo in two or three sentences and at most {max_words} words: "
                  "subjects, setting, composition, lighting, colors and mood. Plain factual language, "
                  "no preamble.",
    },
    "custom": {"label": "custom prompt", "prompt": ""},
}
DEFAULT_STYLE       = "sentence"
DEFAULT_MAX_WORDS   = 40
DEFAULT_MAX_EDGE    = 768
DEFAULT_CONCURRENCY = 4
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS  = 160
JPEG_QUALITY        = 85
MIME_JPEG           = "image/jpeg"
SAMPLE_KEEP         = 5
PREAMBLE = re.compile(r"^(?:this (?:image|photo|picture) (?:shows|depicts|features|is of)|"
                      r"the (?:image|photo|picture) (?:shows|depicts|features))\s*", re.IGNORECASE)

# ------------------------------------------------------------------------------------
# Functions


def set_images(set_name):
    set_dir = paths.image_set_dir(set_name)
    if not os.path.isdir(set_dir):
        raise ValueError("no image set at " + set_dir)
    return set_dir, sorted(os.path.join(set_dir, n) for n in os.listdir(set_dir)
                           if not n.startswith(".") and n.lower().endswith(IMAGE_EXTS))


def sidecar_for(image_path):
    return os.path.splitext(image_path)[0] + CAPTION_SUFFIX


def prompt_for(style, custom_prompt, max_words):
    if style == "custom" or (custom_prompt or "").strip():
        text = (custom_prompt or "").strip()
        if not text:
            raise ValueError("custom style needs a prompt")
        return text
    if style not in STYLES:
        raise ValueError("unknown style: " + str(style) + " (valid: " + ", ".join(STYLES) + ")")
    return STYLES[style]["prompt"].format(max_words=int(max_words))


def prepare_image(path, max_edge=DEFAULT_MAX_EDGE):
    """Downscaled JPEG as base64. Tokens are what a caption costs."""
    from PIL import Image
    with Image.open(path) as handle:
        img = handle.convert("RGB")
        img.thumbnail((int(max_edge), int(max_edge)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def vision_messages(provider, prompt, image_b64, mime=MIME_JPEG):
    """One user turn carrying the picture, in the shape this provider's API reads."""
    if (provider or {}).get("system_message_style") == "field":      # Anthropic
        content = [{"type": "image", "source": {"type": "base64", "media_type": mime, "data": image_b64}},
                   {"type": "text", "text": prompt}]
    else:                                                             # OpenAI-compatible
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + image_b64}}]
    return [{"role": "user", "content": content}]


def clean_caption(text, max_words):
    text = " ".join(str(text or "").replace("\n", " ").split()).strip().strip('"“”\'')
    text = PREAMBLE.sub("", text)
    if text:
        text = text[0].upper() + text[1:]
    words = text.split()
    if max_words and len(words) > int(max_words):
        text = " ".join(words[:int(max_words)]).rstrip(",;:")
    return text


def caption_one(client, path, prompt, max_edge=DEFAULT_MAX_EDGE, max_words=DEFAULT_MAX_WORDS,
                temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS):
    messages = vision_messages(getattr(client, "provider", {}), prompt, prepare_image(path, max_edge))
    text = client.complete(messages, temperature=temperature, max_tokens=max_tokens)
    caption = clean_caption(text, max_words)
    if not caption:
        raise ValueError("empty caption")
    return caption


def _write_sidecar(path, caption):
    target = sidecar_for(path)
    with open(target + ".tmp", "w", encoding="utf-8") as handle:
        handle.write(caption + "\n")
    os.replace(target + ".tmp", target)


def caption_set(set_name, provider=None, model=None, style=DEFAULT_STYLE, prompt=None,
                max_words=DEFAULT_MAX_WORDS, max_edge=DEFAULT_MAX_EDGE, overwrite=False,
                concurrency=DEFAULT_CONCURRENCY, limit=0, temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS, progress=None, should_stop=None, client=None):
    """Caption every picture in the set that lacks a sidecar (or all of them with
    overwrite). `progress(done, total, name, caption_or_None)` is called per picture;
    `should_stop()` is polled between batches. Returns the report."""
    _set_dir, images = set_images(set_name)
    todo = [p for p in images if overwrite or not os.path.isfile(sidecar_for(p))]
    if limit:
        todo = todo[:int(limit)]
    text_prompt = prompt_for(style, prompt, max_words)
    report = {"set": set_name, "images": len(images), "already_captioned": len(images) - len(todo)
              if not overwrite else 0, "todo": len(todo), "done": 0, "failed": 0, "stopped": False,
              "samples": [], "errors": [], "seconds": 0.0, "prompt": text_prompt}
    if not todo:
        return report
    if client is None:
        from veritate_core.plugin import get_teacher_client
        client = get_teacher_client(provider, model)
        if client is None:
            raise ValueError("no teacher configured: pick a provider and a vision model")
    started = time.perf_counter()
    workers = max(1, int(concurrency))
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(todo), workers):
            if should_stop is not None and should_stop():
                report["stopped"] = True
                break
            batch = todo[start:start + workers]
            jobs = {pool.submit(caption_one, client, p, text_prompt, max_edge, max_words,
                                temperature, max_tokens): p for p in batch}
            for job in futures.as_completed(jobs):
                path = jobs[job]
                name = os.path.basename(path)
                try:
                    caption = job.result()
                except Exception as e:
                    report["failed"] += 1
                    if len(report["errors"]) < SAMPLE_KEEP:
                        report["errors"].append({"name": name, "error": type(e).__name__ + ": " + str(e)})
                    if progress is not None:
                        progress(report["done"] + report["failed"], len(todo), name, None)
                    continue
                _write_sidecar(path, caption)
                report["done"] += 1
                report["samples"] = (report["samples"] + [{"name": name, "caption": caption}])[-SAMPLE_KEEP:]
                if progress is not None:
                    progress(report["done"] + report["failed"], len(todo), name, caption)
    report["seconds"] = round(time.perf_counter() - started, 1)
    return report


def main():
    ap = argparse.ArgumentParser(description="Describe every picture in a set with a vision teacher.")
    ap.add_argument("set_name")
    ap.add_argument("--provider", default=None, help="teacher provider id (default: the configured one)")
    ap.add_argument("--model", default=None, help="vision model name (default: the configured one)")
    ap.add_argument("--style", default=DEFAULT_STYLE, choices=tuple(STYLES))
    ap.add_argument("--prompt", default=None, help="custom prompt (implies --style custom)")
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    ap.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE, help="downscale before sending")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--overwrite", action="store_true", help="redo pictures that already have a caption")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    def show(done, total, name, caption):
        print(f"[{done}/{total}] {name}: {caption if caption is not None else 'FAILED'}", flush=True)

    rep = caption_set(args.set_name, provider=args.provider, model=args.model, style=args.style,
                      prompt=args.prompt, max_words=args.max_words, max_edge=args.max_edge,
                      overwrite=args.overwrite, concurrency=args.concurrency, limit=args.limit,
                      progress=show)
    print(f"{rep['images']} pictures, {rep['already_captioned']} already captioned, "
          f"{rep['done']} captioned now, {rep['failed']} failed, {rep['seconds']}s")
    for err in rep["errors"]:
        print("  " + err["name"] + ": " + err["error"])
    return 0 if not rep["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
