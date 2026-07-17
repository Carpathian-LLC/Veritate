# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - persistent c-engine subprocess running `veritate.exe chat_traced`. one process
#   per server, reused across requests. binary frame parser yields per-token raw
#   activation slices for the orchestration layer to convert into mri json frames.
# - protocol mirrors engine/src/main.c chat_traced_loop:
#     stdin  text:  "<temp> <top_k> <max_new>\n<prompt>\n"
#     stdout binary: TFRM frame per token, TEND marker per turn.
# - shape (layers, hidden, ffn, heads, seq, vocab) is read from the bin header at
#   subprocess spawn. all per-frame buffer sizes derive from it. supports any
#   model, not just the 80M fixed shape.
# veritate_mri/inference/backends/c_engine.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import struct
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.normpath(os.path.join(HERE, "..")) not in sys.path:
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
try:
    from runtime import logs as logmod
except Exception:
    class _NoLog:
        def info(self, *a, **kw): pass
        def warn(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def ok(self, *a, **kw): pass
    logmod = _NoLog()

# ------------------------------------------------------------------------------------
# Constants

VERITATE_MODEL_MAGIC = b"VRTE"
HEADER_BYTES         = 32
HEADER_FMT           = "<4sIIIIIII"
STATE_CACHE_DIRNAME  = "state_cache"
# v13 batched prefill width (VERITATE_PREFILL_BATCH). Amortizes the local-block
# weight streaming across the chunk; bitwise-identical to sequential prefill
# (tests/engine/test_prefill_batch.py). 32 measured best on the i7-9700T.
PREFILL_BATCH        = 32

DLA_TOPK            = 12
DLA_ENTRY_BYTES     = 16   # u8 layer, u8 pad, u16 neuron, i32 act, i32 w, i32 contrib
CAND_TOPK           = 12   # v8: per-candidate DLA count, must match VERITATE_CAND_TOPK in veritate.h
CONFIDENCE_BYTES    = 5 * 4

# drain bounds: when a client disconnects mid-stream we consume the engine's
# remaining frames so the next request starts on a clean pipe. caps prevent
# stalling subsequent requests if the engine emits many more frames; on overrun
# we kill + respawn instead.
DRAIN_MAX_FRAMES    = 4096
DRAIN_WALL_S        = 2.0
# A stream holds self.lock for the whole generation to serialize the one stateful
# subprocess. If an SSE client vanishes while its generator is suspended at a
# yield inside the lock, that generator is never closed and the lock is never
# released, wedging every later request (they emit meta, then block on the lock
# forever). A new stream that cannot take the lock within this window declares the
# holder dead and reclaims the backend (kill+respawn+fresh lock) instead of
# hanging. Sized well above a normal full-length generation.
STREAM_LOCK_TIMEOUT_S = 30.0
# How often a waiting request re-checks the holder's liveness. Reclaim triggers
# only when the holder has produced NO frame for STREAM_LOCK_TIMEOUT_S, so a
# slow-but-healthy long generation (which keeps emitting frames) is never killed.
RECLAIM_POLL_S = 2.0
# v8 tail: u16 cand_count + u8[CAND_TOPK] cand_bytes + dla_entry[CAND_TOPK][DLA_TOPK]
# + i16 ablation_layer + i16 ablation_neuron.
V8_TAIL_BYTES       = 2 + CAND_TOPK + CAND_TOPK * DLA_TOPK * DLA_ENTRY_BYTES + 2 + 2

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# chat_traced_loop reads the prompt with fgets(prompt_line, seq + PROMPT_FGETS_BUFSIZE).
# fgets stores at most (size - 1) chars incl the newline, so the payload (bytes before
# '\n') caps at seq + PROMPT_FGETS_BUFSIZE - 2. A longer line leaves residue in stdin
# and permanently desyncs the persistent subprocess; clamp the tail (newest bytes).
PROMPT_FGETS_BUFSIZE = 4

DLA_DTYPE = np.dtype([
    ("layer",   np.uint8),
    ("pad",     np.uint8),
    ("neuron",  np.uint16),
    ("act",     np.int32),
    ("w",       np.int32),
    ("contrib", np.int32),
], align=False)
assert DLA_DTYPE.itemsize == DLA_ENTRY_BYTES

# Compact trace frame (magic 'TFRC', version 9). The engine performs the four heavy
# per-layer reductions itself; these counts + packed dtypes MUST match veritate.h
# (VERITATE_FFN_BUCKET_TARGET / FFN_TOPK / INFO_FLOW_TOPK / LENS_TOPK) and the on-wire
# field order emitted by chat_traced_loop.
FFN_BUCKET_TARGET = 256
FFN_TOPK          = 8
INFO_FLOW_TOPK    = 8
LENS_TOPK         = 3
FFN_TOP_DTYPE  = np.dtype([("id", np.uint16), ("v", np.float32)], align=False)   # 6 B
LENS_DTYPE     = np.dtype([("b", np.uint8),   ("p", np.float32)], align=False)   # 5 B
FLOW_DTYPE     = np.dtype([("p", np.uint16),  ("w", np.float32)], align=False)   # 6 B
assert FFN_TOP_DTYPE.itemsize == 6 and LENS_DTYPE.itemsize == 5 and FLOW_DTYPE.itemsize == 6

# ------------------------------------------------------------------------------------
# Functions

def _read_exact(f, n):
    out = bytearray()
    while len(out) < n:
        chunk = f.read(n - len(out))
        if not chunk:
            return None
        out += chunk
    return bytes(out)


def _read_bin_shape(path):
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"veritate.bin not found: {path}")
    with open(path, "rb") as f:
        hdr = f.read(HEADER_BYTES)
    if len(hdr) < HEADER_BYTES:
        raise RuntimeError(f"veritate.bin truncated: {path}")
    magic, version, vocab, hidden, layers, ffn, heads, seq = struct.unpack(HEADER_FMT, hdr)
    if magic != VERITATE_MODEL_MAGIC:
        raise RuntimeError(f"veritate.bin bad magic {magic!r}: {path}")
    # Pre-empt the one genuinely retired version with a clear message; for
    # everything else, defer to the C engine's actual dispatch (model.c).
    if int(version) == 10:
        raise RuntimeError(
            f"veritate.bin v10 is retired (assigned twice on different branches "
            f"during the v11 merge). Re-export from the latest .pt checkpoint. "
            f"Path: {path}"
        )
    return {
        "version": int(version),
        "vocab":   int(vocab),
        "hidden":  int(hidden),
        "layers":  int(layers),
        "ffn":     int(ffn),
        "heads":   int(heads),
        "seq":     int(seq),
    }


class CTracedSubprocess:
    def __init__(self, exe, model_path, state_cache=True):
        if not exe or not os.path.isfile(exe):
            raise RuntimeError(f"c engine not found: {exe}")
        self.exe = exe
        self.model_path = model_path
        self.state_cache = state_cache
        self.lock = threading.Lock()
        # Bumped whenever the backend is reclaimed from a wedged stream. A
        # generator captures the epoch at start; its cleanup no-ops if the epoch
        # moved on, so an abandoned stream cannot drain frames off the fresh proc.
        self._epoch = 0
        # Guards the reclaim mutation so two timed-out waiters cannot both
        # reclaim. _last_frame_time is the holder's heartbeat (monotonic),
        # refreshed per frame; a reclaimer only acts when it goes stale.
        self._reclaim_lock = threading.Lock()
        self._last_frame_time = 0.0
        # False while a request is mid-flight or ended without a clean TEND. The
        # next request respawns to guarantee a protocol-clean pipe: _ensure_alive
        # only respawns a DEAD proc, but a desynced-but-alive proc (e.g. after an
        # SSE client vanished before the drain reached TEND) would otherwise echo
        # its own header as output until a manual reload.
        self._last_clean = True
        self.proc = None
        # shape comes from the bin header. all frame-size math derives from it.
        self.shape = _read_bin_shape(model_path)
        self._derive_frame_sizes()
        # last stream's per-frame timing trace. populated by stream() each call.
        # list of dicts: t_read_pipe_ms, t_parse_ms, t_engine_inter_ms, frame_size_bytes.
        self.last_trace = []
        self.last_total_wall_ms = 0.0
        self.last_total_bytes = 0
        # reused per-frame parse buffers. consumer must copy or serialize before next yield.
        s = self.shape
        self._buf_residual_pre  = np.empty((s["layers"], s["hidden"]),         dtype=np.int16)
        self._buf_residual_post = np.empty((s["layers"], s["hidden"]),         dtype=np.int16)
        self._buf_ffn_neurons   = np.empty((s["layers"], s["ffn"]),            dtype=np.int8)
        self._buf_attn          = np.empty((s["layers"], s["heads"], s["seq"]),dtype=np.float32)
        self._buf_lens_logits   = np.empty((s["layers"], s["vocab"]),          dtype=np.int32)
        self._spawn()

    def _derive_frame_sizes(self):
        s = self.shape
        self._res_bytes_per_layer        = s["hidden"] * 2
        self._ffn_bytes_per_layer        = s["ffn"]
        self._attn_q_bytes_per_layer     = s["heads"] * s["seq"]
        self._attn_scale_bytes_per_layer = s["heads"] * 4
        self._lens_bytes_per_layer       = s["vocab"] * 4
        self._layer_bytes = (self._res_bytes_per_layer * 2
                             + self._ffn_bytes_per_layer
                             + self._attn_q_bytes_per_layer
                             + self._attn_scale_bytes_per_layer
                             + self._lens_bytes_per_layer)
        self._decision_trace_bytes = s["layers"] * 4 * 2 + 2 * DLA_TOPK * DLA_ENTRY_BYTES
        self._frame_payload_bytes = (self._layer_bytes * s["layers"]
                                     + s["hidden"]
                                     + s["vocab"] * 4
                                     + self._decision_trace_bytes
                                     + CONFIDENCE_BYTES
                                     + V8_TAIL_BYTES)

        # Compact (TFRC v9) geometry. DS/BUCKETS mirror _build_c_mri_frame exactly.
        self._ds      = max(1, s["ffn"] // FFN_BUCKET_TARGET)
        self._buckets = s["ffn"] // self._ds
        # per layer: ffn_full + ffn_argmax (uint8 each) + ffn_top + res/contrib f32 + lens
        self._compact_layer_bytes = (2 * self._buckets
                                     + FFN_TOPK * FFN_TOP_DTYPE.itemsize
                                     + 4 + 4
                                     + LENS_TOPK * LENS_DTYPE.itemsize)
        # once per frame: u8 flow_count + INFO_FLOW_TOPK packed (pos,w) slots
        self._compact_flow_bytes = 1 + INFO_FLOW_TOPK * FLOW_DTYPE.itemsize
        # trailer is identical to TFRM minus final_act (dashboard never reads it).
        self._compact_payload_bytes = (self._compact_layer_bytes * s["layers"]
                                       + self._compact_flow_bytes
                                       + s["vocab"] * 4
                                       + self._decision_trace_bytes
                                       + CONFIDENCE_BYTES
                                       + V8_TAIL_BYTES)

    def _spawn(self):
        env = os.environ.copy()
        if self.model_path:
            env["VERITATE_MODEL_PATH"] = self.model_path
        # Let the engine load act_boost>1 models. The engine's act_boost>1 refusal
        # is a magnitude heuristic; QAT-trained checkpoints can still produce
        # act_boost>1 when embeddings are small. The authoritative QAT signal lives
        # in config.json training_args.qat_enabled; the dashboard checks it. Force
        # the bypass here (not setdefault) so a parent-shell "0" cannot override.
        env["VERITATE_ALLOW_HIGH_ACT_BOOST"] = "1"
        # Persistent prompt/state cache next to the model. setdefault so a parent
        # env value still wins; disabled by constructing with state_cache=False.
        if self.state_cache and self.model_path:
            env.setdefault("VERITATE_STATE_CACHE",
                           os.path.join(os.path.dirname(self.model_path), STATE_CACHE_DIRNAME))
        # Batched prefill off the TTFB path; setdefault so a parent env value wins.
        env.setdefault("VERITATE_PREFILL_BATCH", str(PREFILL_BATCH))
        # VERITATE_ADDONS=<csv> is inherited from the parent env automatically
        # via os.environ.copy(). set it before starting the MRI server to
        # enable engine-side addons. per-request addon selection from the
        # dashboard is python-only today; the C subprocess holds whatever
        # chain it had at spawn time. Restart the C backend in the dashboard
        # to pick up an env-var change. spec: documentation/addons/c_engine_port.md.
        self.proc = subprocess.Popen(
            [self.exe, "chat_traced"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env,
            creationflags=_NO_WINDOW,
        )
        # Engine may print warnings (e.g. "model_load: act_boost=...") to stderr
        # before "ready". Drain lines until "ready" arrives, the process dies,
        # or a bounded prelude budget is exceeded.
        warnings = []
        ready = False
        for _ in range(16):
            line = self.proc.stderr.readline()
            if not line:
                break
            if line.strip().startswith(b"ready"):
                ready = True
                break
            warnings.append(line.strip())
        if not ready:
            # kill before draining: a refusal that doesn't exit (random-fallback
            # branch in chat_traced_loop) leaves stderr open, so stderr.read()
            # would block forever waiting for EOF.
            try: self.proc.kill()
            except Exception: pass
            try: tail = self.proc.stderr.read()
            except Exception: tail = b""
            raise RuntimeError(f"chat_traced not ready: {warnings!r} {tail!r}")
        for w in warnings:
            logmod.warn("c engine", w.decode("latin-1", "replace"))
        # Drain stderr for the life of this process. Without this, a mid-generation
        # stderr flood fills the ~64 KB OS pipe buffer, blocks the engine's write,
        # stalls stdout frames, and wedges stream() until the reclaim kills it.
        # Daemon thread bound to THIS proc's stderr; exits on EOF when it dies.
        err_pipe = self.proc.stderr
        def _pump_stderr():
            try:
                for line in iter(err_pipe.readline, b""):
                    s = line.strip()
                    if s:
                        logmod.warn("c engine", s.decode("latin-1", "replace"))
            except Exception:
                pass
        threading.Thread(target=_pump_stderr, daemon=True).start()
        self._last_clean = True

    def _ensure_alive(self):
        if self.proc is None or self.proc.poll() is not None:
            self._spawn()

    def stream(self, prompt, temperature, top_k, max_new,
               ablate_layer=-1, ablate_neuron=-1, addons_csv="",
               rep_window=0, rep_penalty=0.0, no_repeat_ngram=0, do_trace=True,
               compact=False):
        # Acquire the per-generation lock. A live generation refreshes
        # _last_frame_time on every frame; reclaim only when the holder has gone
        # silent for STREAM_LOCK_TIMEOUT_S (truly wedged, e.g. an abandoned SSE
        # generator parked at a yield), never a slow-but-emitting one. The
        # reclaim mutation is serialized by _reclaim_lock so two waiters cannot
        # both reclaim, and it kills+respawns + bumps the epoch so the wedged
        # generator stops at its next read/epoch check instead of racing us.
        acquired = self.lock.acquire(timeout=RECLAIM_POLL_S)
        while not acquired:
            with self._reclaim_lock:
                if self.lock.acquire(timeout=0):
                    acquired = True
                else:
                    idle = time.monotonic() - self._last_frame_time
                    if idle >= STREAM_LOCK_TIMEOUT_S:
                        logmod.error("c_engine",
                                     f"stream stalled {idle:.0f}s (no frame); reclaiming backend")
                        try:
                            if self.proc is not None:
                                self.proc.kill()
                        except Exception:
                            pass
                        self._epoch += 1
                        self.lock = threading.Lock()
                        self.lock.acquire()
                        self.proc = None
                        acquired = True
            if not acquired:
                acquired = self.lock.acquire(timeout=RECLAIM_POLL_S)
        my_lock = self.lock
        my_epoch = self._epoch
        self._last_frame_time = time.monotonic()
        try:
            self._ensure_alive()
            # Force a fresh proc if the previous request did not end on a clean
            # TEND: a protocol-desynced but alive engine survives _ensure_alive.
            if not self._last_clean:
                self._kill_and_respawn()
            self._last_clean = False
            # newlines ride the line-based protocol as 0x01; chat_traced maps
            # them back so chat-template prompts keep their trained framing.
            p = prompt.replace("\r", "").replace("\n", "\x01")
            # fgets(prompt_line, seq+4) stores <= seq+3 chars incl '\n'; payload caps at
            # seq+2. tail-clamp keeps the newest bytes so an over-long line can't leave
            # residue and desync the subprocess (engine also drains residue defensively).
            # generation budget is seq - prompt_len with no window slide, so also
            # reserve reply room; reserve caps at seq//2 so a pathological max_new
            # cannot wipe out the context.
            prompt_bytes = p.encode("latin-1", "replace")
            seq = self.shape["seq"]
            max_payload = min(seq + PROMPT_FGETS_BUFSIZE - 2,
                              seq - min(int(max_new), seq // 2))
            if len(prompt_bytes) > max_payload:
                prompt_bytes = prompt_bytes[-max_payload:]
            # addons token: empty -> use whatever chain was set at spawn time
            # (env var path); "-" -> clear any prior chain; otherwise comma-
            # separated id list. token must contain no whitespace.
            csv_token = (addons_csv or "").strip().replace(" ", "")
            if not csv_token:
                csv_token = "-"  # explicit clear when caller passed nothing
            # rep_* tail is optional in the protocol; 0/0 leaves the engine
            # sampler bitwise-identical to pre-repetition-control behavior.
            # compact is only meaningful with tracing on; the engine emits TFRC v9
            # (reduced) frames when both are set, else the raw TFRM v8 frames.
            use_compact = bool(compact and do_trace)
            header = (f"{float(temperature):.4f} {int(top_k)} {int(max_new)} "
                      f"{int(ablate_layer)} {int(ablate_neuron)} {csv_token} "
                      f"{int(rep_window)} {float(rep_penalty):.4f} {int(no_repeat_ngram)} "
                      f"{1 if do_trace else 0} {1 if use_compact else 0}\n").encode("ascii")
            try:
                self.proc.stdin.write(header)
                self.proc.stdin.write(prompt_bytes + b"\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                # subprocess died mid-write. respawn and retry once.
                self._spawn()
                try:
                    self.proc.stdin.write(header)
                    self.proc.stdin.write(prompt_bytes + b"\n")
                    self.proc.stdin.flush()
                except (BrokenPipeError, OSError) as e2:
                    raise RuntimeError(f"chat_traced respawn failed: {e2!r}")

            # Bind this generation to the subprocess that received our header.
            # A concurrent reclaim kills THIS proc and installs a fresh one on
            # self.proc; reading my_proc (not self.proc) plus the epoch checks
            # below keep us from ever reading the successor's pipe.
            my_proc = self.proc
            saw_tend = False
            respawned = False
            trace = []
            t_stream_start = time.perf_counter_ns()
            t_prev_frame_done = t_stream_start
            try:
                while True:
                    if my_epoch != self._epoch:
                        return  # reclaimed; stop before touching the successor pipe
                    t_read0 = time.perf_counter_ns()
                    marker = _read_exact(my_proc.stdout, 4)
                    if marker is None:
                        if my_epoch != self._epoch:
                            return  # our proc was killed by a reclaim; not ours to respawn
                        logmod.error("c_engine", "stdout closed mid-frame; respawning")
                        self._kill_and_respawn(); respawned = True
                        raise RuntimeError("chat_traced stdout closed mid-frame; subprocess respawned")
                    if marker == b"TEND":
                        _read_exact(my_proc.stdout, 4)
                        saw_tend = True
                        self._last_clean = True
                        return
                    if marker == b"FFRM":
                        # fast frame: 12-byte tail, no payload. do_trace=0 path.
                        rest = _read_exact(my_proc.stdout, 12)
                        if rest is None:
                            if my_epoch != self._epoch:
                                return
                            logmod.error("c_engine", "stdout closed mid-fast-frame; respawning")
                            self._kill_and_respawn(); respawned = True
                            raise RuntimeError("chat_traced stdout closed mid-fast-frame; subprocess respawned")
                        pos, real_len = struct.unpack("<II", rest[:8])
                        t_read1 = time.perf_counter_ns()
                        trace.append({
                            "t_read_pipe_ms":    (t_read1 - t_read0) / 1e6,
                            "t_parse_ms":        0.0,
                            "t_engine_inter_ms": (t_read0 - t_prev_frame_done) / 1e6,
                            "frame_size_bytes":  16,
                        })
                        t_prev_frame_done = t_read1
                        self._last_frame_time = time.monotonic()
                        yield {"pos": int(pos), "real_len": int(real_len),
                               "byte": int(rest[8]), "argmax_byte": int(rest[9]),
                               "fast": True}
                        continue
                    # TFRM = raw v8 frame; TFRC = compact v9 frame (engine-reduced).
                    is_compact = (marker == b"TFRC")
                    if marker != b"TFRM" and not is_compact:
                        # Pipe is desynced reading garbage instead of frame magic.
                        # Cannot recover by draining; the next read offset is unknown.
                        # Kill the subprocess and respawn so the next request starts clean.
                        logmod.error("c_engine",
                                     f"bad frame marker {marker!r} (expected TFRM/TFRC/TEND); "
                                     f"killing+respawning subprocess to clear pipe desync")
                        self._kill_and_respawn(); respawned = True
                        raise RuntimeError(
                            f"chat_traced pipe desync (got {marker!r}); subprocess respawned, "
                            "retry the request"
                        )

                    rest = _read_exact(my_proc.stdout, 12)
                    if rest is None:
                        if my_epoch != self._epoch:
                            return
                        logmod.error("c_engine", "stdout closed mid-header; respawning")
                        self._kill_and_respawn(); respawned = True
                        raise RuntimeError("chat_traced stdout closed mid-header; subprocess respawned")
                    pos, real_len = struct.unpack("<II", rest[:8])
                    byte = rest[8]
                    argmax_byte = rest[9]

                    payload_bytes = self._compact_payload_bytes if is_compact else self._frame_payload_bytes
                    payload = _read_exact(my_proc.stdout, payload_bytes)
                    if payload is None:
                        if my_epoch != self._epoch:
                            return
                        logmod.error("c_engine", "stdout closed mid-payload; respawning")
                        self._kill_and_respawn(); respawned = True
                        raise RuntimeError("chat_traced stdout closed mid-payload; subprocess respawned")
                    t_read1 = time.perf_counter_ns()

                    if is_compact:
                        parsed = self._parse_compact_frame(payload, pos=pos, real_len=real_len,
                                                           byte=byte, argmax_byte=argmax_byte)
                    else:
                        parsed = self._parse_frame(payload, pos=pos, real_len=real_len,
                                                   byte=byte, argmax_byte=argmax_byte)
                    t_parse1 = time.perf_counter_ns()

                    frame_bytes = 4 + 12 + payload_bytes
                    trace.append({
                        "t_read_pipe_ms":    (t_read1 - t_read0) / 1e6,
                        "t_parse_ms":        (t_parse1 - t_read1) / 1e6,
                        "t_engine_inter_ms": (t_read0 - t_prev_frame_done) / 1e6,
                        "frame_size_bytes":  frame_bytes,
                    })
                    t_prev_frame_done = t_parse1
                    self._last_frame_time = time.monotonic()  # heartbeat

                    yield parsed
            finally:
                # Skip if reclaimed (epoch moved) — the subprocess is no longer
                # ours. Skip the drain after a respawn too: the pipe is already
                # fresh+idle, so draining it would block on a read that never
                # returns (the fresh proc emits nothing until the next request).
                if my_epoch == self._epoch:
                    t_stream_end = time.perf_counter_ns()
                    self.last_trace = trace
                    self.last_total_wall_ms = (t_stream_end - t_stream_start) / 1e6
                    self.last_total_bytes = sum(f["frame_size_bytes"] for f in trace)
                    if not saw_tend and not respawned:
                        self._drain_to_tend()
        finally:
            my_lock.release()

    def _kill_and_respawn(self):
        """Hard-kill the subprocess and start a fresh one. Used when the pipe is
        confirmed desynced (bad frame marker) the read offset is unrecoverable,
        so a clean restart is the only safe path."""
        try:
            if self.proc is not None:
                self.proc.kill()
                self.proc.wait(timeout=3)
        except Exception:
            pass
        self.proc = None
        try:
            self._spawn()
            logmod.ok("c_engine", "subprocess respawned after pipe desync")
        except Exception as e:
            logmod.error("c_engine", f"respawn failed: {e}")
            raise

    def _drain_to_tend(self):
        # consume any pending frames + the TEND marker so the next request starts on a
        # clean pipe. called when the stream generator exits early (client disconnect,
        # exception). bounded by DRAIN_MAX_FRAMES and DRAIN_WALL_S; on overrun, kill +
        # respawn so the next request gets a clean pipe instead of waiting for ours.
        deadline = time.time() + DRAIN_WALL_S
        try:
            for _ in range(DRAIN_MAX_FRAMES):
                if time.time() > deadline:
                    self._kill_and_respawn()
                    return
                marker = _read_exact(self.proc.stdout, 4)
                if marker is None: return
                if marker == b"TEND":
                    _read_exact(self.proc.stdout, 4)
                    self._last_clean = True
                    return
                if marker == b"TFRM" or marker == b"TFRC":
                    pbytes = self._compact_payload_bytes if marker == b"TFRC" else self._frame_payload_bytes
                    if _read_exact(self.proc.stdout, 12) is None: return
                    if _read_exact(self.proc.stdout, pbytes) is None: return
                else:
                    return
            self._kill_and_respawn()
        except Exception:
            return

    def _parse_frame(self, buf, pos, real_len, byte, argmax_byte):
        s = self.shape
        layers, hidden, ffn = s["layers"], s["hidden"], s["ffn"]
        heads, seq, vocab   = s["heads"], s["seq"], s["vocab"]
        off = 0
        residual_pre   = self._buf_residual_pre
        residual_post  = self._buf_residual_post
        ffn_neurons    = self._buf_ffn_neurons
        attn           = self._buf_attn
        lens_logits    = self._buf_lens_logits
        for L in range(layers):
            residual_pre[L]  = np.frombuffer(buf, dtype=np.int16, count=hidden, offset=off); off += self._res_bytes_per_layer
            residual_post[L] = np.frombuffer(buf, dtype=np.int16, count=hidden, offset=off); off += self._res_bytes_per_layer
            ffn_neurons[L]   = np.frombuffer(buf, dtype=np.int8,  count=ffn,    offset=off); off += self._ffn_bytes_per_layer
            attn_q     = np.frombuffer(buf, dtype=np.int8,    count=heads * seq, offset=off).reshape(heads, seq); off += self._attn_q_bytes_per_layer
            attn_scale = np.frombuffer(buf, dtype=np.float32, count=heads,       offset=off);                     off += self._attn_scale_bytes_per_layer
            attn[L] = attn_q.astype(np.float32) * attn_scale[:, None]
            lens_logits[L]   = np.frombuffer(buf, dtype=np.int32, count=vocab,  offset=off); off += self._lens_bytes_per_layer
        final_act = np.frombuffer(buf, dtype=np.int8, count=hidden, offset=off).copy(); off += hidden
        logits    = np.frombuffer(buf, dtype=np.int32, count=vocab, offset=off).copy(); off += vocab * 4
        decisiveness = np.frombuffer(buf, dtype=np.float32, count=layers, offset=off).copy(); off += layers * 4
        bd_scale     = np.frombuffer(buf, dtype=np.float32, count=layers, offset=off).copy(); off += layers * 4
        dla_picked = np.frombuffer(buf, dtype=DLA_DTYPE, count=DLA_TOPK, offset=off).copy(); off += DLA_TOPK * DLA_ENTRY_BYTES
        dla_argmax = np.frombuffer(buf, dtype=DLA_DTYPE, count=DLA_TOPK, offset=off).copy(); off += DLA_TOPK * DLA_ENTRY_BYTES
        conf = np.frombuffer(buf, dtype=np.float32, count=5, offset=off); off += CONFIDENCE_BYTES
        cand_count = int(np.frombuffer(buf, dtype=np.uint16, count=1, offset=off)[0]); off += 2
        cand_bytes = np.frombuffer(buf, dtype=np.uint8, count=CAND_TOPK, offset=off).copy(); off += CAND_TOPK
        dla_cand   = np.frombuffer(buf, dtype=DLA_DTYPE, count=CAND_TOPK * DLA_TOPK,
                                   offset=off).reshape(CAND_TOPK, DLA_TOPK).copy()
        off += CAND_TOPK * DLA_TOPK * DLA_ENTRY_BYTES
        ablation_layer  = int(np.frombuffer(buf, dtype=np.int16, count=1, offset=off)[0]); off += 2
        ablation_neuron = int(np.frombuffer(buf, dtype=np.int16, count=1, offset=off)[0]); off += 2
        return {
            "pos": int(pos), "real_len": int(real_len), "byte": int(byte),
            "argmax_byte": int(argmax_byte),
            "residual_pre": residual_pre, "residual_post": residual_post,
            "ffn_neurons": ffn_neurons, "attention": attn,
            "lens_logits": lens_logits.copy(),
            "final_act": final_act, "logits": logits,
            "decisiveness": decisiveness,
            "bd_scale": bd_scale,
            "dla_picked": dla_picked, "dla_argmax": dla_argmax,
            "margin":           float(conf[0]),
            "entropy":          float(conf[1]),
            "lens_consistency": float(conf[2]),
            "residual_stab":    float(conf[3]),
            "confidence":       float(conf[4]),
            "cand_count":       cand_count,
            "cand_bytes":       cand_bytes,
            "dla_cand":         dla_cand,
            "ablation_layer":   ablation_layer,
            "ablation_neuron":  ablation_neuron,
        }

    def _parse_compact_frame(self, buf, pos, real_len, byte, argmax_byte):
        """Parse a TFRC v9 frame: the engine already reduced the four heavy arrays,
        so this reads the compact per-layer summaries + an unchanged decision-trace
        trailer (minus final_act). Field order MUST match chat_traced_loop's TFRC
        writer in main.c. Returns a dict tagged compact for _build_c_mri_frame_compact."""
        s = self.shape
        layers, vocab = s["layers"], s["vocab"]
        B = self._buckets
        off = 0
        ffn_full, ffn_argmax, ffn_top = [], [], []
        res, contrib, lens = [], [], []
        for _L in range(layers):
            ffn_full.append(np.frombuffer(buf, dtype=np.uint8, count=B, offset=off).copy());   off += B
            ffn_argmax.append(np.frombuffer(buf, dtype=np.uint8, count=B, offset=off).copy()); off += B
            ffn_top.append(np.frombuffer(buf, dtype=FFN_TOP_DTYPE, count=FFN_TOPK, offset=off).copy())
            off += FFN_TOPK * FFN_TOP_DTYPE.itemsize
            rc = np.frombuffer(buf, dtype=np.float32, count=2, offset=off); off += 8
            res.append(float(rc[0])); contrib.append(float(rc[1]))
            lens.append(np.frombuffer(buf, dtype=LENS_DTYPE, count=LENS_TOPK, offset=off).copy())
            off += LENS_TOPK * LENS_DTYPE.itemsize
        flow_count = int(np.frombuffer(buf, dtype=np.uint8, count=1, offset=off)[0]); off += 1
        info_flow  = np.frombuffer(buf, dtype=FLOW_DTYPE, count=INFO_FLOW_TOPK, offset=off).copy()
        off += INFO_FLOW_TOPK * FLOW_DTYPE.itemsize
        # trailer — identical layout to _parse_frame's tail, minus final_act.
        logits       = np.frombuffer(buf, dtype=np.int32,   count=vocab,  offset=off).copy(); off += vocab * 4
        decisiveness = np.frombuffer(buf, dtype=np.float32, count=layers, offset=off).copy(); off += layers * 4
        bd_scale     = np.frombuffer(buf, dtype=np.float32, count=layers, offset=off).copy(); off += layers * 4
        dla_picked = np.frombuffer(buf, dtype=DLA_DTYPE, count=DLA_TOPK, offset=off).copy(); off += DLA_TOPK * DLA_ENTRY_BYTES
        dla_argmax = np.frombuffer(buf, dtype=DLA_DTYPE, count=DLA_TOPK, offset=off).copy(); off += DLA_TOPK * DLA_ENTRY_BYTES
        conf = np.frombuffer(buf, dtype=np.float32, count=5, offset=off); off += CONFIDENCE_BYTES
        cand_count = int(np.frombuffer(buf, dtype=np.uint16, count=1, offset=off)[0]); off += 2
        cand_bytes = np.frombuffer(buf, dtype=np.uint8, count=CAND_TOPK, offset=off).copy(); off += CAND_TOPK
        dla_cand = np.frombuffer(buf, dtype=DLA_DTYPE, count=CAND_TOPK * DLA_TOPK,
                                 offset=off).reshape(CAND_TOPK, DLA_TOPK).copy()
        off += CAND_TOPK * DLA_TOPK * DLA_ENTRY_BYTES
        ablation_layer  = int(np.frombuffer(buf, dtype=np.int16, count=1, offset=off)[0]); off += 2
        ablation_neuron = int(np.frombuffer(buf, dtype=np.int16, count=1, offset=off)[0]); off += 2
        return {
            "compact": True,
            "pos": int(pos), "real_len": int(real_len), "byte": int(byte),
            "argmax_byte": int(argmax_byte),
            "ffn_full": ffn_full, "ffn_argmax": ffn_argmax, "ffn_top": ffn_top,
            "res": res, "contrib": contrib, "lens": lens,
            "info_flow": info_flow[:flow_count],
            "logits": logits,
            "decisiveness": decisiveness, "bd_scale": bd_scale,
            "dla_picked": dla_picked, "dla_argmax": dla_argmax,
            "margin":           float(conf[0]),
            "entropy":          float(conf[1]),
            "lens_consistency": float(conf[2]),
            "residual_stab":    float(conf[3]),
            "confidence":       float(conf[4]),
            "cand_count":       cand_count,
            "cand_bytes":       cand_bytes,
            "dla_cand":         dla_cand,
            "ablation_layer":   ablation_layer,
            "ablation_neuron":  ablation_neuron,
        }

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try: self.proc.kill()
            except Exception: pass
