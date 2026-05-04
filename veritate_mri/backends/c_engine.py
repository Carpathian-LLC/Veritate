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
# ------------------------------------------------------------------------------------

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
    import logs as logmod
except Exception:
    class _NoLog:
        def info(self, *a, **kw): pass
        def warn(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def ok(self, *a, **kw): pass
    logmod = _NoLog()


VERITATE_MODEL_MAGIC = b"VRTE"
HEADER_BYTES         = 32
HEADER_FMT           = "<4sIIIIIII"

DLA_TOPK            = 12
DLA_ENTRY_BYTES     = 16   # u8 layer, u8 pad, u16 neuron, i32 act, i32 w, i32 contrib
CAND_TOPK           = 12   # v8: per-candidate DLA count, must match VERITATE_CAND_TOPK in veritate.h
CONFIDENCE_BYTES    = 5 * 4
# v8 tail: u16 cand_count + u8[CAND_TOPK] cand_bytes + dla_entry[CAND_TOPK][DLA_TOPK]
# + i16 ablation_layer + i16 ablation_neuron.
V8_TAIL_BYTES       = 2 + CAND_TOPK + CAND_TOPK * DLA_TOPK * DLA_ENTRY_BYTES + 2 + 2

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

DLA_DTYPE = np.dtype([
    ("layer",   np.uint8),
    ("pad",     np.uint8),
    ("neuron",  np.uint16),
    ("act",     np.int32),
    ("w",       np.int32),
    ("contrib", np.int32),
], align=False)
assert DLA_DTYPE.itemsize == DLA_ENTRY_BYTES


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
    def __init__(self, exe, model_path):
        if not exe or not os.path.isfile(exe):
            raise RuntimeError(f"c engine not found: {exe}")
        self.exe = exe
        self.model_path = model_path
        self.lock = threading.Lock()
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

    def _spawn(self):
        env = os.environ.copy()
        if self.model_path:
            env["VERITATE_MODEL_PATH"] = self.model_path
        self.proc = subprocess.Popen(
            [self.exe, "chat_traced"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env,
            creationflags=_NO_WINDOW,
        )
        line = self.proc.stderr.readline()
        if not line.strip().startswith(b"ready"):
            err = self.proc.stderr.read()
            try: self.proc.kill()
            except Exception: pass
            raise RuntimeError(f"chat_traced not ready: {line!r} {err!r}")

    def _ensure_alive(self):
        if self.proc is None or self.proc.poll() is not None:
            self._spawn()

    def stream(self, prompt, temperature, top_k, max_new,
               ablate_layer=-1, ablate_neuron=-1):
        with self.lock:
            self._ensure_alive()
            p = prompt.replace("\r", "").replace("\n", " ")
            header = (f"{float(temperature):.4f} {int(top_k)} {int(max_new)} "
                      f"{int(ablate_layer)} {int(ablate_neuron)}\n").encode("ascii")
            try:
                self.proc.stdin.write(header)
                self.proc.stdin.write(p.encode("latin-1", "replace") + b"\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                # subprocess died mid-write. respawn and retry once.
                self._spawn()
                try:
                    self.proc.stdin.write(header)
                    self.proc.stdin.write(p.encode("latin-1", "replace") + b"\n")
                    self.proc.stdin.flush()
                except (BrokenPipeError, OSError) as e2:
                    raise RuntimeError(f"chat_traced respawn failed: {e2!r}")

            saw_tend = False
            trace = []
            t_stream_start = time.perf_counter_ns()
            t_prev_frame_done = t_stream_start
            try:
                while True:
                    t_read0 = time.perf_counter_ns()
                    marker = _read_exact(self.proc.stdout, 4)
                    if marker is None:
                        logmod.error("c_engine", "stdout closed mid-frame; respawning")
                        self._kill_and_respawn()
                        raise RuntimeError("chat_traced stdout closed mid-frame; subprocess respawned")
                    if marker == b"TEND":
                        _read_exact(self.proc.stdout, 4)
                        saw_tend = True
                        return
                    if marker != b"TFRM":
                        # Pipe is desynced reading garbage instead of frame magic.
                        # Cannot recover by draining; the next read offset is unknown.
                        # Kill the subprocess and respawn so the next request starts clean.
                        logmod.error("c_engine",
                                     f"bad frame marker {marker!r} (expected TFRM/TEND); "
                                     f"killing+respawning subprocess to clear pipe desync")
                        self._kill_and_respawn()
                        raise RuntimeError(
                            f"chat_traced pipe desync (got {marker!r}); subprocess respawned, "
                            "retry the request"
                        )

                    rest = _read_exact(self.proc.stdout, 12)
                    if rest is None:
                        logmod.error("c_engine", "stdout closed mid-header; respawning")
                        self._kill_and_respawn()
                        raise RuntimeError("chat_traced stdout closed mid-header; subprocess respawned")
                    pos, real_len = struct.unpack("<II", rest[:8])
                    byte = rest[8]
                    argmax_byte = rest[9]

                    payload = _read_exact(self.proc.stdout, self._frame_payload_bytes)
                    if payload is None:
                        logmod.error("c_engine", "stdout closed mid-payload; respawning")
                        self._kill_and_respawn()
                        raise RuntimeError("chat_traced stdout closed mid-payload; subprocess respawned")
                    t_read1 = time.perf_counter_ns()

                    parsed = self._parse_frame(payload, pos=pos, real_len=real_len,
                                               byte=byte, argmax_byte=argmax_byte)
                    t_parse1 = time.perf_counter_ns()

                    frame_bytes = 4 + 12 + self._frame_payload_bytes
                    trace.append({
                        "t_read_pipe_ms":    (t_read1 - t_read0) / 1e6,
                        "t_parse_ms":        (t_parse1 - t_read1) / 1e6,
                        "t_engine_inter_ms": (t_read0 - t_prev_frame_done) / 1e6,
                        "frame_size_bytes":  frame_bytes,
                    })
                    t_prev_frame_done = t_parse1

                    yield parsed
            finally:
                t_stream_end = time.perf_counter_ns()
                self.last_trace = trace
                self.last_total_wall_ms = (t_stream_end - t_stream_start) / 1e6
                self.last_total_bytes = sum(f["frame_size_bytes"] for f in trace)
                if not saw_tend:
                    self._drain_to_tend()

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
        # exception). silently swallows errors, best-effort cleanup.
        try:
            while True:
                marker = _read_exact(self.proc.stdout, 4)
                if marker is None: return
                if marker == b"TEND":
                    _read_exact(self.proc.stdout, 4)
                    return
                if marker == b"TFRM":
                    if _read_exact(self.proc.stdout, 12) is None: return
                    if _read_exact(self.proc.stdout, self._frame_payload_bytes) is None: return
                else:
                    return
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

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try: self.proc.kill()
            except Exception: pass
