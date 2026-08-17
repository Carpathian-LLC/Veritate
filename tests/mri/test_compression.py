# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - response compression contract: opt-in on Accept-Encoding, byte-identical payload,
#   size floor, binary skip, and per-event flushing on sse.
# - the sse test is the latency guard: it asserts the first event is readable from the
#   first flushed chunk, which only holds while Z_SYNC_FLUSH runs per event.
# tests/mri/test_compression.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import zlib

from flask import Flask, Response
from runtime import compression

# ------------------------------------------------------------------------------------
# Constants

GZIP = {"Accept-Encoding": "gzip"}
EVENTS = 8
BIG_JSON = {"frames": [{"layer": i, "acts": [0.5] * 64} for i in range(64)]}
SMALL_JSON = {"ok": True}
BINARY_BODY = b"\x00\xff" * 4096

# ------------------------------------------------------------------------------------
# Functions

def _event(i):
    return f"data: {json.dumps({'kind': 'token', 'i': i, 'text': 'roman empire'})}\n\n"


def _app(compressed=True):
    app = Flask(__name__)
    if compressed:
        compression.register(app)

    @app.route("/json")
    def big():
        return BIG_JSON

    @app.route("/small")
    def small():
        return SMALL_JSON

    @app.route("/sse")
    def sse():
        return Response((_event(i) for i in range(EVENTS)), mimetype="text/event-stream")

    @app.route("/bin")
    def binary():
        return Response(BINARY_BODY, mimetype="application/octet-stream")

    return app.test_client()


def test_client_without_accept_encoding_gets_untouched_bytes():
    """A client that never advertises gzip receives the pre-change bytes and headers."""
    plain = _app(compressed=False).get("/json")
    served = _app().get("/json")
    assert served.get_data() == plain.get_data()
    assert "Content-Encoding" not in served.headers


def test_gzip_client_gets_encoded_body_that_decodes_identically():
    """With Accept-Encoding: gzip the body is gzip and decompresses to the plain bytes."""
    plain = _app(compressed=False).get("/json").get_data()
    r = _app().get("/json", headers=GZIP)
    assert r.headers["Content-Encoding"] == "gzip"
    assert r.headers["Vary"] == "Accept-Encoding"
    assert len(r.get_data()) < len(plain)
    assert zlib.decompress(r.get_data(), compression.GZIP_WBITS) == plain


def test_response_below_threshold_is_left_alone():
    """A body under MIN_COMPRESS_BYTES stays uncompressed, since gzip would grow it."""
    r = _app().get("/small", headers=GZIP)
    assert len(r.get_data()) < compression.MIN_COMPRESS_BYTES
    assert "Content-Encoding" not in r.headers


def test_binary_content_type_is_skipped():
    """Binary payloads (model bins, images) are never gzipped regardless of size."""
    r = _app().get("/bin", headers=GZIP)
    assert "Content-Encoding" not in r.headers
    assert r.get_data() == BINARY_BODY


def test_sse_flushes_every_event_instead_of_buffering():
    """Each sse event ships as its own flushed gzip chunk, decodable on arrival."""
    r = _app().get("/sse", headers=GZIP)
    chunks = [c for c in r.response if c]
    assert r.headers["Content-Encoding"] == "gzip"
    assert "Content-Length" not in r.headers
    assert len(chunks) > EVENTS - 1
    d = zlib.decompressobj(compression.GZIP_WBITS)
    assert d.decompress(chunks[0]) == _event(0).encode()
    assert b"".join(d.decompress(c) for c in chunks[1:]) == "".join(
        _event(i) for i in range(1, EVENTS)).encode()
