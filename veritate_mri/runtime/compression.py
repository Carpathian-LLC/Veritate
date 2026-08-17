# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - single owner of http response compression. app.py calls register(app); no route
#   module gzips anything itself.
# - sse streams are compressed with a Z_SYNC_FLUSH per source chunk, so every event
#   still leaves the server the moment the engine produces it.
# veritate_mri/runtime/compression.py
# ------------------------------------------------------------------------------------
# Imports:

import zlib

from flask import request

# ------------------------------------------------------------------------------------
# Constants

# gzip container (header + crc trailer) around raw deflate.
GZIP_WBITS = 16 + zlib.MAX_WBITS
# Measured on cardinal-01 (i7-9700T pinned at 800 MHz) over a real 945 KB, 32-event
# /generate trace stream: level 1 = 9.85x at 0.65 ms/event, level 6 = 11.89x at
# 1.49 ms/event, level 9 = 12.47x at 6.62 ms/event. The engine spends ~29 ms
# producing an event, so level 6 buys 17% fewer bytes for under 5% of that budget
# and level 9 costs 23% of it for 5% more.
GZIP_LEVEL = 6
# A body under one TCP segment cannot save a round trip, and the gzip envelope
# makes the smallest ones larger.
MIN_COMPRESS_BYTES = 1024
# Everything outside this set is binary or already compressed (model .bin, images,
# archives): gzip would only burn CPU.
COMPRESSIBLE_MIMETYPES = frozenset((
    "text/event-stream",
    "text/html",
    "text/css",
    "text/plain",
    "text/csv",
    "text/javascript",
    "application/javascript",
    "application/json",
    "image/svg+xml",
))
HTTP_OK = 200

# ------------------------------------------------------------------------------------
# Functions

def register(app):
    """Install the response-compression hook. Register before the route modules:
    flask runs after_request handlers in reverse, so this one runs last and sees
    the final body."""
    app.after_request(_compress)


def _gzip_stream(chunks):
    """Compress a response iterator, syncing the deflate stream at every source
    chunk. Without the flush, zlib holds bytes back until its window fills and an
    sse event would sit in the compressor instead of reaching the client."""
    c = zlib.compressobj(GZIP_LEVEL, zlib.DEFLATED, GZIP_WBITS)
    try:
        for chunk in chunks:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            out = c.compress(chunk) + c.flush(zlib.Z_SYNC_FLUSH)
            if out:
                yield out
        yield c.flush()
    finally:
        # Release the wrapped generator (engine locks) or file handle at once,
        # including when the client disconnects mid-stream.
        closer = getattr(chunks, "close", None)
        if closer is not None:
            closer()


def _compress(response):
    """Gzip an eligible response in place. Untouched unless the client advertised
    gzip, so a client that does not gets exactly the bytes it got before."""
    if (response.status_code != HTTP_OK
            or response.mimetype not in COMPRESSIBLE_MIMETYPES
            or "Content-Encoding" in response.headers
            or not request.accept_encodings["gzip"]):
        return response
    known = response.content_length
    if known is not None and known < MIN_COMPRESS_BYTES:
        return response
    if response.is_streamed:
        # A file response declares a length that no longer matches once gzipped.
        response.headers.pop("Content-Length", None)
        response.response = _gzip_stream(response.response)
    else:
        body = response.get_data()
        if len(body) < MIN_COMPRESS_BYTES:
            return response
        response.set_data(b"".join(_gzip_stream([body])))
    response.headers["Content-Encoding"] = "gzip"
    response.vary.add("Accept-Encoding")
    return response
