# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - log snapshot + sse log stream.
# veritate_mri/routes/logs_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import json

from flask import Response, request

from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

STREAM_KEEPALIVE_SECS = 15.0

# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/logs/snapshot")
    def logs_snapshot():
        after = int(request.args.get("after", "0"))
        limit = request.args.get("limit")
        rows = logmod.snapshot(after_seq=after, limit=int(limit) if limit else None)
        return {"latest_seq": logmod.latest_seq(), "entries": rows}

    @app.route("/logs/stream")
    def logs_stream():
        q = logmod.subscribe()

        def stream():
            try:
                while True:
                    try:
                        entry = q.get(timeout=STREAM_KEEPALIVE_SECS)
                        yield f"data: {json.dumps(entry)}\n\n"
                    except Exception:
                        yield ": keepalive\n\n"
            except GeneratorExit:
                return
            finally:
                logmod.unsubscribe(q)

        return Response(stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
