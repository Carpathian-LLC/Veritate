# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - in-memory pub/sub for the live training stream (v8, tier 4). trainers call
#   publish(payload) per-step when --mri-stream is set; the dashboard subscribes
#   via an SSE route in app.py and renders incoming TFRM-lite frames.
# - frames are not persisted. canonical record stays the per-checkpoint dump
#   suite. this is purely the live "watch the brain form" channel.
# veritate_mri/training/train_stream.py
# ------------------------------------------------------------------------------------
# Imports:

import queue
import threading

# ------------------------------------------------------------------------------------
# Constants

QUEUE_MAX        = 256
SUBSCRIBER_LOCK  = threading.Lock()
SUBSCRIBERS      = []

# Wakeup period for subscribe()'s blocking get. Lets the route emit a keepalive
# and lets generator close drain the queue when the client disconnects.
SUBSCRIBE_TICK_S = 15.0

# ------------------------------------------------------------------------------------
# Functions

def publish(payload):
    """Trainers call this once per step (or per-N steps) when --mri-stream is on.
    Drops the payload if a subscriber's queue is full so the trainer is never
    blocked by a slow client. payload is a dict; the route json-encodes it."""
    with SUBSCRIBER_LOCK:
        live = list(SUBSCRIBERS)
    for q in live:
        try:
            q.put_nowait(payload)
        except queue.Full:
            pass


def subscribe():
    """Generator: registers a fresh queue, yields each payload as it arrives.
    Yields None on each SUBSCRIBE_TICK_S timeout so the SSE route can emit a
    keepalive comment and so a disconnected client's GeneratorExit fires
    promptly (rather than blocking forever in q.get())."""
    q = queue.Queue(maxsize=QUEUE_MAX)
    with SUBSCRIBER_LOCK:
        SUBSCRIBERS.append(q)
    try:
        while True:
            try:
                payload = q.get(timeout=SUBSCRIBE_TICK_S)
            except queue.Empty:
                yield None
                continue
            yield payload
    finally:
        with SUBSCRIBER_LOCK:
            try:
                SUBSCRIBERS.remove(q)
            except ValueError:
                pass


def subscriber_count():
    with SUBSCRIBER_LOCK:
        return len(SUBSCRIBERS)
