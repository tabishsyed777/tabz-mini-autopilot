"""Web dashboard: live camera, Tesla-style scene view, and remote control.

Open  http://<car-ip>:5000  from a phone or laptop on the same Wi-Fi.
"""
import math
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

STATIC = Path(__file__).parent / "static"


class SharedState:
    """Thread-safe box the driving loop writes into and the web server reads from."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = b""
        self.telemetry = {}
        # commands coming from the browser
        self.mode = "manual"            # manual | auto | classic
        self.record = False
        self.manual_steer = 0.0
        self.manual_throttle = 0.0
        self.manual_stamp = 0.0
        self.estop = False

    def controls(self):
        with self.lock:
            return {
                "mode": self.mode,
                "record": self.record,
                "manual_steer": self.manual_steer,
                "manual_throttle": self.manual_throttle,
                "manual_stamp": self.manual_stamp,
                "estop": self.estop,
            }

    def publish(self, frame_bgr, telemetry: dict):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self.lock:
            if ok:
                self.jpeg = buf.tobytes()
            self.telemetry = telemetry


def create_app(state: SharedState) -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/video")
    def video():
        def gen():
            last = None
            while True:
                with state.lock:
                    jpg = state.jpeg
                if jpg and jpg is not last:
                    last = jpg
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.03)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    @app.get("/api/state")
    def get_state():
        with state.lock:
            t = dict(state.telemetry)
            t.update(mode=state.mode, record=state.record, estop=state.estop)
        return jsonify(t)

    @app.post("/api/control")
    def control():
        d = request.get_json(force=True, silent=True) or {}
        try:
            steer = float(d.get("steer", 0))
            throttle = float(d.get("throttle", 0))
        except (TypeError, ValueError):
            return jsonify(ok=False, error="steer and throttle must be numbers"), 400
        if not math.isfinite(steer) or not math.isfinite(throttle):
            return jsonify(ok=False, error="steer and throttle must be finite numbers"), 400
        with state.lock:
            if state.estop:
                return jsonify(ok=False, error="emergency stop is latched"), 423
            state.manual_steer = max(-1.0, min(1.0, steer))
            state.manual_throttle = max(-1.0, min(1.0, throttle))
            state.manual_stamp = time.time()
        return jsonify(ok=True)

    @app.post("/api/mode")
    def mode():
        m = (request.get_json(force=True, silent=True) or {}).get("mode")
        with state.lock:
            if m in ("manual", "auto", "classic") and not state.estop:
                state.mode = m
            return jsonify(mode=state.mode, estop=state.estop)

    @app.post("/api/record")
    def record():
        with state.lock:
            state.record = bool((request.get_json(force=True, silent=True) or {}).get("on"))
            return jsonify(record=state.record)

    @app.post("/api/estop")
    def estop():
        with state.lock:
            state.estop = True
            state.mode = "manual"
            state.manual_steer = 0.0
            state.manual_throttle = 0.0
            state.manual_stamp = 0.0
        return jsonify(ok=True)

    @app.post("/api/estop/reset")
    def reset_estop():
        # Reset only clears the latch. The car remains stopped until a new
        # manual command arrives or the operator explicitly selects autonomy.
        with state.lock:
            state.estop = False
            state.manual_steer = 0.0
            state.manual_throttle = 0.0
            state.manual_stamp = 0.0
        return jsonify(ok=True, estop=False)

    return app


def start(state: SharedState, host: str, port: int):
    app = create_app(state)
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    t = threading.Thread(target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False),
                         daemon=True)
    t.start()
    return t
