"""Mini Autopilot: main driving loop.

Examples
  python run.py                            # safe simulator, no car required
  python run.py --camera usb               # laptop webcam, no motors (dry run)
  python run.py --camera pi                # Pi camera/sensor dry run
  python run.py --camera pi --enable-motors # explicitly arm the real car
Open http://localhost:5000 to see the dashboard and switch modes.
"""
import argparse
import time

import cv2
import numpy as np

from autopilot.config import CFG
from autopilot.perception import LaneDetector, SignDetector
from autopilot.planner import Planner
from autopilot.recorder import Recorder
from dashboard.server import SharedState, start as start_dashboard

MANUAL_TIMEOUT = 0.4      # manual commands older than this are ignored -> car stops


class NullUltrasonic:
    def read_cm(self):
        return 999.0

    def close(self):
        pass


class NullCar:
    def set(self, steer, throttle):
        pass

    def stop(self):
        pass

    def close(self):
        pass


def build_io(args):
    if args.sim:
        from autopilot.hardware.sim_io import SimRig
        rig = SimRig(seed=args.seed)
        return rig, rig, rig
    from autopilot.hardware.camera import CvCamera, PiCamera
    if args.camera == "pi":
        cam = PiCamera()
    elif args.camera == "usb":
        cam = CvCamera(0)
    else:
        cam = CvCamera(args.camera)          # path to a video file
    if args.camera == "pi":
        from autopilot.hardware.ultrasonic import RealUltrasonic
        us = RealUltrasonic()
        if args.enable_motors and not args.no_motors:
            from autopilot.hardware.car import RealCar
            return cam, RealCar(), us
        return cam, NullCar(), us
    return cam, NullCar(), NullUltrasonic()


def annotate(frame, lane, dets):
    out = frame.copy()
    if len(lane.left) > 1:
        cv2.polylines(out, [np.array(lane.left, "int32")], False, (255, 160, 60), 2)
        cv2.polylines(out, [np.array(lane.right, "int32")], False, (255, 160, 60), 2)
        cv2.polylines(out, [np.array(lane.center, "int32")], False, (80, 220, 120), 2)
    colors = {"stop_sign": (60, 60, 240), "red_light": (60, 60, 240),
              "green_light": (80, 220, 80), "obstacle": (40, 160, 250)}
    for d in dets:
        x, y, w, h = d.box
        c = colors.get(d.kind, (255, 255, 255))
        cv2.rectangle(out, (x - 2, y - 2), (x + w + 2, y + h + 2), c, 1)
        cv2.putText(out, d.kind.replace("_", " "), (x, max(10, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--sim", dest="sim", action="store_true", help="built-in simulator (default)")
    source.add_argument("--camera", dest="camera", help="pi | usb | path/to/video.mp4")
    ap.add_argument("--mode", default=None, choices=["manual", "auto", "classic"],
                    help="default: classic in --sim, manual on the car")
    ap.add_argument("--model", default=str(CFG.model_path))
    ap.add_argument("--enable-motors", action="store_true",
                    help="arm physical motors (Pi camera only; default is dry run)")
    ap.add_argument("--no-motors", action="store_true", help="force camera/sensor dry run")
    ap.add_argument("--dashboard-host", default=CFG.dash_host,
                    help="dashboard bind address; default localhost (LAN access has no authentication)")
    ap.add_argument("--record", action="store_true", help="start recording straight away")
    ap.add_argument("--seed", type=int, default=0)
    ap.set_defaults(sim=False, camera=None)
    args = ap.parse_args()

    # No source flag means simulator; selecting --camera opts into a camera source.
    if args.camera is None:
        args.sim = True
        args.camera = "pi"

    if args.enable_motors and (args.sim or args.camera != "pi" or args.no_motors):
        ap.error("--enable-motors requires --camera pi and cannot be combined with --no-motors")

    if args.mode is None:
        args.mode = "classic" if args.sim else "manual"
    cam, car, us = build_io(args)
    lane, signs, planner = LaneDetector(), SignDetector(), Planner()
    state = SharedState()
    state.mode, state.record = args.mode, args.record
    start_dashboard(state, args.dashboard_host, CFG.dash_port)
    print(f"Dashboard: http://localhost:{CFG.dash_port}  (mode: {args.mode})")
    if args.dashboard_host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: dashboard is unauthenticated and includes driving controls; expose it only on a trusted network.")
    if args.camera == "pi" and not args.sim:
        print("Hardware mode: motors " + ("ARMED" if args.enable_motors else "disabled") + "; ultrasonic sensor active.")

    pilot = None
    try:
        from autopilot.neural import NeuralPilot
        pilot = NeuralPilot(args.model)
        print("Neural model loaded:", args.model)
    except Exception as e:
        print(f"No neural model ({type(e).__name__}). 'auto' mode will use the classic lane detector.")

    recorder = None
    fps, t_prev, steer_out, thr_out = 0.0, time.time(), 0.0, 0.0
    try:
        while True:
            frame = cam.read()
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now

            l = lane.estimate(frame)
            dets = signs.detect(frame)
            dist = us.read_cm()
            controls = state.controls()
            mode = controls["mode"]
            net_steer = None
            if pilot is not None:
                net_steer = pilot.predict(frame)

            if controls["estop"]:
                steer_out, thr_out, plan_state, reason = 0.0, 0.0, "E-STOP", "Emergency stop pressed"
            elif mode == "manual":
                fresh = now - controls["manual_stamp"] < MANUAL_TIMEOUT
                steer_out = controls["manual_steer"] if fresh else 0.0
                thr_out = controls["manual_throttle"] * 0.6 if fresh else 0.0
                plan_state, reason = "MANUAL", "Human driving"
            else:
                steer_cmd = net_steer if (mode == "auto" and net_steer is not None) else l.steer
                plan = planner.update(steer_cmd, dets, dist)
                steer_out, thr_out, plan_state, reason = plan.steer, plan.throttle, plan.state, plan.reason

            car.set(steer_out, thr_out)

            # --- data engine: record human driving, and hard cases from the autopilot ---
            if controls["record"] and recorder is None:
                recorder = Recorder("manual" if mode == "manual" else "auto")
            if not controls["record"] and recorder is not None:
                recorder.close(); recorder = None
            if recorder is not None:
                if mode == "manual":
                    recorder.add(frame, steer_out, thr_out)
                elif net_steer is not None and l.confidence > 0.6 and abs(net_steer - l.steer) > 0.45:
                    recorder.add(frame, l.steer, thr_out)    # model and lane detector disagree

            W = frame.shape[1]
            objs = [{"kind": d.kind, "dist": d.dist_m,
                     "lateral": (d.box[0] + d.box[2] / 2 - W / 2) / (W / 2)} for d in dets]
            if dist < 150:
                objs.append({"kind": "obstacle", "dist": dist / 100.0, "lateral": 0.0})
            state.publish(annotate(frame, l, dets), {
                "steer": round(steer_out, 3), "throttle": round(thr_out, 3),
                "power": round(abs(thr_out) * 100),
                "plan": plan_state, "reason": reason, "fps": round(fps, 1),
                "lane_conf": round(l.confidence, 2), "lane_offset": round(l.offset, 3),
                "lane_curve": round(
                    ((l.center[-1][0] - l.center[0][0]) / (W / 2)) if len(l.center) > 3 else 0.0, 3),
                "ultrasonic_cm": round(min(dist, 400), 1), "objects": objs,
                "net_steer": None if net_steer is None else round(net_steer, 3),
                "lane_steer": round(l.steer, 3),
                "recording": recorder.count if recorder else None,
                "has_model": pilot is not None,
            })
            if args.sim:
                time.sleep(max(0.0, 1 / 30 - (time.time() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        car.stop(); car.close(); cam.close(); us.close()
        if recorder:
            recorder.close()
        print("\nStopped safely.")


if __name__ == "__main__":
    main()
