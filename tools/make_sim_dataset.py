"""Generate training data in the simulator, no car needed.

A classical lane-follower drives; we randomly shove the steering now and then
(so the data contains recoveries) but record the CLEAN steering as the label.

    python tools/make_sim_dataset.py --frames 6000
    python train.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autopilot.hardware.sim_io import SimRig
from autopilot.perception import LaneDetector
from autopilot.recorder import Recorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rig = SimRig(seed=args.seed, with_events=False, realtime=False)
    rig.world.s = float(rng.uniform(0, 200))          # random start on the track
    lane = LaneDetector()
    rec = Recorder(tag="sim", every_n=2)
    push, push_left = 0.0, 0
    n = 0
    while rec.count < args.frames:
        frame = rig.read()
        r = lane.estimate(frame)
        if r.confidence > 0.3:
            rec.add(frame, r.steer, 0.35)
        if push_left == 0 and rng.random() < 0.012:
            push, push_left = float(rng.uniform(-0.6, 0.6)), int(rng.integers(6, 18))
        steer = r.steer + (push if push_left > 0 else 0.0)
        push_left = max(0, push_left - 1)
        rig.set(float(np.clip(steer, -1, 1)), 0.35)
        n += 1
        if n % 900 == 0:                               # hop to another stretch of road
            rig.world.s += float(rng.uniform(5, 60)); rig.world.reset_to_road()
        if n > args.frames * 6:
            break
    rec.close()
    print(f"Saved {rec.count} frames to {rec.dir}")


if __name__ == "__main__":
    main()
