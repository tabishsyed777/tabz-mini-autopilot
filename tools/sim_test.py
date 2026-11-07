"""Headless self-test: drives the simulator with the classical autopilot.
Run:  python tools/sim_test.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autopilot.hardware.sim_io import SimRig
from autopilot.perception import LaneDetector, SignDetector
from autopilot.planner import Planner
import autopilot.planner as P


def main(seconds=90, verbose=True):
    rig = SimRig(seed=1, realtime=False)
    lane, signs, plan = LaneDetector(), SignDetector(), Planner()
    # simulated clock so the test runs fast
    t = [0.0]
    P.time.time = lambda: t[0]
    max_off, states, crashes = 0.0, set(), 0
    for i in range(int(seconds * 30)):
        t[0] += 1 / 30
        frame = rig.read()
        l = lane.estimate(frame)
        d = signs.detect(frame)
        p = plan.update(l.steer, d, rig.read_cm())
        rig.set(p.steer, p.throttle)
        states.add(p.state)
        max_off = max(max_off, abs(rig.world.x))
        if verbose and i % 150 == 0:
            print(f"t={t[0]:5.1f}s s={rig.world.s:5.1f}m x={rig.world.x:+.2f} "
                  f"steer={p.steer:+.2f} thr={p.throttle:.2f} {p.state:10s} {p.reason}")
    print("distance:", round(rig.world.s, 1), "m | max offset:", round(max_off, 2),
          "m | states:", sorted(states))
    return rig.world.s, max_off, states

if __name__ == "__main__":
    main()
