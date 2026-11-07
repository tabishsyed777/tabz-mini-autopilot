"""Behaviour planner: decides speed + whether to stop. Pure logic (easy to test).

States: DRIVE -> STOP_SIGN (wait N seconds) -> DRIVE
        DRIVE -> RED_LIGHT (wait until green)  -> DRIVE
        any   -> EMERGENCY (obstacle too close) -> DRIVE when clear
"""
import time
from dataclasses import dataclass
from typing import List, Optional

from .config import CFG
from .perception import Detection


@dataclass
class Plan:
    steer: float
    throttle: float
    state: str
    reason: str


class Planner:
    STOP_TRIGGER_M = 0.90     # start the stop when the sign is this close (m)
    LIGHT_TRIGGER_M = 0.80

    def __init__(self):
        self.state = "DRIVE"
        self._state_t = time.time()
        self._ignore_stop_until = 0.0
        self._smoothed = 0.0

    def _set(self, state):
        if state != self.state:
            self.state, self._state_t = state, time.time()

    def update(self, steer: float, dets: List[Detection], ultrasonic_cm: float,
               curve_hint: float = 0.0) -> Plan:
        now = time.time()
        a = CFG.steer_smoothing
        self._smoothed = a * self._smoothed + (1 - a) * steer
        steer = self._smoothed

        stop = next((d for d in dets if d.kind == "stop_sign"), None)
        red = next((d for d in dets if d.kind == "red_light"), None)
        green = any(d.kind == "green_light" for d in dets)

        # 1. Emergency brake always wins
        if ultrasonic_cm < CFG.brake_distance_cm:
            self._set("EMERGENCY")
            return Plan(0.0, 0.0, self.state, f"Obstacle {ultrasonic_cm:.0f} cm")
        if self.state == "EMERGENCY":
            self._set("DRIVE")

        # 2. Stop sign
        if self.state == "STOP_SIGN":
            left = CFG.stop_sign_pause_s - (now - self._state_t)
            if left > 0:
                return Plan(0.0, 0.0, self.state, f"Stop sign: {left:.1f}s")
            self._ignore_stop_until = now + 4.0
            self._set("DRIVE")
        elif (stop and stop.dist_m is not None and stop.dist_m < self.STOP_TRIGGER_M
              and now > self._ignore_stop_until):
            self._set("STOP_SIGN")
            return Plan(0.0, 0.0, self.state, "Stop sign ahead")

        # 3. Red light
        if self.state == "RED_LIGHT":
            if green or not red:
                if now - self._state_t > 0.6:
                    self._set("DRIVE")
            else:
                return Plan(0.0, 0.0, self.state, "Red light")
        elif red and red.dist_m is not None and red.dist_m < self.LIGHT_TRIGGER_M and not green:
            self._set("RED_LIGHT")
            return Plan(0.0, 0.0, self.state, "Red light")

        # 4. Normal driving, slow for curves and nearby objects
        speed = CFG.cruise_speed
        reason = "Lane keeping"
        if abs(steer) > 0.55:
            speed = CFG.slow_speed; reason = "Slowing for curve"
        if ultrasonic_cm < CFG.slow_distance_cm:
            speed = min(speed, CFG.slow_speed * 0.8); reason = "Object ahead, slowing"
        return Plan(steer, speed, self.state, reason)
