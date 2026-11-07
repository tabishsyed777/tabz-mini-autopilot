"""Tiny driving simulator so you can test everything without a car.

It draws a perspective road with curves, a stop sign, a traffic light and a
box obstacle, and moves a virtual car with a bicycle-model. The same driving
code that runs on the real car runs against this.
"""
import math

import cv2
import numpy as np

from .config import CFG

HORIZON = 0.38          # horizon row as a fraction of image height
ROAD_HALF_W = 0.35      # metres
FOCAL = 210.0           # pixels (tuned by eye)
CAM_H = 0.12            # camera height in metres
WHEELBASE = 0.26
MAX_DELTA = 0.45        # rad, full steering lock


def curvature(s: float) -> float:
    """Road curvature (1/m) at distance s along the track. +ve = bends right."""
    return 0.35 * math.sin(s * 0.55) + 0.12 * math.sin(s * 1.3 + 1.0)


class SimWorld:
    def __init__(self, seed=0, with_events=True):
        self.rng = np.random.default_rng(seed)
        self.s = 0.0            # distance travelled
        self.x = 0.0            # lateral offset to the right of centre (m)
        self.psi = 0.0          # heading relative to road (rad), +ve = right
        self.v = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.with_events = with_events
        # events placed along the track: (kind, s position)
        self.events = []
        if with_events:
            self.events = [("stop", 9.0), ("light_red", 18.0), ("obstacle", 27.0),
                           ("stop", 38.0), ("light_green", 47.0)]
        self.light_timer = {}
        self.crashed = False

    # ---------------- physics ----------------
    def step(self, steer, throttle, dt):
        self.steer, self.throttle = steer, throttle
        target_v = max(0.0, throttle) * 1.4          # m/s at full throttle
        self.v += (target_v - self.v) * min(1.0, dt * 4.0)
        delta = steer * MAX_DELTA
        k = curvature(self.s)
        self.psi += (self.v * math.tan(delta) / WHEELBASE - self.v * k) * dt
        self.psi = max(-1.2, min(1.2, self.psi))
        self.x += self.v * math.sin(self.psi) * dt
        self.s += self.v * math.cos(self.psi) * dt
        if abs(self.x) > ROAD_HALF_W + 0.25:
            self.crashed = True

    def reset_to_road(self):
        self.x, self.psi, self.crashed = 0.0, 0.0, False

    # ---------------- sensing ----------------
    def ultrasonic_cm(self) -> float:
        for kind, pos in self.events:
            if kind == "obstacle" and 0.0 < pos - self.s < 2.0 and abs(self.x) < 0.3:
                return max(5.0, (pos - self.s) * 100.0)
        return 999.0

    def _light_is_green(self, pos):
        # lights turn green after the car has been near them for a while
        return (self.s > pos - 1.5 and self.light_timer.get(pos, 0) > 4.0)

    def tick_lights(self, dt):
        """Lights turn green and obstacles get 'removed' after the car has waited a while."""
        for kind, pos in list(self.events):
            if kind in ("light_red", "obstacle") and 0.0 < pos - self.s < 1.8 and self.v < 0.05:
                self.light_timer[pos] = self.light_timer.get(pos, 0) + dt
                if kind == "obstacle" and self.light_timer[pos] > 5.0:
                    self.events.remove((kind, pos))

    # ---------------- rendering ----------------
    def _road_x(self, d, lateral):
        """Pixel x for a lateral offset (m) at depth d (m)."""
        return CFG.cam_width / 2 + FOCAL * lateral / d

    def render(self):
        W, H = CFG.cam_width, CFG.cam_height
        hy = int(H * HORIZON)
        img = np.zeros((H, W, 3), np.uint8)
        img[:hy] = (200, 170, 130)               # sky (BGR)
        img[hy:] = (60, 110, 60)                 # grass
        k = curvature(self.s)

        pts_l, pts_r, pts_c = [], [], []
        for y in range(hy + 2, H):
            d = FOCAL * CAM_H / (y - hy)
            center = -self.x - self.psi * d + 0.5 * k * d * d
            pts_l.append((self._road_x(d, center - ROAD_HALF_W), y))
            pts_r.append((self._road_x(d, center + ROAD_HALF_W), y))
            pts_c.append((self._road_x(d, center), y, d))
        poly = np.array(pts_l + pts_r[::-1], np.int32)
        cv2.fillPoly(img, [poly], (70, 70, 72))
        # edge lines with thickness shrinking with distance
        for (xl, y), (xr, _), (xc, _, d) in zip(pts_l, pts_r, pts_c):
            t = max(1, int(FOCAL * 0.025 / d))
            cv2.line(img, (int(xl) - t, y), (int(xl) + t, y), (235, 235, 235), 1)
            cv2.line(img, (int(xr) - t, y), (int(xr) + t, y), (235, 235, 235), 1)
            if int((self.s + d) * 4) % 2 == 0:   # dashed centre line
                cv2.line(img, (int(xc) - 1, y), (int(xc) + 1, y), (60, 200, 230), 1)

        # roadside objects
        for kind, pos in self.events:
            d = pos - self.s
            if not (0.35 < d < 4.0):
                continue
            y = int(hy + FOCAL * CAM_H / d)
            center = -self.x - self.psi * d + 0.5 * k * d * d
            if kind == "stop":
                cx = int(self._road_x(d, center + ROAD_HALF_W + 0.10))
                self._draw_stop(img, cx, y, d)
            elif kind.startswith("light"):
                cx = int(self._road_x(d, center - ROAD_HALF_W - 0.18))
                green = kind == "light_green" or self._light_is_green(pos)
                self._draw_light(img, cx, y, d, green)
            elif kind == "obstacle":
                cx = int(self._road_x(d, center))
                s = int(FOCAL * 0.16 / d)
                cv2.rectangle(img, (cx - s, y - 2 * s), (cx + s, y), (30, 140, 245), -1)
                cv2.rectangle(img, (cx - s, y - 2 * s), (cx + s, y), (20, 90, 180), 1)
        # sensor noise + slight blur so it isn't pixel-perfect
        noise = self.rng.normal(0, 4, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def _draw_stop(self, img, cx, ground_y, d):
        r = max(3, int(FOCAL * 0.11 / d))
        top = ground_y - int(FOCAL * 0.30 / d)
        cv2.line(img, (cx, top), (cx, ground_y), (150, 150, 150), max(1, r // 4))
        pts = np.array([[cx + r * math.cos(math.radians(22.5 + 45 * i)),
                         top + r * math.sin(math.radians(22.5 + 45 * i))] for i in range(8)], np.int32)
        cv2.fillPoly(img, [pts], (35, 35, 200))
        cv2.line(img, (cx - r // 2, top), (cx + r // 2, top), (245, 245, 245), max(1, r // 5))

    def _draw_light(self, img, cx, ground_y, d, green):
        r = max(3, int(FOCAL * 0.05 / d))
        top = ground_y - int(FOCAL * 0.30 / d)
        cv2.line(img, (cx, top), (cx, ground_y), (150, 150, 150), max(1, r // 3))
        cv2.rectangle(img, (cx - r - 1, top - r * 2 - 1), (cx + r + 1, top + r * 2 + 1), (25, 25, 25), -1)
        col = (60, 220, 60) if green else (40, 40, 230)
        cy = top + (r if green else -r)
        cv2.circle(img, (cx, cy), r, col, -1)
