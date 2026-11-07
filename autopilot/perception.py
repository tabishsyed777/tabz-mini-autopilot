"""Computer vision: lane finding, sign/light detection, model preprocessing.

Two ways to steer:
  1. LaneDetector : classical OpenCV (works instantly, no training needed).
  2. NeuralPilot  : end-to-end CNN trained on YOUR driving (see train.py).
Detectors for stop signs and traffic lights are classical colour+shape.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import CFG


# ----------------------------------------------------------------------
# Preprocessing shared by training and driving (they MUST match)
# ----------------------------------------------------------------------
def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame -> float32 array shaped (3, H, W) in 0..1 for the network."""
    h = frame_bgr.shape[0]
    crop = frame_bgr[int(h * CFG.crop_top_frac):, :, :]
    small = cv2.resize(crop, (CFG.net_width, CFG.net_height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)


# ----------------------------------------------------------------------
# Classical lane detector
# ----------------------------------------------------------------------
@dataclass
class LaneResult:
    steer: float = 0.0                 # -1..+1
    confidence: float = 0.0            # 0..1
    offset: float = 0.0                # lane-centre offset, -1..+1 (+ = car is right of centre)
    left: List[Tuple[int, int]] = field(default_factory=list)    # image-space polyline
    right: List[Tuple[int, int]] = field(default_factory=list)
    center: List[Tuple[int, int]] = field(default_factory=list)


class LaneDetector:
    """Find the two lane lines in the lower part of the image and steer to their middle.

    Offsets are measured in *lane widths* at fixed look-ahead rows, so the result does
    not depend on camera height or lens. If only one line is visible we assume the
    lane is as wide as it was the last time we saw both.
    """

    NEAR_ROW = 0.80      # fraction of image height
    FAR_ROW = 0.62
    K_OFFSET = 1.5
    K_HEADING = 1.3

    def __init__(self):
        self.last_steer = 0.0
        self.width_by_row = {}          # row -> last good lane width in pixels

    def _mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if CFG.line_is_bright:
            thr = max(150, int(np.percentile(gray, 97) * 0.85))
            _, m = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        else:
            thr = min(90, int(np.percentile(gray, 3) * 1.4 + 10))
            _, m = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    def _row_center(self, mask, y, hint, W):
        """Return (centre_x, lane_width, left_x, right_x, both_seen) for one row, or None."""
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            return None
        lx, rx = xs[xs < hint], xs[xs >= hint]
        l = int(np.median(lx[-5:])) if lx.size else None
        r = int(np.median(rx[:5])) if rx.size else None
        if l is not None and r is not None and 25 < r - l < W * 1.6:
            self.width_by_row[y // 4] = r - l
            return (l + r) / 2, r - l, l, r, True
        known = self.width_by_row.get(y // 4)
        if known is None and self.width_by_row:
            k = min(self.width_by_row, key=lambda kk: abs(kk - y // 4))
            known = self.width_by_row[k]
        if known is None:
            return None
        # only one line seen: decide which side from where it sits relative to the hint
        if l is not None and r is None:
            return l + known / 2, known, l, l + known, False
        if r is not None and l is None:
            return r - known / 2, known, r - known, r, False
        return None

    def estimate(self, frame) -> LaneResult:
        H, W = frame.shape[:2]
        mask = self._mask(frame)
        mask[:int(H * 0.50)] = 0
        res = LaneResult()
        mid = W / 2
        hint = mid
        centers, widths = [], []
        for y in range(H - 4, int(H * 0.52), -4):
            row = self._row_center(mask, y, hint, W)
            if row is None:
                continue
            c, w, l, r, both = row
            if both:
                hint = c
            res.left.append((int(l), y)); res.right.append((int(r), y))
            res.center.append((int(c), y))
            centers.append((y, c)); widths.append((y, w))
        if len(centers) < 5:
            res.steer = self.last_steer * 0.9
            res.confidence = 0.0
            return res

        ys = np.array([c[0] for c in centers], float)
        cs = np.array([c[1] for c in centers], float)
        ws = np.array([w[1] for w in widths], float)
        order = np.argsort(ys)
        ys, cs, ws = ys[order], cs[order], ws[order]

        def at(row_frac):
            y = H * row_frac
            c = float(np.interp(y, ys, cs))
            w = max(float(np.interp(y, ys, ws)), 10.0)
            return (c - mid) / w         # offset in lane widths (+ = centre is right of us)

        e_near, e_far = at(self.NEAR_ROW), at(self.FAR_ROW)
        raw = self.K_OFFSET * e_near + self.K_HEADING * (e_far - e_near)
        res.confidence = min(1.0, len(centers) / 18.0)
        res.offset = float(np.clip(-e_near, -1, 1))
        res.steer = float(np.clip(raw, -1, 1))
        self.last_steer = res.steer
        return res


# ----------------------------------------------------------------------
# Stop sign + traffic light detector (colour + shape)
# ----------------------------------------------------------------------
@dataclass
class Detection:
    kind: str                       # "stop_sign" | "red_light" | "green_light" | "obstacle"
    box: Tuple[int, int, int, int]  # x, y, w, h in frame pixels
    conf: float
    dist_m: Optional[float] = None  # rough distance estimate


class SignDetector:
    """Looks for red octagons (stop sign), red/green circles (traffic light) and orange boxes."""

    REAL_STOP_H_M = CFG.sign_height_m
    REAL_LIGHT_H_M = CFG.light_diam_m
    FOCAL_PX = CFG.focal_px

    @staticmethod
    def _dark_surround(hsv, x, y, w, h) -> bool:
        """Traffic lights sit in a black housing. Check pixels just above/below the blob."""
        H, W = hsv.shape[:2]
        pad_y, pad_x = int(h * 0.9), max(1, int(w * 0.15))
        x0, x1 = max(0, x - pad_x), min(W, x + w + pad_x)
        up = hsv[max(0, y - pad_y):y, x0:x1, 2]
        dn = hsv[y + h:min(H, y + h + pad_y), x0:x1, 2]
        px = np.concatenate([up.ravel(), dn.ravel()]) if up.size + dn.size else np.array([255])
        return float((px < 70).mean()) > 0.4

    def detect(self, frame) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out: List[Detection] = []

        red = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
        green = cv2.inRange(hsv, (45, 120, 110), (85, 255, 255))
        orange = cv2.inRange(hsv, (8, 150, 150), (22, 255, 255))
        k = np.ones((3, 3), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k)

        H = frame.shape[0]
        for cnt in cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(cnt)
            if area < 40:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.7 < w / h < 1.3):
                continue
            peri = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (peri * peri + 1e-6)
            approx = len(cv2.approxPolyDP(cnt, 0.04 * peri, True))
            fill = area / (w * h)
            if circ < 0.75 or fill < 0.68:
                continue
            if self._dark_surround(hsv, x, y, w, h):
                # round red blob inside a dark housing -> traffic light
                out.append(Detection("red_light", (x, y, w, h), 0.8,
                                     self.REAL_LIGHT_H_M * self.FOCAL_PX / max(h, 1)))
            elif w >= 9 and 6 <= approx <= 10 and y < H * 0.7:
                out.append(Detection("stop_sign", (x, y, w, h), min(1, area / 150),
                                     self.REAL_STOP_H_M * self.FOCAL_PX / max(h, 1)))
        for cnt in cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if area < 12 or not (0.7 < w / h < 1.4) or y > H * 0.6:
                continue
            peri = cv2.arcLength(cnt, True)
            if 4 * np.pi * area / (peri * peri + 1e-6) > 0.7:
                out.append(Detection("green_light", (x, y, w, h), 0.8,
                                     self.REAL_LIGHT_H_M * self.FOCAL_PX / max(h, 1)))
        for cnt in cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            out.append(Detection("obstacle", (x, y, w, h), 0.7, None))
        return out
