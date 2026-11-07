"""Saves (camera frame, steering) pairs while you drive. This is the training data."""
import csv
import time
from pathlib import Path

import cv2

from .config import CFG


class Recorder:
    def __init__(self, tag="session", folder=None, every_n=2):
        name = f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.dir = Path(folder) if folder else CFG.data_dir / name
        (self.dir / "images").mkdir(parents=True, exist_ok=True)
        self._csv_path = self.dir / "labels.csv"
        new = not self._csv_path.exists()
        self._f = open(self._csv_path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(["image", "steer", "throttle", "time"])
        self.count = 0
        self._tick = 0
        self.every_n = every_n          # save every Nth frame (30fps -> 15fps)

    def add(self, frame, steer, throttle):
        self._tick += 1
        if self._tick % self.every_n:
            return False
        name = f"{self.count:06d}.jpg"
        cv2.imwrite(str(self.dir / "images" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self._w.writerow([f"images/{name}", f"{steer:.4f}", f"{throttle:.3f}", f"{time.time():.3f}"])
        self.count += 1
        if self.count % 50 == 0:
            self._f.flush()
        return True

    def close(self):
        self._f.flush()
        self._f.close()
