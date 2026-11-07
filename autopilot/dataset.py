"""Loading + augmenting driving data. Numpy/OpenCV only (no torch needed)."""
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .config import CFG
from .perception import preprocess

Sample = Tuple[Path, float]


def list_samples(data_dir: Path = None, min_throttle: float = 0.05) -> List[Sample]:
    """Read every data/*/labels.csv. Frames where the car was stopped are skipped."""
    data_dir = Path(data_dir or CFG.data_dir)
    out: List[Sample] = []
    for csv_path in sorted(data_dir.glob("*/labels.csv")):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    if float(row.get("throttle", 1)) < min_throttle:
                        continue
                    out.append((csv_path.parent / row["image"], float(row["steer"])))
                except (KeyError, ValueError):
                    continue
    return out


def balance(samples: List[Sample], keep_straight: float = 0.35, seed: int = 0) -> List[Sample]:
    """Most driving is straight. Drop some straight frames so the net learns to turn too."""
    rng = np.random.default_rng(seed)
    return [s for s in samples if abs(s[1]) > 0.08 or rng.random() < keep_straight]


def augment(img: np.ndarray, steer: float, rng: np.random.Generator):
    """Random changes that make the network robust. img is BGR full-size."""
    h, w = img.shape[:2]
    # 1. mirror: flipping the picture flips the steering
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        steer = -steer
    # 2. sideways shift (+ matching steering correction, approximate)
    shift = int(rng.integers(-w // 12, w // 12 + 1))
    if shift:
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        steer += 1.2 * shift / w
    # 3. brightness / contrast
    img = np.clip(img.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25), 0, 255)
    # 4. random shadow
    if rng.random() < 0.4:
        x1, x2 = sorted(rng.integers(0, w, 2))
        shade = img.copy()
        poly = np.array([[x1, 0], [x2, 0], [x2 + int(rng.integers(-40, 40)), h],
                         [x1 + int(rng.integers(-40, 40)), h]], np.int32)
        cv2.fillPoly(shade, [poly], (0, 0, 0))
        img = cv2.addWeighted(img, 0.6, shade, 0.4, 0)
    # 5. blur / noise
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img, float(np.clip(steer, -1, 1))


def load_xy(sample: Sample, train: bool, rng: np.random.Generator):
    path, steer = sample
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    if train:
        img, steer = augment(img, steer, rng)
    return preprocess(img), np.float32(steer)
