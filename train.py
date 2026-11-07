"""Train the steering network on your recorded driving data.

    python train.py                       # uses everything in data/
    python train.py --epochs 30 --batch 64
Output: models/autopilot.pt and models/autopilot.onnx (copy the .onnx to the car).
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from autopilot.config import CFG
from autopilot.dataset import balance, list_samples, load_xy
from autopilot.model import PilotNet


class DrivingSet(Dataset):
    def __init__(self, samples, train):
        self.samples, self.train = samples, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        # fresh RNG per call so DataLoader workers don't repeat each other's augmentations
        x, y = load_xy(self.samples[i], self.train, np.random.default_rng())
        return torch.from_numpy(x), torch.tensor(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.15, help="fraction kept for validation")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.epochs < 1 or args.batch < 1 or not 0.0 < args.val < 0.5:
        raise SystemExit("epochs and batch must be positive; val must be between 0 and 0.5")

    samples = list_samples()
    if len(samples) < 200:
        raise SystemExit(f"Only {len(samples)} frames found in {CFG.data_dir}. "
                         "Record at least a few thousand (see docs/BUILD_GUIDE.md).")
    samples = balance(samples)
    # Split by contiguous chunks, not randomly: neighbouring frames look identical,
    # so a random split would leak training images into validation.
    chunk = 60
    blocks = [samples[i:i + chunk] for i in range(0, len(samples), chunk)]
    rng = np.random.default_rng(0)
    rng.shuffle(blocks)
    n_val = max(1, int(len(blocks) * args.val))
    val = [s for b in blocks[:n_val] for s in b]
    train = [s for b in blocks[n_val:] for s in b]
    print(f"train {len(train)} | val {len(val)} frames")

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", dev)
    model = PilotNet().to(dev)
    tl = DataLoader(DrivingSet(train, True), args.batch, shuffle=True, num_workers=args.workers)
    vl = DataLoader(DrivingSet(val, False), args.batch, num_workers=args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                 total_steps=args.epochs * max(1, len(tl)))
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    best = 1e9
    CFG.model_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path = CFG.model_path.with_suffix(".pt")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        model.eval()
        vtot, mae = 0.0, 0.0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                p = model(x)
                vtot += loss_fn(p, y).item() * len(x)
                mae += (p - y).abs().sum().item()
        vloss, mae = vtot / len(val), mae / len(val)
        flag = ""
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), pt_path)
            flag = " *saved*"
        print(f"epoch {ep:3d}/{args.epochs} train {tot/len(train):.4f} "
              f"val {vloss:.4f} | steering error {mae:.3f} | {time.time()-t0:.0f}s{flag}")

    # ---- export best weights to ONNX ----
    model.load_state_dict(torch.load(pt_path, map_location=dev))
    model.eval().cpu()
    dummy = torch.zeros(1, 3, CFG.net_height, CFG.net_width)
    torch.onnx.export(model, dummy, str(CFG.model_path), input_names=["image"],
                      output_names=["steer"], opset_version=17, dynamic_axes=None)
    print(f"\nExported {CFG.model_path}. Copy it to the car's models/ folder.")


if __name__ == "__main__":
    main()
