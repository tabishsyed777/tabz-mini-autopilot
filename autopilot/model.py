"""The neural network: a small PilotNet-style CNN. Camera image in, steering out."""
import torch
import torch.nn as nn

from .config import CFG


class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            block(3, 16, 5, 2),
            block(16, 32, 5, 2),
            block(32, 48, 3, 2),
            block(48, 64, 3, 1),
            block(64, 64, 3, 2),
        )
        self.features.eval()                      # don't touch BatchNorm stats with a dummy input
        with torch.no_grad():
            n = self.features(torch.zeros(1, 3, CFG.net_height, CFG.net_width)).numel()
        self.features.train()
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(n, 100), nn.ReLU(inplace=True),
            nn.Linear(100, 24), nn.ReLU(inplace=True),
            nn.Linear(24, 1), nn.Tanh(),          # steering in -1..+1
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)
