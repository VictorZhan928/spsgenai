# app/ebm_model.py
import torch
import torch.nn as nn

class EnergyCNN(nn.Module):
    """
    Small CNN that outputs a scalar 'energy' for a 3x32x32 image (CIFAR-10).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,128,4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128,256,4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256,512,4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512*2*2, 1)  # scalar energy
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,)
