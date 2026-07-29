from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
from modules import Conv, C3k2, C2PSA



class YOLOv11BackboneVariantV4(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 256, 3, 2)

        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = YOLOv11BackboneVariantV4()
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print(y.shape)
    torch.save(model.state_dict(), "yolov11_backbone_variant_v4.pt")

