from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
from modules import Conv, C3k2, C2PSA



"""

                   from  n    params  module                                       arguments

0 -1 1 464 ultralytics.nn.modules.conv.Conv [3, 16, 3, 2]  
 1 -1 1 4672 ultralytics.nn.modules.conv.Conv [16, 32, 3, 2]  
 2 -1 1 6640 ultralytics.nn.modules.block.C3k2 [32, 64, 1, False, 0.25]  
 3 -1 1 36992 ultralytics.nn.modules.conv.Conv [64, 64, 3, 2]  
 4 -1 1 26080 ultralytics.nn.modules.block.C3k2 [64, 128, 1, False, 0.25]  
 5 -1 1 147712 ultralytics.nn.modules.conv.Conv [128, 128, 3, 2]  
 6 -1 1 87040 ultralytics.nn.modules.block.C3k2 [128, 128, 1, True]  
 7 -1 1 295424 ultralytics.nn.modules.conv.Conv [128, 256, 3, 2]  
 8 -1 1 346112 ultralytics.nn.modules.block.C3k2 [256, 256, 1, True]  
 9 -1 1 249728 ultralytics.nn.modules.block.C2PSA [256, 256, 1]  

"""

class YOLOv11Backbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            C3k2(32, 64, 1, False, 0.25),
            Conv(64, 64, 3, 2),
            C3k2(64, 128, 1, False, 0.25),
            Conv(128, 128, 3, 2),
            C3k2(128, 128, 1, True),
            Conv(128, 256, 3, 2),
            C3k2(256, 256, 1, True),
            C2PSA(256, 256)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = YOLOv11Backbone()
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print(y.shape)

