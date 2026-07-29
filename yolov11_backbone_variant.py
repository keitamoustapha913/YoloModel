from __future__ import annotations

import math
import argparse

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
from modules import ( Conv,
                     PSABlock, PSA, C3k2, C2PSA, Attention, C2fPSA,
                     AAttn, ABlock, A2C2f )



class YOLOv11BackboneVariantV1(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),

        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV2(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 128, 3, 2)

        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV3(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 128, 3, 2)

        )

    def forward(self, x):
        return self.model(x)



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



class YOLOv11BackboneVariantV5(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 256, 3, 2),

        )

    def forward(self, x):
        return self.model(x)

class YOLOv11BackboneVariantV6(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 256, 3, 2),
            C2PSA(256, 256, 1)

        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV7(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 256, 3, 2),
            PSABlock(256, 0.5, 4, True)

        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV8(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 256, 3, 2),
            PSA(256, 256, 0.5)

        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV9(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 256, 3, 2),
            Attention(256, 4, 0.5)

        )

    def forward(self, x):
        return self.model(x)




class YOLOv11BackboneVariantV10(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 256, 3, 2),
            C2fPSA(256, 256, 1)

        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV11(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 256, 3, 2),
            AAttn(256, 8, 4)

        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV12(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 256, 3, 2),
            ABlock(256, 8, 1.2, 4)

        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV13(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 256, 3, 2),
            A2C2f(256, 256, 1, True, 4)

        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV14(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2)
        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV15(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2)
        )

    def forward(self, x):
        return self.model(x)

class YOLOv11BackboneVariantV16(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            PSABlock(128, 0.5, 4, True)
        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV17(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            C2fPSA(128, 128, 1)
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV18(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            A2C2f(128, 128, 1, True, 4)
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV19(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            ABlock(128, 8, 1.2, 4)
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV20(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2)
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV21(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2),
            PSABlock(64, 0.5, 4, True)
    )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV22(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            C2PSA(128, 128, 1)
    )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV23(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2),
            ABlock(64, 8, 1.2, 4)
        )

    def forward(self, x):
        return self.model(x)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default="v1",
        help="Model class variants of YOLOv11BackboneVariantVx; default: YOLOv11BackboneVariantV1",
    )

    return parser.parse_args()


def available_models() -> dict[str, type[nn.Module]]:
    # use globals() to get all classes defined in this module
    variant_name = "YOLOv11BackboneVariant".lower()
    return {
        name.lower().split(variant_name)[-1] : cls
        for name, cls in globals().items()
        if isinstance(cls, type) and issubclass(cls, nn.Module) and name.startswith("YOLOv11BackboneVariant")
    }




if __name__ == "__main__":

    args = parse_args()
    x = torch.randn(1, 3, 640, 640)
    model = available_models()[args.version]()
    y = model(x)
    print(y.shape)
    torch.save(model.state_dict(), f"yolov11_backbone_variant_{args.version}.pt")
