from __future__ import annotations

import math
import argparse

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
from modules import (
    A2C2f,
    AAttn,
    ABlock,
    Attention,
    C2fPSA,
    C2PSA,
    C3k2,
    Conv,
    FeatureSpec,
    PSA,
    PSABlock,
    TransposeDecoderV2,
    TransposeDecoderV3,
    TransposeDecoderV4,
    LogicalReconstructionNetR2V2,
    LogicalReconstructionNetR2V3,
    LogicalReconstructionNetR2V4,
    LogicalReconstructionNetR2V5,
)



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
            Conv(64, 128, 3, 2),
            ABlock(128, 8, 1.2, 4)
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV24(nn.Module):
    """Patch-strided backbone with attention and a pooled latent output."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 32, k=8, s=8, p=0),
            Conv(32, 64, k=4, s=4, p=0),
            ABlock(64, 8, 1.2, 4),
            Conv(64, 32, k=3, s=2),
            Conv(32, 32, k=3, s=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV25(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2),
            PSABlock(64, 0.5, 4, True),
            Conv(64, 32, k=3, s=2),
            Conv(32, 32, k=3, s=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.model(x)




class YOLOv11BackboneVariantV26(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=128,
            width=128,
        )
        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2),
            PSABlock(64, 0.5, 4, True),
            Conv(64, 32, k=3, s=2),
            Conv(32, self.latent_dim, k=3, s=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            TransposeDecoderV2(
                latent_dim=self.latent_dim,
                output_spec=self.feature_spec,
            ),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV27(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=128,
            width=128,
        )
        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2),
            PSABlock(64, 0.5, 4, True),
            Conv(64, 32, k=3, s=2),
            Conv(32, self.latent_dim, k=3, s=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            TransposeDecoderV3(
                latent_dim=self.latent_dim,
                output_spec=self.feature_spec,
            ),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV28(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=128,
            width=128,
        )
        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2),
            PSABlock(64, 0.5, 4, True),
            Conv(64, 32, k=3, s=2),
            Conv(32, self.latent_dim, k=3, s=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            TransposeDecoderV4(
                latent_dim=self.latent_dim,
                output_spec=self.feature_spec,
            ),
        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV29(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=128,
            width=128,
        )
        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2),
            PSABlock(64, 0.5, 4, True),
            Conv(64, self.latent_dim, k=3, s=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            TransposeDecoderV4(
                latent_dim=self.latent_dim,
                output_spec=self.feature_spec,
            ),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV30(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=128,
            width=128,
        )
        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 2),
            Conv(64, self.latent_dim, k=3, s=2),
            PSABlock(self.latent_dim, 0.5, 4, True),
            nn.AdaptiveAvgPool2d((1, 1)),
            TransposeDecoderV4(
                latent_dim=self.latent_dim,
                output_spec=self.feature_spec,
            ),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV31(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=128,
            width=128,
        )
        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, self.latent_dim, 3, 2),
            PSABlock(self.latent_dim, 0.5, 4, True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV32(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=128,
            width=128,
        )
        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, self.latent_dim, 3, 2),
            PSABlock(self.latent_dim, 0.5, 4, True),
            nn.AdaptiveAvgPool2d((1, 1)),
            TransposeDecoderV4(
                latent_dim=self.latent_dim,
                output_spec=self.feature_spec,
            ),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV33(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        output_specs = [
            FeatureSpec(128, 128, 128),
            FeatureSpec(128, 128, 128),
            FeatureSpec(128, 128, 128),
        ]

        image_size = 640

        self.model = nn.Sequential(
            LogicalReconstructionNetR2V2(
                image_size=image_size,
                output_specs=output_specs,
            ),
        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV34(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        output_specs = [
            FeatureSpec(128, 128, 128),
            FeatureSpec(128, 128, 128),
            FeatureSpec(128, 128, 128),
        ]

        image_size = 640

        self.model = nn.Sequential(
            LogicalReconstructionNetR2V3(
                image_size=image_size,
                output_specs=output_specs,
            ),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV35(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        output_specs = [
            FeatureSpec(128, 20, 20),
            FeatureSpec(128, 10, 10),
            FeatureSpec(128, 5, 5),
        ]

        image_size = 640

        self.model = nn.Sequential(
            LogicalReconstructionNetR2V4(
                image_size=image_size,
                output_specs=output_specs,
            ),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV36(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 4),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV37(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        output_specs = [
            FeatureSpec(128, 80, 80),
            FeatureSpec(128, 40, 40),
            FeatureSpec(128, 20, 20),
        ]

        image_size = 640

        self.model = nn.Sequential(
            LogicalReconstructionNetR2V5(
                image_size=image_size,
                output_specs=output_specs,
            ),
        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV38(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        output_specs = [
            FeatureSpec(128, 32, 32),
            FeatureSpec(128, 16, 16),
            FeatureSpec(128, 8, 8),
        ]

        image_size = 256

        self.model = nn.Sequential(
            LogicalReconstructionNetR2V5(
                image_size=image_size,
                output_specs=output_specs,
            ),
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
