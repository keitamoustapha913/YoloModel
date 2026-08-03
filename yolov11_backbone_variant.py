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
    TransposeDecoderV5,
    TransposeDecoderV6,
    TransposeDecoderV7,
    LogicalReconstructionNetR2V2,
    LogicalReconstructionNetR2V3,
    LogicalReconstructionNetR2V4,
    LogicalReconstructionNetR2V5,
    LogicalReconstructionNetR2V6,
    MuDeNetReconstructionV1,
    MuDeNetReconstructionsV1,
    ChannelAttention,
    SpatialAttention,
    MuDeNetReconstructionV2,
    MuDeNetReconstructionV3,
    FrozenResNet18PyramidV2,
    FrozenMuDeNetTeacherV2,
    
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
            FeatureSpec(64, 80, 80),
            FeatureSpec(64, 40, 40),
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
    




class YOLOv11BackboneVariantV39(nn.Module):
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
            C2PSA(256, 256, 1),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV40(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            # C3k2(128, 128, 1, True),
            C2PSA(128, 128, 1),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV41(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            C2PSA(128, 128, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(128, self.latent_dim, 1, 1),
        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV42(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            PSABlock(128, 1.0, 4, True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(128, self.latent_dim, 1, 1),
        )

    def forward(self, x):
        return self.model(x)

class YOLOv11BackboneVariantV43(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            C3k2(128, 128, 1, True)
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV44(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.latent_dim = 32

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            C3k2(128, 128, 1, False, 0.5, False)
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV45(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 4),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            C3k2(128, 128, 1, True, 0.5, False)
        )

    def forward(self, x):
        return self.model(x)

class YOLOv11BackboneVariantV46(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 4),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            Conv(128, 128, 3, 1),
        )

    def forward(self, x):
        return self.model(x)
    

class YOLOv11BackboneVariantV47(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            Conv(in_channels, 16, 3, 4),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 1),
            Conv(64, 64, 3, 2),
            Conv(64, 128, 3, 1),
            Conv(128, 64, 3, 2),
            C3k2(64, 128, 1, False, 0.5, True)
        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV48(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            MuDeNetReconstructionV1(in_channels)
        )

    def forward(self, x):
        return self.model(x)
    

class YOLOv11BackboneVariantV49(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            MuDeNetReconstructionsV1()
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV50(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 4),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            # Conv(32, 64, 3, 2),
            # Conv(64, 128, 3, 2),
            # Conv(128, 128, 3, 1),
        )

    def forward(self, x):
        return self.model(x)
    
class YOLOv11BackboneVariantV51(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 4),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv(8, 64, 3, 1),
        )

    def forward(self, x):
        return self.model(x)
    
    
class YOLOv11BackboneVariantV52(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            MuDeNetReconstructionV2(3)
        )

    def forward(self, x):
        return self.model(x)



    
class YOLOv11BackboneVariantV53(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            FrozenResNet18PyramidV2()
        )

    def forward(self, x):
        return self.model(x) # ((1, 64, 160, 160), (1, 128, 80, 80), (1, 256, 40, 40))
    

class YOLOv11BackboneVariantV54(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            FrozenMuDeNetTeacherV2()
        )

    def forward(self, x):
        return self.model(x) # ((1, 64, 160, 160), (1, 128, 80, 80), (1, 256, 40, 40))
    

class YOLOv11BackboneVariantV55(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=64,
            height=80,
            width=80,
        )
        self.latent_dim = 256

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 32, 3, 2),
            PSABlock(32, 0.5, 4, True),
            # Conv(32, 64, k=3, s=2),
            # Conv(64, 128, k=3, s=2),
            # C3k2(64, 128, 1, False, 0.5, False),
            nn.Conv2d(
                32,
                self.latent_dim,
                kernel_size=5,
                stride=4,
                padding=0,
            ),
            # Conv(256, 256, k=5, s=1),
            # Conv(128, 256, k=3, s=2),
            # TransposeDecoderV6(
            #     latent_dim=self.latent_dim,
            #     output_spec=self.feature_spec,
            # ),
        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV56(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=20,
            width=20,
        )
        self.latent_dim = 64

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 32, 3, 2),
            PSABlock(32, 0.5, 4, True),

            # Conv(32, 64, k=3, s=2),
            # Conv(64, 128, k=3, s=2),
            # C3k2(64, 64, 1, False, 0.5, False),
            # nn.Conv2d(
            #     32,
            #     self.latent_dim,
            #     kernel_size=3,
            #     stride=4,
            #     padding=0,
            # ),
            # nn.Conv2d(
            #     64,
            #     self.latent_dim,
            #     kernel_size=5,
            #     stride=1,
            #     padding=0,
            # ),
            # Conv(256, 256, k=5, s=1),
            # Conv(128, 256, k=3, s=2),

            nn.Flatten(1),
            nn.Linear(32 * 20 * 20, 64),
            nn.Unflatten(1, (64, 1, 1)),
            TransposeDecoderV6(
                latent_dim=self.latent_dim,
                output_spec=self.feature_spec,
            ),
        )

    def forward(self, x):
        return self.model(x)



class YOLOv11BackboneVariantV57(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=20,
            width=20,
        )
        self.latent_dim = 64

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 32, 3, 2),
            PSABlock(32, 0.5, 4, True),

            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(32, 32, 1, 2),

            # nn.Flatten(1),
            # nn.Linear(32 * 20 * 20, 64),
            # nn.Unflatten(1, (64, 1, 1)),
            # TransposeDecoderV6(
            #     latent_dim=self.latent_dim,
            #     output_spec=self.feature_spec,
            # ),
        )

    def forward(self, x):
        return self.model(x)




class YOLOv11BackboneVariantV58(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        output_specs = [
            FeatureSpec(64, 80, 80),
            FeatureSpec(64, 40, 40),
            FeatureSpec(128, 20, 20),
        ]

        image_size = 640

        self.model = nn.Sequential(
            LogicalReconstructionNetR2V6(
                image_size=image_size,
                output_specs=output_specs,
            ),
        )

    def forward(self, x):
        return self.model(x)
    


class YOLOv11BackboneVariantV59(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=20,
            width=20,
        )
        latent_dim = 64

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 32, 3, 2),
            PSABlock(32, 0.5, 4, True),

            nn.Flatten(1),
            nn.Linear(32 * 20 * 20, latent_dim),
            nn.Unflatten(1, (latent_dim, 1, 1)),
            # TransposeDecoderV6(
            #     latent_dim=latent_dim,
            #     output_spec=self.feature_spec,
            # ),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV60(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 32, 3, 2),
            PSABlock(32, 0.5, 4, True),
            Conv(32, 64, k=3, s=2),
            Conv(64, 128, k=3, s=2),
            nn.Flatten(1),
            nn.Linear(128 * 5 * 5 , latent_dim),
            nn.Unflatten(1, (latent_dim, 1, 1))
        )

    def forward(self, x):
        return self.model(x)
    

class YOLOv11BackboneVariantV61(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()

        self.feature_spec = FeatureSpec(
            channels=128,
            height=20,
            width=20,
        )

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 32, 3, 2),
            PSABlock(32, 0.5, 4, True),
            Conv(32, 64, k=3, s=2),
            Conv(64, 128, k=3, s=2),
            nn.Flatten(1),
            nn.Linear(128 * 5 * 5 , latent_dim),
            nn.Unflatten(1, (latent_dim, 1, 1)),
            TransposeDecoderV6(
                latent_dim=latent_dim,
                output_spec=self.feature_spec,
            ),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV62(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()

        self.feature_spec = FeatureSpec(
            channels=64,
            height=80,
            width=80,
        )

        self.model = nn.Sequential(
            Conv(in_channels, 8, 3, 2),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 32, 3, 2),
            PSABlock(32, 0.5, 4, True),
            Conv(32, 64, k=3, s=2),
            Conv(64, 128, k=3, s=2),
            nn.Flatten(1),
            nn.Linear(128 * 5 * 5 , latent_dim),
            nn.Unflatten(1, (latent_dim, 1, 1)),
            TransposeDecoderV6(
                latent_dim=latent_dim,
                output_spec=self.feature_spec,
            ),
        )

    def forward(self, x):
        return self.model(x)


class YOLOv11BackboneVariantV63(nn.Module):
    """Low-latency learned compression and reconstruction at 20x20.

    The encoder uses a non-overlapping 16x16 patch stem to avoid expensive
    high-resolution convolution stages. It then builds a compact hierarchy at
    40x40, 20x20, 10x10, and 5x5. A PSA block refines the inexpensive 5x5
    feature map before a learned 5x5 projection creates the 1x1 latent tensor.
    TransposeDecoderV6 reconstructs 20x20 directly through the exact
    1-to-5-to-10-to-20 spatial path without interpolation.
    """

    def __init__(self, in_channels=3, latent_dim=32):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=128,
            height=20,
            width=20,
        )
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # Explicit padding=0 creates exactly 40x40 non-overlapping patches
            # from a 640x640 image; automatic padding would produce 41x41.
            Conv(in_channels, 16, 16, 16, p=0),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            # Apply spatial attention at 5x5, where its token interactions are
            # substantially cheaper than at earlier high-resolution stages.
            PSABlock(128, 0.5, 4, True),
            # Learned full-spatial projection: 128x5x5 -> latent_dim x1x1.
            nn.Conv2d(
                128,
                self.latent_dim,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            TransposeDecoderV6(
                latent_dim=self.latent_dim,
                output_spec=self.feature_spec,
            ),
        )

    def forward(self, x):
        return self.model(x)

class YOLOv11BackboneVariantV64(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.feature_spec = FeatureSpec(
            channels=64,
            height=80,
            width=80,
        )
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # Explicit padding=0 creates exactly 40x40 non-overlapping patches
            # from a 640x640 image; automatic padding would produce 41x41.
            Conv(in_channels, 16, 16, 16, p=0),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
            Conv(64, 128, 3, 2),
            # Apply spatial attention at 5x5, where its token interactions are
            # substantially cheaper than at earlier high-resolution stages.
            PSABlock(128, 0.5, 4, True),
            # Learned full-spatial projection: 128x5x5 -> latent_dim x1x1.
            nn.Conv2d(
                128,
                self.latent_dim,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            TransposeDecoderV7(
                latent_dim=self.latent_dim,
                output_spec=self.feature_spec,
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
