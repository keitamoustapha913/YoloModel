from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Sequence, Optional, Tuple, List, Union
from torchvision.models import ResNet18_Weights, resnet18


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

######################### BLOCkS #########################

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y




class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q * self.scale).transpose(-2, -1) @ k
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, all_head_dim, 7, 1, 3, g=all_head_dim, act=False)

    def __setstate__(self, state):
        """Add missing all_head_dim attribute to old checkpoints."""
        super().__setstate__(state)
        if not hasattr(self, "all_head_dim"):
            self.all_head_dim = self.head_dim * self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, _, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q * (self.head_dim**-0.5)).transpose(-2, -1) @ k
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, self.all_head_dim)
            v = v.reshape(B // self.area, N * self.area, self.all_head_dim)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y



class PatchStridedBackbone(nn.Module):
    """
    Alternative backbone using Conv :
    Based on ViT-style patch embedding but with convolutional layers for better efficiency:
    [B,3,640,640] -> [B,256,80,80] for patch_size=8
    [B,3,640,640] -> [B,256,40,40] for patch_size=16
    [B,3,640,640] -> [B,256,20,20] for patch_size=32
    """
    _SUPPORTED_PATCH_SIZES = (8, 16, 32, 64)

    def __init__(self, in_ch=3, out_ch=256, patch_size=16):
        super().__init__()
        if patch_size not in self._SUPPORTED_PATCH_SIZES:
            raise ValueError(
                f"patch_size must be one of {self._SUPPORTED_PATCH_SIZES}, got {patch_size}."
            )
        if out_ch <= 0:
            raise ValueError(f"out_ch must be > 0, got {out_ch}.")

        self.patch_size = int(patch_size)
        num_conv_layers = int(math.log2(self.patch_size))
        stage_channels = self._build_stage_channels(out_ch=out_ch, num_stages=num_conv_layers)

        layers = []
        c_in = in_ch
        for c_out in stage_channels:
            layers.append(Conv(c_in, c_out, k=3, s=2, act=True))
            c_in = c_out
        self.model = nn.Sequential(*layers)

    @staticmethod
    def _build_stage_channels(out_ch: int, num_stages: int):
        channels = []
        for i in range(1, num_stages):
            c = max(32, out_ch // (2 ** (num_stages - i)))
            channels.append(c)
        channels.append(out_ch)
        return channels

    def forward(self, x):
        return self.model(x)



###########################################

@dataclass(frozen=True)
class FeatureSpec:
    channels: int
    height: int
    width: int


def _build_progressive_stage_channels(
    latent_dim: int,
    output_channels: int,
    num_stages: int,
) -> list[int]:
    """Create a gradual channel schedule for progressive decoder stages.

    Each entry describes the output-channel count of one spatial decoder
    stage. The schedule begins at ``latent_dim``, grows by powers of two, and
    finishes at exactly ``output_channels``. Growth levels are distributed as
    evenly as possible across ``num_stages``; repeated channel counts are
    intentional because they avoid increasing channel width too early at
    expensive spatial resolutions.

    This helper only constructs model configuration during ``__init__``. It
    does not process tensors and adds no inference-time operations.

    Args:
        latent_dim: Number of channels in the latent input. Must be positive.
        output_channels: Required channels in the final decoder output. It
            must be greater than or equal to ``latent_dim``.
        num_stages: Number of spatial decoder stages, including the initial
            stage that generates the first feature map. Must be positive.

    Returns:
        A list of ``num_stages`` non-decreasing channel counts. Its final
        entry is always exactly ``output_channels``.

    Raises:
        ValueError: If a dimension is non-positive or if ``output_channels``
            is smaller than ``latent_dim``.

    Example:
        A six-stage decoder growing from 32 latent channels to 128 output
        channels receives the following schedule:

        >>> _build_progressive_stage_channels(32, 128, 6)
        [32, 32, 32, 64, 64, 128]
    """
    if latent_dim < 1:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
    if output_channels < latent_dim:
        raise ValueError(
            "output_channels must be greater than or equal to latent_dim, "
            f"got {output_channels} and {latent_dim}."
        )
    if num_stages < 1:
        raise ValueError(f"num_stages must be positive, got {num_stages}.")
    if num_stages == 1:
        return [output_channels]

    # Number of channel doublings needed to reach or pass the target width.
    growth_levels = math.ceil(math.log2(output_channels / latent_dim))
    channels: list[int] = []
    for stage_index in range(num_stages):
        # Map the spatial stage onto a channel-growth level. Using floor keeps
        # channels narrow until the next complete growth level is reached.
        level = math.floor(stage_index * growth_levels / (num_stages - 1))
        channels.append(min(output_channels, latent_dim * (2**level)))

    # Handle non-power-of-two targets such as 96 channels exactly.
    channels[-1] = output_channels
    return channels


def _build_progressive_reduction_stage_channels(
    latent_dim: int,
    output_channels: int,
    num_stages: int,
) -> list[int]:
    """Create the reverse of the progressive V5 channel-growth schedule.

    Each entry is the output-channel count of one spatial decoder stage. The
    schedule starts at ``latent_dim``, progressively halves channel width, and
    finishes at exactly ``output_channels``. Reductions occur early enough to
    mirror the increasing schedule in reverse, limiting wide channel tensors
    at the largest and most expensive spatial resolutions.

    This helper only builds decoder configuration during ``__init__`` and adds
    no inference-time operations.

    Args:
        latent_dim: Number of channels entering the decoder. It must be
            greater than or equal to ``output_channels``.
        output_channels: Required number of channels in the final output.
        num_stages: Number of spatial decoder stages, including the initial
            learned seed stage.

    Returns:
        A non-increasing list containing one channel count per stage. The last
        entry is always exactly ``output_channels``.

    Example:
        Reversing a five-stage 64-to-256 growth schedule gives:

        >>> _build_progressive_reduction_stage_channels(256, 64, 5)
        [256, 128, 128, 64, 64]
    """

    if latent_dim < 1:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
    if output_channels < 1:
        raise ValueError(
            f"output_channels must be positive, got {output_channels}."
        )
    if latent_dim < output_channels:
        raise ValueError(
            "latent_dim must be greater than or equal to output_channels, "
            f"got {latent_dim} and {output_channels}."
        )
    if num_stages < 1:
        raise ValueError(f"num_stages must be positive, got {num_stages}.")
    if num_stages == 1:
        return [output_channels]

    reduction_levels = math.ceil(math.log2(latent_dim / output_channels))
    channels: list[int] = []
    for stage_index in range(num_stages):
        # Ceil applies reductions toward the beginning of the spatial path.
        # This is the temporal reverse of V5 growth, which uses floor to delay
        # channel increases until later stages.
        level = math.ceil(
            stage_index * reduction_levels / (num_stages - 1)
        )
        channels.append(
            max(output_channels, latent_dim // (2**level))
        )

    # Support ratios that are not exact powers of two without undershooting
    # or overshooting the requested final channel count.
    channels[-1] = output_channels
    return channels


def _build_contract_expand_stage_channels(
    latent_dim: int,
    output_channels: int,
    num_stages: int,
) -> list[int]:
    """Build a channel hourglass for progressive spatial reconstruction.

    The schedule halves channels whenever possible, but reserves enough
    remaining spatial stages to reach ``output_channels`` using at most one
    channel doubling per stage. The final stage is always exactly the requested
    output width. This avoids holding every spatial stage at the final width
    while also preventing an unrecoverably narrow channel bottleneck.

    Args:
        latent_dim: Number of channels in the 1x1 latent tensor.
        output_channels: Required channels in the final feature map.
        num_stages: Number of learned spatial stages, including the seed.

    Returns:
        One channel count per spatial stage. Values contract toward the
        narrowest safe width and expand again when required to reach the final
        output width.

    Examples:
        A 20x20 decoder has three stages at 5x5, 10x10, and 20x20::

            >>> _build_contract_expand_stage_channels(256, 128, 3)
            [128, 64, 128]

        An 80x80 decoder has five stages and can contract further::

            >>> _build_contract_expand_stage_channels(256, 64, 5)
            [128, 64, 32, 32, 64]
    """

    if latent_dim < 1:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
    if output_channels < 1:
        raise ValueError(
            f"output_channels must be positive, got {output_channels}."
        )
    if num_stages < 1:
        raise ValueError(f"num_stages must be positive, got {num_stages}.")
    if num_stages == 1:
        return [output_channels]

    channels: list[int] = []
    previous_channels = latent_dim
    for stage_index in range(num_stages):
        remaining_stages = num_stages - stage_index - 1
        if remaining_stages == 0:
            stage_channels = output_channels
        else:
            halved_channels = max(1, previous_channels // 2)
            # If R stages remain, C must be at least ceil(target / 2**R)
            # so repeated doubling can still reach the requested output.
            minimum_recoverable_channels = math.ceil(
                output_channels / (2**remaining_stages)
            )
            stage_channels = max(
                halved_channels,
                minimum_recoverable_channels,
            )

        channels.append(stage_channels)
        previous_channels = stage_channels

    return channels



class TransposeDecoder(nn.Module):
    """Decode [B, Z, 1, 1] into one requested teacher-map shape."""

    def __init__(
        self,
        latent_dim: int,
        output_spec: FeatureSpec,
        base_channels: int = 32,
        max_channels: int = 256,
    ) -> None:
        super().__init__()
        self.output_spec = output_spec

        target_extent = max(output_spec.height, output_spec.width)
        generated_extent = max(4, 2 ** math.ceil(math.log2(target_extent)))

        layers: list[nn.Module] = []
        current_extent = 1
        in_channels = latent_dim
        hidden_channels = max_channels

        # 1x1 -> 4x4
        is_final = generated_extent == 4
        first_out = output_spec.channels if is_final else hidden_channels
        layers.append(
            nn.ConvTranspose2d(
                in_channels,
                first_out,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True,
            )
        )
        current_extent = 4
        if not is_final:
            layers.append(nn.ReLU(inplace=True))
        in_channels = first_out

        while current_extent < generated_extent:
            next_extent = current_extent * 2
            is_final = next_extent == generated_extent
            if is_final:
                out_channels = output_spec.channels
            else:
                hidden_channels = max(base_channels, hidden_channels // 2)
                out_channels = hidden_channels

            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True,
                )
            )
            if not is_final:
                layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            current_extent = next_extent

        self.network = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        output = self.network(latent)
        target_size = (self.output_spec.height, self.output_spec.width)
        if output.shape[-2:] != target_size:
            output = F.interpolate(output, size=target_size, mode="bilinear", align_corners=False)
        return output
class TransposeDecoderV2(nn.Module):
    """Progressively grow spatial resolution and channels from a latent map."""

    def __init__(self, latent_dim: int, output_spec: FeatureSpec) -> None:
        super().__init__()
        self.output_spec = output_spec

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if output_spec.channels < latent_dim:
            raise ValueError(
                "TransposeDecoderV2 requires output channels to be greater "
                f"than or equal to latent_dim, got {output_spec.channels} "
                f"and {latent_dim}."
            )

        target_extent = max(output_spec.height, output_spec.width)
        generated_extent = max(4, 2 ** math.ceil(math.log2(target_extent)))
        num_spatial_stages = int(math.log2(generated_extent)) - 1
        stage_channels = _build_progressive_stage_channels(
            latent_dim=latent_dim,
            output_channels=output_spec.channels,
            num_stages=num_spatial_stages,
        )

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=4,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        if num_spatial_stages > 1:
            layers.append(nn.ReLU(inplace=True))

        for index, (in_channels, out_channels) in enumerate(
            zip(stage_channels[:-1], stage_channels[1:])
        ):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=True,
                )
            )
            if index < len(stage_channels) - 2:
                layers.append(nn.ReLU(inplace=True))

        self.stage_channels = tuple(stage_channels)
        self.network = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        output = self.network(latent)
        target_size = (self.output_spec.height, self.output_spec.width)
        if output.shape[-2:] != target_size:
            output = F.interpolate(
                output,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return output


class TransposeDecoderV3(nn.Module):
    """Progressive 3x3 decoder with lightweight high-resolution stages."""

    def __init__(self, latent_dim: int, output_spec: FeatureSpec) -> None:
        super().__init__()
        self.output_spec = output_spec

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if output_spec.channels < latent_dim:
            raise ValueError(
                "TransposeDecoderV3 requires output channels to be greater "
                f"than or equal to latent_dim, got {output_spec.channels} "
                f"and {latent_dim}."
            )

        target_extent = max(output_spec.height, output_spec.width)
        generated_extent = max(4, 2 ** math.ceil(math.log2(target_extent)))
        num_spatial_stages = int(math.log2(generated_extent)) - 1
        stage_channels = _build_progressive_stage_channels(
            latent_dim=latent_dim,
            output_channels=output_spec.channels,
            num_stages=num_spatial_stages,
        )

        # A 3x3 transpose convolution with stride 2 and output_padding 1
        # expands the 1x1 latent map directly to 4x4.
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=1,
                bias=True,
            )
        ]
        if num_spatial_stages > 1:
            layers.append(nn.ReLU(inplace=True))

        channel_pairs = list(zip(stage_channels[:-1], stage_channels[1:]))
        grouped_stage_start = max(0, len(channel_pairs) - 2)
        for index, (in_channels, out_channels) in enumerate(channel_pairs):
            # Keep low-resolution stages dense for complete channel mixing.
            # Group only the two largest spatial stages, where dense 3x3
            # transpose convolutions are disproportionately expensive.
            groups = (
                math.gcd(in_channels, out_channels)
                if index >= grouped_stage_start
                else 1
            )
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    groups=groups,
                    bias=True,
                )
            )
            if index < len(channel_pairs) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.stage_channels = tuple(stage_channels)
        self.network = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        output = self.network(latent)
        target_size = (self.output_spec.height, self.output_spec.width)
        if output.shape[-2:] != target_size:
            output = F.interpolate(
                output,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return output




class TransposeDecoderV4(nn.Module):
    """Progressive 3x3 decoder with dense channel mixing at every stage."""

    def __init__(self, latent_dim: int, output_spec: FeatureSpec) -> None:
        super().__init__()
        self.output_spec = output_spec

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if output_spec.channels < latent_dim:
            raise ValueError(
                "TransposeDecoderV4 requires output channels to be greater "
                f"than or equal to latent_dim, got {output_spec.channels} "
                f"and {latent_dim}."
            )

        target_extent = max(output_spec.height, output_spec.width)
        generated_extent = max(4, 2 ** math.ceil(math.log2(target_extent)))
        num_spatial_stages = int(math.log2(generated_extent)) - 1
        stage_channels = _build_progressive_stage_channels(
            latent_dim=latent_dim,
            output_channels=output_spec.channels,
            num_stages=num_spatial_stages,
        )

        # A 3x3 transpose convolution with stride 2 and output_padding 1
        # expands the 1x1 latent map directly to 4x4.
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=1,
                bias=True,
            )
        ]
        if num_spatial_stages > 1:
            layers.append(nn.ReLU(inplace=True))

        channel_pairs = list(zip(stage_channels[:-1], stage_channels[1:]))
        for index, (in_channels, out_channels) in enumerate(channel_pairs):
            # V4 keeps every stage dense so each output channel can learn
            # from every input channel, including at high resolutions.
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    groups=1,
                    bias=True,
                )
            )
            if index < len(channel_pairs) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.stage_channels = tuple(stage_channels)
        self.model = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        output = self.model(latent)
        target_size = (self.output_spec.height, self.output_spec.width)
        if output.shape[-2:] != target_size:
            output = F.interpolate(
                output,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return output


class TransposeDecoderV5(nn.Module):
    """Generate square power-of-two and odd-base feature maps exactly.

    Power-of-two targets retain the V4 spatial architecture::

        1 -> 4 -> 8 -> 16 -> ... -> target

    A non-power-of-two target divisible by two is decomposed into an odd base
    multiplied by a power of two. The decoder learns the odd-sized seed
    directly and then doubles it until it reaches the requested extent. For
    example, 80 is ``5 * 2**4``, which produces::

        1 -> 5 -> 10 -> 20 -> 40 -> 80

    This avoids generating a larger power-of-two map and resizing it back to
    the requested shape. Every channel-changing stage remains dense
    (``groups=1``), as in V4.
    """

    def __init__(self, latent_dim: int, output_spec: FeatureSpec) -> None:
        super().__init__()
        self.output_spec = output_spec

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if output_spec.channels < latent_dim:
            raise ValueError(
                "TransposeDecoderV5 requires output channels to be greater "
                f"than or equal to latent_dim, got {output_spec.channels} "
                f"and {latent_dim}."
            )
        if output_spec.height != output_spec.width:
            raise ValueError(
                "TransposeDecoderV5 requires a square output so it can reach "
                "the requested size without interpolation, got "
                f"{output_spec.height}x{output_spec.width}."
            )

        target_extent = output_spec.height
        if target_extent < 4:
            raise ValueError(
                "TransposeDecoderV5 requires an output extent of at least 4, "
                f"got {target_extent}."
            )

        power_of_two_target = is_power_of_two(target_extent)
        if power_of_two_target:
            seed_extent = 4
            num_doublings = int(math.log2(target_extent)) - 2
        else:
            if target_extent % 2 != 0:
                raise ValueError(
                    "TransposeDecoderV5 requires the output extent to be a "
                    "power of two or divisible by 2, got "
                    f"{target_extent}."
                )

            # Remove every factor of two. The remaining odd factor is the
            # smallest exact seed from which repeated doubling reaches the
            # target. For 80 this yields a 5x5 seed and four doublings.
            seed_extent = target_extent
            num_doublings = 0
            while seed_extent % 2 == 0:
                seed_extent //= 2
                num_doublings += 1

            if seed_extent == 1:
                raise RuntimeError(
                    "Internal V5 extent decomposition failed for a "
                    "non-power-of-two target."
                )

        num_spatial_stages = num_doublings + 1
        stage_channels = _build_progressive_stage_channels(
            latent_dim=latent_dim,
            output_channels=output_spec.channels,
            num_stages=num_spatial_stages,
        )

        if power_of_two_target:
            # Match V4 exactly: a dense 3x3 transposed convolution maps the
            # latent 1x1 tensor to the initial 4x4 feature map.
            first_stage = nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=1,
                groups=1,
                bias=True,
            )
        else:
            # A stride-1 transposed convolution maps 1x1 directly to the odd
            # seed extent. For an 80x80 target, kernel_size=5 creates 5x5.
            first_stage = nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=seed_extent,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                bias=True,
            )

        layers: list[nn.Module] = [first_stage]
        if num_spatial_stages > 1:
            layers.append(nn.ReLU(inplace=True))

        channel_pairs = list(zip(stage_channels[:-1], stage_channels[1:]))
        for index, (in_channels, out_channels) in enumerate(channel_pairs):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    groups=1,
                    bias=True,
                )
            )
            if index < len(channel_pairs) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.seed_extent = seed_extent
        self.stage_channels = tuple(stage_channels)
        self.model = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        output = self.model(latent)
        expected_size = (self.output_spec.height, self.output_spec.width)
        if output.shape[-2:] != expected_size:
            raise RuntimeError(
                "TransposeDecoderV5 failed to generate its requested output "
                f"size {expected_size}; got {output.shape[-2:]}."
            )
        return output


class TransposeDecoderV6(nn.Module):
    """Increase spatial size while adapting channel width progressively.

    V6 includes both channel directions. When ``latent_dim`` is smaller than
    the requested output channels, it uses V5's progressive growth schedule.
    When ``latent_dim`` is greater than or equal to the output channels, it
    uses the reverse progressive reduction schedule. Power-of-two spatial
    targets retain V4's ``1 -> 4`` seed path; even non-power-of-two targets
    use V5's exact odd-base path and therefore require no interpolation.

    For a 256-channel latent tensor and a 64x80x80 output, the complete learned
    schedule is::

        spatial:  1 ->   5 ->  10 ->  20 -> 40 -> 80
        channels: 256 -> 256 -> 128 -> 128 -> 64 -> 64

    All transposed convolutions use ``groups=1`` for dense channel mixing.
    """

    def __init__(self, latent_dim: int, output_spec: FeatureSpec) -> None:
        super().__init__()
        self.output_spec = output_spec

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if output_spec.channels <= 0:
            raise ValueError(
                "TransposeDecoderV6 requires positive output channels, got "
                f"{output_spec.channels}."
            )
        if output_spec.height != output_spec.width:
            raise ValueError(
                "TransposeDecoderV6 requires a square output so it can reach "
                "the requested size without interpolation, got "
                f"{output_spec.height}x{output_spec.width}."
            )

        target_extent = output_spec.height
        if target_extent < 4:
            raise ValueError(
                "TransposeDecoderV6 requires an output extent of at least 4, "
                f"got {target_extent}."
            )

        power_of_two_target = is_power_of_two(target_extent)
        if power_of_two_target:
            seed_extent = 4
            num_doublings = int(math.log2(target_extent)) - 2
        else:
            if target_extent % 2 != 0:
                raise ValueError(
                    "TransposeDecoderV6 requires the output extent to be a "
                    "power of two or divisible by 2, got "
                    f"{target_extent}."
                )

            # Factor target_extent into odd_seed * 2**num_doublings. This
            # reaches targets such as 80 exactly through 5, 10, 20, 40, 80.
            seed_extent = target_extent
            num_doublings = 0
            while seed_extent % 2 == 0:
                seed_extent //= 2
                num_doublings += 1

            if seed_extent == 1:
                raise RuntimeError(
                    "Internal V6 extent decomposition failed for a "
                    "non-power-of-two target."
                )

        num_spatial_stages = num_doublings + 1
        if latent_dim < output_spec.channels:
            stage_channels = _build_progressive_stage_channels(
                latent_dim=latent_dim,
                output_channels=output_spec.channels,
                num_stages=num_spatial_stages,
            )
            self.channel_direction = "growth"
        else:
            stage_channels = _build_progressive_reduction_stage_channels(
                latent_dim=latent_dim,
                output_channels=output_spec.channels,
                num_stages=num_spatial_stages,
            )
            self.channel_direction = "reduction"

        if power_of_two_target:
            first_stage = nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=1,
                groups=1,
                bias=True,
            )
        else:
            first_stage = nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=seed_extent,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                bias=True,
            )

        layers: list[nn.Module] = [first_stage]
        if num_spatial_stages > 1:
            layers.append(nn.ReLU(inplace=True))

        channel_pairs = list(zip(stage_channels[:-1], stage_channels[1:]))
        for index, (in_channels, out_channels) in enumerate(channel_pairs):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    groups=1,
                    bias=True,
                )
            )
            if index < len(channel_pairs) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.seed_extent = seed_extent
        self.stage_channels = tuple(stage_channels)
        self.model = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        output = self.model(latent)
        expected_size = (self.output_spec.height, self.output_spec.width)
        if output.shape[-2:] != expected_size:
            raise RuntimeError(
                "TransposeDecoderV6 failed to generate its requested output "
                f"size {expected_size}; got {output.shape[-2:]}."
            )
        return output


class TransposeDecoderV7(nn.Module):
    """Expressive narrow-seed decoder with dense spatial reconstruction.

    V6 preserves the wide latent channel count in its learned spatial seed.
    That is expressive but expensive when, for example, a 256-channel latent
    tensor creates a 256-channel 5x5 map. V7 maps the complete latent vector
    directly into a narrower spatial seed, continues contracting while enough
    stages remain, and expands channels again to meet the requested output.

    Unlike projecting channels down at 1x1, the direct spatial seed exposes
    many more learned values than the latent contains and can therefore retain
    the complete latent rank. V7 halves channels only when the remaining
    spatial stages can still double back to the requested output width. For a
    256-channel latent and a 64x80x80 target, the path is::

        spatial:   1 ->  5 -> 10 -> 20 -> 40 -> 80
        channels: 256 -> 128 -> 64 -> 32 -> 32 -> 64

    All learned channel mappings use ``groups=1``. Spatial expansion remains
    fully learned through transposed convolutions, and no interpolation or
    sub-pixel rearrangement is used.
    """

    def __init__(
        self,
        latent_dim: int,
        output_spec: FeatureSpec,
    ) -> None:
        super().__init__()
        self.output_spec = output_spec

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if output_spec.channels <= 0:
            raise ValueError(
                "TransposeDecoderV7 requires positive output channels, got "
                f"{output_spec.channels}."
            )
        if output_spec.height != output_spec.width:
            raise ValueError(
                "TransposeDecoderV7 requires a square output so it can reach "
                "the requested size without interpolation, got "
                f"{output_spec.height}x{output_spec.width}."
            )

        target_extent = output_spec.height
        if target_extent < 4:
            raise ValueError(
                "TransposeDecoderV7 requires an output extent of at least 4, "
                f"got {target_extent}."
            )

        power_of_two_target = is_power_of_two(target_extent)
        if power_of_two_target:
            seed_extent = 4
            num_doublings = int(math.log2(target_extent)) - 2
        else:
            if target_extent % 2 != 0:
                raise ValueError(
                    "TransposeDecoderV7 requires the output extent to be a "
                    "power of two or divisible by 2, got "
                    f"{target_extent}."
                )

            seed_extent = target_extent
            num_doublings = 0
            while seed_extent % 2 == 0:
                seed_extent //= 2
                num_doublings += 1

            if seed_extent == 1:
                raise RuntimeError(
                    "Internal V7 extent decomposition failed for a "
                    "non-power-of-two target."
                )

        num_spatial_stages = num_doublings + 1
        stage_channels = _build_contract_expand_stage_channels(
            latent_dim=latent_dim,
            output_channels=output_spec.channels,
            num_stages=num_spatial_stages,
        )

        if power_of_two_target:
            first_stage = nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=1,
                groups=1,
                bias=True,
            )
        else:
            first_stage = nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=seed_extent,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                bias=True,
            )

        layers: list[nn.Module] = [first_stage]
        if num_spatial_stages > 1:
            layers.append(nn.ReLU(inplace=True))

        channel_pairs = list(zip(stage_channels[:-1], stage_channels[1:]))
        for index, (in_channels, out_channels) in enumerate(channel_pairs):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    groups=1,
                    bias=True,
                )
            )
            if index < len(channel_pairs) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.latent_dim = latent_dim
        self.seed_channels = stage_channels[0]
        self.channel_direction = "contract_expand"
        self.seed_extent = seed_extent
        self.stage_channels = tuple(stage_channels)
        self.model = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        if (
            latent.ndim != 4
            or latent.shape[1] != self.latent_dim
            or latent.shape[-2:] != (1, 1)
        ):
            raise ValueError(
                "TransposeDecoderV7 expects a latent tensor shaped "
                f"[B, {self.latent_dim}, 1, 1], got {tuple(latent.shape)}."
            )

        output = self.model(latent)
        expected_size = (self.output_spec.height, self.output_spec.width)
        if output.shape[-2:] != expected_size:
            raise RuntimeError(
                "TransposeDecoderV7 failed to generate its requested output "
                f"size {expected_size}; got {output.shape[-2:]}."
            )
        return output


class TransposeDecoderV8(nn.Module):
    """Channel-hourglass decoder with a fixed 64-channel spatial seed.

    V8 maps the complete latent tensor directly into a learned 64-channel
    spatial seed. Starting from 64 channels substantially reduces the dominant
    seed parameters while preserving a direct dense mapping from every latent
    channel into every seed channel and spatial position.

    The remaining stages use the same recoverable hourglass rule as V7: halve
    channels when possible, but reserve enough stages to double back to the
    exact requested output width. With a 256-channel latent, examples are::

        FeatureSpec(128, 20, 20)
        spatial:   1 ->  5 -> 10 ->  20
        channels: 256 -> 64 -> 64 -> 128

        FeatureSpec(64, 80, 80)
        spatial:   1 ->  5 -> 10 -> 20 -> 40 -> 80
        channels: 256 -> 64 -> 32 -> 16 -> 32 -> 64

    All convolutions are dense (``groups=1``), all spatial expansion is
    learned, and no interpolation or sub-pixel rearrangement is used.
    """

    seed_channels = 64

    def __init__(self, latent_dim: int, output_spec: FeatureSpec) -> None:
        super().__init__()
        self.output_spec = output_spec

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if output_spec.channels <= 0:
            raise ValueError(
                "TransposeDecoderV8 requires positive output channels, got "
                f"{output_spec.channels}."
            )
        if output_spec.height != output_spec.width:
            raise ValueError(
                "TransposeDecoderV8 requires a square output so it can reach "
                "the requested size without interpolation, got "
                f"{output_spec.height}x{output_spec.width}."
            )

        target_extent = output_spec.height
        if target_extent < 4:
            raise ValueError(
                "TransposeDecoderV8 requires an output extent of at least 4, "
                f"got {target_extent}."
            )

        power_of_two_target = is_power_of_two(target_extent)
        if power_of_two_target:
            seed_extent = 4
            num_doublings = int(math.log2(target_extent)) - 2
        else:
            if target_extent % 2 != 0:
                raise ValueError(
                    "TransposeDecoderV8 requires the output extent to be a "
                    "power of two or divisible by 2, got "
                    f"{target_extent}."
                )

            seed_extent = target_extent
            num_doublings = 0
            while seed_extent % 2 == 0:
                seed_extent //= 2
                num_doublings += 1

            if seed_extent == 1:
                raise RuntimeError(
                    "Internal V8 extent decomposition failed for a "
                    "non-power-of-two target."
                )

        num_spatial_stages = num_doublings + 1
        if num_spatial_stages == 1:
            if output_spec.channels != self.seed_channels:
                raise ValueError(
                    "A one-stage V8 decoder can only output its fixed "
                    f"{self.seed_channels} seed channels, got "
                    f"{output_spec.channels}."
                )
            stage_channels = [self.seed_channels]
        else:
            stage_channels = [self.seed_channels]
            stage_channels.extend(
                _build_contract_expand_stage_channels(
                    latent_dim=self.seed_channels,
                    output_channels=output_spec.channels,
                    num_stages=num_spatial_stages - 1,
                )
            )

        if power_of_two_target:
            first_stage = nn.ConvTranspose2d(
                latent_dim,
                self.seed_channels,
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=1,
                groups=1,
                bias=True,
            )
        else:
            first_stage = nn.ConvTranspose2d(
                latent_dim,
                self.seed_channels,
                kernel_size=seed_extent,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                bias=True,
            )

        layers: list[nn.Module] = [first_stage]
        if num_spatial_stages > 1:
            layers.append(nn.ReLU(inplace=True))

        channel_pairs = list(zip(stage_channels[:-1], stage_channels[1:]))
        for index, (in_channels, out_channels) in enumerate(channel_pairs):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    groups=1,
                    bias=True,
                )
            )
            if index < len(channel_pairs) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.latent_dim = latent_dim
        self.seed_extent = seed_extent
        self.stage_channels = tuple(stage_channels)
        self.model = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        if (
            latent.ndim != 4
            or latent.shape[1] != self.latent_dim
            or latent.shape[-2:] != (1, 1)
        ):
            raise ValueError(
                "TransposeDecoderV8 expects a latent tensor shaped "
                f"[B, {self.latent_dim}, 1, 1], got {tuple(latent.shape)}."
            )

        output = self.model(latent)
        expected_size = (self.output_spec.height, self.output_spec.width)
        if output.shape[-2:] != expected_size:
            raise RuntimeError(
                "TransposeDecoderV8 failed to generate its requested output "
                f"size {expected_size}; got {output.shape[-2:]}."
            )
        return output


class TransposeDecoderV9(nn.Module):
    """Minimum-cost hourglass decoder with an at-most-32-channel seed.

    V9 prioritizes parameters and latency. If ``latent_dim`` is at least 32,
    the complete latent tensor maps directly into a learned 32-channel spatial
    seed. If the latent is narrower than 32 channels, V9 keeps that width and
    lets the recoverable hourglass schedule determine all following stages.

    With a 256-channel latent, the requested reference paths are::

        FeatureSpec(128, 20, 20)
        spatial:   1 ->  5 -> 10 ->  20
        channels: 256 -> 32 -> 64 -> 128

        FeatureSpec(64, 80, 80)
        spatial:   1 ->  5 -> 10 -> 20 -> 40 -> 80
        channels: 256 -> 32 -> 16 -> 16 -> 32 -> 64

    Every convolution uses ``groups=1``. Spatial growth remains fully learned
    and exact, with no interpolation or sub-pixel rearrangement.
    """

    maximum_seed_channels = 32

    def __init__(self, latent_dim: int, output_spec: FeatureSpec) -> None:
        super().__init__()
        self.output_spec = output_spec

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if output_spec.channels <= 0:
            raise ValueError(
                "TransposeDecoderV9 requires positive output channels, got "
                f"{output_spec.channels}."
            )
        if output_spec.height != output_spec.width:
            raise ValueError(
                "TransposeDecoderV9 requires a square output so it can reach "
                "the requested size without interpolation, got "
                f"{output_spec.height}x{output_spec.width}."
            )

        target_extent = output_spec.height
        if target_extent < 4:
            raise ValueError(
                "TransposeDecoderV9 requires an output extent of at least 4, "
                f"got {target_extent}."
            )

        seed_channels = min(latent_dim, self.maximum_seed_channels)
        power_of_two_target = is_power_of_two(target_extent)
        if power_of_two_target:
            seed_extent = 4
            num_doublings = int(math.log2(target_extent)) - 2
        else:
            if target_extent % 2 != 0:
                raise ValueError(
                    "TransposeDecoderV9 requires the output extent to be a "
                    "power of two or divisible by 2, got "
                    f"{target_extent}."
                )

            seed_extent = target_extent
            num_doublings = 0
            while seed_extent % 2 == 0:
                seed_extent //= 2
                num_doublings += 1

            if seed_extent == 1:
                raise RuntimeError(
                    "Internal V9 extent decomposition failed for a "
                    "non-power-of-two target."
                )

        num_spatial_stages = num_doublings + 1
        if num_spatial_stages == 1:
            stage_channels = [output_spec.channels]
        else:
            stage_channels = [seed_channels]
            stage_channels.extend(
                _build_contract_expand_stage_channels(
                    latent_dim=seed_channels,
                    output_channels=output_spec.channels,
                    num_stages=num_spatial_stages - 1,
                )
            )

        if power_of_two_target:
            first_stage = nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=1,
                groups=1,
                bias=True,
            )
        else:
            first_stage = nn.ConvTranspose2d(
                latent_dim,
                stage_channels[0],
                kernel_size=seed_extent,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                bias=True,
            )

        layers: list[nn.Module] = [first_stage]
        if num_spatial_stages > 1:
            layers.append(nn.ReLU(inplace=True))

        channel_pairs = list(zip(stage_channels[:-1], stage_channels[1:]))
        for index, (in_channels, out_channels) in enumerate(channel_pairs):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    groups=1,
                    bias=True,
                )
            )
            if index < len(channel_pairs) - 1:
                layers.append(nn.ReLU(inplace=True))

        self.latent_dim = latent_dim
        self.seed_channels = seed_channels
        self.seed_extent = seed_extent
        self.stage_channels = tuple(stage_channels)
        self.model = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        if (
            latent.ndim != 4
            or latent.shape[1] != self.latent_dim
            or latent.shape[-2:] != (1, 1)
        ):
            raise ValueError(
                "TransposeDecoderV9 expects a latent tensor shaped "
                f"[B, {self.latent_dim}, 1, 1], got {tuple(latent.shape)}."
            )

        output = self.model(latent)
        expected_size = (self.output_spec.height, self.output_spec.width)
        if output.shape[-2:] != expected_size:
            raise RuntimeError(
                "TransposeDecoderV9 failed to generate its requested output "
                f"size {expected_size}; got {output.shape[-2:]}."
            )
        return output


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


class ReconstructionResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.activation(self.conv1(inputs))
        output = self.conv2(output)
        return self.activation(inputs + output)


class ReconstructionResidualBlockV2(nn.Module):
    """Dense bottleneck residual block with a dilated spatial convolution."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        bottleneck_ratio: int = 4,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}.")
        if dilation < 1:
            raise ValueError(f"dilation must be positive, got {dilation}.")
        if bottleneck_ratio < 1:
            raise ValueError(
                f"bottleneck_ratio must be positive, got {bottleneck_ratio}."
            )

        bottleneck_channels = max(1, channels // bottleneck_ratio)
        self.bottleneck_channels = bottleneck_channels
        self.reduce = nn.Conv2d(
            channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
        )
        self.spatial = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=1,
            bias=True,
        )
        self.expand = nn.Conv2d(
            bottleneck_channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.activation(self.reduce(inputs))
        output = self.activation(self.spatial(output))
        output = self.expand(output)
        return self.activation(inputs + output)


class LogicalReconstructionNetR2(nn.Module):
    """
    R2 replication.

    The network shares an image stem and produces three progressively more
    contextual feature stages (dilation 1, 2, 4), followed by level-specific
    1x1 projections and exact spatial alignment to A's outputs.
    """

    @staticmethod
    def _select_stem_ratio(image_size: int, largest_height: int) -> int:
        if image_size % largest_height != 0:
            raise ValueError("Teacher map size must divide the input image size.")

        ratio = image_size // largest_height
        if ratio < 1 or not is_power_of_two(ratio):
            raise ValueError(
                f"Input/map resolution ratio must be a power of two; received {ratio}."
            )
        return ratio

    @staticmethod
    def _build_stem_channels(hidden_channels: int, ratio: int) -> list[int]:
        """Return the original fixed-width channel schedule for the R2 stem."""

        num_downsampling_stages = int(math.log2(ratio))
        return [hidden_channels] * max(1, num_downsampling_stages)

    @staticmethod
    def _select_stem_kernel_size(stride: int) -> int:
        """Keep the original R2 stem on 3x3 convolutions."""

        return 3

    @staticmethod
    def _build_residual_block(channels: int, dilation: int) -> nn.Module:
        """Build the original full-width residual block."""

        return ReconstructionResidualBlock(channels, dilation)

    def __init__(
        self,
        image_size: int,
        output_specs: Sequence[FeatureSpec],
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        if len(output_specs) != 3:
            raise ValueError("R2 requires exactly three output specifications.")
        if hidden_channels < 1:
            raise ValueError(
                f"hidden_channels must be positive, got {hidden_channels}."
            )
        self.output_specs = tuple(output_specs)

        largest_height = max(spec.height for spec in output_specs)
        largest_width = max(spec.width for spec in output_specs)
        if largest_height != largest_width:
            raise ValueError("This replication expects square teacher maps.")
        ratio = self._select_stem_ratio(image_size, largest_height)

        stem_channels = self._build_stem_channels(hidden_channels, ratio)
        stem_layers: list[nn.Module] = []
        in_channels = 3
        current_ratio = 1
        for out_channels in stem_channels:
            stride = 2 if current_ratio < ratio else 1
            kernel_size = self._select_stem_kernel_size(stride)
            stem_layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        groups=1,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
            current_ratio *= stride

        self.stem = nn.Sequential(*stem_layers)
        self.stage1 = self._build_residual_block(hidden_channels, dilation=1)
        self.stage2 = self._build_residual_block(hidden_channels, dilation=2)
        self.stage3 = self._build_residual_block(hidden_channels, dilation=4)
        self.heads = nn.ModuleList(
            [nn.Conv2d(hidden_channels, spec.channels, kernel_size=1) for spec in output_specs]
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        base = self.stem(image)
        stages = (
            self.stage1(base),
            None,
            None,
        )
        level1 = stages[0]
        level2 = self.stage2(level1)
        level3 = self.stage3(level2)
        stage_outputs = (level1, level2, level3)

        outputs: list[Tensor] = []
        for stage, head, spec in zip(stage_outputs, self.heads, self.output_specs):
            output = head(stage)
            target_size = (spec.height, spec.width)
            if output.shape[-2:] != target_size:
                output = F.interpolate(output, size=target_size, mode="bilinear", align_corners=False)
            outputs.append(output)
        return tuple(outputs)  # type: ignore[return-value]


class LogicalReconstructionNetR2V2(LogicalReconstructionNetR2):
    """Hierarchical R2 variant with three progressively downsampled stages."""

    def __init__(
        self,
        image_size: int,
        output_specs: Sequence[FeatureSpec],
        hidden_channels: int = 128,
    ) -> None:
        nn.Module.__init__(self)
        if len(output_specs) != 3:
            raise ValueError("R2 V2 requires exactly three output specifications.")
        if image_size < 32:
            raise ValueError(
                "image_size must support five stride-2 reductions, "
                f"got {image_size}."
            )
        if hidden_channels < 32:
            raise ValueError(
                "hidden_channels must be at least 32 for the progressive "
                f"stage schedule, got {hidden_channels}."
            )

        self.output_specs = tuple(output_specs)
        first_stage_channels = hidden_channels // 4
        stage_channels = _build_progressive_stage_channels(
            latent_dim=first_stage_channels,
            output_channels=hidden_channels,
            num_stages=3,
        )
        self.stage_channels = tuple(stage_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, groups=1),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        in_channels = 16
        for out_channels, dilation in zip(stage_channels, (1, 2, 4)):
            stages.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=1,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                    self._build_residual_block(out_channels, dilation),
                )
            )
            in_channels = out_channels

        self.stage1, self.stage2, self.stage3 = stages
        self.heads = nn.ModuleList(
            [
                nn.Conv2d(channels, spec.channels, kernel_size=1, groups=1)
                for channels, spec in zip(stage_channels, output_specs)
            ]
        )

    @staticmethod
    def _build_residual_block(channels: int, dilation: int) -> nn.Module:
        """Build the four-times-narrower dense bottleneck residual block."""

        return ReconstructionResidualBlockV2(
            channels=channels,
            dilation=dilation,
            bottleneck_ratio=4,
        )


class LogicalReconstructionNetR2V3(nn.Module):
    """Independent hierarchical reconstruction network.

    V3 follows the efficient spatial hierarchy introduced by V2, but it is a
    standalone ``nn.Module`` rather than a subclass of R2 or R2V2. For the
    default 640x640 input and 128 hidden channels, its feature path is::

        RGB input       [B,   3, 640, 640]
        stem output     [B,  16, 160, 160]
        stage 1 output  [B,  32,  80,  80]
        stage 2 output  [B,  64,  40,  40]
        stage 3 output  [B, 128,  20,  20]

    Each stage starts with a dense 3x3 stride-2 convolution and then applies a
    bottleneck ``ReconstructionResidualBlockV2``. A dense 1x1 head projects
    each stage to its requested channel count, and bilinear interpolation
    aligns it to the requested spatial dimensions.
    """

    def __init__(
        self,
        image_size: int,
        output_specs: Sequence[FeatureSpec],
        hidden_channels: int = 128,
        bottleneck_ratio: int = 4,
    ) -> None:
        super().__init__()
        if image_size < 32 or image_size % 32 != 0:
            raise ValueError(
                "image_size must be a positive multiple of 32, "
                f"got {image_size}."
            )
        if len(output_specs) != 3:
            raise ValueError("R2 V3 requires exactly three output specifications.")
        if hidden_channels < 32 or hidden_channels % 4 != 0:
            raise ValueError(
                "hidden_channels must be at least 32 and divisible by 4, "
                f"got {hidden_channels}."
            )
        if bottleneck_ratio < 1:
            raise ValueError(
                f"bottleneck_ratio must be positive, got {bottleneck_ratio}."
            )
        if any(
            spec.channels < 1 or spec.height < 1 or spec.width < 1
            for spec in output_specs
        ):
            raise ValueError("All output feature dimensions must be positive.")

        self.image_size = image_size
        self.output_specs = tuple(output_specs)
        self.stage_channels = (
            hidden_channels // 4,
            hidden_channels // 2,
            hidden_channels,
        )

        # Two inexpensive spatial reductions produce the shared 160x160 map.
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                8,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                8,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        # Channel width doubles while spatial extent halves at every stage.
        stage_inputs = (16, *self.stage_channels[:-1])
        dilations = (1, 2, 4)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=1,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                    ReconstructionResidualBlockV2(
                        channels=out_channels,
                        dilation=dilation,
                        bottleneck_ratio=bottleneck_ratio,
                    ),
                )
                for in_channels, out_channels, dilation in zip(
                    stage_inputs,
                    self.stage_channels,
                    dilations,
                )
            ]
        )

        # One head corresponds to each stage and each output specification.
        self.heads = nn.ModuleList(
            [
                nn.Conv2d(
                    stage_channels,
                    spec.channels,
                    kernel_size=1,
                    groups=1,
                    bias=True,
                )
                for stage_channels, spec in zip(
                    self.stage_channels,
                    self.output_specs,
                )
            ]
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        expected_size = (self.image_size, self.image_size)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "Expected an RGB image tensor shaped [B, 3, H, W], "
                f"got {tuple(image.shape)}."
            )
        if image.shape[-2:] != expected_size:
            raise ValueError(
                f"Expected spatial size {expected_size}, got {image.shape[-2:]}."
            )

        features: list[Tensor] = []
        feature = self.stem(image)
        for stage in self.stages:
            feature = stage(feature)
            features.append(feature)

        outputs: list[Tensor] = []
        for feature, head, spec in zip(features, self.heads, self.output_specs):
            output = head(feature)
            target_size = (spec.height, spec.width)
            if output.shape[-2:] != target_size:
                output = F.interpolate(
                    output,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            outputs.append(output)

        return outputs[0], outputs[1], outputs[2]


class LogicalReconstructionNetR2V4(nn.Module):
    """Sub-0.5-ms reconstruction with native-resolution stage outputs.

    A dense 16x16 patch stem performs the expensive spatial reduction in one
    CUDA operation. Three dense 3x3 stages then grow channels while halving the
    feature map. For a 640x640 image and 128 hidden channels, the path is::

        stem     [B,  16, 40, 40]
        stage 1  [B,  32, 20, 20]
        stage 2  [B,  64, 10, 10]
        stage 3  [B, 128,  5,  5]

    Each stage has its own groups=1 pointwise head and returns its native
    spatial resolution. No interpolation, transposed convolution, dilation, or
    shared deepest-feature projection is used. This preserves true
    level-specific outputs while avoiding expensive spatial reconstruction.
    """

    def __init__(
        self,
        image_size: int,
        output_specs: Sequence[FeatureSpec],
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        if image_size < 128 or image_size % 128 != 0:
            raise ValueError(
                "image_size must be a positive multiple of 128, "
                f"got {image_size}."
            )
        if len(output_specs) != 3:
            raise ValueError("R2 V4 requires exactly three output specifications.")
        if hidden_channels < 8 or hidden_channels % 8 != 0:
            raise ValueError(
                "hidden_channels must be at least 8 and divisible by 8, "
                f"got {hidden_channels}."
            )
        if any(
            spec.channels < 1 or spec.height < 1 or spec.width < 1
            for spec in output_specs
        ):
            raise ValueError("All output feature dimensions must be positive.")

        native_sizes = (
            (image_size // 32, image_size // 32),
            (image_size // 64, image_size // 64),
            (image_size // 128, image_size // 128),
        )
        for index, (spec, native_size) in enumerate(
            zip(output_specs, native_sizes),
            start=1,
        ):
            if (spec.height, spec.width) != native_size:
                raise ValueError(
                    f"R2 V4 output {index} must use its native stage size "
                    f"{native_size}, got {(spec.height, spec.width)}."
                )

        self.image_size = image_size
        self.output_specs = tuple(output_specs)
        stem_channels = hidden_channels // 8
        self.stage_channels = (
            hidden_channels // 4,
            hidden_channels // 2,
            hidden_channels,
        )

        # Non-overlapping learned patches cover the complete input exactly.
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                stem_channels,
                kernel_size=16,
                stride=16,
                padding=0,
                groups=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        stage_inputs = (stem_channels, *self.stage_channels[:-1])
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=1,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                )
                for in_channels, out_channels in zip(
                    stage_inputs,
                    self.stage_channels,
                )
            ]
        )

        self.heads = nn.ModuleList(
            [
                nn.Conv2d(
                    stage_channels,
                    spec.channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    bias=True,
                )
                for stage_channels, spec in zip(
                    self.stage_channels,
                    self.output_specs,
                )
            ]
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        expected_size = (self.image_size, self.image_size)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "Expected an RGB image tensor shaped [B, 3, H, W], "
                f"got {tuple(image.shape)}."
            )
        if image.shape[-2:] != expected_size:
            raise ValueError(
                f"Expected spatial size {expected_size}, got {image.shape[-2:]}."
            )

        feature = self.stem(image)
        outputs: list[Tensor] = []
        for stage, head in zip(self.stages, self.heads):
            feature = stage(feature)
            outputs.append(head(feature))

        return outputs[0], outputs[1], outputs[2]


class LogicalReconstructionNetR2V5(nn.Module):
    """Native-resolution reconstruction from a compact sequential encoder.

    For a 640x640 input, the encoder and selected stage outputs are::

        Conv(3,  8, 3, 4) -> [B,  8, 160, 160]
        Conv(8, 16, 3, 2) -> [B, 16,  80,  80]  stage 1
        Conv(16,32, 3, 2) -> [B, 32,  40,  40]  stage 2
        Conv(32,64, 3, 2) -> [B, 64,  20,  20]  stage 3

    Each selected stage has an independent dense 1x1 output head. Outputs stay
    at their native resolutions, so the network performs no interpolation,
    transposed convolution, or dilated convolution.
    """

    def __init__(
        self,
        image_size: int,
        output_specs: Sequence[FeatureSpec],
    ) -> None:
        super().__init__()
        if image_size < 32 or image_size % 32 != 0:
            raise ValueError(
                "image_size must be a positive multiple of 32, "
                f"got {image_size}."
            )
        if len(output_specs) != 3:
            raise ValueError("R2 V5 requires exactly three output specifications.")
        if any(
            spec.channels < 1 or spec.height < 1 or spec.width < 1
            for spec in output_specs
        ):
            raise ValueError("All output feature dimensions must be positive.")

        native_sizes = (
            (image_size // 8, image_size // 8),
            (image_size // 16, image_size // 16),
            (image_size // 32, image_size // 32),
        )
        for index, (spec, native_size) in enumerate(
            zip(output_specs, native_sizes),
            start=1,
        ):
            if (spec.height, spec.width) != native_size:
                raise ValueError(
                    f"R2 V5 output {index} must use its native stage size "
                    f"{native_size}, got {(spec.height, spec.width)}."
                )

        self.image_size = image_size
        self.output_specs = tuple(output_specs)
        self.model = nn.Sequential(
            Conv(3, 8, 3, 4),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
        )
        self.network = nn.ModuleList(
            [
                nn.Conv2d(
                    stage_channels,
                    spec.channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    bias=True,
                )
                for stage_channels, spec in zip(
                    (16, 32, 64),
                    self.output_specs,
                )
            ]
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        expected_size = (self.image_size, self.image_size)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "Expected an RGB image tensor shaped [B, 3, H, W], "
                f"got {tuple(image.shape)}."
            )
        if image.shape[-2:] != expected_size:
            raise ValueError(
                f"Expected spatial size {expected_size}, got {image.shape[-2:]}."
            )

        outputs: list[Tensor] = []
        feature = image
        for index, layer in enumerate(self.model):
            feature = layer(feature)
            if index > 0:
                outputs.append(self.network[index - 1](feature))

        return outputs[0], outputs[1], outputs[2]





class MuDeNetReconstructionV1(nn.Module):
    """
    Input:
        [B, in_channels, 640, 640]

    Outputs:
        y1: [B,  64, 80, 80]
        y2: [B,  64, 40, 40]
        y3: [B, 128, 20, 20]
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.stage160 = Conv(
            in_channels,
            16,
            k=3,
            s=4,
        )

        self.stage80 = Conv(
            16,
            32,
            k=3,
            s=2,
        )

        self.stage40 = Conv(
            32,
            64,
            k=3,
            s=2,
        )

        self.stage20 = nn.Sequential(
            Conv(
                64,
                128,
                k=3,
                s=2,
            ),
            Conv(
                128,
                128,
                k=3,
                s=1,
            ),
        )

        # Linear output projections.
        self.output80 = nn.Conv2d(
            32,
            64,
            kernel_size=1,
        )

        self.output40 = nn.Conv2d(
            64,
            64,
            kernel_size=1,
        )

        self.output20 = nn.Conv2d(
            128,
            128,
            kernel_size=1,
        )

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:

        x160 = self.stage160(x)
        x80 = self.stage80(x160)
        x40 = self.stage40(x80)
        x20 = self.stage20(x40)

        y1 = self.output80(x80)
        y2 = self.output40(x40)
        y3 = self.output20(x20)

        return y1, y2, y3
    

class MuDeNetReconstructionsV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.r1 = MuDeNetReconstructionV1(in_channels=3)
        self.r2 = MuDeNetReconstructionV1(in_channels=3)

    def forward(self, x):
        r1_outputs = self.r1(x)
        r2_outputs = self.r2(x)

        return r1_outputs, r2_outputs
    

class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))

class MuDeNetReconstructionV2(nn.Module):
    """
    Input:
        [B, in_channels, 640, 640]

    Outputs:
        y1: [B,  64, 80, 80]
        y2: [B,  64, 40, 40]
        y3: [B, 128, 20, 20]
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.stage160 = Conv(
            in_channels,
            8,
            k=3,
            s=4,
        )

        self.stage80 = Conv(
            8,
            16,
            k=3,
            s=2,
        )

        self.stage40 = Conv(
            16,
            32,
            k=3,
            s=2,
        )

        # Linear output projections.
        self.output80 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv(8, 16, 3, 1),
            Conv(16, 32, 3, 1),
            Conv(32, 64, 3, 1),
        )

        self.output40 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv(16, 32, 3, 1),
            Conv(32, 64, 3, 1),
        )

        self.output20 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv(32, 64, 3, 1),
            Conv(64, 128, 3, 1),
        )


    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:

        x160 = self.stage160(x)
        x80 = self.stage80(x160)
        x40 = self.stage40(x80)

        y1 = self.output80(x160)
        y2 = self.output40(x80)
        y3 = self.output20(x40)

        return y1, y2 , y3
    



class MuDeNetReconstructionV3(nn.Module):
    """
    Input:
        [B, in_channels, 640, 640]

    Outputs:
        y1: [B,  64, 80, 80]
        y2: [B,  64, 40, 40]
        y3: [B, 128, 20, 20]
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.stage160 = Conv(
            in_channels,
            8,
            k=3,
            s=4,
        )

        self.stage80 = Conv(
            8,
            16,
            k=3,
            s=2,
        )

        self.stage40 = Conv(
            16,
            32,
            k=3,
            s=2,
        )

        # Linear output projections.
        self.output80 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Conv(8, 16, 3, 1),
            Conv(16, 32, 3, 1),
            Conv(32, 64, 3, 1),
        )

        # self.output40 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     Conv(16, 32, 3, 1),
        #     Conv(32, 64, 3, 1),
        # )

        # self.output20 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     Conv(32, 64, 3, 1),
        #     Conv(64, 128, 3, 1),
        # )


    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:

        x160 = self.stage160(x)
        # x80 = self.stage80(x160)
        # x40 = self.stage40(x80)

        y1 = self.output80(x160)
        # y2 = self.output40(x80)
        # y3 = self.output20(x40)

        return y1

class FrozenResNet18PyramidV2(nn.Module):
    """
    Frozen ImageNet-pretrained ResNet-18 used only to create
    pre-distillation targets.

    Expected input:
        [B, 3, 640, 640]

    Returned features:
        layer1: [B,  64, 160, 160]
        layer2: [B, 128,  80,  80]
        layer3: [B, 256,  40,  40]
    """

    def __init__(self) -> None:
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.eval()

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Initial ResNet stem, before max pooling.
        f1 = self.relu(self.bn1(self.conv1(image)))

        # First residual stage.
        x = self.maxpool(f1)
        layer1 = self.layer1(x)

        # Second residual stage.
        layer2 = self.layer2(layer1)
        
        # Third residual stage.
        layer3 = self.layer3(layer2)

        return layer1, layer2, layer3


FeaturePyramid = Tuple[Tensor, Tensor, Tensor]


class UnifiedEmbeddingBuilderV2(nn.Module):
    """
    Creates the MuDeNet-style unified target embedding E.

    Steps:
        1. Spatially align ResNet feature maps.
        2. Concatenate along the channel dimension.
        3. Select a fixed random subset of channels.
        4. Apply channel-wise z-score normalization.
    """

    def __init__(
        self,
        output_channels: int = 128,
        random_seed: int = 42,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        # f1: 64 channels
        # f2: 128 channels
        # f3: 256 channels

        ######################################

        # Exact spatial reduction by two:
        # 160 -> 80
        # 80  -> 40
        # 40  -> 20
        self.spatial_downsample = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.epsilon = float(epsilon)

        # -------------------------------------------------
        # Create fixed channel indices.
        #
        # These are generated only once and are saved in
        # the model state_dict as buffers.
        # -------------------------------------------------

        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_seed)

        # layer2: select 64 from 128 channels.
        layer2_indices = torch.randperm(
            128,
            generator=generator,
        )[:64]

        # layer3: select 128 from 256 channels.
        layer3_indices = torch.randperm(
            256,
            generator=generator,
        )[:128]

        # Sorting does not change which channels were chosen.
        # It gives a more regular channel-access pattern.
        layer2_indices = layer2_indices.sort().values
        layer3_indices = layer3_indices.sort().values

        self.register_buffer(
            "layer2_indices",
            layer2_indices,
            persistent=True,
        )

        self.register_buffer(
            "layer3_indices",
            layer3_indices,
            persistent=True,
        )

        # -------------------------------------------------
        # Normalization statistics.
        #
        # They are initialized as identity normalization:
        # mean = 0
        # std  = 1
        #
        # They are replaced during calibration.
        # -------------------------------------------------

        self.register_buffer(
            "t1_mean",
            torch.zeros(1, 64, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "t1_std",
            torch.ones(1, 64, 1, 1),
            persistent=True,
        )

        self.register_buffer(
            "t2_mean",
            torch.zeros(1, 64, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "t2_std",
            torch.ones(1, 64, 1, 1),
            persistent=True,
        )

        self.register_buffer(
            "t3_mean",
            torch.zeros(1, 128, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "t3_std",
            torch.ones(1, 128, 1, 1),
            persistent=True,
        )

        self.register_buffer(
            "statistics_ready",
            torch.tensor(False),
            persistent=True,
        )

    def raw_embedding_descriptors(
        self,
        features: Tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        f1, f2, f3 = features

        ########################################

        # -----------------------------------------------
        # T1
        #
        # layer1 already has 64 channels, so no selection.
        # [B,64,160,160] -> [B,64,80,80]
        # -----------------------------------------------

        t1 = self.spatial_downsample(f1)

        # -----------------------------------------------
        # T2
        #
        # Select before resizing:
        # [B,128,80,80]
        #     -> [B,64,80,80]
        #     -> [B,64,40,40]
        # -----------------------------------------------

        f2_selected = torch.index_select(
            f2,
            dim=1,
            index=self.layer2_indices,
        )

        t2 = self.spatial_downsample(f2_selected)

        # -----------------------------------------------
        # T3
        #
        # Select before resizing:
        # [B,256,40,40]
        #     -> [B,128,40,40]
        #     -> [B,128,20,20]
        # -----------------------------------------------

        f3_selected = torch.index_select(
            f3,
            dim=1,
            index=self.layer3_indices,
        )

        t3 = self.spatial_downsample(f3_selected)

        return t1, t2, t3
    
    def normalize_descriptors(
        self,
        descriptors: Tuple[Tensor, Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply independently calibrated channel-wise statistics.
        """

        # if not bool(self.statistics_ready.item()):
        #     raise RuntimeError(
        #         "Teacher statistics have not been calibrated. "
        #         "Run calibrate_teacher_statistics() first, or call "
        #         "forward(..., normalize=False)."
        #     )

        t1, t2, t3 = descriptors

        t1_normalized = (
            t1 - self.t1_mean
        ) / self.t1_std.clamp_min(self.epsilon)

        t2_normalized = (
            t2 - self.t2_mean
        ) / self.t2_std.clamp_min(self.epsilon)

        t3_normalized = (
            t3 - self.t3_mean
        ) / self.t3_std.clamp_min(self.epsilon)

        return (
            t1_normalized,
            t2_normalized,
            t3_normalized,
        )

    def forward(
        self,
        features: Tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        descriptors = self.raw_embedding_descriptors(features)

        return self.normalize_descriptors(descriptors)



class FrozenMuDeNetTeacherV2(nn.Module):
    """
    Frozen teacher architecture compatible with the prior ResNet-18
    distillation replication used in this project.

    Input:  [B, 3, H, W]
    Output: three [B, embedding_channels, H/2, W/2] maps.
    """

    def __init__(self, embedding_channels: int = 128, random_seed: int = 42) -> None:
        super().__init__()
        self.backbone = FrozenResNet18PyramidV2()
        self.embedding_builder = UnifiedEmbeddingBuilderV2(
            output_channels=embedding_channels,
            random_seed=random_seed,
        )

    @torch.inference_mode()
    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.backbone(image)
        embedding = self.embedding_builder(features)
        return embedding



class ContextualResidualR2Head(nn.Module):
    """Add neighboring-stage context before the terminal channel projection.

    The correction starts at exactly zero, so a newly migrated contextual head
    initially computes the same function as its legacy 1x1 projection. The
    3x3 layer is deliberately linear and residual to match the architecture
    proven by the probe-only investigation.
    """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        if input_channels < 1 or output_channels < 1:
            raise ValueError("Contextual head channel counts must be positive.")
        self.correction = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        )
        self.projection = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
        )
        nn.init.zeros_(self.correction.weight)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features + self.correction(features))

class LogicalReconstructionNetR2V6(nn.Module):
    """Native-resolution reconstruction from a compact sequential encoder.

    For a 640x640 input, the encoder and selected stage outputs are::

        Conv(3,  8, 3, 4) -> [B,  8, 160, 160]
        Conv(8, 16, 3, 2) -> [B, 16,  80,  80]  stage 1
        Conv(16,32, 3, 2) -> [B, 32,  40,  40]  stage 2
        Conv(32,64, 3, 2) -> [B, 64,  20,  20]  stage 3

    Each selected stage has an independent zero-initialized 3x3 residual
    correction followed by a dense 1x1 output projection. Outputs stay at
    their native resolutions, so the network performs no interpolation or
    transposed convolution.
    """

    def __init__(
        self,
        image_size: int,
        output_specs: Sequence[FeatureSpec],
    ) -> None:
        super().__init__()
        if image_size < 32 or image_size % 32 != 0:
            raise ValueError(
                "image_size must be a positive multiple of 32, "
                f"got {image_size}."
            )
        if len(output_specs) != 3:
            raise ValueError("R2 V5 requires exactly three output specifications.")
        if any(
            spec.channels < 1 or spec.height < 1 or spec.width < 1
            for spec in output_specs
        ):
            raise ValueError("All output feature dimensions must be positive.")

        native_sizes = (
            (image_size // 8, image_size // 8),
            (image_size // 16, image_size // 16),
            (image_size // 32, image_size // 32),
        )
        for index, (spec, native_size) in enumerate(
            zip(output_specs, native_sizes),
            start=1,
        ):
            if (spec.height, spec.width) != native_size:
                raise ValueError(
                    f"R2 V5 output {index} must use its native stage size "
                    f"{native_size}, got {(spec.height, spec.width)}."
                )

        self.image_size = image_size
        self.output_specs = tuple(output_specs)
        self.encoder = nn.Sequential(
            Conv(3, 8, 3, 4),
            Conv(8, 16, 3, 2),
            Conv(16, 32, 3, 2),
            Conv(32, 64, 3, 2),
        )
        self.heads = nn.ModuleList(
            [
                ContextualResidualR2Head(
                    input_channels=stage_channels,
                    output_channels=spec.channels,
                )
                for stage_channels, spec in zip(
                    (16, 32, 64),
                    self.output_specs,
                )
            ]
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        expected_size = (self.image_size, self.image_size)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "Expected an RGB image tensor shaped [B, 3, H, W], "
                f"got {tuple(image.shape)}."
            )
        if image.shape[-2:] != expected_size:
            raise ValueError(
                f"Expected spatial size {expected_size}, got {image.shape[-2:]}."
            )

        outputs: list[Tensor] = []
        feature = image
        for index, layer in enumerate(self.encoder):
            feature = layer(feature)
            if index > 0:
                outputs.append(self.heads[index - 1](feature))

        return outputs[0], outputs[1], outputs[2]
