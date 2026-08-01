from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Sequence


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
    """Sub-0.5-ms reconstruction network with a learned feature hierarchy.

    A dense 16x16 patch stem performs the expensive spatial reduction in one
    CUDA operation. Three dense 3x3 stages then grow channels while halving the
    feature map. For a 640x640 image and 128 hidden channels, the path is::

        stem     [B,  16, 40, 40]
        stage 1  [B,  32, 20, 20]
        stage 2  [B,  64, 10, 10]
        stage 3  [B, 128,  5,  5]

    One groups=1 projection creates all three output channel groups from the
    deepest hierarchical feature. A single bilinear resize materializes the
    fused output, and channel views expose the three requested tensors. This
    avoids separate head and resize launches while retaining learned stem and
    multistage processing. No dilated convolution is used.
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

        target_sizes = {(spec.height, spec.width) for spec in output_specs}
        if len(target_sizes) != 1:
            raise ValueError(
                "R2 V4 requires all outputs to share one spatial size so the "
                "expensive resize can be performed once."
            )

        self.image_size = image_size
        self.output_specs = tuple(output_specs)
        self.target_size = next(iter(target_sizes))
        self.output_channels = tuple(spec.channels for spec in output_specs)
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

        self.output_projection = nn.Conv2d(
            hidden_channels,
            sum(self.output_channels),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
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
        for stage in self.stages:
            feature = stage(feature)

        output = self.output_projection(feature)
        if output.shape[-2:] != self.target_size:
            output = F.interpolate(
                output,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )
        outputs = output.split(self.output_channels, dim=1)
        return outputs[0], outputs[1], outputs[2]
