"""Profile the standalone YOLOv11 backbone.

Examples:
    uv run python profile_backbone.py
    uv run python profile_backbone.py --model YOLOv11Backbone
    uv run python profile_backbone.py --model YOLOv11BackboneVariantV1
    uv run python profile_backbone.py --device cuda --dtype fp16
    uv run python profile_backbone.py --height 640 --width 640 --iterations 200

MAC counting includes convolution and transposed-convolution MACs, plus the
matrix multiplications in the PSA attention blocks. Elementwise operations,
batch normalization, activation, softmax, pooling, interpolation,
concatenation, and tensor reshaping are not included in the MAC total.
"""

from __future__ import annotations

import argparse
import inspect
import importlib
import time
from dataclasses import dataclass

import torch
from torch import nn

from modules import Attention


@dataclass
class MacResult:
    macs: int
    output: object


def format_output_shapes(output: object) -> str:
    """Format tensor shapes from a tensor or a nested model output."""

    if isinstance(output, torch.Tensor):
        return str(tuple(output.shape))
    if isinstance(output, tuple):
        shapes = ", ".join(format_output_shapes(item) for item in output)
        if len(output) == 1:
            shapes += ","
        return f"({shapes})"
    if isinstance(output, list):
        shapes = ", ".join(format_output_shapes(item) for item in output)
        return f"[{shapes}]"
    if isinstance(output, dict):
        shapes = ", ".join(
            f"{key!r}: {format_output_shapes(value)}"
            for key, value in output.items()
        )
        return f"{{{shapes}}}"
    return f"<{type(output).__name__}>"


def find_linear_stage_container(
    module: nn.Module,
) -> tuple[str, nn.Sequential | nn.ModuleList] | None:
    """Return a nested container that represents a module's linear pipeline.

    Some composite stages, including the transpose decoders, keep their
    ordered operations in a ``model`` or ``network`` attribute. Restricting
    recursive reporting to these known pipeline attributes avoids treating
    branch-oriented containers such as a C2f ``ModuleList`` as though every
    child consumed the previous child's complete output.
    """

    if isinstance(module, (nn.Sequential, nn.ModuleList)):
        return "", module

    for attribute_name in ("model", "network"):
        container = getattr(module, attribute_name, None)
        if isinstance(container, (nn.Sequential, nn.ModuleList)):
            return attribute_name, container

    return None


def print_nested_stage_macs(
    module: nn.Module,
    inputs: object,
    stage_path: str,
    indentation: int = 4,
) -> None:
    """Print MACs and output shapes for a composite stage's linear children."""

    nested = find_linear_stage_container(module)
    if nested is None:
        return

    container_name, children = nested
    child_input = inputs
    container_path = (
        f"{stage_path}.{container_name}" if container_name else stage_path
    )
    for index, child in enumerate(children):
        child_result = count_macs(child, child_input)
        child_path = f"{container_path}.{index}"
        print(
            f"{' ' * indentation}{child_path} "
            f"[{type(child).__name__}]: "
            f"{child_result.macs / 1e6:10.3f} MMACs "
            f"-> {format_output_shapes(child_result.output)}"
        )
        print_nested_stage_macs(
            child,
            child_input,
            child_path,
            indentation=indentation + 2,
        )
        child_input = child_result.output


def count_macs(module: nn.Module, inputs: object) -> MacResult:
    """Count convolution, transposed-convolution, and attention MACs."""

    total = 0
    hooks = []

    def count_conv(conv_module: nn.Conv2d, hook_inputs, output) -> None:
        nonlocal total
        x = hook_inputs[0]
        channels_in = x.shape[1]
        output_height, output_width = output.shape[-2:]
        channels_out = conv_module.out_channels
        kernel_height, kernel_width = conv_module.kernel_size
        groups = conv_module.groups

        total += (
            x.shape[0]
            * output_height
            * output_width
            * channels_out
            * (channels_in // groups)
            * kernel_height
            * kernel_width
        )

    def count_attention(attention_module: Attention, hook_inputs, _output) -> None:
        nonlocal total
        x = hook_inputs[0]
        batch, _channels, height, width = x.shape
        tokens = height * width

        # q @ k and v @ attention. The qkv, positional encoding, and
        # projection convolutions are counted by count_conv().
        total += (
            batch
            * attention_module.num_heads
            * tokens
            * tokens
            * (attention_module.key_dim + attention_module.head_dim)
        )

    def count_conv_transpose(
        conv_module: nn.ConvTranspose2d, hook_inputs, _output
    ) -> None:
        nonlocal total
        x = hook_inputs[0]
        kernel_height, kernel_width = conv_module.kernel_size

        # Every input position expands through the transposed-convolution
        # kernel into out_channels / groups output channels.
        total += (
            x.shape[0]
            * x.shape[-2]
            * x.shape[-1]
            * conv_module.in_channels
            * (conv_module.out_channels // conv_module.groups)
            * kernel_height
            * kernel_width
        )

    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            hooks.append(child.register_forward_hook(count_conv))
        elif isinstance(child, nn.ConvTranspose2d):
            hooks.append(child.register_forward_hook(count_conv_transpose))
        elif isinstance(child, Attention):
            hooks.append(child.register_forward_hook(count_attention))

    try:
        with torch.inference_mode():
            output = module(inputs)
    finally:
        for hook in hooks:
            hook.remove()

    return MacResult(total, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="YOLOv11Backbone",
        help="Model class exported by models.py; default: YOLOv11Backbone",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List model classes exported by models.py and exit.",
    )
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for profiling and benchmarking.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp32",
        help="Inference data type. fp16/bf16 are usually most useful on CUDA.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def available_models() -> dict[str, type[nn.Module]]:
    """Return nn.Module classes exposed by models.py."""

    models_module = importlib.import_module("models")
    return {
        name: model_class
        for name, model_class in inspect.getmembers(models_module, inspect.isclass)
        if issubclass(model_class, nn.Module) and model_class is not nn.Module
    }


def create_model(model_name: str) -> nn.Module:
    """Create the model class selected from models.py."""

    models = available_models()
    try:
        model_class = models[model_name]
    except KeyError as error:
        available = ", ".join(sorted(models)) or "none"
        raise ValueError(
            f"Model {model_name!r} was not found in models.py. "
            f"Available models: {available}"
        ) from error

    try:
        model = model_class()
    except TypeError as error:
        raise TypeError(
            f"Could not instantiate {model_name} with no arguments. "
            "Add constructor arguments to the profiler if this model requires them."
        ) from error

    if not isinstance(model, nn.Module):
        raise TypeError(f"{model_name} is not a torch.nn.Module")
    return model


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def get_dtype(dtype_name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_name]


def benchmark(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    """Return latency per image in milliseconds and images per second."""

    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(iterations):
            model(inputs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    batch_latency_ms = elapsed / iterations * 1000.0
    latency_per_image_ms = batch_latency_ms / inputs.shape[0]
    images_per_second = inputs.shape[0] * iterations / elapsed
    return latency_per_image_ms, images_per_second


def main() -> None:
    args = parse_args()
    models = available_models()
    if args.list_models:
        for model_name in sorted(models):
            print(model_name)
        return

    if args.batch_size < 1 or args.height < 1 or args.width < 1:
        raise ValueError("batch size, height, and width must be positive")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    model = create_model(args.model).eval().to(device=device, dtype=dtype)
    inputs = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=device,
        dtype=dtype,
    )

    result = count_macs(model, inputs)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"Model:      {args.model}")
    print(f"Device:     {device}")
    if device.type == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(device)}")
    print(f"Dtype:      {dtype}")
    print(f"Input:      {tuple(inputs.shape)}")
    print(f"Output:     {format_output_shapes(result.output)}")
    print(f"Parameters: {parameters:,}")
    print(f"MACs:       {result.macs:,} ({result.macs / 1e9:.6f} GMACs)")
    print(f"GFLOPs:     {2 * result.macs / 1e9:.6f} (2 FLOPs per MAC)")

    stages = getattr(model, "model", None)
    if isinstance(stages, (nn.Sequential, nn.ModuleList)):
        print("\nPer-stage MACs:")
        stage_input = inputs
        for index, stage in enumerate(stages):
            stage_result = count_macs(stage, stage_input)
            print(
                f"  {index}: {stage_result.macs / 1e6:10.3f} MMACs "
                f"-> {format_output_shapes(stage_result.output)}"
            )
            print_nested_stage_macs(stage, stage_input, str(index))
            stage_input = stage_result.output
    else:
        print("\nPer-stage MACs: unavailable (model.model is not sequential)")

    latency_ms, images_per_second = benchmark(
        model, inputs, device, args.warmup, args.iterations
    )
    print("\nSpeed:")
    print(f"  Latency: {latency_ms:.3f} ms/image")
    print(f"  FPS:     {images_per_second:.2f} images/second")


if __name__ == "__main__":
    main()
