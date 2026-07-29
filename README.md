# YOLOv11 Backbone

Standalone PyTorch implementation of the YOLOv11 backbone.

## Profiling

Run the profiling script with `uv`:

```bash
uv run python profile_backbone.py
```

The script reports:

- Number of parameters
- MACs and GMACs
- GFLOPs, using two FLOPs per MAC
- Per-stage operation counts
- Inference latency and FPS

MAC counting includes convolution operations and the matrix multiplications in
the PSA attention blocks. Batch normalization, activations, softmax,
concatenation, reshaping, and other elementwise operations are not included.

### Profiling options

```bash
uv run python profile_backbone.py --device cuda --dtype fp16
uv run python profile_backbone.py --height 640 --width 640 --iterations 200
uv run python profile_backbone.py --batch-size 4
```

Available options:

```text
--height       Input height; default: 640
--width        Input width; default: 640
--batch-size   Batch size; default: 1
--device       auto, cpu, or cuda; default: auto
--dtype        fp32, fp16, or bf16; default: fp32
--warmup       Warmup iterations; default: 20
--iterations   Timed iterations; default: 100
```

For a 640×640 input with batch size 1, the backbone produces a `(1, 256, 20,
20)` feature map and has approximately:

```text
Parameters: 1,200,864
MACs:       3.740518 GMACs
GFLOPs:     7.481037
```

Inference speed depends on the hardware, precision, input size, and batch size.
