# LLFormer

LLFormer is the transformer-based method from **Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method** (AAAI 2023 Oral). It combines row/column axial self-attention, dual-gated feed-forward networks, cross-layer attention fusion, a four-level encoder-decoder, and learned weighted skip connections. Axial attention avoids constructing full two-dimensional spatial attention maps, making the architecture more practical for high-resolution enhancement.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/abs/2212.11548 |
| Official source code | https://github.com/TaoWangzj/LLFormer |
| Default configuration | `libllie/deepLearning/config/LLFormer.yaml` |

## License notice

The integrated LLFormer architecture is adapted from the official implementation, which is released for academic, non-commercial use under **CC BY-NC-SA 4.0**. See `libllie/deepLearning/models/LLFormer_LICENSE.txt`. This restriction is narrower than LibLLIE's general MIT license and applies to this adapted implementation.

## LibLLIE implementation

| Item | Location |
| --- | --- |
| Model | `libllie/deepLearning/models/LLFormer.py` |
| Model name | `LLFormer` (alias: `LL-Former`) |
| Loss | `libllie/deepLearning/loss/LLFormer_Loss.py` |
| Loss name | `llformer` |
| Paired patch dataset | `libllie/data/datasets/LLFormer.py` |

The default architecture matches the official training entrypoint: `dim=16`, blocks `[2, 4, 8, 16]`, heads `[1, 2, 4, 8]`, two refinement blocks, WithBias layer normalization, and no global residual skip.

The official training entrypoint uses PyTorch `SmoothL1Loss`; LibLLIE registers the same objective as `llformer`. Auxiliary Charbonnier, edge, SSIM, and PSNR losses present in the upstream utility file are not part of the published training command and are therefore not enabled by default.

## Dataset and patch training

`LLFormerPairedDataset` supports the usual paired layout:

```text
dataset/
  train/
    low/
    high/
  val/
    low/
    high/
```

Training uses synchronized `128×128` random crops and the same flips/rotations as the official loader. Validation keeps the full image by default. Change `data.train_params.crop_size` for other patch sizes; patch dimensions should be divisible by 16.

## Training

Set `data.root_dir` and run:

```bash
libllie train libllie/deepLearning/config/LLFormer.yaml
```

or:

```python
import libllie as llie

result = llie.train("libllie/deepLearning/config/LLFormer.yaml")
print(result["checkpoint_dir"])
```

The supplied YAML follows the official LOL schedule: Adam at `1e-4`, minimum LR `1e-6`, batch size 8, and 4000 epochs. UHD-LOL4K/8K experiments in the official repository use 300 epochs and batch size 12, which can be adjusted in the same file.

## Prediction

LLFormer pads images internally to a multiple of 16 and crops predictions back to the original size:

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "outputs/LLFormer/checkpoints/best.pt",
    "input.png",
    output="results/LLFormer/output.png",
    device="cuda",
)
```

For memory-efficient 4K/8K inference, enable overlap-and-average tiling. The following reproduces the official UHD patch and half-stride strategy:

```python
enhanced, saved_path = llie.predict(
    "outputs/LLFormer/checkpoints/best.pt",
    "uhd_input.png",
    output="results/LLFormer/uhd_output.png",
    device="cuda",
    config={
        "tile_size": [720, 1280],
        "tile_overlap": [360, 640],
    },
)
```

Tile dimensions must be divisible by 16. Overlapping predictions are averaged, including boundary tiles.

## Official checkpoints

The model retains the official module and parameter names. Official checkpoints store weights under `state_dict` and may prefix keys with `module.` when trained with `DataParallel`; strip that prefix before loading into a default LibLLIE `LLFormer` instance. Checkpoints produced by LibLLIE already include the model configuration and can be passed directly to `llie.predict`.
