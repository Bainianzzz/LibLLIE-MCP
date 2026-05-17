# LLNet

LLNet is a stacked sparse denoising autoencoder model for natural low-light image enhancement.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1016/j.patcog.2016.06.008 |
| Paper PDF | https://web.me.iastate.edu/soumiks/pdf/Journal/LAS16_llnet.pdf |
| Official source code | https://github.com/kglore/llnet_color |
| Official project page | None |

## Model Introduction

LLNet enhances low-light images with a stacked sparse denoising autoencoder. The original method corrupts and vectorizes local image patches, trains autoencoders to reconstruct clean normal-light patches, and then reconstructs the full enhanced image from overlapping patch predictions.

In LibLLIE, LLNet keeps this patch-wise autoencoder idea but wraps patch extraction and aggregation inside a PyTorch image-to-image model. The implementation extracts overlapping patches with `torch.nn.functional.unfold`, processes each patch through a fully connected encoder-decoder network, and folds the reconstructed patches back into an image.

The related LibLLIE loss uses supervised reconstruction error and optional sparse autoencoder regularization terms.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Model implementation | `libllie/deepLearning/models/LLNet.py` |
| Model class name | `LLNet` |
| Default configuration | `libllie/deepLearning/config/LLNet.yaml` |
| Related loss | `libllie/deepLearning/loss/LLNet_Loss.py` |

## Main Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `patch_size` | `int` | `17` | Local patch size used by the autoencoder |
| `patch_stride` | `int` | `3` | Stride used when extracting overlapping patches |
| `hidden_dims` | `list[int]` | `[2000, 1600, 1200]` | Encoder hidden dimensions |
| `activation` | `str` | `"sigmoid"` | Hidden activation: `sigmoid`, `relu`, or `tanh` |
| `output_activation` | `str` | `"sigmoid"` | Output activation: `sigmoid`, `clamp`, or `none` |

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "LLNet",
    "input.jpg",
    output="results/LLNet/output.png",
    device="cuda",
)
```

Training example:

```python
llie.train(
    "libllie/deepLearning/config/LLNet.yaml",
    root_dir="datasets/LOL",
    epochs=10,
    batch_size=1,
)
```

Use smaller hidden dimensions for quick debugging:

```python
llie.train(
    model="LLNet",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="llnet",
    model_params={
        "hidden_dims": [256, 128, 64],
        "patch_stride": 8,
    },
    epochs=2,
    batch_size=1,
)
```
