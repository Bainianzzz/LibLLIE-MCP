# RetinexFormer

RetinexFormer is a one-stage Retinex-based Transformer model for low-light image enhancement.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.pdf |
| Official source code | https://github.com/caiyuanhao1998/Retinexformer |
| Official project page | None |

## Model Introduction

RetinexFormer combines Retinex-based illumination modeling with Transformer-style feature restoration. The model first estimates illumination features and an illumination map, then uses illumination-guided multi-head self-attention to guide image restoration. Compared with multi-stage Retinex decomposition pipelines, RetinexFormer is designed as a one-stage architecture for low-light image enhancement.

In LibLLIE, RetinexFormer is implemented as a registered deep-learning model. The implementation includes an illumination estimator, illumination-guided attention blocks, a U-Net style denoiser, and optional multi-stage stacking. In training mode, the model returns the enhanced image and stage intermediate outputs through the standard LibLLIE model-output dictionary; in inference mode, it returns the enhanced image tensor.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Model implementation | `libllie/deepLearning/models/RetinexFormer.py` |
| Model class name | `RetinexFormer` |
| Default configuration | `libllie/deepLearning/config/RetinexFormer.yaml` |
| Related loss | `libllie/deepLearning/loss/RetinexFormer_Loss.py` |

## Loss Function

The official RetinexFormer training configuration uses pixel-level L1 loss. LibLLIE registers this loss as `retinexformer` and provides an optional illumination consistency term through `illumination_weight`.

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "RetinexFormer",
    "input.jpg",
    output="results/RetinexFormer/output.png",
    device="cuda",
)
```

Training example:

```python
llie.train(
    model="RetinexFormer",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="retinexformer",
    epochs=10,
    batch_size=4,
)
```

YAML configuration:

```python
llie.train("libllie/deepLearning/config/RetinexFormer.yaml")
```
