# PairLIE

PairLIE is the model introduced in **Learning a Simple Low-light Image Enhancer from Paired Low-light Instances** (CVPR 2023). Instead of requiring a normal-light ground truth, it learns from two low-light observations of the same scene. A shared network estimates a denoised representation, illumination, and reflectance, while reflectance consistency links the two observations.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.pdf |
| Official source code | https://github.com/zhenqifu/PairLIE |
| Default configuration | `libllie/deepLearning/config/PairLIE.yaml` |

## LibLLIE implementation

| Item | Location |
| --- | --- |
| Model | `libllie/deepLearning/models/PairLIE.py` |
| Model name | `PairLIE` (alias: `Pair-LIE`) |
| Loss | `libllie/deepLearning/loss/PairLIE_Loss.py` |
| Loss name | `pairlie` |
| Dataset | `libllie/data/datasets/PairLIE.py` |

During inference, PairLIE produces:

```text
enhanced = illumination ** enhancement_gamma * reflectance
```

The default `enhancement_gamma` is `0.2`, matching the official general-purpose setting. The official LOL example uses `0.14`; this can be set as a checkpoint configuration override during prediction.

The integrated objective follows the official implementation:

```text
L = consistency_weight * MSE(R1, R2)
  + reconstruction_weight * L_reconstruction
  + preservation_weight * MSE(input1, denoised1)
```

`L_reconstruction` includes Retinex reconstruction, reflectance estimation, illumination-to-max-RGB fidelity, and illumination total variation. The default preservation weight is `500`.

## Training data

PairLIE needs two different low-light instances of each scene, not a low/normal-light pair. The included `PairLIEInstancesDataset` supports the official layout:

```text
PairLIE-training-dataset/
  1/
    exposure_1.png
    exposure_2.png
    ...
  2/
    exposure_1.png
    exposure_2.png
    ...
```

It also accepts a `root/train/scene/...` layout. Each sample randomly selects two different images from a scene and applies the same random crop to both. Every scene folder must contain at least two supported images.

## Training

Set `data.root_dir` in the YAML file, then run:

```bash
libllie train libllie/deepLearning/config/PairLIE.yaml
```

or use Python:

```python
import libllie as llie

result = llie.train("libllie/deepLearning/config/PairLIE.yaml")
print(result["checkpoint_dir"])
```

The unified Trainer detects PairLIE's paired-forward contract and sends both low-light instances through the same model. Other registered models keep the standard single-input path.

## Prediction

Use a checkpoint written by LibLLIE:

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "outputs/PairLIE/checkpoints/best.pt",
    "input.jpg",
    output="results/PairLIE/output.png",
    device="cuda",
)
```

For LOL-style inference:

```python
enhanced, saved_path = llie.predict(
    "outputs/PairLIE/checkpoints/best.pt",
    "input.jpg",
    output="results/PairLIE/lol.png",
    device="cuda",
    config={"enhancement_gamma": 0.14},
)
```

The implementation retains the official network parameter names, so the released raw `PairLIE.pth` state dictionary can also be loaded into a default `PairLIE` model with `load_state_dict(..., strict=True)`. LibLLIE checkpoints include the additional model configuration required by `llie.predict`.
