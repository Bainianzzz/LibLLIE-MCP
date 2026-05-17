# SCI

SCI is a fast low-light enhancement model already implemented in LibLLIE.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| Official source code | https://github.com/vis-opt-group/SCI |

## Model Introduction

SCI usually stands for Self-Calibrated Illumination. This model targets fast, flexible, and robust low-light image enhancement, progressively optimizing illumination estimation through an enhancement network and a calibration network. Compared with complex multi-stage models, SCI emphasizes a lightweight structure and efficient inference.

In LibLLIE, SCI consists of illumination enhancement and calibration parts. The model configuration can control the number of stages, the number of enhancement network layers, the number of calibration network layers, and the channel count.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Model implementation | `libllie/deepLearning/models/SCI.py` |
| Model class name | `SCI` |
| Default configuration | `libllie/deepLearning/config/SCI.yaml` |
| Related loss | `libllie/deepLearning/loss/Sci_Loss.py` |

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "SCI",
    "input.jpg",
    output="results/SCI/output.png",
    device="cuda",
)
```

Training example:

```python
llie.train(
    model="SCI",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="sci",
    epochs=10,
    batch_size=4,
)
```
