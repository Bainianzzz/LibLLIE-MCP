# KinD

KinD is a Retinex-inspired deep-learning model for practical low-light image enhancement.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1145/3343031.3350926 |
| Paper title | Kindling the Darkness: A Practical Low-light Image Enhancer |
| Official source code | https://github.com/zhangyhuaee/KinD |
| Official project page | None |

## Model Introduction

KinD decomposes an image into reflectance and illumination, restores degraded reflectance, adjusts illumination, and then recombines both components to produce the enhanced result. The original implementation trains the decomposition, restoration, and illumination adjustment networks in separate stages.

In LibLLIE, KinD is implemented as one integrated PyTorch model to match the unified training and prediction pipeline. The model contains:

| Subnetwork | Purpose |
| --- | --- |
| `KinDDecompositionNet` | Estimate reflectance `R` and illumination `I` |
| `KinDRestorationNet` | Restore low-light reflectance guided by illumination |
| `KinDIlluminationAdjustmentNet` | Adjust illumination using an exposure-ratio map |

The related loss combines the main objectives from the official staged training process, including Retinex reconstruction, reflectance consistency, illumination smoothness, reflectance restoration, illumination adjustment, and final enhanced-image supervision.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Model implementation | `libllie/deepLearning/models/KinD.py` |
| Model class name | `KinD` |
| Default configuration | `libllie/deepLearning/config/KinD.yaml` |
| Related loss | `libllie/deepLearning/loss/KinD_Loss.py` |

## Main Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `decomposition_channels` | `int` | `64` | Feature channels used by the decomposition network |
| `decomposition_layers` | `int` | `5` | Number of intermediate decomposition layers |
| `restoration_channels` | `int` | `32` | Base feature channels used by the restoration network |
| `adjustment_channels` | `int` | `32` | Feature channels used by the illumination adjustment network |
| `adjustment_layers` | `int` | `3` | Number of intermediate illumination adjustment layers |
| `illumination_ratio` | `float` | `5.0` | Exposure ratio used by inference illumination adjustment |

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "KinD",
    "input.jpg",
    output="results/KinD/output.png",
    device="cuda",
)
```

Training example:

```python
llie.train(
    "libllie/deepLearning/config/KinD.yaml",
    root_dir="datasets/LOL",
    epochs=10,
    batch_size=2,
)
```

Override inference exposure ratio:

```python
enhanced, saved_path = llie.predict(
    "KinD",
    "input.jpg",
    output="results/KinD/brighter.png",
    device="cuda",
    illumination_ratio=6.0,
)
```
