# Zero-IG

Zero-IG is a zero-shot low-light enhancement and denoising model already implemented in LibLLIE.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.pdf |
| Official source code | https://github.com/Doyle59217/ZeroIG |

## Model Introduction

The full name of Zero-IG is Zero-shot Illumination-Guided joint denoising and adaptive enhancement. The method targets zero-shot scenarios and considers both low-light enhancement and noise suppression, using an illumination-guided mechanism to guide the enhancement and denoising process.

In LibLLIE, Zero-IG contains an enhancement network, a two-stage denoising network, a texture difference module, and local average pooling components. The configuration can set the number of enhancement layers, enhancement channels, denoising channels, and running mode.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Model implementation | `libllie/deepLearning/models/ZeroIG.py` |
| Model class name | `ZeroIG` |
| Default configuration | `libllie/deepLearning/config/ZeroIG.yaml` |
| Related loss | `libllie/deepLearning/loss/ZeroIG_Loss.py` |

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "ZeroIG",
    "input.jpg",
    output="results/ZeroIG/output.png",
    device="cuda",
)
```

Training example:

```python
llie.train(
    model="ZeroIG",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="zeroig",
    epochs=10,
    batch_size=4,
)
```
