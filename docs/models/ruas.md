# RUAS

RUAS is a Retinex-inspired low-light enhancement model already implemented in LibLLIE.

## Links

| Type | URL |
| --- | --- |
| Paper | https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf |
| Official source code | https://github.com/KarelZhang/RUAS |

## Model Introduction

RUAS combines Retinex ideas, unrolled optimization, and architecture search for low-light image enhancement. The model includes enhancement- and denoising-related structures, simulates iterative optimization through unrolling, and introduces cooperative prior architecture search to obtain an effective network structure.

In LibLLIE, RUAS contains illumination enhancement and denoising modules. The default configuration can control the number of IEM iterations, the number of NRM layers, enhancement network channels, and denoising network channels.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Model implementation | `libllie/deepLearning/models/RUAS.py` |
| Model class name | `RUAS` |
| Default configuration | `libllie/deepLearning/config/RUAS.yaml` |
| Related loss | `libllie/deepLearning/loss/RUAS_Loss.py` |

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "RUAS",
    "input.jpg",
    output="results/RUAS/output.png",
    device="cuda",
)
```

Training example:

```python
llie.train(
    model="RUAS",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="ruas",
    epochs=10,
    batch_size=4,
)
```
