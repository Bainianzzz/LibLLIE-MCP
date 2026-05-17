# LEDNet

LEDNet is a low-light enhancement and deblurring model already implemented in LibLLIE.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/pdf/2202.03373 |
| Official source code | https://github.com/sczhou/LEDNet |

## Model Introduction

LEDNet targets low-light image enhancement and deblurring tasks. Its design includes multi-scale feature extraction, dynamic convolution, or attention-related modules to improve brightness, details, and clarity in dark scenes at the same time.

In LibLLIE, LEDNet's configuration includes channel settings, skip connection, side loss, dynamic convolution kernel size, the number of curve-attention iterations, and pyramid pooling bin settings.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Model implementation | `libllie/deepLearning/models/LEDNet.py` |
| Model class name | `LEDNet` |
| Default configuration | `libllie/deepLearning/config/LEDNet.yaml` |
| Related loss | `libllie/deepLearning/loss/LEDNet_Loss.py` |

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "LEDNet",
    "input.jpg",
    output="results/LEDNet/output.png",
    device="cuda",
)
```

Training example:

```python
llie.train(
    model="LEDNet",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="lednet",
    epochs=10,
    batch_size=4,
)
```
