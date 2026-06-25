# HVI-CIDNet

HVI-CIDNet is the Color and Intensity Decoupling Network from **HVI: A New Color Space for Low-light Image Enhancement** (CVPR 2025). It converts RGB images into a learnable HVI color space, processes chromatic H/V features and intensity features in two interacting branches, and couples them with Lighten Cross-Attention (LCA) blocks.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/abs/2502.20272 |
| Official source code | https://github.com/Fediory/HVI-CIDNet |
| Default configuration | `libllie/deepLearning/config/CIDNet.yaml` |

## LibLLIE implementation

| Item | Location |
| --- | --- |
| Model | `libllie/deepLearning/models/CIDNet.py` |
| Model name | `CIDNet` (alias: `HVI-CIDNet`) |
| Loss | `libllie/deepLearning/loss/CIDNet_Loss.py` |
| Loss name | `cidnet` |

The integrated loss follows the official training objective in both RGB and HVI domains:

```text
L = L_RGB + hvi_weight * L_HVI
L_domain = pixel_weight * L1
         + ssim_weight * (1 - SSIM)
         + edge_weight * L_edge
         + perceptual_weight * L_VGG19
```

The default weights are `1.0`, `0.5`, `50.0`, `0.01`, and `hvi_weight=1.0`, matching the upstream defaults. VGG19 perceptual features use `conv1_2`, `conv2_2`, `conv3_4`, and `conv4_4` with MSE distance. The first training run may download the ImageNet VGG19 weights through torchvision. Set `loss.params.use_perceptual: false` for an offline or lightweight smoke test.

CIDNet pads tensors internally to a multiple of eight and crops its output back to the original size, so prediction supports arbitrary image dimensions. `input_gamma`, `saturation_scale`, and `intensity_scale` expose the upstream inference controls; each defaults to `1.0`.

## Training

Edit the dataset path in the YAML file, then run:

```bash
libllie train libllie/deepLearning/config/CIDNet.yaml
```

or use Python:

```python
import libllie as llie

result = llie.train("libllie/deepLearning/config/CIDNet.yaml")
print(result["checkpoint_dir"])
```

The configuration expects a LOLv1-compatible paired layout. Other registered paired datasets can be selected by changing `data.dataset`, paths, and split names.

## Prediction

Use a checkpoint produced by the LibLLIE trainer:

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "outputs/CIDNet/checkpoints/best.pt",
    "input.jpg",
    output="results/CIDNet/output.png",
    device="cuda",
)
```

To tune the optional upstream-style inference controls, pass checkpoint configuration overrides through `Predictor`, or create the model with `LLIEModel.create_model("CIDNet", config={...})`.

## Official raw weights

The architecture keeps the official parameter names, so a raw upstream `.pth` state dictionary can be loaded into a newly created model:

```python
import torch
from libllie.deepLearning.models import LLIEModel

model = LLIEModel.create_model("CIDNet", config={"device": "cuda"})
state = torch.load("LOLv1.pth", map_location=model.device)
model.load_state_dict(state, strict=True)
model.eval_mode()
```

LibLLIE training checkpoints contain additional configuration and optimizer metadata; use those checkpoints directly with `llie.predict`.
