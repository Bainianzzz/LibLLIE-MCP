# Custom Deep-Learning Models

This document explains how to implement custom low-light enhancement models based on LibLLIE's `LLIEModel` base class.

## 1. Basic Mechanism

The deep-learning model base class is:

```python
from libllie.deepLearning.models import LLIEModel
```

All model classes that inherit `LLIEModel` are automatically registered when their modules are imported. After registration, they can be used as follows:

```python
import libllie as llie

llie.predict("MyModel", "input.jpg")
llie.train(model="MyModel", root_dir="datasets/LOL")
```

Note: automatic registration happens when Python imports the model module. Therefore, after adding a new model file, you need to import the model in `libllie/deepLearning/models/__init__.py`, or manually import the module containing the model before use.

## 2. Minimal Model Template

It is recommended to create the model in `libllie/deepLearning/models/MyModel.py`.

```python
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from libllie.deepLearning.models import LLIEModel


class MyModel(LLIEModel):
    """Example custom low-light enhancement model."""

    def _init_model(self) -> None:
        """Initialize network layers."""
        in_channels = self.config.get("input_channels", 3)
        hidden_channels = self.config.get("hidden_channels", 32)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run forward pass.

        Args:
            x: Low-light input tensor with shape [B, C, H, W].
            **kwargs: Optional model-specific forward parameters.

        Returns:
            In inference mode, returns enhanced image tensor.
            In training mode, returns standardized model output dictionary.
        """
        pred = self.net(x)

        if self.config.get("mode") == "train":
            return self._format_output(pred=pred)

        return pred
```

Then add the following to `libllie/deepLearning/models/__init__.py`:

```python
from .MyModel import MyModel
```

## 3. Configuration Parameters

`LLIEModel` automatically merges three types of configuration:

1. Default configuration returned by `_get_default_config()`.
2. The `config` dictionary passed to the constructor.
3. Keyword arguments passed to the constructor.

Examples:

```python
model = MyModel(config={"hidden_channels": 64})
model = MyModel(hidden_channels=64)
```

If the model needs extra default parameters, override `_get_default_config()`:

```python
def _get_default_config(self):
    config = super()._get_default_config()
    config.update({
        "hidden_channels": 32,
        "num_blocks": 4,
    })
    return config
```

If parameter validation is needed, override `_validate_config()`:

```python
def _validate_config(self):
    super()._validate_config()
    if self.config["hidden_channels"] <= 0:
        raise ValueError("hidden_channels must be positive.")
```

## 4. Training Output Convention

To stay compatible with `Trainer` and custom loss functions, it is recommended to return the standard structure in training mode:

```python
return self._format_output(
    pred=pred,
    loss_inputs={
        "aux": aux_tensor,
    },
    meta={
        "debug": "optional metadata",
    },
)
```

The standard output contains:

| Field | Meaning |
| --- | --- |
| `pred` | Final enhanced image |
| `loss_inputs` | Intermediate variables needed by loss functions |
| `meta` | Debug or extra information |

If the model directly returns a Tensor, `Trainer` can also handle it; however, for complex models, `_format_output()` is recommended.

## 5. Prediction Call

After registration, it can be called directly:

```python
import libllie as llie

llie.predict(
    "MyModel",
    "input.jpg",
    output="results/MyModel",
    device="cuda",
)
```

It can also be used through the unified Predictor:

```python
from libllie import Predictor

predictor = Predictor("MyModel", backend="deep", device="cuda")
predictor("input.jpg", output="results/MyModel")
```

## 6. Training Call

Training can be started directly with keyword arguments:

```python
import libllie as llie

llie.train(
    model="MyModel",
    dataset="LOLv1Dataset",
    root_dir="datasets/LOL",
    loss="charbonnier",
    epochs=10,
    batch_size=4,
    device="cuda",
)
```

You can also create a YAML configuration file:

```yaml
model:
  name: MyModel
  params:
    hidden_channels: 32

data:
  dataset: LOLv1Dataset
  root_dir: datasets/LOL
  batch_size: 4
  num_workers: 4

loss:
  name: charbonnier
  params: {}

optimizer:
  name: adam
  lr: 0.0001
  params: {}

scheduler:
  name: null
  params: {}

train:
  epochs: 10
  output_dir: checkpoints/MyModel_LOL
  device: cuda
```

Then call:

```python
llie.train("path/to/MyModel.yaml")
```

## 7. Check Registration Status

```python
from libllie.deepLearning.models import LLIEModel

print(LLIEModel.list_registered_models())
```

Or:

```python
import libllie as llie

print(llie.list_models())
print(llie.list_available())
```

## 8. FAQ

1. Model not found.
   This is usually because the model file has not been imported. Import the model class in `models/__init__.py`, or manually import the module before use.

2. Prediction not found during training.
   Confirm that forward returns a Tensor, or returns a dictionary containing fields such as `pred`, `enhanced`, `output`, or `prediction`. `_format_output(pred=...)` is recommended.

3. Input and output sizes are inconsistent.
   `Trainer` tries to align spatial sizes, but the number of channels must be consistent. Low-light enhancement models should usually output the same number of channels as the input.
