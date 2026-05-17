# Custom Loss Functions

This document explains how to implement custom training losses based on LibLLIE's `BaseLoss` base class.

## 1. Basic Mechanism

The loss function base class is:

```python
from libllie.deepLearning.loss import BaseLoss
```

All loss classes that inherit `BaseLoss` are automatically registered when their modules are imported. After registration, they can be used through the training configuration:

```python
import libllie as llie

llie.train(
    model="ZeroDCE",
    dataset="LOLv1Dataset",
    root_dir="datasets/LOL",
    loss="my_loss",
)
```

Note: after adding a new loss file, you need to import the loss class in `libllie/deepLearning/loss/__init__.py`, or manually import the module before use.

## 2. Supervised Loss with Reference

Reference-based losses are used for paired training. The inputs are the predicted image and the normal-light reference image.

It is recommended to create `libllie/deepLearning/loss/MyLoss.py`:

```python
from typing import List

import torch
import torch.nn as nn

from libllie.deepLearning.loss import BaseLoss


class MyLoss(BaseLoss):
    """Example supervised loss."""

    name = "my_loss"
    aliases: List[str] = ["myloss"]
    requires_target = True

    def __init__(self, l1_weight: float = 1.0) -> None:
        """Initialize custom loss.

        Args:
            l1_weight: Weight for_teach the L1 loss term.
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.l1 = nn.L1Loss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            prediction: Enhanced image tensor.
            target: Normal-light reference tensor.

        Returns:
            Scalar loss tensor.
        """
        return self.l1_weight * self.l1(prediction, target)
```

Then add the following to `libllie/deepLearning/loss/__init__.py`:

```python
from .MyLoss import MyLoss
```

## 3. No-Reference or Model-Specific Losses

Some low-light enhancement methods do not require a paired reference. They rely on the input image and intermediate model outputs. In this case, set:

```python
requires_target = False
```

The forward signature should be:

```python
forward(input_tensor, model_output)
```

Example:

```python
from typing import Any, List

import torch

from libllie.deepLearning.loss import BaseLoss


class MyZeroReferenceLoss(BaseLoss):
    """Example zero-reference loss."""

    name = "my_zero_ref"
    aliases: List[str] = ["my_zero_reference"]
    requires_target = False

    def __init__(self, exposure_weight: float = 1.0) -> None:
        """Initialize zero-reference loss.

        Args:
            exposure_weight: Weight for_teach the exposure regularization term.
        """
        super().__init__()
        self.exposure_weight = exposure_weight

    def forward(
        self,
        input_tensor: torch.Tensor,
        model_output: Any,
    ) -> torch.Tensor:
        """Compute zero-reference loss.

        Args:
            input_tensor: Low-light input tensor.
            model_output: Raw output returned by the model.

        Returns:
            Scalar loss tensor.
        """
        if isinstance(model_output, dict):
            pred = model_output["pred"]
        else:
            pred = model_output

        exposure_loss = torch.mean((pred.mean(dim=(2, 3), keepdim=True) - 0.6) ** 2)
        return self.exposure_weight * exposure_loss
```

## 4. Read Model Intermediate Variables

If a model in training mode returns:

```python
return self._format_output(
    pred=pred,
    loss_inputs={
        "illumination": illumination,
        "reflectance": reflectance,
    },
)
```

The loss function can read them as follows:

```python
def forward(self, input_tensor, model_output):
    pred = model_output["pred"]
    illumination = model_output["loss_inputs"]["illumination"]
    reflectance = model_output["loss_inputs"]["reflectance"]
    ...
```

This is suitable for model-specific losses such as Retinex, curve estimation, and illumination estimation.

## 5. Use in Training Configuration

YAML example:

```yaml
loss:
  name: my_loss
  params:
    l1_weight: 1.0
```

Python call example:

```python
import libllie as llie

llie.train(
    model="MyModel",
    dataset="LOLv1Dataset",
    root_dir="datasets/LOL",
    loss="my_loss",
    loss_params={
        "l1_weight": 1.0,
    },
)
```

## 6. Check Registration Status

```python
from libllie.deepLearning.loss import BaseLoss

print(BaseLoss.list_registered_losses())
```

Or:

```python
import libllie as llie

print(llie.list_losses())
```

## 7. Implementation Recommendations

1. Reference-based losses should keep `forward(prediction, target)`.
2. No-reference losses should set `requires_target = False` and keep `forward(input_tensor, model_output)`.
3. Loss functions should return a scalar Tensor.
4. Do not execute `backward()` inside the loss.
5. If multiple loss terms are used, keep their weights as class parameters for YAML configuration.
