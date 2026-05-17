# 自定义损失函数

本文档说明如何基于 LibLLIE 的 `BaseLoss` 基类实现自定义训练损失。

## 1. 基本机制

损失函数基类是：

```python
from libllie.deepLearning.loss import BaseLoss
```

所有继承 `BaseLoss` 的损失类会在模块被导入时自动注册。注册后可以通过训练配置使用：

```python
import libllie as llie

llie.train(
    model="ZeroDCE",
    dataset="LOLv1Dataset",
    root_dir="datasets/LOL",
    loss="my_loss",
)
```

注意：新增损失文件后，需要在 `libllie/deepLearning/loss/__init__.py` 中导入该损失类，或者在使用前手动导入该模块。

## 2. 有参考监督损失

有参考损失用于 paired 训练，输入是预测图像和正常光参考图像。

建议在 `libllie/deepLearning/loss/MyLoss.py` 中创建：

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

然后在 `libllie/deepLearning/loss/__init__.py` 中加入：

```python
from .MyLoss import MyLoss
```

## 3. 无参考或模型专用损失

有些低光增强方法不需要 paired reference，而是依赖输入图像和模型中间结果。此时需要设置：

```python
requires_target = False
```

forward 签名应为：

```python
forward(input_tensor, model_output)
```

示例：

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

## 4. 读取模型中间变量

如果模型训练模式下返回：

```python
return self._format_output(
    pred=pred,
    loss_inputs={
        "illumination": illumination,
        "reflectance": reflectance,
    },
)
```

损失函数可以这样读取：

```python
def forward(self, input_tensor, model_output):
    pred = model_output["pred"]
    illumination = model_output["loss_inputs"]["illumination"]
    reflectance = model_output["loss_inputs"]["reflectance"]
    ...
```

这适合 Retinex、curve estimation、illumination estimation 等模型专用损失。

## 5. 在训练配置中使用

YAML 示例：

```yaml
loss:
  name: my_loss
  params:
    l1_weight: 1.0
```

Python 调用示例：

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

## 6. 检查注册状态

```python
from libllie.deepLearning.loss import BaseLoss

print(BaseLoss.list_registered_losses())
```

或者：

```python
import libllie as llie

print(llie.list_losses())
```

## 7. 实现建议

1. 有参考损失保持 `forward(prediction, target)`。
2. 无参考损失设置 `requires_target = False`，并保持 `forward(input_tensor, model_output)`。
3. 损失函数应返回标量 Tensor。
4. 不要在 loss 内部执行 `backward()`。
5. 如果使用多个 loss term，建议在类中保留权重参数，便于 YAML 配置。

