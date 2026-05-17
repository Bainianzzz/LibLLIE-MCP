# 自定义深度学习模型

本文档说明如何基于 LibLLIE 的 `LLIEModel` 基类实现自定义低光增强模型。

## 1. 基本机制

深度学习模型基类是：

```python
from libllie.deepLearning.models import LLIEModel
```

所有继承 `LLIEModel` 的模型类会在模块被导入时自动注册。注册后可以通过以下方式使用：

```python
import libllie as llie

llie.predict("MyModel", "input.jpg")
llie.train(model="MyModel", root_dir="datasets/LOL")
```

注意：自动注册发生在 Python 导入模型模块时。因此，新增模型文件后，需要在 `libllie/deepLearning/models/__init__.py` 中导入该模型，或者在使用前手动导入该模型所在模块。

## 2. 最小模型模板

建议在 `libllie/deepLearning/models/MyModel.py` 中创建模型。

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

然后在 `libllie/deepLearning/models/__init__.py` 中加入：

```python
from .MyModel import MyModel
```

## 3. 配置参数

`LLIEModel` 会自动合并三类配置：

1. `_get_default_config()` 返回的默认配置。
2. 构造函数传入的 `config` 字典。
3. 构造函数传入的关键字参数。

示例：

```python
model = MyModel(config={"hidden_channels": 64})
model = MyModel(hidden_channels=64)
```

如果模型需要额外默认参数，可以重写 `_get_default_config()`：

```python
def _get_default_config(self):
    config = super()._get_default_config()
    config.update({
        "hidden_channels": 32,
        "num_blocks": 4,
    })
    return config
```

如果需要检查参数合法性，可以重写 `_validate_config()`：

```python
def _validate_config(self):
    super()._validate_config()
    if self.config["hidden_channels"] <= 0:
        raise ValueError("hidden_channels must be positive.")
```

## 4. 训练输出约定

为了兼容 `Trainer` 和自定义损失函数，建议训练模式下返回标准结构：

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

标准输出包含：

| 字段 | 含义 |
| --- | --- |
| `pred` | 最终增强图像 |
| `loss_inputs` | 损失函数需要的中间变量 |
| `meta` | 调试或额外信息 |

如果模型直接返回 Tensor，`Trainer` 也可以处理；但对于复杂模型，推荐使用 `_format_output()`。

## 5. 预测调用

注册后可以直接调用：

```python
import libllie as llie

llie.predict(
    "MyModel",
    "input.jpg",
    output="results/MyModel",
    device="cuda",
)
```

也可以通过统一 Predictor：

```python
from libllie import Predictor

predictor = Predictor("MyModel", backend="deep", device="cuda")
predictor("input.jpg", output="results/MyModel")
```

## 6. 训练调用

可以直接使用关键字参数训练：

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

也可以创建 YAML 配置文件：

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

然后调用：

```python
llie.train("path/to/MyModel.yaml")
```

## 7. 检查注册状态

```python
from libllie.deepLearning.models import LLIEModel

print(LLIEModel.list_registered_models())
```

或者：

```python
import libllie as llie

print(llie.list_models())
print(llie.list_available())
```

## 8. 常见问题

1. 模型找不到。
   通常是因为模型文件没有被导入。请在 `models/__init__.py` 中导入模型类，或在使用前手动导入模块。

2. 训练时报找不到 prediction。
   确认 forward 返回 Tensor，或者返回包含 `pred`、`enhanced`、`output`、`prediction` 等字段的字典。推荐使用 `_format_output(pred=...)`。

3. 输入输出尺寸不一致。
   `Trainer` 会尝试对齐空间尺寸，但通道数必须一致。低光增强模型通常应输出与输入相同的通道数。

