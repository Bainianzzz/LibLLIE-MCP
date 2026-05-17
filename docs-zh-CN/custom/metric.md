# 自定义评估指标

本文档说明如何基于 LibLLIE 的 `BaseMetric` 基类实现自定义图像质量评估指标。

## 1. 基本机制

指标基类是：

```python
from libllie.evaluation import BaseMetric
```

所有继承 `BaseMetric` 的指标类会在模块被导入时自动注册。注册后可以通过评估接口使用：

```python
import libllie as llie

llie.evaluate(
    en_img_dir="results/MyModel",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["MyMetric"],
)
```

注意：新增指标文件后，需要保证该模块在使用前被导入。可以将指标类放入 `libllie/evaluation/metrics.py`，或在 `libllie/evaluation/__init__.py` 中导入自定义指标模块。

## 2. BaseMetric 输入约定

`BaseMetric.compute()` 会负责：

- 检查输入是否为 torch Tensor。
- 将 `[C, H, W]` 输入扩展为 `[B, C, H, W]`。
- 将输入移动到指标的 device。
- 当 enhanced 和 reference 空间尺寸不一致时尝试插值对齐。
- 调用子类 `_compute_impl()`。

子类只需要实现：

```python
def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor) -> float:
    ...
```

## 3. 有参考指标模板

有参考指标需要增强图像和参考图像。

```python
import torch
import torch.nn.functional as F

from libllie.evaluation import BaseMetric


class MyMetric(BaseMetric):
    """Example full-reference metric."""

    def __init__(self, device=None, scale: float = 1.0, **kwargs):
        """Initialize metric.

        Args:
            device: Metric computation device.
            scale: Optional score scale.
            **kwargs: Extra metric parameters.
        """
        super().__init__(device=device, **kwargs)
        self.scale = scale

    def _compute_impl(
        self,
        enImg: torch.Tensor,
        Refer: torch.Tensor,
    ) -> float:
        """Compute metric value.

        Args:
            enImg: Enhanced image tensor with shape [B, C, H, W].
            Refer: Reference image tensor with shape [B, C, H, W].

        Returns:
            Metric value.
        """
        score = F.l1_loss(enImg, Refer).item()
        return float(score * self.scale)

    @property
    def requires_reference(self) -> bool:
        """Whether this metric requires reference images."""
        return True

    @property
    def higher_is_better(self) -> bool:
        """Whether larger values indicate better quality."""
        return False
```

类名建议以 `Metric` 结尾。使用时可以写 `My` 或 `MyMetric`，因为 `BaseMetric.create_metric()` 会自动补全 `Metric` 后缀。

## 4. 无参考指标模板

无参考指标不需要 reference，需要重写 `requires_reference`：

```python
import torch

from libllie.evaluation import BaseMetric


class MyNoReferenceMetric(BaseMetric):
    """Example no-reference metric."""

    def _compute_impl(
        self,
        enImg: torch.Tensor,
        Refer: torch.Tensor = None,
    ) -> float:
        """Compute no-reference score.

        Args:
            enImg: Enhanced image tensor with shape [B, C, H, W].
            Refer: Unused reference tensor.

        Returns:
            Metric value.
        """
        brightness = enImg.mean().item()
        return float(brightness)

    @property
    def requires_reference(self) -> bool:
        """Whether this metric requires reference images."""
        return False

    @property
    def higher_is_better(self) -> bool:
        """Whether larger values indicate better quality."""
        return True
```

## 5. 在 Evaluator 中使用

有参考评估：

```python
import libllie as llie

results = llie.evaluate(
    en_img_dir="results/MyModel",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["MyMetric"],
    save_path="results/eval_my_metric.json",
)
```

无参考评估：

```python
results = llie.evaluate(
    en_img_dir="results/MyModel",
    metrics=["MyNoReferenceMetric"],
)
```

如果自定义指标未放入 LibLLIE 包内，需要在使用前导入：

```python
import my_project.metrics
import libllie as llie

llie.evaluate(..., metrics=["MyMetric"])
```

## 6. 检查注册状态

```python
from libllie.evaluation import BaseMetric

print(BaseMetric.list_available_metrics())
```

或者：

```python
import libllie as llie

print(llie.list_metrics())
```

## 7. 实现建议

1. `_compute_impl()` 返回 Python float。
2. 指标内部不需要处理 `[C, H, W]` 到 `[B, C, H, W]` 的扩展，基类已经完成。
3. 有参考指标保持 `requires_reference=True`。
4. 无参考指标必须重写 `requires_reference=False`。
5. 正确设置 `higher_is_better`，Evaluator 会用它标记指标方向。
6. 如果依赖第三方库，建议在指标初始化时报出清晰的 ImportError。

