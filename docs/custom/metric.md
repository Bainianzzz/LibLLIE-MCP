# Custom Evaluation Metrics

This document explains how to implement custom image quality evaluation metrics based on LibLLIE's `BaseMetric` base class.

## 1. Basic Mechanism

The metric base class is:

```python
from libllie.evaluation import BaseMetric
```

All metric classes that inherit `BaseMetric` are automatically registered when their modules are imported. After registration, they can be used through the evaluation interface:

```python
import libllie as llie

llie.evaluate(
    en_img_dir="results/MyModel",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["MyMetric"],
)
```

Note: after adding a new metric file, make sure the module is imported before use. You can place the metric class in `libllie/evaluation/metrics.py`, or import the custom metric module in `libllie/evaluation/__init__.py`.

## 2. BaseMetric Input Convention

`BaseMetric.compute()` is responsible for:

- Checking whether the input is a torch Tensor.
- Expanding `[C, H, W]` inputs to `[B, C, H, W]`.
- Moving inputs to the metric device.
- Trying interpolation alignment when enhanced and reference spatial sizes are inconsistent.
- Calling the subclass `_compute_impl()`.

Subclasses only need to implement:

```python
def _compute_impl(self, enImg: torch.Tensor, Refer: torch.Tensor) -> float:
    ...
```

## 3. Full-Reference Metric Template

Full-reference metrics require both enhanced images and reference images.

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

Class names are recommended to end with `Metric`. When used, you can write `My` or `MyMetric`, because `BaseMetric.create_metric()` automatically completes the `Metric` suffix.

## 4. No-Reference Metric Template

No-reference metrics do not require a reference image, so `requires_reference` must be overridden:

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

## 5. Use in Evaluator

Full-reference evaluation:

```python
import libllie as llie

results = llie.evaluate(
    en_img_dir="results/MyModel",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["MyMetric"],
    save_path="results/eval_my_metric.json",
)
```

No-reference evaluation:

```python
results = llie.evaluate(
    en_img_dir="results/MyModel",
    metrics=["MyNoReferenceMetric"],
)
```

If the custom metric is not placed inside the LibLLIE package, import it before use:

```python
import my_project.metrics
import libllie as llie

llie.evaluate(..., metrics=["MyMetric"])
```

## 6. Check Registration Status

```python
from libllie.evaluation import BaseMetric

print(BaseMetric.list_available_metrics())
```

Or:

```python
import libllie as llie

print(llie.list_metrics())
```

## 7. Implementation Recommendations

1. `_compute_impl()` returns a Python float.
2. The metric does not need to handle expansion from `[C, H, W]` to `[B, C, H, W]`; the base class already handles it.
3. Full-reference metrics should keep `requires_reference=True`.
4. No-reference metrics must override `requires_reference=False`.
5. Correctly set `higher_is_better`; Evaluator uses it to mark the metric direction.
6. If a third-party library is required, report a clear ImportError during metric initialization.
