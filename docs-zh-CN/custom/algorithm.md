# 自定义传统增强算法

本文档说明如何基于 LibLLIE 的 `LLIEnhancer` 基类实现自定义传统低光增强算法。

## 1. 基本机制

传统算法基类是：

```python
from libllie.traditional.algorithms import LLIEnhancer
```

所有继承 `LLIEnhancer` 的算法类会在模块被导入时自动注册。注册后可以通过统一预测接口使用：

```python
import libllie as llie

llie.predict("my_algorithm", "input.jpg")
llie.enhance("my_algorithm", "images/", output="results/my_algorithm")
```

注意：新增算法文件后，需要在 `libllie/traditional/algorithms/__init__.py` 中导入该算法类，或者在使用前手动导入该模块。

## 2. LLIEnhancer 输入输出约定

`LLIEnhancer` 会负责：

- 使用 `ImageReader` 读取输入。
- 将输入转换为 NumPy 图像。
- 校验图像形状和 dtype。
- 调用子类 `_enhance()`。
- 根据 `keep_dtype` 和 `clip_output` 后处理输出。
- 根据 `output_type` 返回 NumPy、PIL、bytes、base64 或 file。

子类只需要实现：

```python
def _enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
    ...
```

注意：三通道 NumPy 图像默认按 OpenCV 风格 BGR 顺序处理。

## 3. 最小算法模板

建议在 `libllie/traditional/algorithms/MyAlgorithm.py` 中创建：

```python
from typing import Any, Dict

import numpy as np

from libllie.traditional.algorithms import LLIEnhancer


class MyAlgorithm(LLIEnhancer):
    """Example traditional enhancement algorithm."""

    name = "my_algorithm"
    aliases = ["myalg", "my"]

    def __init__(
        self,
        gain: float = 1.2,
        **kwargs: Any,
    ) -> None:
        """Initialize the enhancer.

        Args:
            gain: Brightness gain.
            **kwargs: Base enhancer parameters.
        """
        super().__init__(**kwargs)
        self.gain = gain

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Enhance one image.

        Args:
            image: Input image array in BGR order for_teach color images.
            **kwargs: Optional runtime parameters.

        Returns:
            Enhanced image array.
        """
        gain = kwargs.get("gain", self.gain)
        enhanced = image.astype(np.float32) * gain
        return enhanced

    def get_params(self) -> Dict[str, Any]:
        """Get enhancer parameters.

        Returns:
            Dictionary containing base parameters and algorithm parameters.
        """
        params = super().get_params()
        params.update({"gain": self.gain})
        return params
```

然后在 `libllie/traditional/algorithms/__init__.py` 中加入：

```python
from .MyAlgorithm import MyAlgorithm
```

## 4. 参数设置方式

算法参数可以在构造时传入：

```python
from libllie.traditional.algorithms import LLIEnhancer

enhancer = LLIEnhancer.create_enhancer("my_algorithm", gain=1.5)
```

也可以通过统一预测接口传入：

```python
import libllie as llie

llie.predict(
    "my_algorithm",
    "input.jpg",
    output="results/my_algorithm",
    gain=1.5,
)
```

如果参数是预测时动态覆盖，可以在 `_enhance()` 中读取 `kwargs`：

```python
gain = kwargs.get("gain", self.gain)
```

## 5. 输出格式

`LLIEnhancer` 支持以下输出类型：

```python
output_type="numpy"
output_type="pil"
output_type="bytes"
output_type="base64"
output_type="file"
```

传统算法预测器内部会强制使用 `output_type="numpy"`，然后统一保存图像。直接使用 enhancer 时可以指定其他格式：

```python
enhancer = MyAlgorithm(output_type="pil")
result = enhancer("input.jpg")
```

## 6. 在 Predictor 中使用

单图：

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "my_algorithm",
    "input.jpg",
    output="results/my_algorithm/output.png",
)
```

文件夹：

```python
saved_paths = llie.predict(
    "my_algorithm",
    "images/",
    output="results/my_algorithm",
)
```

## 7. 检查注册状态

```python
from libllie.traditional.algorithms import LLIEnhancer

print(LLIEnhancer.list_registered_enhancers())
```

或者：

```python
import libllie as llie

print(llie.list_algorithms())
```

## 8. 实现建议

1. `_enhance()` 必须返回 `np.ndarray`。
2. 不要在 `_enhance()` 内部读取文件路径，输入读取已经由 `LLIEnhancer` 完成。
3. 不要在 `_enhance()` 内部保存图像，保存由 Predictor 或 ImageWriter 负责。
4. 对颜色空间敏感的算法应明确 BGR/RGB 转换。
5. 参数校验建议放在 `__init__()` 或单独的 `_validate_*()` 方法中。
