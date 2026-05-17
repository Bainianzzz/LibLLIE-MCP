# Gamma

Gamma 是 LibLLIE 中已经实现的伽马校正增强算法。

## 链接

| 类型 | URL |
| --- | --- |
| 论文链接 | None |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

Gamma correction 是经典的幂律灰度变换方法。它通过如下形式调整图像亮度：

```text
output = input ^ gamma
```

当图像归一化到 `[0, 1]` 后：

- `gamma < 1` 通常会提亮暗区域。
- `gamma > 1` 通常会压暗图像。

在 LibLLIE 当前实现中，默认 `gamma=0.6`。用户可以根据低光增强需求传入更合适的 gamma 值。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/Gamma.py` |
| 算法类名 | `Gamma` |
| 注册名称 | `Gamma` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 实现说明

LibLLIE 中的 Gamma 会先将图像转换到浮点范围，再进行幂律变换，最后恢复原始 dtype。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `gamma` | `float` | `0.6` | 幂律指数，必须为正数 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "Gamma",
    "input.jpg",
    output="results/gamma/output.jpg",
    gamma=0.6,
)
```

如果希望通过统一 Predictor 显式使用传统后端：

```python
from libllie import Predictor

predictor = Predictor("Gamma", backend="traditional")
predictor("input.jpg", output="results/gamma/output.jpg", gamma=0.6)
```
