# DCP

DCP 是 LibLLIE 中已经实现的暗通道先验增强算法。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/TPAMI.2010.168 |
| 论文标题 | Single Image Haze Removal Using Dark Channel Prior |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

暗通道先验最初用于单幅图像去雾。对于低光增强，可以将低光图像看作一种退化形式，并借助暗通道估计透射图和大气光，从而恢复更明亮、更清晰的图像。

LibLLIE 中的 DCP 实现包含暗通道计算、大气光估计、透射图估计以及引导滤波细化。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/DCP.py` |
| 算法类名 | `DCP` |
| 注册名称 | `dcp` |
| 别名 | `darkchannel`, `dark_channel_prior` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 实现说明

DCP 方法通常对参数比较敏感，尤其是暗通道窗口大小、透射图下界以及引导滤波参数。实际使用时可以根据图像尺寸和低光程度进行调整。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `size` | `int` | `15` | 暗通道腐蚀核大小 |
| `omega` | `float` | `0.95` | 透射图估计权重 |
| `t_min` | `float` | `0.1` | 最小透射率 |
| `guided_radius` | `int` | `60` | 引导滤波半径 |
| `guided_eps` | `float` | `1e-4` | 引导滤波正则项 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "dcp",
    "input.jpg",
    output="results/dcp/output.jpg",
    size=15,
    omega=0.95,
)
```

也可以使用别名：

```python
enhanced, saved_path = llie.predict(
    "darkchannel",
    "input.jpg",
    output="results/dcp/output.jpg",
)
```
