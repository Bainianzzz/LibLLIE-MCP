# HE

HE 是 LibLLIE 中已经实现的传统直方图均衡化算法。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 / 经典来源 | https://doi.org/10.1016/S0146-664X(77)80011-7 |
| 相关早期文献 | https://doi.org/10.1109/T-C.1974.223892 |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

HE，即 Histogram Equalization，是经典的全局对比度增强方法。它计算图像灰度分布并构造累积分布函数，将原始灰度值重新映射到更均匀的强度分布，从而提升整体图像对比度。

在低光增强中，HE 可以提高暗区域可见性，但也可能导致局部过增强、噪声放大或颜色偏移。因此，对于彩色图像，通常需要选择合适的色彩空间，并只对亮度通道执行均衡化。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/HE.py` |
| 算法类名 | `HE` |
| 注册名称 | `he` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 实现说明

LibLLIE 的 HE 支持灰度图像和多种色彩空间：

- `rgb` / `bgr`
- `hsv`
- `hls`
- `yuv` / `ycbcr`
- `lab`

当选择与亮度相关的色彩空间时，算法会均衡化亮度通道，然后转换回 BGR 图像。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"rgb"` | 执行直方图均衡化的色彩空间 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "he",
    "input.jpg",
    output="results/he/output.jpg",
    color_space="hsv",
)
```

文件夹批量处理：

```python
saved_paths = llie.predict(
    "he",
    "images/",
    output="results/he",
    color_space="yuv",
)
```
