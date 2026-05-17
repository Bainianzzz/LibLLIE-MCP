# AHE

AHE 是 LibLLIE 中已经实现的自适应直方图均衡化算法。

## 链接

| 类型 | URL |
| --- | --- |
| 相关经典来源 | https://doi.org/10.1016/B978-0-12-336156-1.50061-6 |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

AHE，即 Adaptive Histogram Equalization，会在局部图像区域内执行直方图均衡化，而不是像 HE 那样对整幅图像使用同一个全局映射。

这种局部处理方式可以增强暗区域中的细节，但也更容易放大噪声。LibLLIE 中的 AHE 可以作为局部对比度增强的传统算法基线。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/AHE.py` |
| 算法类名 | `AHE` |
| 注册名称 | `ahe` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 实现说明

LibLLIE 的 AHE 通过 OpenCV 的局部直方图均衡化流程实现，并支持在不同色彩空间中处理亮度通道。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | 执行均衡化的色彩空间 |
| `tile_grid_size` | `tuple` | `(8, 8)` | 局部网格大小 |

支持的色彩空间：

- `rgb` / `bgr`
- `hsv`
- `hls`
- `yuv` / `ycbcr`
- `lab`

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "ahe",
    "input.jpg",
    output="results/ahe/output.jpg",
    color_space="yuv",
)
```

文件夹批量处理：

```python
saved_paths = llie.predict(
    "ahe",
    "images/",
    output="results/ahe",
)
```
