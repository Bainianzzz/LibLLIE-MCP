# CLAHE

CLAHE 是 LibLLIE 中已经实现的限制对比度自适应直方图均衡化算法。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 / 经典来源 | https://ieeexplore.ieee.org/document/109340/ |
| Graphics Gems 章节 | https://doi.org/10.1016/B978-0-12-336156-1.50061-6 |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

CLAHE，即 Contrast Limited Adaptive Histogram Equalization，是 AHE 的改进版本。它在局部直方图均衡化的基础上引入裁剪阈值，用于限制局部对比度放大，从而降低噪声过度增强的问题。

CLAHE 常用于医学图像、低光图像、遥感图像以及其他低对比度图像增强任务。与 HE 相比，CLAHE 更关注局部细节；与 AHE 相比，CLAHE 对噪声更稳健。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/CLAHE.py` |
| 算法类名 | `CLAHE` |
| 注册名称 | `clahe` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 实现说明

LibLLIE 的 CLAHE 基于 OpenCV 的 `cv2.createCLAHE()` 实现，支持在不同色彩空间中处理亮度通道或全部通道。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | 执行 CLAHE 的色彩空间 |
| `clip_limit` | `float` | `2.0` | 对比度裁剪阈值 |
| `tile_grid_size` | `tuple` | `(8, 8)` | 局部网格大小 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "clahe",
    "input.jpg",
    output="results/clahe/output.jpg",
    color_space="lab",
    clip_limit=2.0,
    tile_grid_size=(8, 8),
)
```

文件夹批量处理：

```python
saved_paths = llie.predict(
    "clahe",
    "images/",
    output="results/clahe",
    progress_bar=True,
)
```
