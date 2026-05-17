# RCLAHE

RCLAHE 是 LibLLIE 中已经实现的递归 CLAHE 增强算法。

## 链接

| 类型 | URL |
| --- | --- |
| 相关算法 | https://ieeexplore.ieee.org/document/109340/ |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

RCLAHE，即 Recursive CLAHE，会多次执行 CLAHE 以获得更强的局部对比度增强效果。它可以作为比单次 CLAHE 更激进的传统低光增强基线。

由于递归处理可能放大噪声和局部伪影，实际使用时需要谨慎设置递归次数和裁剪阈值。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/RCLAHE.py` |
| 算法类名 | `RCLAHE` |
| 注册名称 | `rclahe` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 实现说明

RCLAHE 复用 CLAHE 的色彩空间处理逻辑，并按照 `iterations` 参数重复执行增强。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `color_space` | `str` | `"yuv"` | 执行 CLAHE 的色彩空间 |
| `clip_limit` | `float` | `2.0` | 对比度裁剪阈值 |
| `tile_grid_size` | `tuple` | `(8, 8)` | 局部网格大小 |
| `iterations` | `int` | `2` | 递归执行 CLAHE 的次数 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "rclahe",
    "input.jpg",
    output="results/rclahe/output.jpg",
    iterations=2,
)
```
