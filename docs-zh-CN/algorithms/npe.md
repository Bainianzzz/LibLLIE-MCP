# NPE

NPE 是一种面向非均匀光照图像的自然性保持增强算法。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/TIP.2013.2261309 |
| 论文标题 | Naturalness Preserved Enhancement Algorithm for Non-Uniform Illumination Images |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

NPE 用于在保持自然视觉外观的同时增强非均匀光照图像。该方法估计光照，分离类反射细节，使用自然性保持变换映射光照，并将增强后的细节与原图进行融合。

在 LibLLIE 中，NPE 使用 bright-pass 光照估计、高斯平滑、类 bi-log 光照变换以及自然性融合权重。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/NPE.py` |
| 算法类名 | `NPE` |
| 注册名称 | `NPE` |
| 别名 | `npe`, `naturalness_preserved_enhancement` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `sigma` | `float` | `15.0` | bright-pass 光照滤波的高斯尺度 |
| `illumination_floor` | `float` | `0.05` | 光照图下界 |
| `enhancement_strength` | `float` | `4.0` | bi-log 光照映射强度 |
| `naturalness` | `float` | `0.35` | 保持原始自然性的融合权重 |
| `detail_weight` | `float` | `1.0` | 作用于反射细节恢复的权重 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "npe",
    "input.jpg",
    output="results/npe/output.jpg",
)
```

调整自然性权重：

```python
enhanced, saved_path = llie.predict(
    "npe",
    "input.jpg",
    output="results/npe/natural.jpg",
    naturalness=0.5,
    enhancement_strength=3.0,
)
```
