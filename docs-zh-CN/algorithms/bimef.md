# BIMEF

BIMEF 是一种面向低光图像增强的生物启发式多曝光融合算法。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.48550/arXiv.1711.00591 |
| 论文标题 | A Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

BIMEF 通过生成输入图像的曝光增强版本，并将其与原图进行融合来增强低光图像。融合权重由对比度、饱和度和良好曝光度计算得到，从而在细节增强和自然观感之间取得平衡。

在 LibLLIE 中，曝光比例既可以由用户手动指定，也可以根据图像亮度均值自动估计。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/BIMEF.py` |
| 算法类名 | `BIMEF` |
| 注册名称 | `BIMEF` |
| 别名 | `bimef`, `bio_inspired_multi_exposure_fusion` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `exposure_ratio` | `Optional[float]` | `None` | 手动曝光比例；如果为 `None`，则自动估计 |
| `target_mean` | `float` | `0.55` | 自动曝光估计时的目标亮度均值 |
| `max_ratio` | `float` | `5.0` | 自动曝光比例的最大值 |
| `well_exposed_sigma` | `float` | `0.2` | 良好曝光度权重中的 sigma |
| `contrast_weight` | `float` | `1.0` | 对比度权重指数 |
| `saturation_weight` | `float` | `1.0` | 饱和度权重指数 |
| `well_exposed_weight` | `float` | `1.0` | 良好曝光度权重指数 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "bimef",
    "input.jpg",
    output="results/bimef/output.jpg",
)
```

手动设置曝光比例：

```python
enhanced, saved_path = llie.predict(
    "bimef",
    "input.jpg",
    output="results/bimef/manual.jpg",
    exposure_ratio=3.0,
)
```
