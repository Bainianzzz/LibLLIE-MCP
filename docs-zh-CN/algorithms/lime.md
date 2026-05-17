# LIME

LIME 是一种基于光照图估计的传统低光图像增强算法。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/TIP.2016.2639450 |
| 论文标题 | Low-light Image Enhancement via Illumination Map Estimation |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

LIME 从输入图像的最大颜色通道估计光照图。随后使用边缘保持平滑细化光照图，并利用细化后的光照图恢复更明亮的类反射图像。

在 LibLLIE 中，光照图通过引导滤波进行细化。增强图像通过将输入图像除以 gamma 调整后的光照图得到。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/LIME.py` |
| 算法类名 | `LIME` |
| 注册名称 | `LIME` |
| 别名 | `lime`, `illumination_map_estimation` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `gamma` | `float` | `0.8` | 作用于细化光照图的 gamma |
| `guided_radius` | `int` | `15` | 引导滤波半径 |
| `guided_eps` | `float` | `1e-3` | 引导滤波正则项 |
| `illumination_floor` | `float` | `0.05` | 光照图下界 |
| `exposure` | `float` | `1.0` | 全局曝光倍率 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "lime",
    "input.jpg",
    output="results/lime/output.jpg",
    gamma=0.8,
)
```

文件夹批量处理：

```python
saved_paths = llie.predict(
    "lime",
    "images/",
    output="results/lime",
    progress_bar=True,
)
```
