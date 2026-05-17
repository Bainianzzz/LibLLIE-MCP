# GCP

GCP 是 LibLLIE 中实现的 Gamma Correction Prior 低光增强算法，对应论文 “Low-light image enhancement using gamma correction prior in mixed color spaces”。

## 链接

| 类型 | URL |
| --- | --- |
| 论文链接 | https://www.sciencedirect.com/science/article/abs/pii/S0031320323006994 |
| 官方源码 | https://github.com/TripleJ2543/Low_Light_Pattern_Recognition_2023 |
| 官方项目页 | None |

## 算法简介

GCP 不是使用固定指数的普通 Gamma 校正。它利用 gamma correction prior 构造像素级自适应 gamma，并结合暗通道思想估计大气光和透射图，从而提升低光图像的亮度和可见细节。

LibLLIE 的实现参考了官方开源脚本。主要流程包括：

1. 将图像转换到归一化浮点空间。
2. 对输入图像进行高斯平滑并取反，得到用于估计的低光退化表示。
3. 计算暗通道，并从暗通道响应较高的区域估计大气光。
4. 根据大气光对各通道进行归一化。
5. 由每个像素的最大通道值生成自适应 gamma，并估计透射图。
6. 根据透射图恢复增强图像，并通过百分位动态范围拉伸得到最终结果。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/GCP.py` |
| 算法类名 | `GCP` |
| 注册名称 | `GCP` |
| 别名 | `gcp`, `gamma_correction_prior` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `gamma_max` | `float` | `6.0` | 像素自适应 gamma 的最大值 |
| `erosion_window` | `int` | `15` | 暗通道腐蚀核大小 |
| `atmospheric_bins` | `int` | `200` | 大气光估计中使用的直方图 bin 数量 |
| `atmospheric_percentile` | `float` | `0.99` | 选择大气光候选区域时使用的暗通道百分位比例 |
| `t_min` | `float` | `0.1` | 透射图下界 |
| `blur_ksize` | `int` | `7` | 高斯平滑核大小，必须为正奇数 |
| `high_percentile` | `float` | `99.5` | 最终动态范围拉伸的高百分位 |
| `low_percentile` | `float` | `0.5` | 最终动态范围拉伸的低百分位 |
| `eps` | `float` | `1e-6` | 避免除零的小数值 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "gcp",
    "input.jpg",
    output="results/gcp/output.jpg",
)
```

通过统一 Predictor 显式调用传统算法后端：

```python
from libllie import Predictor

predictor = Predictor("gcp", backend="traditional")
predictor("images/", output="results/gcp")
```

传入自定义参数：

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "gcp",
    "input.jpg",
    output="results/gcp_custom.png",
    gamma_max=5.0,
    erosion_window=11,
    high_percentile=99.0,
    low_percentile=1.0,
)
```
