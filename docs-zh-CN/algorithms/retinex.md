# Retinex

Retinex 是 LibLLIE 中已经实现的一组传统低光增强算法。当前实现包括 SSR、MSR 和 MSRCR。

## 链接

| 类型 | URL |
| --- | --- |
| SSR 相关论文 | https://doi.org/10.1109/83.557356 |
| MSR / MSRCR 相关论文 | https://doi.org/10.1109/83.597272 |
| 官方源码 | None |
| 官方项目页 | None |

## 算法简介

Retinex 方法基于照明和反射分量分离的思想。对于低光增强，Retinex 会抑制缓慢变化的照明分量，并突出与反射相关的图像细节。

LibLLIE 当前实现了三个 Retinex 变体：

| 算法 | 含义 |
| --- | --- |
| `SSR` | Single Scale Retinex |
| `MSR` | Multi Scale Retinex |
| `MSRCR` | Multi Scale Retinex with Color Restoration |

## SSR

SSR 使用单个高斯环绕尺度估计照明：

```text
R(x, y) = log(I(x, y)) - log(G_sigma(x, y) * I(x, y))
```

SSR 简单高效，但单一尺度不一定能同时兼顾局部细节和全局照明。

## MSR

MSR 会对多个高斯环绕尺度下的 Retinex 响应进行平均：

```text
MSR = mean(SSR_sigma_1, SSR_sigma_2, ..., SSR_sigma_n)
```

相比 SSR，MSR 能更好地平衡细节增强和整体色调一致性。

## MSRCR

MSRCR 在 MSR 基础上加入颜色恢复项，用于减少逐通道 Retinex 处理带来的颜色失真问题，常用于彩色低光图像。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `libllie/traditional/algorithms/Retinex.py` |
| SSR 类名 | `SSR` |
| MSR 类名 | `MSR` |
| MSRCR 类名 | `MSRCR` |
| 注册名称 | `SSR`, `MSR`, `MSRCR` |
| 别名 | `ssr`, `single_scale_retinex`, `msr`, `multi_scale_retinex`, `msrcr`, `multi_scale_retinex_color_restoration` |
| 基类 | `libllie/traditional/algorithms/BaseEnhancer.py` 中的 `LLIEnhancer` |

## 主要参数

通用参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `low_clip` | `float` | `1.0` | 显示归一化使用的低百分位 |
| `high_clip` | `float` | `99.0` | 显示归一化使用的高百分位 |
| `eps` | `float` | `1e-6` | 避免 log 和除法不稳定的小数值 |

SSR 参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `sigma` | `float` | `80.0` | 高斯环绕尺度 |

MSR 参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `scales` | `Sequence[float]` | `(15.0, 80.0, 250.0)` | 高斯环绕尺度 |

MSRCR 参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `scales` | `Sequence[float]` | `(15.0, 80.0, 250.0)` | 高斯环绕尺度 |
| `alpha` | `float` | `125.0` | 颜色恢复强度增益 |
| `beta` | `float` | `46.0` | 颜色恢复对数增益 |
| `gain` | `float` | `1.0` | 作用在恢复后 Retinex 响应上的全局增益 |
| `offset` | `float` | `0.0` | 显示归一化前加入的全局偏移 |

## 调用示例

SSR：

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "ssr",
    "input.jpg",
    output="results/ssr/output.jpg",
    sigma=80.0,
)
```

MSR：

```python
enhanced, saved_path = llie.predict(
    "msr",
    "input.jpg",
    output="results/msr/output.jpg",
    scales=(15.0, 80.0, 250.0),
)
```

MSRCR：

```python
enhanced, saved_path = llie.predict(
    "msrcr",
    "input.jpg",
    output="results/msrcr/output.jpg",
    alpha=125.0,
    beta=46.0,
)
```

文件夹批量处理：

```python
saved_paths = llie.predict(
    "msrcr",
    "images/",
    output="results/msrcr",
    progress_bar=True,
)
```
