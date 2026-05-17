# KinD++

KinD++ 是一种改进的基于 Retinex 的低光增强模型，在 KinD 基础上加入了多尺度照明注意力。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1007/s11263-020-01407-x |
| 论文标题 | Beyond Brightening Low-light Images |
| 官方源码 | https://github.com/zhangyhuaee/KinD_plus |
| 官方项目页 | None |

## 模型简介

KinD++ 在原始 KinD 框架基础上提升了低光增强的稳健性。它保留了 Retinex 风格的 decomposition-restoration-adjustment 流程，但在反射恢复网络中引入了 multi-scale illumination attention module（MSIA）。MSIA 利用估计出的照明图引导多尺度特征恢复，从而减少非均匀亮斑和过平滑伪影。

在 LibLLIE 中，由于 Python 类名不能包含 `++`，模型类名实现为 `KinDPlusPlus`。默认 YAML 文件保留论文风格名称：

```text
libllie/deepLearning/config/KinD++.yaml
```

LibLLIE 实现包含：

| 子网络 | 用途 |
| --- | --- |
| `KinDPPDecompositionNet` | 估计反射分量和照明分量 |
| `KinDPPRestorationNet` | 使用 MSIA 引导特征恢复反射分量 |
| `KinDPPIlluminationAdjustmentNet` | 使用曝光比例图调整照明分量 |

对应损失函数整合了官方分阶段训练代码中的主要目标：分解重建、反射一致性、互照明约束、输入感知照明平滑、反射恢复、照明调整以及最终增强图像监督。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/KinDPlusPlus.py` |
| 模型类名 | `KinDPlusPlus` |
| 默认配置 | `libllie/deepLearning/config/KinD++.yaml` |
| 相关损失 | `libllie/deepLearning/loss/KinDPlusPlus_Loss.py` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `decomposition_channels` | `int` | `32` | decomposition network 使用的基础特征通道数 |
| `restoration_channels` | `int` | `32` | MSIA restoration network 使用的基础特征通道数 |
| `adjustment_channels` | `int` | `32` | illumination adjustment network 使用的特征通道数 |
| `illumination_ratio` | `float` | `5.0` | 推理时 illumination adjustment 使用的曝光比例 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "KinDPlusPlus",
    "input.jpg",
    output="results/KinDPlusPlus/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    "libllie/deepLearning/config/KinD++.yaml",
    root_dir="datasets/LOL",
    epochs=10,
    batch_size=2,
)
```

覆盖推理曝光比例：

```python
enhanced, saved_path = llie.predict(
    "KinDPlusPlus",
    "input.jpg",
    output="results/KinDPlusPlus/brighter.png",
    device="cuda",
    illumination_ratio=6.0,
)
```
