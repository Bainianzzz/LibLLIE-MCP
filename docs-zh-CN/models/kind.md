# KinD

KinD 是一种受 Retinex 启发的实用低光图像增强深度学习模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1145/3343031.3350926 |
| 论文标题 | Kindling the Darkness: A Practical Low-light Image Enhancer |
| 官方源码 | https://github.com/zhangyhuaee/KinD |
| 官方项目页 | None |

## 模型简介

KinD 将图像分解为反射分量和照明分量，随后恢复退化的反射分量、调整照明分量，并将二者重新组合得到增强结果。原始实现采用分阶段方式训练 decomposition、restoration 和 illumination adjustment 三个网络。

在 LibLLIE 中，KinD 被实现为一个集成式 PyTorch 模型，以适配统一训练和预测流水线。模型包含：

| 子网络 | 用途 |
| --- | --- |
| `KinDDecompositionNet` | 估计反射分量 `R` 和照明分量 `I` |
| `KinDRestorationNet` | 在照明引导下恢复低光反射分量 |
| `KinDIlluminationAdjustmentNet` | 使用曝光比例图调整照明分量 |

对应损失函数整合了官方分阶段训练中的主要目标，包括 Retinex 重建、反射一致性、照明平滑、反射恢复、照明调整以及最终增强图像监督。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/KinD.py` |
| 模型类名 | `KinD` |
| 默认配置 | `libllie/deepLearning/config/KinD.yaml` |
| 相关损失 | `libllie/deepLearning/loss/KinD_Loss.py` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `decomposition_channels` | `int` | `64` | decomposition network 使用的特征通道数 |
| `decomposition_layers` | `int` | `5` | decomposition network 的中间层数量 |
| `restoration_channels` | `int` | `32` | restoration network 使用的基础特征通道数 |
| `adjustment_channels` | `int` | `32` | illumination adjustment network 使用的特征通道数 |
| `adjustment_layers` | `int` | `3` | illumination adjustment network 的中间层数量 |
| `illumination_ratio` | `float` | `5.0` | 推理时 illumination adjustment 使用的曝光比例 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "KinD",
    "input.jpg",
    output="results/KinD/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    "libllie/deepLearning/config/KinD.yaml",
    root_dir="datasets/LOL",
    epochs=10,
    batch_size=2,
)
```

覆盖推理曝光比例：

```python
enhanced, saved_path = llie.predict(
    "KinD",
    "input.jpg",
    output="results/KinD/brighter.png",
    device="cuda",
    illumination_ratio=6.0,
)
```
