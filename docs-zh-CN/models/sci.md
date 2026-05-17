# SCI

SCI 是 LibLLIE 中已经实现的快速低光增强模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| 官方源码 | https://github.com/vis-opt-group/SCI |

## 模型简介

SCI 的全称通常对应 Self-Calibrated Illumination。该模型面向快速、灵活、鲁棒的低光图像增强，通过增强网络和校准网络逐步优化照明估计。相比复杂的多阶段模型，SCI 强调轻量结构和高效推理。

在 LibLLIE 中，SCI 由 illumination enhancement 和 calibration 两部分组成。模型配置中可以控制 stage 数量、增强网络层数、校准网络层数和通道数。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/SCI.py` |
| 模型类名 | `SCI` |
| 默认配置 | `libllie/deepLearning/config/SCI.yaml` |
| 相关损失 | `libllie/deepLearning/loss/Sci_Loss.py` |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "SCI",
    "input.jpg",
    output="results/SCI/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    model="SCI",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="sci",
    epochs=10,
    batch_size=4,
)
```

