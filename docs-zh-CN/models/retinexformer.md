# RetinexFormer

RetinexFormer 是一个基于 Retinex 的单阶段 Transformer 低光图像增强模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.pdf |
| 官方源码 | https://github.com/caiyuanhao1998/Retinexformer |
| 官方项目页 | None |

## 模型简介

RetinexFormer 将 Retinex 照明建模与 Transformer 风格的特征恢复结合起来。模型首先估计照明特征和照明图，然后使用 illumination-guided multi-head self-attention 引导图像恢复。相比多阶段 Retinex 分解流程，RetinexFormer 被设计为用于低光增强的单阶段结构。

在 LibLLIE 中，RetinexFormer 被实现为一个已注册的深度学习模型。当前实现包含照明估计器、照明引导注意力块、U-Net 风格去噪恢复模块，以及可选的多 stage 堆叠。训练模式下，模型通过 LibLLIE 标准模型输出字典返回增强图像和 stage 中间结果；推理模式下直接返回增强图像 Tensor。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/RetinexFormer.py` |
| 模型类名 | `RetinexFormer` |
| 默认配置 | `libllie/deepLearning/config/RetinexFormer.yaml` |
| 相关损失 | `libllie/deepLearning/loss/RetinexFormer_Loss.py` |

## 损失函数

RetinexFormer 官方训练配置使用 pixel-level L1 loss。LibLLIE 将该损失注册为 `retinexformer`，并额外提供可选的 illumination consistency 项，可以通过 `illumination_weight` 控制。

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "RetinexFormer",
    "input.jpg",
    output="results/RetinexFormer/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    model="RetinexFormer",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="retinexformer",
    epochs=10,
    batch_size=4,
)
```

YAML 配置：

```python
llie.train("libllie/deepLearning/config/RetinexFormer.yaml")
```
