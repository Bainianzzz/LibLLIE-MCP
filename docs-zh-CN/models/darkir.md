# DarkIR

DarkIR 是 LibLLIE 中已经实现的鲁棒低光图像恢复模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://arxiv.org/pdf/2412.13443 |
| CVPR 论文页 | https://openaccess.thecvf.com/content/CVPR2025/papers/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.pdf |
| 官方源码 | https://github.com/cidautai/DarkIR |

## 模型简介

DarkIR 面向鲁棒低光图像恢复任务，不仅关注亮度提升，也关注低光场景中常见的噪声、颜色退化和细节损失问题。模型采用编码器-解码器式结构，并包含多尺度或深度可分离相关模块以增强暗光恢复能力。

在 LibLLIE 中，DarkIR 的默认配置包含网络宽度、编码器/解码器 block 数量、dilation 设置、是否使用 extra depth-wise 模块，以及是否启用 side loss。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/DarkIR.py` |
| 模型类名 | `DarkIR` |
| 默认配置 | `libllie/deepLearning/config/DarkIR.yaml` |
| 相关损失 | `libllie/deepLearning/loss/DarkIR_Loss.py` |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "DarkIR",
    "input.jpg",
    output="results/DarkIR/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    model="DarkIR",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="darkir",
    epochs=10,
    batch_size=4,
)
```

