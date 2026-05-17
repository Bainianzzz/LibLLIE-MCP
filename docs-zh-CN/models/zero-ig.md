# Zero-IG

Zero-IG 是 LibLLIE 中已经实现的零样本低光增强和去噪模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.pdf |
| 官方源码 | https://github.com/Doyle59217/ZeroIG |

## 模型简介

Zero-IG 的全称是 Zero-shot Illumination-Guided joint denoising and adaptive enhancement。该方法面向零样本场景，同时考虑低光增强和噪声抑制，通过 illumination-guided 机制引导增强和去噪过程。

在 LibLLIE 中，Zero-IG 包含增强网络、两阶段去噪网络、纹理差异模块和局部平均池化等组件。配置中可以设置增强层数、增强通道数、去噪通道数和运行模式。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/ZeroIG.py` |
| 模型类名 | `ZeroIG` |
| 默认配置 | `libllie/deepLearning/config/ZeroIG.yaml` |
| 相关损失 | `libllie/deepLearning/loss/ZeroIG_Loss.py` |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "ZeroIG",
    "input.jpg",
    output="results/ZeroIG/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    model="ZeroIG",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="zeroig",
    epochs=10,
    batch_size=4,
)
```

