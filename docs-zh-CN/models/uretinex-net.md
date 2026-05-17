# URetinex-Net

URetinex-Net 是 LibLLIE 中已经实现的 Retinex-based deep unfolding 低光增强模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf |
| 官方源码 | https://github.com/AndersonYong/URetinex-Net |

## 模型简介

URetinex-Net 基于 Retinex 理论，将低光图像增强表示为可展开的优化过程。模型通过多个 unfolding round 对反射分量、照明分量和增强结果进行联合建模，适合处理复杂照明退化下的低光增强问题。

在 LibLLIE 中，URetinex-Net 的实现类名为 `URetinexNet`。配置中可以设置 unfolding 轮数、Retinex 相关权重、照明调整比例以及是否使用自适应 ratio。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/URetinex.py` |
| 模型类名 | `URetinexNet` |
| 默认配置 | `libllie/deepLearning/config/URetinexNet.yaml` |
| 相关损失 | `libllie/deepLearning/loss/URetinex_Loss.py` |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "URetinexNet",
    "input.jpg",
    output="results/URetinexNet/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    model="URetinexNet",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="uretinex",
    epochs=10,
    batch_size=4,
)
```

