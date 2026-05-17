# Zero-DCE

Zero-DCE 是 LibLLIE 中已经实现的零参考低光增强模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf |
| 官方源码 | https://github.com/Li-Chongyi/Zero-DCE |
| 官方项目页 | https://li-chongyi.github.io/Proj_Zero-DCE.html |

## 模型简介

Zero-DCE 将低光增强建模为图像特定的曲线估计问题。模型不依赖成对的低光/正常光图像进行监督训练，而是通过一组无参考约束学习逐像素增强曲线。该方法的核心特点是结构轻量、推理速度快，并且适合零参考低光增强场景。

在 LibLLIE 中，Zero-DCE 输出增强图像以及曲线参数。训练模式下，模型会通过标准输出结构返回 `pred` 和 loss 所需的中间变量；推理模式下直接返回增强图像。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/ZeroDCE.py` |
| 模型类名 | `ZeroDCE` |
| 默认配置 | `libllie/deepLearning/config/ZeroDCE.yaml` |
| 相关损失 | `libllie/deepLearning/loss/ZeroDCE_Loss.py` |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/ZeroDCE/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    model="ZeroDCE",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="zerodce",
    epochs=10,
    batch_size=4,
)
```

