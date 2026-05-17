# LEDNet

LEDNet 是 LibLLIE 中已经实现的低光增强和去模糊模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://arxiv.org/pdf/2202.03373 |
| 官方源码 | https://github.com/sczhou/LEDNet |

## 模型简介

LEDNet 面向低光图像增强和去模糊任务。模型设计中包含多尺度特征提取、动态卷积或注意力相关模块，用于同时提升暗光场景中的亮度、细节和清晰度。

在 LibLLIE 中，LEDNet 的配置包含通道设置、skip connection、side loss、动态卷积 kernel size、曲线注意力迭代次数和 pyramid pooling bin 设置。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/LEDNet.py` |
| 模型类名 | `LEDNet` |
| 默认配置 | `libllie/deepLearning/config/LEDNet.yaml` |
| 相关损失 | `libllie/deepLearning/loss/LEDNet_Loss.py` |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "LEDNet",
    "input.jpg",
    output="results/LEDNet/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    model="LEDNet",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="lednet",
    epochs=10,
    batch_size=4,
)
```

