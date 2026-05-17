# Zero-DCE++

Zero-DCE++ 是 LibLLIE 中已经实现的轻量化零参考低光增强模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://ieeexplore.ieee.org/document/9369102/ |
| 官方源码 | https://github.com/Li-Chongyi/Zero-DCE_extension |
| 官方项目页 | https://li-chongyi.github.io/Proj_Zero-DCE++.html |

## 模型简介

Zero-DCE++ 是 Zero-DCE 的轻量扩展版本，目标是在保持零参考训练优势的同时进一步降低模型复杂度。该模型通过更紧凑的网络结构估计增强曲线，适合对速度和参数量要求较高的低光增强应用。

在 LibLLIE 中，Zero-DCE++ 保留了曲线估计类模型的接口风格。训练模式下返回增强结果和曲线相关中间变量；推理模式下返回增强图像。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/ZeroDCEPlusPlus.py` |
| 模型类名 | `ZeroDCEPlusPlus` |
| 默认配置 | `libllie/deepLearning/config/ZeroDCE++.yaml` |
| 相关损失 | `libllie/deepLearning/loss/ZeroDCE_Loss.py` |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "ZeroDCEPlusPlus",
    "input.jpg",
    output="results/ZeroDCEPlusPlus/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    model="ZeroDCEPlusPlus",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="zerodce_extension",
    epochs=10,
    batch_size=4,
)
```

