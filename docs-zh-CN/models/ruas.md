# RUAS

RUAS 是 LibLLIE 中已经实现的 Retinex-inspired 低光增强模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf |
| 官方源码 | https://github.com/KarelZhang/RUAS |

## 模型简介

RUAS 将 Retinex 思想、展开优化和结构搜索结合起来，用于低光图像增强。模型包含增强和去噪相关结构，通过 unrolling 的方式模拟迭代优化过程，并引入 cooperative prior architecture search 来获得有效网络结构。

在 LibLLIE 中，RUAS 包含 illumination enhancement 和 denoising 模块。默认配置中可以控制 IEM 迭代次数、NRM 层数、增强网络通道数和去噪网络通道数。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/RUAS.py` |
| 模型类名 | `RUAS` |
| 默认配置 | `libllie/deepLearning/config/RUAS.yaml` |
| 相关损失 | `libllie/deepLearning/loss/RUAS_Loss.py` |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "RUAS",
    "input.jpg",
    output="results/RUAS/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    model="RUAS",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="ruas",
    epochs=10,
    batch_size=4,
)
```

