# LLNet

LLNet 是一种用于自然低光图像增强的堆叠稀疏去噪自编码器模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1016/j.patcog.2016.06.008 |
| 论文 PDF | https://web.me.iastate.edu/soumiks/pdf/Journal/LAS16_llnet.pdf |
| 官方源码 | https://github.com/kglore/llnet_color |
| 官方项目页 | None |

## 模型简介

LLNet 使用堆叠稀疏去噪自编码器增强低光图像。原方法会对局部图像 patch 进行退化和向量化，训练自编码器从退化低光 patch 重建干净的正常光 patch，然后再由重叠 patch 的预测结果重建完整增强图像。

在 LibLLIE 中，LLNet 保留了这种 patch-wise autoencoder 思路，但将 patch 提取和聚合封装在 PyTorch 图像到图像模型内部。当前实现使用 `torch.nn.functional.unfold` 提取重叠 patch，通过全连接编码器-解码器网络处理每个 patch，并使用 fold 将重建 patch 聚合回图像。

对应的 LibLLIE 损失函数使用监督重建误差，并提供可选的稀疏自编码器正则项。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/LLNet.py` |
| 模型类名 | `LLNet` |
| 默认配置 | `libllie/deepLearning/config/LLNet.yaml` |
| 相关损失 | `libllie/deepLearning/loss/LLNet_Loss.py` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `patch_size` | `int` | `17` | 自编码器使用的局部 patch 大小 |
| `patch_stride` | `int` | `3` | 提取重叠 patch 时使用的步长 |
| `hidden_dims` | `list[int]` | `[2000, 1600, 1200]` | 编码器隐藏层维度 |
| `activation` | `str` | `"sigmoid"` | 隐藏层激活函数：`sigmoid`、`relu` 或 `tanh` |
| `output_activation` | `str` | `"sigmoid"` | 输出激活函数：`sigmoid`、`clamp` 或 `none` |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "LLNet",
    "input.jpg",
    output="results/LLNet/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    "libllie/deepLearning/config/LLNet.yaml",
    root_dir="datasets/LOL",
    epochs=10,
    batch_size=1,
)
```

快速调试时可以使用更小的隐藏层维度：

```python
llie.train(
    model="LLNet",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="llnet",
    model_params={
        "hidden_dims": [256, 128, 64],
        "patch_stride": 8,
    },
    epochs=2,
    batch_size=1,
)
```
