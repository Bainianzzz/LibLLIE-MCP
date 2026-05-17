# EnlightenGAN

EnlightenGAN 是一种注意力引导的生成对抗低光图像增强模型，面向无配对监督场景。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/TIP.2021.3051462 |
| 论文标题 | EnlightenGAN: Deep Light Enhancement Without Paired Supervision |
| 官方源码 | https://github.com/VITA-Group/EnlightenGAN |
| 官方项目页 | https://github.com/VITA-Group/EnlightenGAN |

## 模型简介

EnlightenGAN 不依赖成对的低光/正常光图像监督即可进行低光增强。原方法使用 attention-guided U-Net 生成器、全局和局部判别器，以及 self feature preserving regularization，使增强图像在变亮的同时保留输入图像的结构和颜色特征。

在 LibLLIE 中，EnlightenGAN 被适配到统一模型和训练器接口。模型包含：

| 组件 | 用途 |
| --- | --- |
| `EnlightenGANGenerator` | 注意力引导的 U-Net 生成器 |
| `global_discriminator` | 用于整幅图像的 PatchGAN 判别器 |
| `local_discriminator` | 用于局部裁剪 patch 的 PatchGAN 判别器 |
| attention map | 根据输入图像暗度估计的注意力图 |

对应损失函数整合了全局/局部 LSGAN 损失、判别器损失、自特征保持正则、曝光正则和总变分平滑约束。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/EnlightenGAN.py` |
| 模型类名 | `EnlightenGAN` |
| 默认配置 | `libllie/deepLearning/config/EnlightenGAN.yaml` |
| 相关损失 | `libllie/deepLearning/loss/EnlightenGAN_Loss.py` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `generator_channels` | `int` | `32` | 生成器使用的基础特征通道数 |
| `discriminator_channels` | `int` | `32` | 判别器使用的基础特征通道数 |
| `discriminator_layers` | `int` | `3` | PatchGAN 下采样层数 |
| `use_attention` | `bool` | `True` | 是否将 attention map 拼接到生成器输入中 |
| `local_patch_ratio` | `float` | `0.5` | 局部对抗损失使用的中心 patch 比例 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "EnlightenGAN",
    "input.jpg",
    output="results/EnlightenGAN/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    "libllie/deepLearning/config/EnlightenGAN.yaml",
    root_dir="datasets/LOL",
    epochs=10,
    batch_size=2,
)
```

快速调试配置：

```python
llie.train(
    model="EnlightenGAN",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="enlightengan",
    model_params={
        "generator_channels": 16,
        "discriminator_channels": 16,
    },
    epochs=2,
    batch_size=1,
)
```
