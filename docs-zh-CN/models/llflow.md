# LLFlow

LLFlow 是一种用于低光图像增强的条件 normalizing flow 模型。

## 链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1609/aaai.v36i3.20162 |
| 论文标题 | Low-Light Image Enhancement with Normalizing Flow |
| 官方源码 | https://github.com/wyf0912/LLFlow |
| 官方项目页 | https://github.com/wyf0912/LLFlow |

## 模型简介

LLFlow 使用 normalizing flow 对给定低光图像条件下的正常曝光图像条件分布进行建模。它不只是预测一个确定性输出，而是学习图像空间和 latent 空间之间的可逆映射，并以低光输入作为条件。原方法使用负对数似然目标训练 flow，并可通过采样或确定性 latent code 生成增强图像。

在 LibLLIE 中，LLFlow 被适配到统一的 image-to-image 模型接口。当前实现包含：

| 组件 | 用途 |
| --- | --- |
| `LLFlowConditionEncoder` | 提取低光条件特征 |
| `LLFlowAffineCoupling` | 条件 affine coupling flow 层 |
| `LLFlow` | 堆叠条件 flow 层并执行正向/反向变换 |

训练时，损失函数会将成对的正常光目标图像映射到 latent 空间，并计算 flow negative log-likelihood。推理时，模型默认使用零 latent 张量生成确定性增强结果。设置 `sample_temperature > 0` 可以采样随机增强结果。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/LLFlow.py` |
| 模型类名 | `LLFlow` |
| 默认配置 | `libllie/deepLearning/config/LLFlow.yaml` |
| 相关损失 | `libllie/deepLearning/loss/LLFlow_Loss.py` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `condition_channels` | `int` | `32` | 低光条件特征通道数 |
| `condition_blocks` | `int` | `4` | condition encoder 中的残差块数量 |
| `flow_layers` | `int` | `8` | 条件 affine coupling 层数量 |
| `flow_hidden_channels` | `int` | `64` | 每个 coupling network 中的隐藏通道数 |
| `scale_clamp` | `float` | `2.0` | affine coupling log-scale 的裁剪范围 |
| `sample_temperature` | `float` | `0.0` | 推理时使用的 latent 采样温度 |

## 调用示例

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "LLFlow",
    "input.jpg",
    output="results/LLFlow/output.png",
    device="cuda",
)
```

训练示例：

```python
llie.train(
    "libllie/deepLearning/config/LLFlow.yaml",
    root_dir="datasets/LOL",
    epochs=10,
    batch_size=2,
)
```

随机采样推理：

```python
enhanced, saved_path = llie.predict(
    "LLFlow",
    "input.jpg",
    output="results/LLFlow/sample.png",
    device="cuda",
    sample_temperature=0.7,
)
```
