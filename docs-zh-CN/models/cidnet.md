# HVI-CIDNet

HVI-CIDNet 是 CVPR 2025 论文 **HVI: A New Color Space for Low-light Image Enhancement** 提出的颜色—强度解耦网络。模型先把 RGB 图像转换到可学习的 HVI 颜色空间，再分别处理 H/V 色度特征和 I 强度特征，并通过 Lighten Cross-Attention（LCA）模块进行双分支交互。

## 相关链接

| 类型 | 地址 |
| --- | --- |
| 论文 | https://arxiv.org/abs/2502.20272 |
| 官方代码 | https://github.com/Fediory/HVI-CIDNet |
| 默认配置 | `libllie/deepLearning/config/CIDNet.yaml` |

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/CIDNet.py` |
| 模型名称 | `CIDNet`（别名：`HVI-CIDNet`） |
| 损失实现 | `libllie/deepLearning/loss/CIDNet_Loss.py` |
| 损失名称 | `cidnet` |

集成后的损失与官方训练目标一致，同时在 RGB 和 HVI 两个域计算：

```text
L = L_RGB + hvi_weight * L_HVI
L_domain = pixel_weight * L1
         + ssim_weight * (1 - SSIM)
         + edge_weight * L_edge
         + perceptual_weight * L_VGG19
```

默认权重依次为 `1.0`、`0.5`、`50.0`、`0.01`，并设置 `hvi_weight=1.0`。VGG19 感知损失使用 `conv1_2`、`conv2_2`、`conv3_4`、`conv4_4` 四层特征及 MSE 距离。首次训练可能由 torchvision 下载 ImageNet VGG19 权重；离线或快速冒烟测试可设置 `loss.params.use_perceptual: false`。

CIDNet 会在内部把输入补齐到 8 的倍数，并在输出时裁回原始尺寸，因此可直接预测任意尺寸图像。`input_gamma`、`saturation_scale` 和 `intensity_scale` 对应上游推理调节能力，默认均为 `1.0`。

## 训练

先修改 YAML 中的数据集路径，然后运行：

```bash
libllie train libllie/deepLearning/config/CIDNet.yaml
```

也可通过 Python API 训练：

```python
import libllie as llie

result = llie.train("libllie/deepLearning/config/CIDNet.yaml")
print(result["checkpoint_dir"])
```

默认配置使用 LOLv1 兼容的成对数据布局。若使用其他成对数据集，可调整 `data.dataset`、路径和 split 名称。

## 预测

使用 LibLLIE 训练器生成的 checkpoint：

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "outputs/CIDNet/checkpoints/best.pt",
    "input.jpg",
    output="results/CIDNet/output.png",
    device="cuda",
)
```

## 加载官方原始权重

当前实现保留了官方参数名称，因此可以把上游 `.pth` 状态字典加载到新建模型中：

```python
import torch
from libllie.deepLearning.models import LLIEModel

model = LLIEModel.create_model("CIDNet", config={"device": "cuda"})
state = torch.load("LOLv1.pth", map_location=model.device)
model.load_state_dict(state, strict=True)
model.eval_mode()
```

LibLLIE 自身生成的训练 checkpoint 还包含配置、优化器等元数据，可以直接交给 `llie.predict` 使用。
