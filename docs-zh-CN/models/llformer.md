# LLFormer

LLFormer 来自 AAAI 2023 Oral 论文 **Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method**。模型组合了行/列轴向自注意力、双门控前馈网络、跨层注意力融合、四级编码器—解码器和可学习加权跳连。轴向注意力避免构造完整的二维空间注意力图，更适合高分辨率低照增强。

## 相关链接

| 类型 | 地址 |
| --- | --- |
| 论文 | https://arxiv.org/abs/2212.11548 |
| 官方代码 | https://github.com/TaoWangzj/LLFormer |
| 默认配置 | `libllie/deepLearning/config/LLFormer.yaml` |

## 许可证说明

集成的 LLFormer 架构改编自官方实现，其许可证为 **CC BY-NC-SA 4.0**，仅供学术、非商业研究使用。完整声明见 `libllie/deepLearning/models/LLFormer_LICENSE.txt`。该限制严于 LibLLIE 总体采用的 MIT 许可证，并适用于本模型实现。

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/LLFormer.py` |
| 模型名称 | `LLFormer`（别名：`LL-Former`） |
| 损失实现 | `libllie/deepLearning/loss/LLFormer_Loss.py` |
| 损失名称 | `llformer` |
| 成对 patch 数据集 | `libllie/data/datasets/LLFormer.py` |

默认网络与官方训练入口一致：`dim=16`、block 数 `[2, 4, 8, 16]`、head 数 `[1, 2, 4, 8]`、两个 refinement block、WithBias LayerNorm，并关闭全局残差 skip。

官方训练入口实际使用 PyTorch `SmoothL1Loss`，LibLLIE 将同一目标注册为 `llformer`。上游工具文件中附带的 Charbonnier、Edge、SSIM 和 PSNR 损失并未用于论文训练命令，因此默认不启用。

## 数据布局与 patch 训练

`LLFormerPairedDataset` 支持标准成对布局：

```text
dataset/
  train/
    low/
    high/
  val/
    low/
    high/
```

训练默认对低照图和 GT 执行同步 `128×128` 随机裁剪，并使用与官方加载器一致的翻转/旋转增强；验证默认保留完整图像。可以通过 `data.train_params.crop_size` 修改 patch 尺寸，建议尺寸可被 16 整除。

## 训练

修改 `data.root_dir` 后运行：

```bash
libllie train libllie/deepLearning/config/LLFormer.yaml
```

或使用 Python：

```python
import libllie as llie

result = llie.train("libllie/deepLearning/config/LLFormer.yaml")
print(result["checkpoint_dir"])
```

默认 YAML 采用官方 LOL 训练计划：Adam、初始学习率 `1e-4`、最小学习率 `1e-6`、batch size 8、共 4000 epoch。官方 UHD-LOL4K/8K 实验使用 300 epoch 和 batch size 12，可在同一配置文件中调整。

## 预测

模型会自动把输入补齐到 16 的倍数，并在输出时裁回原始尺寸：

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "outputs/LLFormer/checkpoints/best.pt",
    "input.png",
    output="results/LLFormer/output.png",
    device="cuda",
)
```

4K/8K 图像可启用重叠分块推理以降低显存占用。下列配置对应官方 UHD 脚本的 patch 与半步长策略：

```python
enhanced, saved_path = llie.predict(
    "outputs/LLFormer/checkpoints/best.pt",
    "uhd_input.png",
    output="results/LLFormer/uhd_output.png",
    device="cuda",
    config={
        "tile_size": [720, 1280],
        "tile_overlap": [360, 640],
    },
)
```

tile 尺寸必须能被 16 整除。重叠区域与边界 patch 会自动平均融合。

## 官方 checkpoint

当前模型保留了官方模块与参数名称。官方 checkpoint 将权重保存在 `state_dict` 中，多卡训练权重可能带有 `module.` 前缀，加载前需要移除。LibLLIE 自身生成的 checkpoint 已包含模型配置，可直接传给 `llie.predict`。
