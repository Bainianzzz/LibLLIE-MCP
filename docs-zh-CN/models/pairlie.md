# PairLIE

PairLIE 来自 CVPR 2023 论文 **Learning a Simple Low-light Image Enhancer from Paired Low-light Instances**。它不依赖正常光 GT，而是使用同一场景的两张不同低照实例学习增强。共享网络分别估计去噪表示、照明和反射，并通过两张图像之间的反射一致性完成约束。

## 相关链接

| 类型 | 地址 |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.pdf |
| 官方代码 | https://github.com/zhenqifu/PairLIE |
| 默认配置 | `libllie/deepLearning/config/PairLIE.yaml` |

## LibLLIE 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `libllie/deepLearning/models/PairLIE.py` |
| 模型名称 | `PairLIE`（别名：`Pair-LIE`） |
| 损失实现 | `libllie/deepLearning/loss/PairLIE_Loss.py` |
| 损失名称 | `pairlie` |
| 数据集实现 | `libllie/data/datasets/PairLIE.py` |

推理阶段的增强结果为：

```text
enhanced = illumination ** enhancement_gamma * reflectance
```

默认 `enhancement_gamma=0.2`，与官方通用推理设置一致；官方 LOL 示例使用 `0.14`，预测时可通过配置覆盖。

集成后的训练目标遵循官方实现：

```text
L = consistency_weight * MSE(R1, R2)
  + reconstruction_weight * L_reconstruction
  + preservation_weight * MSE(input1, denoised1)
```

其中 `L_reconstruction` 包含 Retinex 重建、反射估计、照明与最大 RGB 通道之间的约束，以及照明 TV 正则。输入保持项默认权重为 `500`。

## 训练数据布局

PairLIE 需要同一场景的两张不同低照实例，而不是常规的低照/正常光图像对。新增的 `PairLIEInstancesDataset` 支持官方布局：

```text
PairLIE-training-dataset/
  1/
    exposure_1.png
    exposure_2.png
    ...
  2/
    exposure_1.png
    exposure_2.png
    ...
```

也支持 `root/train/scene/...` 形式。数据集每次从一个场景目录中随机选择两张不同图像，并对两张图使用完全相同的随机裁剪。每个场景至少需要两张受支持的图像。

## 训练

修改 YAML 中的 `data.root_dir` 后运行：

```bash
libllie train libllie/deepLearning/config/PairLIE.yaml
```

或使用 Python API：

```python
import libllie as llie

result = llie.train("libllie/deepLearning/config/PairLIE.yaml")
print(result["checkpoint_dir"])
```

统一 Trainer 会识别 PairLIE 的成对前向约定，让两张低照实例通过同一个共享模型；其他模型仍保持原有的单输入路径。

## 预测

使用 LibLLIE 生成的 checkpoint：

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "outputs/PairLIE/checkpoints/best.pt",
    "input.jpg",
    output="results/PairLIE/output.png",
    device="cuda",
)
```

若使用 LOL 设置，可在预测时覆盖 gamma：

```python
enhanced, saved_path = llie.predict(
    "outputs/PairLIE/checkpoints/best.pt",
    "input.jpg",
    output="results/PairLIE/lol.png",
    device="cuda",
    config={"enhancement_gamma": 0.14},
)
```

当前实现保留了官方网络参数名称，因此也可以把官方 `PairLIE.pth` 状态字典加载到默认模型中。LibLLIE 自身的 checkpoint 还包含模型配置，可以直接交给 `llie.predict`。
