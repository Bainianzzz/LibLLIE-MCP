# LibLLIE 命令行接口

LibLLIE 安装后会提供命令行接口。注册的命令包括：

```bash
libllie ...
llie ...
```

这两个命令指向同一个 CLI 入口。

## 可用命令

| 命令 | 内部调用的 API |
| --- | --- |
| `predict` | `llie.predict()` |
| `train` | `llie.train()` |
| `evaluate` | `llie.evaluate()` |
| `imwrite` | `llie.imwrite()` |
| `list` | `llie.list_available()` |

图像读取功能不会作为 CLI 命令暴露。

## 列出可用组件

```bash
libllie list
# 或
llie list
```

该命令会打印可用模型、传统算法、评估指标、损失函数和数据集。

## 预测增强

传统算法：

```bash
libllie predict gcp input.jpg -o results/gcp/output.png
```

深度学习 checkpoint：

```bash
libllie predict checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt input.jpg -o results/zerodce/output.png --device cuda
```

文件夹输入：

```bash
libllie predict bimef images/ -o results/bimef
```

常用选项：

| 选项 | 含义 |
| --- | --- |
| `--backend` | Predictor 后端：`auto`、`deep` 或 `traditional` |
| `--device` | 深度学习预测使用的设备 |
| `--output-dir` | Predictor 默认输出目录 |
| `--no-progress` | 禁用文件夹预测进度条 |
| `--no-save` | 单张图像预测时不保存输出 |
| `--output-name` | 保存到文件夹时使用的输出文件名 |
| `--output-ext` | 输出扩展名覆盖 |

额外关键字参数可以通过 `--kwargs` 传入：

```bash
libllie predict lime input.jpg -o results/lime.png --kwargs gamma=0.8 guided_radius=15
```

对于 tuple/list 类型的值，需要在 shell 中给值加引号：

```bash
libllie predict clahe input.jpg -o results/clahe.png --kwargs "tile_grid_size=(8, 8)" clip_limit=2.0
```

## 训练

从 YAML 配置文件训练：

```bash
libllie train libllie/deepLearning/config/ZeroDCE.yaml
```

覆盖训练参数：

```bash
libllie train libllie/deepLearning/config/ZeroDCE.yaml --kwargs epochs=5 batch_size=2 device=cpu
```

`--kwargs` 条目会转发给 `llie.train()`。

## 评估

全参考评估：

```bash
libllie evaluate results/ZeroDCE --ref-img-dir datasets/LOL/eval15/high --metrics PSNR SSIM --save-path results/eval.json
```

无参考评估：

```bash
libllie evaluate results/ZeroDCE --metrics NIQE --save-path results/eval_no_ref.json
```

额外评估器参数可以通过 `--kwargs` 传入：

```bash
libllie evaluate results/ZeroDCE --metrics PSNR SSIM --kwargs device=cpu
```

## Imwrite

通过 `llie.imwrite()` 转换或保存图像：

```bash
libllie imwrite input.jpg -o results/copy.png
```

保存到文件夹并指定输出文件名：

```bash
libllie imwrite input.jpg -o results --output-name copied.png
```

指定保存格式：

```bash
libllie imwrite input.jpg -o results/copied --save-format png
```

## KEY=VALUE 解析

CLI 支持在 `--kwargs` 中使用简单的 Python 风格值：

```bash
--kwargs lr=1e-4 epochs=10 amp=false device=cpu
```

解析规则：

| 输入 | 解析值 |
| --- | --- |
| `true` / `false` | 布尔值 |
| `none` | `None` |
| `10`, `1e-4`, `(8, 8)` | Python 字面量值 |
| 其他字符串 | 原始字符串 |

## 帮助

```bash
libllie --help
libllie predict --help
libllie train --help
libllie evaluate --help
libllie imwrite --help
```
