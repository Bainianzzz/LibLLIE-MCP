# LibLLIE 顶层 API 概览

LibLLIE 的顶层 API 目标是通过一个统一入口完成常见任务：

```python
import libllie as llie
```

随后直接调用 `llie.predict()`、`llie.train()`、`llie.evaluate()`、`llie.imread()`、`llie.imwrite()` 等函数。

## 主要接口

| API | 功能 |
| --- | --- |
| `llie.predict()` | 使用深度学习模型或传统算法进行低光增强 |
| `llie.enhance()` | `llie.predict()` 的别名 |
| `llie.train()` | 启动训练流程 |
| `llie.evaluate()` | 对增强结果进行指标评估 |
| `llie.eval()` | `llie.evaluate()` 的别名 |
| `llie.imread()` | 统一读取图像 |
| `llie.imwrite()` | 统一保存图像 |
| `llie.read_image()` | `llie.imread()` 的别名 |
| `llie.write_image()` | `llie.imwrite()` 的别名 |
| `llie.list_available()` | 查看当前可用的模型、算法、指标、损失函数和数据集 |

## 快速查看可用组件

```python
import libllie as llie

available = llie.list_available()
print(available)
```

返回结果通常包含：

```python
{
    "models": [...],
    "algorithms": [...],
    "metrics": [...],
    "losses": [...],
    "datasets": [...],
}
```

也可以分别查看：

```python
llie.list_models()
llie.list_algorithms()
llie.list_metrics()
llie.list_losses()
llie.list_datasets()
```

## API 设计思路

LibLLIE 内部仍然保留清晰的模块结构：

| 模块 | 职责 |
| --- | --- |
| `libllie.data` | 图像读写、数据集、数据变换 |
| `libllie.traditional` | 传统低光增强算法 |
| `libllie.deepLearning` | 深度学习模型、损失函数、训练器、预测器 |
| `libllie.evaluation` | 评估器和评价指标 |

顶层 API 只是对这些模块进行封装，方便用户完成常见任务。

## 基本使用流程

### 读取和保存图像

```python
import libllie as llie

image = llie.imread("input.jpg", output_format="pil")
saved_path = llie.imwrite(image, output="results/copy.png")
```

### 使用传统算法增强

```python
enhanced, saved_path = llie.predict(
    "gcp",
    "input.jpg",
    output="results/gcp_output.png",
)
```

### 使用深度学习模型增强

```python
enhanced, saved_path = llie.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/zerodce_output.png",
    device="cuda",
)
```

如果只传入 `"ZeroDCE"`，LibLLIE 会创建 ZeroDCE 网络结构，但不会自动加载训练好的权重。实际可用预测通常需要传入 checkpoint 路径。

### 启动训练

```python
result = llie.train(
    model="ZeroDCE",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="zerodce",
    epochs=10,
    batch_size=4,
    device="cuda",
)
```

### 评估增强结果

```python
results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM"],
    save_path="results/eval.json",
)
```

## 推荐阅读顺序

1. `docs/guide/image_io.md`：图像读取和保存。
2. `docs/guide/predict.md`：传统算法和深度学习模型预测。
3. `docs/guide/train.md`：训练流程。
4. `docs/guide/evaluate.md`：评估流程。
