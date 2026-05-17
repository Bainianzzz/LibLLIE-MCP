# 训练 API

LibLLIE 使用 `llie.train()` 作为顶层训练入口。

```python
import libllie as llie
```

## 函数形式

```python
llie.train(config=None, **kwargs)
```

| 参数 | 含义 |
| --- | --- |
| `config` | 配置字典或 YAML 配置文件路径 |
| `**kwargs` | 训练参数，可覆盖配置文件中的字段 |

返回值是训练流程返回的结果字典，通常包含输出目录、checkpoint 路径、训练状态等信息。

## 使用参数直接训练

```python
result = llie.train(
    model="ZeroDCE",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="zerodce",
    optimizer="adam",
    lr=1e-4,
    epochs=10,
    batch_size=4,
    device="cuda",
)

print(result)
```

## 使用 YAML 配置训练

```python
result = llie.train("libllie/deepLearning/config/ZeroDCE.yaml")
```

这种方式适合可复现实验，因为训练参数都集中在配置文件中。

## 配置字典训练

```python
config = {
    "model": "ZeroDCE",
    "dataset": "CommonDataset",
    "root_dir": "datasets/LOL",
    "loss": "zerodce",
    "optimizer": "adam",
    "lr": 1e-4,
    "epochs": 10,
    "batch_size": 4,
    "device": "cuda",
}

result = llie.train(config)
```

## 覆盖配置参数

如果同时传入配置文件和关键字参数，关键字参数可以用于覆盖配置：

```python
result = llie.train(
    "libllie/deepLearning/config/ZeroDCE.yaml",
    epochs=5,
    batch_size=2,
    device="cuda",
)
```

这适合临时调整实验参数。

## 查看可用训练组件

```python
print(llie.list_models())
print(llie.list_losses())
print(llie.list_datasets())
```

或者一次性查看：

```python
llie.list_available()
```

## 常见训练参数

| 参数 | 含义 |
| --- | --- |
| `model` | 模型名称，例如 `"ZeroDCE"` |
| `dataset` | 数据集名称，例如 `"CommonDataset"` |
| `root_dir` | 数据集根目录 |
| `loss` | 损失函数名称 |
| `optimizer` | 优化器名称 |
| `lr` | 学习率 |
| `epochs` | 训练轮数 |
| `batch_size` | 批大小 |
| `device` | 训练设备，例如 `"cuda"` 或 `"cpu"` |
| `output_dir` | 训练输出目录 |
| `resume_path` | 断点恢复 checkpoint 路径 |

具体可用参数以 `Trainer` 当前实现和模型、数据集、损失函数的配置为准。

## 训练输出

训练流程通常会在输出目录中保存：

| 内容 | 说明 |
| --- | --- |
| `checkpoints/last.pt` | 最近一次保存的 checkpoint |
| `checkpoints/best.pt` | 验证集表现最好的 checkpoint |
| 训练日志 | 训练过程中的损失、指标或状态信息 |

完成训练后，可以直接使用保存的 checkpoint 做预测：

```python
enhanced, saved_path = llie.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/zerodce_output.png",
    device="cuda",
)
```

## 断点恢复训练

如果训练中断，可以通过 `resume_path` 恢复：

```python
result = llie.train(
    "libllie/deepLearning/config/ZeroDCE.yaml",
    resume_path="checkpoints/ZeroDCE_CommonDataset/checkpoints/last.pt",
)
```

## 建议

1. 调试时先使用较小的 `epochs` 和 `batch_size`。
2. 正式实验建议使用 YAML 配置文件保存完整参数。
3. 每次训练后记录 checkpoint 路径，预测和评估都应基于明确的 checkpoint。
4. 如果 CUDA 不可用，将 `device` 设置为 `"cpu"`。
