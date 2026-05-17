# 评估 API

LibLLIE 使用 `llie.evaluate()` 作为顶层评估入口，用于计算增强图像的质量指标。

```python
import libllie as llie
```

`llie.eval()` 是 `llie.evaluate()` 的别名。

## 函数形式

```python
llie.evaluate(
    en_img_dir,
    ref_img_dir=None,
    metrics=None,
    save_path=None,
    return_evaluator=False,
    **kwargs,
)
```

| 参数 | 含义 |
| --- | --- |
| `en_img_dir` | 增强图像所在目录 |
| `ref_img_dir` | 参考图像所在目录，可选 |
| `metrics` | 指标名称或指标名称列表 |
| `save_path` | 评估结果保存路径 |
| `return_evaluator` | 是否返回 `Evaluator` 实例 |
| `**kwargs` | 传给 `Evaluator` 的额外参数 |

## 查看可用指标

```python
print(llie.list_metrics())
```

也可以查看所有可用组件：

```python
llie.list_available()
```

## 有参考图像评估

有参考图像时，可以计算 PSNR、SSIM 等全参考指标。

```python
results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM"],
    save_path="results/eval_full_reference.json",
)

print(results)
```

要求增强图像和参考图像能够按文件名或评估器规则对应起来。

## 无参考图像评估

无参考图像时，可以只传入增强图像目录，并选择无参考指标：

```python
results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    metrics=["NIQE"],
    save_path="results/eval_no_reference.json",
)
```

具体可用无参考指标取决于当前环境中安装的依赖和 LibLLIE 已注册的指标。

## 同时计算多个指标

```python
results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM", "LOE"],
)
```

如果 `metrics=None`，评估器会使用默认指标设置，具体行为以当前 `Evaluator` 实现为准。

## 保存结果

传入 `save_path` 后，评估结果会保存到指定路径：

```python
results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM"],
    save_path="results/evaluation.json",
)
```

## 返回 Evaluator 实例

如果需要进一步访问评估器对象，可以设置 `return_evaluator=True`：

```python
evaluator = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM"],
    return_evaluator=True,
)

print(evaluator.results)
```

## 与预测流程结合

完整流程示例：

```python
import libllie as llie

saved_paths = llie.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "datasets/LOL/eval15/low",
    output="results/ZeroDCE",
    device="cuda",
)

results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM"],
    save_path="results/ZeroDCE_eval.json",
)
```

## 常见问题

### 指标不可用

先运行：

```python
llie.list_metrics()
```

确认指标是否已经注册。部分指标可能依赖额外第三方库。

### 结果为空或图像无法匹配

检查增强图像目录和参考图像目录中的文件名是否能够对应，尤其是批量预测后是否保持了原始文件名。

### 没有参考图像

不要传 `ref_img_dir`，并选择无参考指标，例如 `NIQE` 或其他当前已注册的无参考指标。
