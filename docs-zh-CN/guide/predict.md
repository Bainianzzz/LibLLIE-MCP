# 预测增强 API

LibLLIE 使用 `llie.predict()` 作为统一预测入口，可调用传统低光增强算法，也可调用深度学习模型。

```python
import libllie as llie
```

`llie.enhance()` 是 `llie.predict()` 的别名。

## 函数形式

```python
llie.predict(target, source, output=None, **kwargs)
```

| 参数 | 含义 |
| --- | --- |
| `target` | 深度学习模型名称、checkpoint 路径、传统算法名称、模型实例或算法实例 |
| `source` | 单张图像输入或图像文件夹 |
| `output` | 输出文件路径或输出文件夹 |
| `**kwargs` | Predictor 构造参数或预测调用参数 |

## 查看可用模型和算法

```python
print(llie.list_models())
print(llie.list_algorithms())
print(llie.list_available())
```

## 传统算法预测

单张图像：

```python
enhanced, saved_path = llie.predict(
    "he",
    "input.jpg",
    output="results/he_output.jpg",
)
```

文件夹批量预测：

```python
saved_paths = llie.predict(
    "gcp",
    "images/",
    output="results/gcp",
    progress_bar=True,
)
```

批量预测时，LibLLIE 会自动读取文件夹中的图像文件，并保持原始文件名和后缀。

## 传入传统算法参数

不同传统算法支持不同参数，可以直接通过 `llie.predict()` 传入：

```python
enhanced, saved_path = llie.predict(
    "clahe",
    "input.jpg",
    output="results/clahe_output.png",
    color_space="lab",
    clip_limit=2.0,
    tile_grid_size=(8, 8),
)
```

GCP 示例：

```python
enhanced, saved_path = llie.predict(
    "gcp",
    "input.jpg",
    output="results/gcp_output.png",
    gamma_max=5.0,
    erosion_window=11,
)
```

## 深度学习模型预测

如果传入模型名称：

```python
enhanced, saved_path = llie.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/zerodce_output.png",
    device="cuda",
)
```

这种写法会创建 ZeroDCE 网络结构，但不会自动加载训练好的 checkpoint。除非模型类内部显式加载权重，否则它使用的是未训练或默认初始化的模型参数。

实际使用时，推荐传入训练好的 checkpoint：

```python
enhanced, saved_path = llie.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/zerodce_output.png",
    device="cuda",
)
```

## checkpoint 预测

当 `target` 是 `.pt` 或 `.pth` 文件时，LibLLIE 会把它识别为深度学习 checkpoint：

```python
enhanced, saved_path = llie.predict(
    "checkpoints/model/best.pt",
    "input.jpg",
    output="results/output.png",
    device="cuda",
)
```

checkpoint 内部需要包含模型类别和参数，例如训练流程保存的 `model_class` 和 `model_state_dict`。

## 文件夹批量预测

深度学习模型也支持文件夹输入：

```python
saved_paths = llie.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "images/",
    output="results/zerodce",
    device="cuda",
    progress_bar=True,
)
```

输出规则：

| 输入类型 | 输出行为 |
| --- | --- |
| 单张图像 + 文件路径输出 | 保存到指定文件 |
| 单张图像 + 文件夹输出 | 使用原始图像名保存到该文件夹 |
| 文件夹输入 + 文件夹输出 | 批量保存，并保持原始文件名和相对目录结构 |
| 文件夹输入 + 未指定输出 | 保存到 `results/{模型名或算法名}` |

## 不保存结果

单张图像预测时可以设置 `save=False`：

```python
enhanced, saved_path = llie.predict(
    "gcp",
    "input.jpg",
    save=False,
)

print(saved_path)  # None
```

## 显式使用统一 Predictor

如果需要重复调用同一个模型或算法，建议创建 `Predictor` 实例：

```python
from libllie import Predictor

predictor = Predictor("gcp", backend="traditional")
predictor("input1.jpg", output="results/1.png")
predictor("input2.jpg", output="results/2.png")
```

深度学习 checkpoint：

```python
from libllie import Predictor

predictor = Predictor(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    backend="deep",
    device="cuda",
)

predictor("input.jpg", output="results/output.png")
```

## 常见问题

### 找不到模型或算法

先运行：

```python
llie.list_available()
```

确认名称是否已经注册。自定义组件需要在使用前先导入。

### 只传模型名称但效果异常

只传 `"ZeroDCE"` 这类模型名称时，LibLLIE 只知道模型结构，不知道应该加载哪个训练权重。请传入训练好的 `.pt` 或 `.pth` checkpoint。

### 文件夹预测没有输出

确认输入目录存在，并且目录中包含支持的图像后缀，例如 `.jpg`、`.png`、`.bmp`、`.tif`。
