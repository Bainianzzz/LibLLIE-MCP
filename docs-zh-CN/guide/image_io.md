# 图像读写 API

LibLLIE 提供 `llie.imread()` 和 `llie.imwrite()` 作为顶层图像读写入口。

```python
import libllie as llie
```

## 读取图像

基础用法：

```python
image = llie.imread("input.jpg")
```

默认返回 PIL 图像，因为 `output_format` 默认值是 `"pil"`。

```python
image = llie.imread("input.jpg", output_format="pil")
```

## 输出格式

`llie.imread()` 支持通过 `output_format` 指定返回类型：

| 参数值 | 返回类型 |
| --- | --- |
| `"pil"` | `PIL.Image.Image` |
| `"numpy"` | `numpy.ndarray` |
| `"tensor"` | `torch.Tensor` |
| `"bytes"` | 图像二进制数据 |
| `"base64"` | base64 字符串 |
| `"file"` | 文件路径 |

示例：

```python
pil_image = llie.imread("input.jpg", output_format="pil")
np_image = llie.imread("input.jpg", output_format="numpy")
tensor = llie.imread("input.jpg", output_format="tensor")
```

## 输入类型

`llie.imread()` 的输入可以是常见图像来源：

| 输入 | 说明 |
| --- | --- |
| 本地文件路径 | 例如 `"input.jpg"` |
| `pathlib.Path` | 路径对象 |
| `PIL.Image.Image` | PIL 图像 |
| `numpy.ndarray` | NumPy 图像数组 |
| `torch.Tensor` | PyTorch Tensor |
| `bytes` / `bytearray` | 图像二进制数据 |
| base64 字符串 | base64 编码图像 |
| URL | 网络图像地址 |

示例：

```python
from pathlib import Path
import libllie as llie

image = llie.imread(Path("input.jpg"), output_format="pil")
```

## 保存图像

基础用法：

```python
saved_path = llie.imwrite(image, output="results/output.png")
```

`output` 可以是文件路径，也可以是文件夹路径。

### 保存到文件路径

```python
saved_path = llie.imwrite(
    image,
    output="results/enhanced.png",
)
```

这种情况下，文件名和后缀由 `output` 决定。

### 保存到文件夹

```python
saved_path = llie.imwrite(
    image,
    output="results",
    output_name="enhanced.png",
)
```

当 `output` 是文件夹时，建议显式传入 `output_name`。

### 默认保存位置

如果不传入 `output`，图像默认保存到 `results` 目录。

```python
saved_path = llie.imwrite(image)
```

## 指定保存格式

可以通过 `save_format` 指定保存格式：

```python
saved_path = llie.imwrite(
    image,
    output="results/enhanced",
    save_format="png",
)
```

当 `output` 是文件路径且带有后缀时，保存格式通常由后缀决定；当需要强制格式时，可以使用 `save_format`。

## 别名

以下接口等价：

```python
llie.imread(...)
llie.read_image(...)
```

以下接口等价：

```python
llie.imwrite(...)
llie.write_image(...)
```

## 与预测接口配合

预测接口内部也会使用 LibLLIE 的图像读写能力。因此，`llie.predict()` 可以直接接受多种图像输入：

```python
image = llie.imread("input.jpg", output_format="pil")

enhanced, saved_path = llie.predict(
    "gcp",
    image,
    output="results/gcp_output.png",
)
```

这种方式适合在预测前自己完成图像读取、预处理或格式转换。
