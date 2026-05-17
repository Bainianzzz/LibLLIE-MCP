# Image I/O API

LibLLIE provides `llie.imread()` and `llie.imwrite()` as top-level image I/O entry points.

```python
import libllie as llie
```

## Read Images

Basic usage:

```python
image = llie.imread("input.jpg")
```

The default return type is a PIL image because the default value of `output_format` is `"pil"`.

```python
image = llie.imread("input.jpg", output_format="pil")
```

## Output Formats

`llie.imread()` supports specifying the return type through `output_format`:

| Parameter value | Return type |
| --- | --- |
| `"pil"` | `PIL.Image.Image` |
| `"numpy"` | `numpy.ndarray` |
| `"tensor"` | `torch.Tensor` |
| `"bytes"` | Image binary data |
| `"base64"` | Base64 string |
| `"file"` | File path |

Examples:

```python
pil_image = llie.imread("input.jpg", output_format="pil")
np_image = llie.imread("input.jpg", output_format="numpy")
tensor = llie.imread("input.jpg", output_format="tensor")
```

## Input Types

The input to `llie.imread()` can be common image sources:

| Input | Description |
| --- | --- |
| Local file path | For example `"input.jpg"` |
| `pathlib.Path` | Path object |
| `PIL.Image.Image` | PIL image |
| `numpy.ndarray` | NumPy image array |
| `torch.Tensor` | PyTorch Tensor |
| `bytes` / `bytearray` | Image binary data |
| base64 string | Base64-encoded image |
| URL | Network image address |

Example:

```python
from pathlib import Path
import libllie as llie

image = llie.imread(Path("input.jpg"), output_format="pil")
```

## Save Images

Basic usage:

```python
saved_path = llie.imwrite(image, output="results/output.png")
```

`output` can be either a file path or a folder path.

### Save to a File Path

```python
saved_path = llie.imwrite(
    image,
    output="results/enhanced.png",
)
```

In this case, the file name and suffix are determined by `output`.

### Save to a Folder

```python
saved_path = llie.imwrite(
    image,
    output="results",
    output_name="enhanced.png",
)
```

When `output` is a folder, it is recommended to explicitly pass `output_name`.

### Default Save Location

If `output` is not passed, the image is saved to the `results` directory by default.

```python
saved_path = llie.imwrite(image)
```

## Specify Save Format

You can specify the save format through `save_format`:

```python
saved_path = llie.imwrite(
    image,
    output="results/enhanced",
    save_format="png",
)
```

When `output` is a file path with a suffix, the save format is usually determined by the suffix; when a forced format is needed, use `save_format`.

## Aliases

The following APIs are equivalent:

```python
llie.imread(...)
llie.read_image(...)
```

The following APIs are equivalent:

```python
llie.imwrite(...)
llie.write_image(...)
```

## Use with the Prediction API

The prediction interface also uses LibLLIE's image I/O capabilities internally. Therefore, `llie.predict()` can directly accept multiple image input types:

```python
image = llie.imread("input.jpg", output_format="pil")

enhanced, saved_path = llie.predict(
    "gcp",
    image,
    output="results/gcp_output.png",
)
```

This approach is suitable when you want to read, preprocess, or convert image formats yourself before prediction.
