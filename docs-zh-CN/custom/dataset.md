# 自定义数据集

本文档说明如何基于 LibLLIE 的 `BaseDataset` 基类实现自定义训练数据集。

## 1. 基本机制

数据集基类是：

```python
from libllie.data.datasets import BaseDataset
```

所有继承 `BaseDataset` 的数据集类会在模块被导入时自动注册。注册后可以通过训练接口使用：

```python
import libllie as llie

llie.train(
    model="ZeroDCE",
    dataset="MyDataset",
    root_dir="datasets/MyDataset",
)
```

注意：新增数据集文件后，需要在 `libllie/data/datasets/__init__.py` 中导入该数据集类，或者在使用前手动导入该模块。

## 2. BaseDataset 的数据约定

`BaseDataset` 默认处理 paired low-light enhancement 数据。每个样本通常返回：

```python
(low_tensor, high_tensor, filename)
```

其中：

| 字段 | 含义 |
| --- | --- |
| `low_tensor` | 低光照输入图像 |
| `high_tensor` | 正常光参考图像；无参考场景下可以为 None |
| `filename` | 原始低光图像文件名 |

`BaseDataset` 已经实现：

- 图像读取。
- low/high 图像配对。
- transform 应用。
- 文件名返回。
- 数据集注册。
- 基础统计信息。

子类通常只需要实现 `_resolve_pair_dirs()`，用于告诉基类 low/high 图像目录在哪里。

## 3. 最小数据集模板

假设你的数据集目录如下：

```text
datasets/MyDataset/
    train/
        input/
            001.png
            002.png
        target/
            001.png
            002.png
    val/
        input/
        target/
```

可以在 `libllie/data/datasets/MyDataset.py` 中创建：

```python
from pathlib import Path
from typing import Optional, Tuple

from libllie.data.datasets import BaseDataset


class MyDataset(BaseDataset):
    """Example paired low-light dataset."""

    name = "MyDataset"
    aliases = ["my_dataset", "mydata"]

    def _resolve_pair_dirs(
        self,
        low_dir: Optional[Path],
        high_dir: Optional[Path],
    ) -> Tuple[Path, Optional[Path]]:
        """Resolve low-light and reference image directories.

        Args:
            low_dir: Optional explicit low-light directory.
            high_dir: Optional explicit reference directory.

        Returns:
            Tuple of low-light directory and optional reference directory.
        """
        if low_dir is not None:
            return low_dir, high_dir

        split_dir = self.root_dir / self.split
        return split_dir / "input", split_dir / "target"
```

然后在 `libllie/data/datasets/__init__.py` 中加入：

```python
from .MyDataset import MyDataset
```

## 4. 支持多种 split 名称

如果数据集有不同目录命名，例如 `Train`、`Test`、`validation`，可以在子类中处理：

```python
class MyDataset(BaseDataset):
    name = "MyDataset"
    aliases = ["my_dataset"]

    split_aliases = {
        "train": ("train", "Train"),
        "val": ("val", "Val", "validation"),
        "_test": ("test", "Test"),
    }

    def _resolve_pair_dirs(self, low_dir, high_dir):
        if low_dir is not None:
            return low_dir, high_dir

        split_names = self.split_aliases.get(self.split.lower(), (self.split,))
        for split_name in split_names:
            candidate_low = self.root_dir / split_name / "low"
            candidate_high = self.root_dir / split_name / "high"
            if candidate_low.exists() and candidate_high.exists():
                return candidate_low, candidate_high

        first = split_names[0]
        return self.root_dir / first / "low", self.root_dir / first / "high"
```

## 5. 自定义配对规则

默认 `_build_pairs()` 使用文件名 stem 进行配对，例如 `001.png` 匹配 `001.jpg`。

如果你的数据集配对规则不同，可以重写 `_build_pairs()`：

```python
def _build_pairs(self):
    pairs = []
    low_files = self._list_images(self.low_dir)

    for low_path in low_files:
        high_name = low_path.stem.replace("_low", "_normal") + low_path.suffix
        high_path = self.high_dir / high_name
        if high_path.exists():
            pairs.append((low_path, high_path))

    return pairs
```

## 6. 在训练中使用

Python 调用：

```python
import libllie as llie

llie.train(
    model="ZeroDCE",
    dataset="MyDataset",
    root_dir="datasets/MyDataset",
    train_split="train",
    val_split="val",
    batch_size=4,
)
```

YAML 配置：

```yaml
data:
  dataset: MyDataset
  root_dir: datasets/MyDataset
  batch_size: 4
  num_workers: 4
  train_split: train
  val_split: val
  return_filename: true
```

如果目录不符合默认解析方式，也可以显式指定：

```python
llie.train(
    model="ZeroDCE",
    dataset="MyDataset",
    root_dir="datasets/MyDataset",
    train_low_dir="datasets/MyDataset/train/input",
    train_high_dir="datasets/MyDataset/train/target",
)
```

## 7. 检查注册状态

```python
from libllie.data.datasets import BaseDataset

print(BaseDataset.list_registered_datasets())
```

或者：

```python
import libllie as llie

print(llie.list_datasets())
```

## 8. 实现建议

1. paired 数据集优先继承 `BaseDataset`，只实现 `_resolve_pair_dirs()`。
2. 文件名配对规则特殊时，再重写 `_build_pairs()`。
3. 数据集类的 `name` 应保持清晰，`aliases` 用于兼容简写。
4. 如果是无参考训练数据，允许 `high_dir=None`，但损失函数也应是无参考损失。
5. transform 由 `BaseDataset` 统一管理，优先通过构造参数传入，不建议在 `_read_image()` 中硬编码增强。

