# Custom Datasets

This document explains how to implement custom training datasets based on LibLLIE's `BaseDataset` base class.

## 1. Basic Mechanism

The dataset base class is:

```python
from libllie.data.datasets import BaseDataset
```

All dataset classes that inherit `BaseDataset` are automatically registered when their modules are imported. After registration, they can be used through the training interface:

```python
import libllie as llie

llie.train(
    model="ZeroDCE",
    dataset="MyDataset",
    root_dir="datasets/MyDataset",
)
```

Note: after adding a new dataset file, you need to import the dataset class in `libllie/data/datasets/__init__.py`, or manually import the module before use.

## 2. BaseDataset Data Convention

`BaseDataset` handles paired low-light enhancement data by default. Each sample usually returns:

```python
(low_tensor, high_tensor, filename)
```

where:

| Field | Meaning |
| --- | --- |
| `low_tensor` | Low-light input image |
| `high_tensor` | Normal-light reference image; can be None in no-reference scenarios |
| `filename` | Original low-light image file name |

`BaseDataset` already implements:

- Image reading.
- Low/high image pairing.
- Transform application.
- File name return.
- Dataset registration.
- Basic statistics.

Subclasses usually only need to implement `_resolve_pair_dirs()` to tell the base class where the low/high image directories are.

## 3. Minimal Dataset Template

Assume your dataset directory is:

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

You can create `libllie/data/datasets/MyDataset.py`:

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

Then add the following to `libllie/data/datasets/__init__.py`:

```python
from .MyDataset import MyDataset
```

## 4. Support Multiple Split Names

If the dataset uses different directory names, such as `Train`, `Test`, or `validation`, handle them in the subclass:

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

## 5. Custom Pairing Rules

The default `_build_pairs()` pairs files by file-name stem, for example `001.png` matches `001.jpg`.

If your dataset uses different pairing rules, override `_build_pairs()`:

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

## 6. Use in Training

Python call:

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

YAML configuration:

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

If the directory does not match the default parsing method, you can also specify it explicitly:

```python
llie.train(
    model="ZeroDCE",
    dataset="MyDataset",
    root_dir="datasets/MyDataset",
    train_low_dir="datasets/MyDataset/train/input",
    train_high_dir="datasets/MyDataset/train/target",
)
```

## 7. Check Registration Status

```python
from libllie.data.datasets import BaseDataset

print(BaseDataset.list_registered_datasets())
```

Or:

```python
import libllie as llie

print(llie.list_datasets())
```

## 8. Implementation Recommendations

1. For paired datasets, prefer inheriting `BaseDataset` and only implementing `_resolve_pair_dirs()`.
2. Override `_build_pairs()` only when file-name pairing rules are special.
3. The dataset class `name` should be clear, and `aliases` should be used for compatible abbreviations.
4. If it is no-reference training data, allow `high_dir=None`, but the loss function should also be no-reference.
5. Transforms are uniformly managed by `BaseDataset`; prefer passing them through constructor parameters and avoid hardcoding augmentation in `_read_image()`.
