# Custom Traditional Enhancement Algorithms

This document explains how to implement custom traditional low-light enhancement algorithms based on LibLLIE's `LLIEnhancer` base class.

## 1. Basic Mechanism

The traditional algorithm base class is:

```python
from libllie.traditional.algorithms import LLIEnhancer
```

All algorithm classes that inherit `LLIEnhancer` are automatically registered when their modules are imported. After registration, they can be used through the unified prediction interface:

```python
import libllie as llie

llie.predict("my_algorithm", "input.jpg")
llie.enhance("my_algorithm", "images/", output="results/my_algorithm")
```

Note: after adding a new algorithm file, you need to import the algorithm class in `libllie/traditional/algorithms/__init__.py`, or manually import the module before use.

## 2. LLIEnhancer Input and Output Conventions

`LLIEnhancer` is responsible for:

- Reading inputs with `ImageReader`.
- Converting inputs to NumPy images.
- Validating image shape and dtype.
- Calling the subclass `_enhance()`.
- Postprocessing output according to `keep_dtype` and `clip_output`.
- Returning NumPy, PIL, bytes, base64, or file according to `output_type`.

Subclasses only need to implement:

```python
def _enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
    ...
```

Note: three-channel NumPy images are processed in OpenCV-style BGR order by default.

## 3. Minimal Algorithm Template

It is recommended to create `libllie/traditional/algorithms/MyAlgorithm.py`:

```python
from typing import Any, Dict

import numpy as np

from libllie.traditional.algorithms import LLIEnhancer


class MyAlgorithm(LLIEnhancer):
    """Example traditional enhancement algorithm."""

    name = "my_algorithm"
    aliases = ["myalg", "my"]

    def __init__(
        self,
        gain: float = 1.2,
        **kwargs: Any,
    ) -> None:
        """Initialize the enhancer.

        Args:
            gain: Brightness gain.
            **kwargs: Base enhancer parameters.
        """
        super().__init__(**kwargs)
        self.gain = gain

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Enhance one image.

        Args:
            image: Input image array in BGR order for_teach color images.
            **kwargs: Optional runtime parameters.

        Returns:
            Enhanced image array.
        """
        gain = kwargs.get("gain", self.gain)
        enhanced = image.astype(np.float32) * gain
        return enhanced

    def get_params(self) -> Dict[str, Any]:
        """Get enhancer parameters.

        Returns:
            Dictionary containing base parameters and algorithm parameters.
        """
        params = super().get_params()
        params.update({"gain": self.gain})
        return params
```

Then add the following to `libllie/traditional/algorithms/__init__.py`:

```python
from .MyAlgorithm import MyAlgorithm
```

## 4. Parameter Setting

Algorithm parameters can be passed during construction:

```python
from libllie.traditional.algorithms import LLIEnhancer

enhancer = LLIEnhancer.create_enhancer("my_algorithm", gain=1.5)
```

They can also be passed through the unified prediction interface:

```python
import libllie as llie

llie.predict(
    "my_algorithm",
    "input.jpg",
    output="results/my_algorithm",
    gain=1.5,
)
```

If a parameter is dynamically overridden during prediction, read it from `kwargs` in `_enhance()`:

```python
gain = kwargs.get("gain", self.gain)
```

## 5. Output Formats

`LLIEnhancer` supports the following output types:

```python
output_type="numpy"
output_type="pil"
output_type="bytes"
output_type="base64"
output_type="file"
```

The traditional algorithm predictor internally forces `output_type="numpy"` and then saves images uniformly. When using an enhancer directly, other formats can be specified:

```python
enhancer = MyAlgorithm(output_type="pil")
result = enhancer("input.jpg")
```

## 6. Use in Predictor

Single image:

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "my_algorithm",
    "input.jpg",
    output="results/my_algorithm/output.png",
)
```

Folder:

```python
saved_paths = llie.predict(
    "my_algorithm",
    "images/",
    output="results/my_algorithm",
)
```

## 7. Check Registration Status

```python
from libllie.traditional.algorithms import LLIEnhancer

print(LLIEnhancer.list_registered_enhancers())
```

Or:

```python
import libllie as llie

print(llie.list_algorithms())
```

## 8. Implementation Recommendations

1. `_enhance()` must return `np.ndarray`.
2. Do not read file paths inside `_enhance()`; input reading is already handled by `LLIEnhancer`.
3. Do not save images inside `_enhance()`; saving is handled by Predictor or ImageWriter.
4. Algorithms sensitive to color spaces should explicitly handle BGR/RGB conversion.
5. Parameter validation is recommended in `__init__()` or a separate `_validate_*()` method.
