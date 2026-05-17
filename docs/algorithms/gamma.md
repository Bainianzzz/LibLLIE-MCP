# Gamma

Gamma is a gamma correction enhancement algorithm already implemented in LibLLIE.

## Links

| Type | URL |
| --- | --- |
| Paper link | None |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

Gamma correction is a classic power-law gray-level transformation method. It adjusts image brightness in the following form:

```text
output = input ^ gamma
```

After the image is normalized to `[0, 1]`:

- `gamma < 1` usually brightens dark regions.
- `gamma > 1` usually darkens the image.

In the current LibLLIE implementation, the default is `gamma=0.6`. Users can pass a more suitable gamma value according to low-light enhancement needs.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Algorithm implementation | `libllie/traditional/algorithms/Gamma.py` |
| Algorithm class name | `Gamma` |
| Registered name | `Gamma` |
| Base class | `LLIEnhancer` in `libllie/traditional/algorithms/BaseEnhancer.py` |

## Implementation Notes

LibLLIE's Gamma first converts the image to a floating-point range, then applies the power-law transform, and finally restores the original dtype.

Main parameter:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `gamma` | `float` | `0.6` | Power-law exponent, must be positive |

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "Gamma",
    "input.jpg",
    output="results/gamma/output.jpg",
    gamma=0.6,
)
```

If you want to explicitly use the traditional backend through the unified Predictor:

```python
from libllie import Predictor

predictor = Predictor("Gamma", backend="traditional")
predictor("input.jpg", output="results/gamma/output.jpg", gamma=0.6)
```
