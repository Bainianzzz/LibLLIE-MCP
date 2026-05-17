# Log

Log is a logarithmic transform enhancement algorithm already implemented in LibLLIE.

## Links

| Type | URL |
| --- | --- |
| Paper link | None |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

Logarithmic transform is a classic point-operation enhancement method. It compresses highlight regions and boosts responses in dark regions through a logarithmic function, and is often used to enhance details in low gray-level ranges.

The transform form in LibLLIE is:

```text
output = log(1 + gain * input) / log(1 + gain)
```

The input is usually normalized to `[0, 1]`. A larger `gain` produces stronger enhancement in dark regions.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Algorithm implementation | `libllie/traditional/algorithms/Log.py` |
| Algorithm class name | `Log` |
| Registered name | `log` |
| Alias | `Log` |
| Base class | `LLIEnhancer` in `libllie/traditional/algorithms/BaseEnhancer.py` |

## Implementation Notes

LibLLIE's Log algorithm automatically normalizes according to the input dtype and restores the original dtype after enhancement.

Main parameter:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `gain` | `float` | `10.0` | Logarithmic enhancement gain, must be positive |

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "log",
    "input.jpg",
    output="results/log/output.jpg",
    gain=10.0,
)
```
