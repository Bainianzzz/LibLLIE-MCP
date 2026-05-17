# HE

HE is a traditional histogram equalization algorithm already implemented in LibLLIE.

## Links

| Type | URL |
| --- | --- |
| Paper / classic source | https://doi.org/10.1016/S0146-664X(77)80011-7 |
| Related early literature | https://doi.org/10.1109/T-C.1974.223892 |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

HE, or Histogram Equalization, is a classic global contrast enhancement method. It computes the gray-level distribution of an image and constructs a cumulative distribution function to remap original gray levels to a more uniform intensity distribution, thereby improving overall image contrast.

In low-light enhancement, HE can improve the visibility of dark regions, but it may also cause local over-enhancement, noise amplification, or color shifts. Therefore, for color images, it is usually necessary to choose an appropriate color space and equalize only the luminance channel.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Algorithm implementation | `libllie/traditional/algorithms/HE.py` |
| Algorithm class name | `HE` |
| Registered name | `he` |
| Base class | `LLIEnhancer` in `libllie/traditional/algorithms/BaseEnhancer.py` |

## Implementation Notes

LibLLIE's HE supports grayscale images and multiple color spaces:

- `rgb` / `bgr`
- `hsv`
- `hls`
- `yuv` / `ycbcr`
- `lab`

When a luminance-related color space is selected, the algorithm equalizes the luminance channel and then converts back to a BGR image.

Main parameter:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `color_space` | `str` | `"rgb"` | Color space where histogram equalization is performed |

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "he",
    "input.jpg",
    output="results/he/output.jpg",
    color_space="hsv",
)
```

Folder batch processing:

```python
saved_paths = llie.predict(
    "he",
    "images/",
    output="results/he",
    color_space="yuv",
)
```
