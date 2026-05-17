# LIME

LIME is a traditional low-light image enhancement algorithm based on illumination map estimation.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1109/TIP.2016.2639450 |
| Paper title | Low-light Image Enhancement via Illumination Map Estimation |
| Official source code | None |
| Official project page | None |

## Algorithm Introduction

LIME estimates an illumination map from the maximum color channel of the input image. The illumination map is then refined with edge-preserving smoothing and used to recover a brighter reflectance-like image.

In LibLLIE, the illumination map is refined with a guided filter. The enhanced image is computed by dividing the input image by the gamma-adjusted illumination map.

## Location in LibLLIE

| Item | Location |
| --- | --- |
| Algorithm implementation | `libllie/traditional/algorithms/LIME.py` |
| Algorithm class name | `LIME` |
| Registered name | `LIME` |
| Aliases | `lime`, `illumination_map_estimation` |
| Base class | `LLIEnhancer` in `libllie/traditional/algorithms/BaseEnhancer.py` |

## Main Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `gamma` | `float` | `0.8` | Gamma applied to the refined illumination map |
| `guided_radius` | `int` | `15` | Guided filter radius |
| `guided_eps` | `float` | `1e-3` | Guided filter regularization term |
| `illumination_floor` | `float` | `0.05` | Lower bound for illumination |
| `exposure` | `float` | `1.0` | Global exposure multiplier |

## Usage Example

```python
import libllie as llie

enhanced, saved_path = llie.predict(
    "lime",
    "input.jpg",
    output="results/lime/output.jpg",
    gamma=0.8,
)
```

Folder batch processing:

```python
saved_paths = llie.predict(
    "lime",
    "images/",
    output="results/lime",
    progress_bar=True,
)
```
