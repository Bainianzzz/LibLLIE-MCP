# Prediction Enhancement API

LibLLIE uses `llie.predict()` as the unified prediction entry point. It can call traditional low-light enhancement algorithms and deep-learning models.

```python
import libllie as llie
```

`llie.enhance()` is an alias of `llie.predict()`.

## Function Form

```python
llie.predict(target, source, output=None, **kwargs)
```

| Parameter | Meaning |
| --- | --- |
| `target` | Deep-learning model name, checkpoint path, traditional algorithm name, model instance, or algorithm instance |
| `source` | Single-image input or image folder |
| `output` | Output file path or output folder |
| `**kwargs` | Predictor construction parameters or prediction-call parameters |

## View Available Models and Algorithms

```python
print(llie.list_models())
print(llie.list_algorithms())
print(llie.list_available())
```

## Traditional Algorithm Prediction

Single image:

```python
enhanced, saved_path = llie.predict(
    "he",
    "input.jpg",
    output="results/he_output.jpg",
)
```

Folder batch prediction:

```python
saved_paths = llie.predict(
    "gcp",
    "images/",
    output="results/gcp",
    progress_bar=True,
)
```

During batch prediction, LibLLIE automatically reads image files in the folder and preserves the original file names and suffixes.

## Pass Traditional Algorithm Parameters

Different traditional algorithms support different parameters, which can be passed directly through `llie.predict()`:

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

GCP example:

```python
enhanced, saved_path = llie.predict(
    "gcp",
    "input.jpg",
    output="results/gcp_output.png",
    gamma_max=5.0,
    erosion_window=11,
)
```

## Deep-Learning Model Prediction

If a model name is passed:

```python
enhanced, saved_path = llie.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/zerodce_output.png",
    device="cuda",
)
```

This creates the ZeroDCE network structure, but it does not automatically load a trained checkpoint. Unless the model class explicitly loads weights internally, it uses untrained or default-initialized model parameters.

In practical use, it is recommended to pass a trained checkpoint:

```python
enhanced, saved_path = llie.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/zerodce_output.png",
    device="cuda",
)
```

## Checkpoint Prediction

When `target` is a `.pt` or `.pth` file, LibLLIE recognizes it as a deep-learning checkpoint:

```python
enhanced, saved_path = llie.predict(
    "checkpoints/model/best.pt",
    "input.jpg",
    output="results/output.png",
    device="cuda",
)
```

The checkpoint must contain model class and parameter information, such as `model_class` and `model_state_dict` saved by the training pipeline.

## Folder Batch Prediction

Deep-learning models also support folder input:

```python
saved_paths = llie.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "images/",
    output="results/zerodce",
    device="cuda",
    progress_bar=True,
)
```

Output rules:

| Input type | Output behavior |
| --- | --- |
| Single image + file-path output | Save to the specified file |
| Single image + folder output | Save to the folder using the original image name |
| Folder input + folder output | Batch save while preserving original file names and relative directory structure |
| Folder input + no specified output | Save to `results/{model name or algorithm name}` |

## Do Not Save Results

For single-image prediction, you can set `save=False`:

```python
enhanced, saved_path = llie.predict(
    "gcp",
    "input.jpg",
    save=False,
)

print(saved_path)  # None
```

## Explicitly Use the Unified Predictor

If you need to repeatedly call the same model or algorithm, it is recommended to create a `Predictor` instance:

```python
from libllie import Predictor

predictor = Predictor("gcp", backend="traditional")
predictor("input1.jpg", output="results/1.png")
predictor("input2.jpg", output="results/2.png")
```

Deep-learning checkpoint:

```python
from libllie import Predictor

predictor = Predictor(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    backend="deep",
    device="cuda",
)

predictor("input.jpg", output="results/output.png")
```

## FAQ

### Model or Algorithm Not Found

Run first:

```python
llie.list_available()
```

Confirm whether the name has been registered. Custom components must be imported before use.

### Abnormal Results When Only a Model Name Is Passed

When only a model name such as `"ZeroDCE"` is passed, LibLLIE only knows the model structure and does not know which trained weights should be loaded. Please pass a trained `.pt` or `.pth` checkpoint.

### No Output from Folder Prediction

Confirm that the input directory exists and contains supported image suffixes such as `.jpg`, `.png`, `.bmp`, and `.tif`.
