# LibLLIE Top-Level API Overview

The goal of LibLLIE's top-level API is to complete common tasks through a unified entry point:

```python
import libllie as llie
```

You can then directly call functions such as `llie.predict()`, `llie.train()`, `llie.evaluate()`, `llie.imread()`, and `llie.imwrite()`.

## Main APIs

| API | Function |
| --- | --- |
| `llie.predict()` | Use a deep-learning model or traditional algorithm for low-light enhancement |
| `llie.enhance()` | Alias of `llie.predict()` |
| `llie.train()` | Start the training pipeline |
| `llie.evaluate()` | Evaluate metrics for enhanced results |
| `llie.eval()` | Alias of `llie.evaluate()` |
| `llie.imread()` | Unified image reading |
| `llie.imwrite()` | Unified image saving |
| `llie.read_image()` | Alias of `llie.imread()` |
| `llie.write_image()` | Alias of `llie.imwrite()` |
| `llie.list_available()` | View currently available models, algorithms, metrics, losses, and datasets |

## Quickly View Available Components

```python
import libllie as llie

available = llie.list_available()
print(available)
```

The returned result usually contains:

```python
{
    "models": [...],
    "algorithms": [...],
    "metrics": [...],
    "losses": [...],
    "datasets": [...],
}
```

You can also view each category separately:

```python
llie.list_models()
llie.list_algorithms()
llie.list_metrics()
llie.list_losses()
llie.list_datasets()
```

## API Design Idea

LibLLIE still keeps a clear internal module structure:

| Module | Responsibility |
| --- | --- |
| `libllie.data` | Image I/O, datasets, and data transforms |
| `libllie.traditional` | Traditional low-light enhancement algorithms |
| `libllie.deepLearning` | Deep-learning models, losses, trainer, and predictor |
| `libllie.evaluation` | Evaluator and evaluation metrics |

The top-level API only wraps these modules to make common tasks easier for users.

## Basic Workflow

### Read and Save Images

```python
import libllie as llie

image = llie.imread("input.jpg", output_format="pil")
saved_path = llie.imwrite(image, output="results/copy.png")
```

### Enhance with a Traditional Algorithm

```python
enhanced, saved_path = llie.predict(
    "gcp",
    "input.jpg",
    output="results/gcp_output.png",
)
```

### Enhance with a Deep-Learning Model

```python
enhanced, saved_path = llie.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/zerodce_output.png",
    device="cuda",
)
```

If you only pass `"ZeroDCE"`, LibLLIE creates the ZeroDCE network structure, but it does not automatically load trained weights. Practically useful prediction usually requires a checkpoint path.

### Start Training

```python
result = llie.train(
    model="ZeroDCE",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="zerodce",
    epochs=10,
    batch_size=4,
    device="cuda",
)
```

### Evaluate Enhanced Results

```python
results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM"],
    save_path="results/eval.json",
)
```

## Recommended Reading Order

1. `docs/guide/image_io.md`: image reading and saving.
2. `docs/guide/predict.md`: prediction with traditional algorithms and deep-learning models.
3. `docs/guide/train.md`: training pipeline.
4. `docs/guide/evaluate.md`: evaluation pipeline.
