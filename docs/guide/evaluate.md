# Evaluation API

LibLLIE uses `llie.evaluate()` as the top-level evaluation entry point for computing quality metrics of enhanced images.

```python
import libllie as llie
```

`llie.eval()` is an alias of `llie.evaluate()`.

## Function Form

```python
llie.evaluate(
    en_img_dir,
    ref_img_dir=None,
    metrics=None,
    save_path=None,
    return_evaluator=False,
    **kwargs,
)
```

| Parameter | Meaning |
| --- | --- |
| `en_img_dir` | Directory containing enhanced images |
| `ref_img_dir` | Directory containing reference images, optional |
| `metrics` | Metric name or list of metric names |
| `save_path` | Path for saving evaluation results |
| `return_evaluator` | Whether to return the `Evaluator` instance |
| `**kwargs` | Extra parameters passed to `Evaluator` |

## View Available Metrics

```python
print(llie.list_metrics())
```

You can also view all available components:

```python
llie.list_available()
```

## Evaluation with Reference Images

When reference images are available, full-reference metrics such as PSNR and SSIM can be computed.

```python
results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM"],
    save_path="results/eval_full_reference.json",
)

print(results)
```

Enhanced images and reference images must be matchable by file name or by the evaluator's matching rules.

## Evaluation without Reference Images

When reference images are unavailable, pass only the enhanced-image directory and choose no-reference metrics:

```python
results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    metrics=["NIQE"],
    save_path="results/eval_no_reference.json",
)
```

The specific available no-reference metrics depend on the dependencies installed in the current environment and the metrics registered in LibLLIE.

## Compute Multiple Metrics

```python
results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM", "LOE"],
)
```

If `metrics=None`, the evaluator uses its default metric settings. The exact behavior depends on the current `Evaluator` implementation.

## Save Results

After passing `save_path`, evaluation results are saved to the specified path:

```python
results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM"],
    save_path="results/evaluation.json",
)
```

## Return the Evaluator Instance

If you need further access to the evaluator object, set `return_evaluator=True`:

```python
evaluator = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM"],
    return_evaluator=True,
)

print(evaluator.results)
```

## Use with the Prediction Pipeline

Complete workflow example:

```python
import libllie as llie

saved_paths = llie.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "datasets/LOL/eval15/low",
    output="results/ZeroDCE",
    device="cuda",
)

results = llie.evaluate(
    en_img_dir="results/ZeroDCE",
    ref_img_dir="datasets/LOL/eval15/high",
    metrics=["PSNR", "SSIM"],
    save_path="results/ZeroDCE_eval.json",
)
```

## FAQ

### Metric Unavailable

Run first:

```python
llie.list_metrics()
```

Confirm whether the metric has been registered. Some metrics may depend on extra third-party libraries.

### Empty Results or Image Matching Failure

Check whether file names in the enhanced-image directory and reference-image directory can be matched, especially whether original file names were preserved after batch prediction.

### No Reference Images

Do not pass `ref_img_dir`; choose no-reference metrics such as `NIQE` or other currently registered no-reference metrics.
