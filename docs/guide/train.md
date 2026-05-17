# Training API

LibLLIE uses `llie.train()` as the top-level training entry point.

```python
import libllie as llie
```

## Function Form

```python
llie.train(config=None, **kwargs)
```

| Parameter | Meaning |
| --- | --- |
| `config` | Configuration dictionary or YAML configuration file path |
| `**kwargs` | Training parameters that can override fields in the configuration file |

The return value is the result dictionary returned by the training pipeline, usually containing the output directory, checkpoint paths, training status, and other information.

## Train Directly with Parameters

```python
result = llie.train(
    model="ZeroDCE",
    dataset="CommonDataset",
    root_dir="datasets/LOL",
    loss="zerodce",
    optimizer="adam",
    lr=1e-4,
    epochs=10,
    batch_size=4,
    device="cuda",
)

print(result)
```

## Train with YAML Configuration

```python
result = llie.train("libllie/deepLearning/config/ZeroDCE.yaml")
```

This approach is suitable for reproducible experiments because all training parameters are centralized in the configuration file.

## Train with a Configuration Dictionary

```python
config = {
    "model": "ZeroDCE",
    "dataset": "CommonDataset",
    "root_dir": "datasets/LOL",
    "loss": "zerodce",
    "optimizer": "adam",
    "lr": 1e-4,
    "epochs": 10,
    "batch_size": 4,
    "device": "cuda",
}

result = llie.train(config)
```

## Override Configuration Parameters

If a configuration file and keyword arguments are passed at the same time, keyword arguments can be used to override the configuration:

```python
result = llie.train(
    "libllie/deepLearning/config/ZeroDCE.yaml",
    epochs=5,
    batch_size=2,
    device="cuda",
)
```

This is suitable for temporarily adjusting experiment parameters.

## View Available Training Components

```python
print(llie.list_models())
print(llie.list_losses())
print(llie.list_datasets())
```

Or view everything at once:

```python
llie.list_available()
```

## Common Training Parameters

| Parameter | Meaning |
| --- | --- |
| `model` | Model name, for example `"ZeroDCE"` |
| `dataset` | Dataset name, for example `"CommonDataset"` |
| `root_dir` | Dataset root directory |
| `loss` | Loss function name |
| `optimizer` | Optimizer name |
| `lr` | Learning rate |
| `epochs` | Number of training epochs |
| `batch_size` | Batch size |
| `device` | Training device, for example `"cuda"` or `"cpu"` |
| `output_dir` | Training output directory |
| `resume_path` | Checkpoint path for resuming training |

The specific available parameters depend on the current `Trainer` implementation and the configurations of the model, dataset, and loss function.

## Training Outputs

The training pipeline usually saves the following in the output directory:

| Content | Description |
| --- | --- |
| `checkpoints/last.pt` | Most recently saved checkpoint |
| `checkpoints/best.pt` | Checkpoint with the best validation performance |
| Training logs | Losses, metrics, or status information during training |

After training is complete, you can directly use the saved checkpoint for prediction:

```python
enhanced, saved_path = llie.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/zerodce_output.png",
    device="cuda",
)
```

## Resume Training

If training is interrupted, you can resume with `resume_path`:

```python
result = llie.train(
    "libllie/deepLearning/config/ZeroDCE.yaml",
    resume_path="checkpoints/ZeroDCE_CommonDataset/checkpoints/last.pt",
)
```

## Recommendations

1. Use smaller `epochs` and `batch_size` values during debugging.
2. For formal experiments, use YAML configuration files to save complete parameters.
3. Record the checkpoint path after each training run. Prediction and evaluation should be based on explicit checkpoints.
4. If CUDA is unavailable, set `device` to `"cpu"`.
