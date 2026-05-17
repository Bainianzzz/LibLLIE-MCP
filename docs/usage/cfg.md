# Training Configuration

LibLLIE uses YAML files and a default Python configuration to drive the deep-learning training pipeline. The configuration system is centered around two locations:

| Location | Purpose |
| --- | --- |
| `libllie/deepLearning/config/` | Built-in YAML templates for supported deep-learning models |
| `libllie/deepLearning/config.py` | Default training configuration and configuration merge utilities |

These files are mainly consumed by `llie.train()` and `libllie.deepLearning.trainer.Trainer`.

## Built-in YAML Files

The `libllie/deepLearning/config/` directory contains ready-to-edit training templates:

| File | Model | Dataset Template | Loss |
| --- | --- | --- | --- |
| `DarkIR.yaml` | `DarkIR` | `CommonDataset` | `darkir` |
| `EnlightenGAN.yaml` | `EnlightenGAN` | `CommonDataset` | `enlightengan` |
| `KinD++.yaml` | `KinDPlusPlus` | `CommonDataset` | `kind++` |
| `KinD.yaml` | `KinD` | `CommonDataset` | `kind` |
| `LEDNet.yaml` | `LEDNet` | `CommonDataset` | `lednet` |
| `LLNet.yaml` | `LLNet` | `CommonDataset` | `llnet` |
| `LLFlow.yaml` | `LLFlow` | `CommonDataset` | `llflow` |
| `RetinexFormer.yaml` | `RetinexFormer` | `CommonDataset` | `retinexformer` |
| `RUAS.yaml` | `RUAS` | `LOLv1Dataset` | `ruas_loss` |
| `SCI.yaml` | `SCI` | `LOLv1Dataset` | `sci_loss` |
| `URetinexNet.yaml` | `URetinexNet` | `CommonDataset` | `uretinex` |
| `ZeroDCE++.yaml` | `ZeroDCEPlusPlus` | `LOLv1Dataset` | `zerodce++` |
| `ZeroDCE.yaml` | `ZeroDCEPlusPlus` | `LOLv1Dataset` | `zerodce_loss` |
| `ZeroIG.yaml` | `ZeroIG` | `CommonDataset` | `zeroig` |

Before using a template, update at least `data.root_dir` so it points to your dataset root directory.

```yaml
data:
  root_dir: path/to/your/dataset/dir
```

Then train with:

```bash
libllie train libllie/deepLearning/config/ZeroDCE.yaml
```

or in Python:

```python
import libllie as llie

result = llie.train("libllie/deepLearning/config/ZeroDCE.yaml")
```

## YAML Structure

Each training YAML file follows the same top-level structure:

```yaml
model:
  name: ZeroDCEPlusPlus
  params: {}

data:
  dataset: LOLv1Dataset
  root_dir: path/to/your/dataset/dir
  train_split: train
  val_split: _test
  batch_size: 4
  num_workers: 4
  pin_memory: true
  return_filename: true

loss:
  name: zerodce_loss
  params: {}
  output_index: null
  output_key: null

optimizer:
  name: adam
  lr: 0.0001
  params:
    betas:
      - 0.9
      - 0.999
    weight_decay: 0.0

scheduler:
  name: cosineannealinglr
  params:
    T_max: 100
    eta_min: 1.0e-06

train:
  epochs: 100
  device: cuda
  output_dir: null
  save_every: 1
  validate_every: 1
  log_every: 10
  grad_clip: 1.0
  amp: false
  seed: 42
  resume: null
```

## `model` Section

The `model` section controls model construction.

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | `None` | Registered model name or checkpoint path |
| `params` | `{}` | Keyword arguments passed to the model constructor |

Example:

```yaml
model:
  name: RetinexFormer
  params:
    input_channels: 3
    output_channels: 3
    feature_channels: 32
    stage: 1
    levels: 2
```

`Trainer` resolves model names through the `LLIEModel` registry. Model names are matched case-insensitively.

## `data` Section

The `data` section controls dataset and dataloader construction.

| Key | Default | Meaning |
| --- | --- | --- |
| `dataset` | `LOLv1Dataset` | Dataset class name or registered dataset name |
| `root_dir` | `None` | Dataset root directory; required for training |
| `batch_size` | `4` | Dataloader batch size |
| `num_workers` | `4` | Number of dataloader workers |
| `pin_memory` | `True` | Enable pinned memory when using CUDA |
| `shuffle` | `True` | Shuffle the training dataloader |
| `drop_last` | `False` | Drop the last incomplete training batch |
| `train_split` | `"train"` | Training split name |
| `val_split` | `"_test"` | Validation split name |
| `return_filename` | `True` | Whether datasets return filenames |
| `params` | `{}` | Shared dataset parameters |
| `train_params` | `{}` | Training-split-specific dataset parameters |
| `val_params` | `{}` | Validation-split-specific dataset parameters |

The trainer also supports explicit directory overrides:

| Key | Meaning |
| --- | --- |
| `train_low_dir` | Explicit low-light image directory for training |
| `train_high_dir` | Explicit normal-light image directory for training |
| `val_low_dir` | Explicit low-light image directory for validation |
| `val_high_dir` | Explicit normal-light image directory for validation |

Example:

```yaml
data:
  dataset: CommonDataset
  root_dir: datasets/LOL
  train_split: train
  val_split: val
  batch_size: 4
  num_workers: 4
  return_filename: true
```

## `loss` Section

The `loss` section controls loss construction and prediction extraction.

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | `None` | Registered loss name; if omitted, the trainer tries to infer one from the model |
| `params` | `{}` | Keyword arguments passed to the loss constructor |
| `output_index` | `None` | Optional index used when the model returns a tuple/list |
| `output_key` | `None` | Optional key used when the model returns a dictionary |

Example:

```yaml
loss:
  name: retinexformer
  params:
    pixel_weight: 1.0
    illumination_weight: 0.0
  output_index: null
  output_key: null
```

`output_index` and `output_key` are useful for models that return multiple tensors. When they are not set, the trainer attempts to extract a compatible prediction tensor automatically.

## `optimizer` Section

The `optimizer` section controls optimizer construction.

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | `"adam"` | Optimizer name |
| `lr` | `1e-4` | Learning rate |
| `params` | `{}` | Extra optimizer parameters |

Supported optimizer names:

| Name | PyTorch Optimizer |
| --- | --- |
| `adam` | `torch.optim.Adam` |
| `adamw` | `torch.optim.AdamW` |
| `sgd` | `torch.optim.SGD` |
| `rmsprop` | `torch.optim.RMSprop` |

Example:

```yaml
optimizer:
  name: adamw
  lr: 0.0001
  params:
    weight_decay: 0.01
```

## `scheduler` Section

The `scheduler` section controls the learning-rate scheduler.

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | `None` | Scheduler name; `None` disables scheduler construction |
| `params` | `{}` | Extra scheduler parameters |

Supported scheduler names:

| Name | PyTorch Scheduler |
| --- | --- |
| `steplr` | `torch.optim.lr_scheduler.StepLR` |
| `multisteplr` | `torch.optim.lr_scheduler.MultiStepLR` |
| `cosineannealinglr` | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| `reducelronplateau` | `torch.optim.lr_scheduler.ReduceLROnPlateau` |

Example:

```yaml
scheduler:
  name: cosineannealinglr
  params:
    T_max: 100
    eta_min: 1.0e-06
```

## `train` Section

The `train` section controls the training loop and output behavior.

| Key | Default | Meaning |
| --- | --- | --- |
| `epochs` | `100` | Number of training epochs |
| `output_dir` | `None` | Output directory; if `None`, the trainer creates `checkpoints/{Model}_{Dataset}` |
| `save_every` | `1` | Save `last.pt` every N epochs |
| `validate_every` | `1` | Run validation every N epochs |
| `log_every` | `10` | Progress-log frequency in training steps |
| `grad_clip` | `None` | Gradient clipping max norm; `None` disables clipping |
| `amp` | `False` | Enable automatic mixed precision on CUDA |
| `resume` | `None` | Checkpoint path used to resume training |
| `seed` | `42` | Random seed |
| `device` | `"cuda"` if available, otherwise `"cpu"` | Training device |

Example:

```yaml
train:
  epochs: 50
  device: cuda
  output_dir: checkpoints/my_experiment
  save_every: 1
  validate_every: 1
  amp: false
  resume: null
```

## `config.py`

`libllie/deepLearning/config.py` defines the default configuration used by the trainer.

### `DEFAULT_TRAIN_CONFIG`

`DEFAULT_TRAIN_CONFIG` is a nested dictionary containing default values for all standard sections:

```python
DEFAULT_TRAIN_CONFIG = {
    "model": {...},
    "data": {...},
    "loss": {...},
    "optimizer": {...},
    "scheduler": {...},
    "train": {...},
}
```

When a YAML file or a Python dictionary is passed to `Trainer`, it is merged into this default configuration. This means a config file only needs to define the fields that differ from the defaults, although in practice the built-in YAML files keep most fields explicit for readability.

### `get_default_train_config()`

```python
get_default_train_config()
```

Returns a deep copy of `DEFAULT_TRAIN_CONFIG`.

Use this when you want to build or inspect a configuration dictionary in Python without mutating the global default values.

```python
from libllie.deepLearning.config import get_default_train_config

config = get_default_train_config()
config["model"]["name"] = "ZeroDCE"
config["data"]["root_dir"] = "datasets/LOL"
```

### `deep_update()`

```python
deep_update(base, updates)
```

Recursively merges `updates` into `base`.

If both `base[key]` and `updates[key]` are dictionaries, their nested fields are merged. Otherwise, the value from `updates` replaces the value in `base`.

Example:

```python
from libllie.deepLearning.config import deep_update, get_default_train_config

config = get_default_train_config()
deep_update(
    config,
    {
        "train": {
            "epochs": 5,
            "device": "cpu",
        },
        "optimizer": {
            "lr": 1e-4,
        },
    },
)
```

## Override Configuration Values

`llie.train()` and `Trainer` support direct keyword overrides. These overrides are converted into the nested config structure and then merged with the YAML configuration.

```python
import libllie as llie

result = llie.train(
    "libllie/deepLearning/config/ZeroDCE.yaml",
    root_dir="datasets/LOL",
    epochs=5,
    batch_size=2,
    device="cpu",
)
```

Equivalent nested meaning:

```yaml
data:
  root_dir: datasets/LOL
  batch_size: 2
train:
  epochs: 5
  device: cpu
```

Common flat override keys:

| Flat key | Config location |
| --- | --- |
| `model`, `model_name` | `model.name` |
| `model_params` | `model.params` |
| `dataset`, `dataset_name` | `data.dataset` |
| `root_dir` | `data.root_dir` |
| `batch_size` | `data.batch_size` |
| `num_workers` | `data.num_workers` |
| `train_split` | `data.train_split` |
| `val_split` | `data.val_split` |
| `loss`, `loss_name` | `loss.name` |
| `loss_params` | `loss.params` |
| `optimizer`, `optimizer_name` | `optimizer.name` |
| `lr` | `optimizer.lr` |
| `optimizer_params` | `optimizer.params` |
| `scheduler`, `scheduler_name` | `scheduler.name` |
| `scheduler_params` | `scheduler.params` |
| `epochs` | `train.epochs` |
| `output_dir` | `train.output_dir` |
| `save_every` | `train.save_every` |
| `validate_every` | `train.validate_every` |
| `log_every` | `train.log_every` |
| `grad_clip` | `train.grad_clip` |
| `amp` | `train.amp` |
| `resume` | `train.resume` |
| `seed` | `train.seed` |
| `device` | `train.device` |

## Minimal Custom Config

Because defaults are provided by `config.py`, a minimal config can be small:

```yaml
model:
  name: ZeroDCE

data:
  dataset: CommonDataset
  root_dir: datasets/LOL

loss:
  name: zerodce_loss

train:
  epochs: 10
  device: cuda
```

The trainer fills unspecified fields from `DEFAULT_TRAIN_CONFIG`.

## Recommended Workflow

1. Copy one YAML file from `libllie/deepLearning/config/`.
2. Update `model.name`, `model.params`, `data.dataset`, and `data.root_dir`.
3. Choose the loss function in `loss.name` and set any `loss.params`.
4. Set `train.output_dir` for reproducible experiment organization.
5. Run training through `llie.train()` or `libllie train`.
6. Use `train.resume` when continuing from an interrupted checkpoint.
