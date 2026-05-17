# 训练配置

LibLLIE 使用 YAML 文件和 Python 默认配置来驱动深度学习训练流程。配置系统主要由两个位置组成：

| 位置 | 用途 |
| --- | --- |
| `libllie/deepLearning/config/` | 支持的深度学习模型内置 YAML 模板 |
| `libllie/deepLearning/config.py` | 默认训练配置和配置合并工具 |

这些文件主要由 `llie.train()` 和 `libllie.deepLearning.trainer.Trainer` 使用。

## 内置 YAML 文件

`libllie/deepLearning/config/` 目录包含可以直接编辑的训练模板：

| 文件 | 模型 | 数据集模板 | 损失函数 |
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

使用模板前，至少需要更新 `data.root_dir`，使其指向你的数据集根目录。

```yaml
data:
  root_dir: path/to/your/dataset/dir
```

然后可以通过命令行训练：

```bash
libllie train libllie/deepLearning/config/ZeroDCE.yaml
```

也可以在 Python 中调用：

```python
import libllie as llie

result = llie.train("libllie/deepLearning/config/ZeroDCE.yaml")
```

## YAML 结构

每个训练 YAML 文件都遵循相同的顶层结构：

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

## `model` 部分

`model` 部分控制模型构建。

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `name` | `None` | 注册模型名称或 checkpoint 路径 |
| `params` | `{}` | 传递给模型构造函数的关键字参数 |

示例：

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

`Trainer` 通过 `LLIEModel` 注册表解析模型名称。模型名称大小写不敏感。

## `data` 部分

`data` 部分控制数据集和 dataloader 构建。

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `dataset` | `LOLv1Dataset` | 数据集类名或注册数据集名称 |
| `root_dir` | `None` | 数据集根目录；训练时必须提供 |
| `batch_size` | `4` | dataloader batch size |
| `num_workers` | `4` | dataloader worker 数量 |
| `pin_memory` | `True` | 使用 CUDA 时启用 pinned memory |
| `shuffle` | `True` | 是否打乱训练 dataloader |
| `drop_last` | `False` | 是否丢弃最后一个不完整训练 batch |
| `train_split` | `"train"` | 训练 split 名称 |
| `val_split` | `"_test"` | 验证 split 名称 |
| `return_filename` | `True` | 数据集是否返回文件名 |
| `params` | `{}` | 共享数据集参数 |
| `train_params` | `{}` | 训练 split 专用数据集参数 |
| `val_params` | `{}` | 验证 split 专用数据集参数 |

trainer 也支持显式目录覆盖：

| 键 | 含义 |
| --- | --- |
| `train_low_dir` | 训练低光图像目录 |
| `train_high_dir` | 训练正常光图像目录 |
| `val_low_dir` | 验证低光图像目录 |
| `val_high_dir` | 验证正常光图像目录 |

示例：

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

## `loss` 部分

`loss` 部分控制损失函数构建和预测张量提取。

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `name` | `None` | 注册损失函数名称；如果省略，trainer 会尝试根据模型自动推断 |
| `params` | `{}` | 传递给损失函数构造函数的关键字参数 |
| `output_index` | `None` | 当模型返回 tuple/list 时使用的可选索引 |
| `output_key` | `None` | 当模型返回 dictionary 时使用的可选键 |

示例：

```yaml
loss:
  name: retinexformer
  params:
    pixel_weight: 1.0
    illumination_weight: 0.0
  output_index: null
  output_key: null
```

`output_index` 和 `output_key` 适用于会返回多个张量的模型。如果它们没有设置，trainer 会尝试自动提取兼容的预测张量。

## `optimizer` 部分

`optimizer` 部分控制优化器构建。

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `name` | `"adam"` | 优化器名称 |
| `lr` | `1e-4` | 学习率 |
| `params` | `{}` | 额外优化器参数 |

支持的优化器名称：

| 名称 | PyTorch 优化器 |
| --- | --- |
| `adam` | `torch.optim.Adam` |
| `adamw` | `torch.optim.AdamW` |
| `sgd` | `torch.optim.SGD` |
| `rmsprop` | `torch.optim.RMSprop` |

示例：

```yaml
optimizer:
  name: adamw
  lr: 0.0001
  params:
    weight_decay: 0.01
```

## `scheduler` 部分

`scheduler` 部分控制学习率调度器。

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `name` | `None` | 调度器名称；`None` 表示不构建 scheduler |
| `params` | `{}` | 额外调度器参数 |

支持的 scheduler 名称：

| 名称 | PyTorch Scheduler |
| --- | --- |
| `steplr` | `torch.optim.lr_scheduler.StepLR` |
| `multisteplr` | `torch.optim.lr_scheduler.MultiStepLR` |
| `cosineannealinglr` | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| `reducelronplateau` | `torch.optim.lr_scheduler.ReduceLROnPlateau` |

示例：

```yaml
scheduler:
  name: cosineannealinglr
  params:
    T_max: 100
    eta_min: 1.0e-06
```

## `train` 部分

`train` 部分控制训练循环和输出行为。

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `epochs` | `100` | 训练 epoch 数 |
| `output_dir` | `None` | 输出目录；如果为 `None`，trainer 会创建 `checkpoints/{Model}_{Dataset}` |
| `save_every` | `1` | 每 N 个 epoch 保存一次 `last.pt` |
| `validate_every` | `1` | 每 N 个 epoch 运行一次验证 |
| `log_every` | `10` | 训练 step 中的进度日志频率 |
| `grad_clip` | `None` | 梯度裁剪最大范数；`None` 表示禁用 |
| `amp` | `False` | 在 CUDA 上启用自动混合精度 |
| `resume` | `None` | 用于恢复训练的 checkpoint 路径 |
| `seed` | `42` | 随机种子 |
| `device` | 如果 CUDA 可用则为 `"cuda"`，否则为 `"cpu"` | 训练设备 |

示例：

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

`libllie/deepLearning/config.py` 定义了 trainer 使用的默认配置。

### `DEFAULT_TRAIN_CONFIG`

`DEFAULT_TRAIN_CONFIG` 是一个嵌套字典，包含所有标准配置部分的默认值：

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

当 YAML 文件或 Python 字典传递给 `Trainer` 时，它会被合并到该默认配置中。这意味着配置文件只需要定义与默认值不同的字段；不过在实际使用中，内置 YAML 文件为了可读性通常会显式写出大多数字段。

### `get_default_train_config()`

```python
get_default_train_config()
```

返回 `DEFAULT_TRAIN_CONFIG` 的深拷贝。

当你想在 Python 中构建或查看配置字典，同时又不希望修改全局默认值时，可以使用它。

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

将 `updates` 递归合并到 `base`。

如果 `base[key]` 和 `updates[key]` 都是字典，则合并它们的嵌套字段。否则，`updates` 中的值会替换 `base` 中的值。

示例：

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

## 覆盖配置值

`llie.train()` 和 `Trainer` 支持直接传入关键字参数进行覆盖。这些覆盖项会被转换为嵌套配置结构，然后与 YAML 配置合并。

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

等价的嵌套含义为：

```yaml
data:
  root_dir: datasets/LOL
  batch_size: 2
train:
  epochs: 5
  device: cpu
```

常用扁平覆盖键：

| 扁平键 | 配置位置 |
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

## 最小自定义配置

由于 `config.py` 提供了默认值，因此最小配置可以很简洁：

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

trainer 会从 `DEFAULT_TRAIN_CONFIG` 中补全未指定的字段。

## 推荐工作流

1. 从 `libllie/deepLearning/config/` 复制一个 YAML 文件。
2. 更新 `model.name`、`model.params`、`data.dataset` 和 `data.root_dir`。
3. 在 `loss.name` 中选择损失函数，并设置需要的 `loss.params`。
4. 设置 `train.output_dir`，便于可复现实验管理。
5. 通过 `llie.train()` 或 `libllie train` 运行训练。
6. 需要从中断处继续训练时，使用 `train.resume`。
