# LibLLIE Command-Line Interface

LibLLIE provides a command-line interface after installation. The registered commands are:

```bash
libllie ...
llie ...
```

Both commands point to the same CLI entry.

## Available Commands

| Command | API used internally |
| --- | --- |
| `predict` | `llie.predict()` |
| `train` | `llie.train()` |
| `evaluate` | `llie.evaluate()` |
| `imwrite` | `llie.imwrite()` |
| `list` | `llie.list_available()` |

Image reading is not exposed as a CLI command.

## List Available Components

```bash
libllie list
# or
llie list
```

This prints available models, traditional algorithms, metrics, losses, and datasets.

## Predict

Traditional algorithm:

```bash
libllie predict gcp input.jpg -o results/gcp/output.png
```

Deep-learning checkpoint:

```bash
libllie predict checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt input.jpg -o results/zerodce/output.png --device cuda
```

Folder input:

```bash
libllie predict bimef images/ -o results/bimef
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--backend` | Predictor backend: `auto`, `deep`, or `traditional` |
| `--device` | Device used by deep-learning prediction |
| `--output-dir` | Default predictor output directory |
| `--no-progress` | Disable progress bar for folder prediction |
| `--no-save` | Do not save single-image prediction output |
| `--output-name` | Output file name when saving to a folder |
| `--output-ext` | Output extension override |

Extra keyword arguments can be passed with `--kwargs`:

```bash
libllie predict lime input.jpg -o results/lime.png --kwargs gamma=0.8 guided_radius=15
```

For tuple/list values, quote the value in your shell:

```bash
libllie predict clahe input.jpg -o results/clahe.png --kwargs "tile_grid_size=(8, 8)" clip_limit=2.0
```

## Train

Train from a YAML configuration:

```bash
libllie train libllie/deepLearning/config/ZeroDCE.yaml
```

Override training parameters:

```bash
libllie train libllie/deepLearning/config/ZeroDCE.yaml --kwargs epochs=5 batch_size=2 device=cpu
```

`--kwargs` entries are forwarded to `llie.train()`.

## Evaluate

Full-reference evaluation:

```bash
libllie evaluate results/ZeroDCE --ref-img-dir datasets/LOL/eval15/high --metrics PSNR SSIM --save-path results/eval.json
```

No-reference evaluation:

```bash
libllie evaluate results/ZeroDCE --metrics NIQE --save-path results/eval_no_ref.json
```

Extra evaluator parameters can be passed with `--kwargs`:

```bash
libllie evaluate results/ZeroDCE --metrics PSNR SSIM --kwargs device=cpu
```

## Imwrite

Convert or save an image through `llie.imwrite()`:

```bash
libllie imwrite input.jpg -o results/copy.png
```

Save to a folder with a specific output name:

```bash
libllie imwrite input.jpg -o results --output-name copied.png
```

Specify save format:

```bash
libllie imwrite input.jpg -o results/copied --save-format png
```

## KEY=VALUE Parsing

The CLI supports simple Python-like values in `--kwargs`:

```bash
--kwargs lr=1e-4 epochs=10 amp=false device=cpu
```

Parsing rules:

| Input | Parsed value |
| --- | --- |
| `true` / `false` | Boolean |
| `none` | `None` |
| `10`, `1e-4`, `(8, 8)` | Python literal value |
| Other strings | Raw string |

## Help

```bash
libllie --help
libllie predict --help
libllie train --help
libllie evaluate --help
libllie imwrite --help
```
