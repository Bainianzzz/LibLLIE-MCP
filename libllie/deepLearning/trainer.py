"""Training utilities for_teach deep-learning low-light enhancement models."""

import importlib
import json
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from libllie.deepLearning.models import LLIEModel
from libllie.deepLearning.loss import BaseLoss
from libllie.deepLearning.config import deep_update, get_default_train_config
from libllie.data.datasets import BaseDataset
from libllie.data import datasets as dataset_module
from libllie.utils import log_info_env, device_display_name


class Trainer:
    """YAML-driven trainer for_teach low-light enhancement models.

    The trainer is intentionally model-agnostic: it creates any registered
    `LLIEModel`, builds a dataset from `libllie.data.datasets`, normalizes
    different model output formats to a final enhanced tensor, and optimizes
    against paired normal-light images.
    """

    def __init__(self, config: Optional[Union[str, Path, Dict[str, Any]]] = None, **kwargs: Any):
        """Initialize a trainer.

        Args:
            config: Optional YAML config path or configuration dictionary.
            **kwargs: Direct configuration overrides. Supported flat arguments
                are mapped into model, data, loss, optimizer, scheduler, and
                train sections.

        Raises:
            FileNotFoundError: If a config path does not exist.
            ValueError: If required model or dataset settings are missing.
            TypeError: If unsupported keyword arguments are provided.
        """
        print('Initializing Trainer...')
        self.config_path = Path(config) if isinstance(config, (str, Path)) else None
        self.user_config = self._load_config(config)
        if kwargs:
            direct_config = self._kwargs_to_config(kwargs)
            self.user_config = deep_update(self.user_config, direct_config)
        self.config = self._with_defaults(self.user_config)
        self._validate_required_config()

        self.device = torch.device(
            self.config["train"].get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )

        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.history = []
        self.training_started_at = None
        self.training_ended_at = None

        self._set_seed(self.config["train"].get("seed"))

        self.model = self._build_model()
        self._resolve_output_dir()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model.config['save_dir'] = self.output_dir
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.criterion = self._build_loss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(self.config["train"].get("amp", False)) and self.device.type == "cuda"
        )
        self._save_training_config()

        resume_path = self.config["train"].get("resume")
        if resume_path:
            self.load_checkpoint(resume_path)
        log_info_env(device=self.device)
        self.print_training_info()

    @staticmethod
    def _load_config(config: Optional[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Any]:
        """Load training configuration.

        Args:
            config: None, configuration dictionary, or YAML config path.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If a config path does not exist.
            ValueError: If the YAML file does not contain a mapping.
        """
        if config is None:
            return {}

        if isinstance(config, dict):
            return dict(config)

        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Training config file does not exist: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        if not isinstance(loaded, dict):
            raise ValueError(f"Training config must be a YAML mapping: {config_path}")
        return loaded

    @staticmethod
    def _with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with default trainer configuration.

        Args:
            config: User configuration dictionary.

        Returns:
            Merged configuration dictionary.
        """
        merged = get_default_train_config()
        return deep_update(merged, dict(config))

    @staticmethod
    def _kwargs_to_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat Trainer keyword arguments to nested configuration.

        Args:
            kwargs: Flat keyword arguments passed to ``Trainer``.

        Returns:
            Nested configuration dictionary.

        Raises:
            TypeError: If an unsupported keyword argument is provided.
        """
        config: Dict[str, Any] = {}
        flat_map = {
            "model": ("model", "name"),
            "model_name": ("model", "name"),
            "model_params": ("model", "params"),
            "dataset": ("data", "dataset"),
            "dataset_name": ("data", "dataset"),
            "root_dir": ("data", "root_dir"),
            "batch_size": ("data", "batch_size"),
            "num_workers": ("data", "num_workers"),
            "pin_memory": ("data", "pin_memory"),
            "shuffle": ("data", "shuffle"),
            "drop_last": ("data", "drop_last"),
            "train_split": ("data", "train_split"),
            "val_split": ("data", "val_split"),
            "return_filename": ("data", "return_filename"),
            "train_low_dir": ("data", "train_low_dir"),
            "train_high_dir": ("data", "train_high_dir"),
            "val_low_dir": ("data", "val_low_dir"),
            "val_high_dir": ("data", "val_high_dir"),
            "data_params": ("data", "params"),
            "train_params": ("data", "train_params"),
            "val_params": ("data", "val_params"),
            "loss": ("loss", "name"),
            "loss_name": ("loss", "name"),
            "loss_params": ("loss", "params"),
            "optimizer": ("optimizer", "name"),
            "optimizer_name": ("optimizer", "name"),
            "lr": ("optimizer", "lr"),
            "optimizer_params": ("optimizer", "params"),
            "scheduler": ("scheduler", "name"),
            "scheduler_name": ("scheduler", "name"),
            "scheduler_params": ("scheduler", "params"),
            "epochs": ("train", "epochs"),
            "output_dir": ("train", "output_dir"),
            "save_every": ("train", "save_every"),
            "validate_every": ("train", "validate_every"),
            "log_every": ("train", "log_every"),
            "grad_clip": ("train", "grad_clip"),
            "amp": ("train", "amp"),
            "resume": ("train", "resume"),
            "seed": ("train", "seed"),
            "device": ("train", "device"),
        }

        for key, value in kwargs.items():
            if key in {"data", "model_config"}:
                target = "data" if key == "data" else "model"
                deep_update(config.setdefault(target, {}), dict(value))
                continue
            if key in {"model", "loss", "optimizer", "scheduler", "train"} and isinstance(value, dict):
                deep_update(config.setdefault(key, {}), dict(value))
                continue
            if key not in flat_map:
                raise TypeError(f"Unsupported Trainer argument: {key}")

            section, option = flat_map[key]
            config.setdefault(section, {})[option] = value

        return config

    def _validate_required_config(self) -> None:
        """Validate required training configuration fields.

        Raises:
            ValueError: If required model or dataset root settings are missing.
        """
        if self.config["model"].get("name") is None:
            raise ValueError("A model is required. Pass model='ZeroDCE' or set model.name in config.")
        if self.config["data"].get("root_dir") is None:
            raise ValueError("A dataset root directory is required. Pass root_dir=... or set data.root_dir in config.")

    @staticmethod
    def _safe_name(value: Any) -> str:
        """Convert a value to a filesystem-safe run name.

        Args:
            value: Value used to infer a name.

        Returns:
            Filesystem-safe name string.
        """
        text = str(value.__class__.__name__ if isinstance(value, LLIEModel) else value)
        text = Path(text).stem if any(sep in text for sep in ("/", "\\")) else text
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_") or "run"

    def _resolve_output_dir(self) -> None:
        """Resolve and store the trainer output directory."""
        output_dir = self.config["train"].get("output_dir")
        if output_dir is None:
            model_name = self.model.__class__.__name__
            dataset_name = self._safe_name(self.config["data"]["dataset"])
            output_dir = Path("checkpoints") / f"{model_name}_{dataset_name}"
            self.config["train"]["output_dir"] = str(output_dir)

        self.output_dir = Path(output_dir)

    @staticmethod
    def _set_seed(seed: Optional[int] = 42) -> None:
        """Set random seeds for_teach reproducible training.

        Args:
            seed: Optional random seed. If None, no seed is set.
        """
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def print_training_info(self) -> None:
        """Print a compact summary of the trainer setup."""
        data_cfg = self.config["data"]

        print("=" * 70)
        print("Trainer initialized")
        print("-" * 70)
        print(f"{'Model':<20}: {self.model.__class__.__name__:<20}")
        print(f"{'Loss':<20}: {self.criterion.__class__.__name__:<20}")
        print(f"{'Training device':<20}: {device_display_name(device=self.device):<20}")
        print(f"{'Optimizer':<20}: {self.optimizer.__class__.__name__:<20} ")
        print(f"{'lr':<20}: {self.optimizer.param_groups[0]['lr']:<20}")
        print(f"{'Scheduler':<20}: {self.scheduler.__class__.__name__ if self.scheduler is not None else 'None':<20}")
        print(f"{'Dataset name':<20}: {str(data_cfg.get('dataset')):<20}")
        print(f"{'Batch size':<20}: {str(data_cfg.get('batch_size')):<20}")
        print(f"{'Root directory':<20}: {str(data_cfg.get('root_dir')):<20}")
        print(f"{'Output directory':<20}: {str(self.output_dir):<20}")
        print("=" * 70)

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        """Resolve a registered model name case-insensitively.

        Args:
            model_name: Requested model name.

        Returns:
            Canonical registered model name.

        Raises:
            ValueError: If the model is not registered.
        """
        registered = LLIEModel.list_registered_models()
        lookup = {name.lower(): name for name in registered}
        key = model_name.lower()
        if key in lookup:
            return lookup[key]
        raise ValueError(f"Model '{model_name}' is not registered. Available models: {registered}")

    def _build_model(self) -> LLIEModel:
        """Build or load the training model.

        Returns:
            Model instance moved to the training device.

        Raises:
            ValueError: If a model name cannot be resolved.
        """
        model_cfg = self.config["model"]
        model_input = model_cfg["name"]
        model_params = dict(model_cfg.get("params") or {})
        model_params["device"] = str(self.device)
        model_params["mode"] = "train"

        if isinstance(model_input, LLIEModel):
            model = model_input
            model.config.update(model_params)
        else:
            model_str = str(model_input)
            if Path(model_str).is_file():
                model = LLIEModel.load_model(
                    checkpoint_path=model_str,
                    device=self.device,
                    strict=True,
                )
                model.config.update(model_params)
            else:
                model_name = self._resolve_model_name(model_str)
                model = LLIEModel.create_model(model_name, config=model_params)

        model.train_mode()
        model.to(self.device)
        return model

    @staticmethod
    def _resolve_dataset_class(dataset_name: str):
        """Resolve a dataset class from a registered name or import path.

        Args:
            dataset_name: Registered dataset name, class name, or dotted import
                path.

        Returns:
            Dataset class.

        Raises:
            TypeError: If a dotted dataset class does not inherit
                ``BaseDataset``.
            ValueError: If the dataset cannot be resolved.
        """
        if "." in dataset_name:
            module_name, class_name = dataset_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            dataset_cls = getattr(module, class_name)
            if not issubclass(dataset_cls, BaseDataset):
                raise TypeError(f"Dataset '{dataset_name}' must inherit BaseDataset.")
            return dataset_cls

        try:
            return BaseDataset.get_dataset_class(dataset_name)
        except ValueError:
            pass

        if hasattr(dataset_module, dataset_name):
            return getattr(dataset_module, dataset_name)

        for name in dir(dataset_module):
            if name.lower() == dataset_name.lower():
                return getattr(dataset_module, name)

        available = BaseDataset.list_registered_datasets()
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available}")

    def _build_dataset(self, split_key: str, split_value: str, low_dir_key: str, high_dir_key: str):
        """Build one dataset split.

        Args:
            split_key: Split prefix used for_teach split-specific params.
            split_value: Dataset split name.
            low_dir_key: Config key for_teach an explicit low-light directory.
            high_dir_key: Config key for_teach an explicit normal-light directory.

        Returns:
            Dataset instance.

        Raises:
            ValueError: If ``data.root_dir`` is missing.
        """
        data_cfg = self.config["data"]
        dataset_cls = self._resolve_dataset_class(data_cfg["dataset"])
        root_dir = data_cfg.get("root_dir")
        if root_dir is None:
            raise ValueError("data.root_dir is required in the training config.")

        dataset_params = dict(data_cfg.get("params", {}))
        split_params = dict(data_cfg.get(f"{split_key}_params", {}))
        dataset_kwargs = {**dataset_params, **split_params}
        dataset_kwargs.update({
            "root_dir": root_dir,
            "split": split_value,
            "return_filename": data_cfg.get("return_filename", True),
        })

        low_dir = data_cfg.get(low_dir_key)
        high_dir = data_cfg.get(high_dir_key)
        if low_dir is not None:
            dataset_kwargs["low_dir"] = low_dir
        if high_dir is not None:
            dataset_kwargs["high_dir"] = high_dir

        dataset_kwargs.setdefault("transform_low", None)
        dataset_kwargs.setdefault("transform_high", None)
        dataset_kwargs.setdefault("common_transform", None)

        return dataset_cls(**dataset_kwargs)

    def _build_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Build train and optional validation dataloaders.

        Returns:
            Tuple containing train dataloader and optional validation dataloader.
        """
        data_cfg = self.config["data"]

        train_dataset = self._build_dataset(
            split_key="train",
            split_value=data_cfg.get("train_split", "train"),
            low_dir_key="train_low_dir",
            high_dir_key="train_high_dir",
        )

        val_loader = None
        use_val = bool(data_cfg.get("val_split") or data_cfg.get("val_low_dir"))
        if use_val:
            val_dataset = self._build_dataset(
                split_key="val",
                split_value=data_cfg.get("val_split", "_test"),
                low_dir_key="val_low_dir",
                high_dir_key="val_high_dir",
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=data_cfg["batch_size"],
                shuffle=False,
                num_workers=data_cfg["num_workers"],
                pin_memory=bool(data_cfg.get("pin_memory", True)) and self.device.type == "cuda",
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=bool(data_cfg.get("shuffle", True)),
            num_workers=data_cfg["num_workers"],
            pin_memory=bool(data_cfg.get("pin_memory", True)) and self.device.type == "cuda",
            drop_last=bool(data_cfg.get("drop_last", False)),
        )

        return train_loader, val_loader

    def _build_loss(self) -> nn.Module:
        """Build the configured loss function.

        Returns:
            Loss module moved to the training device when it is a ``BaseLoss``.

        Raises:
            AttributeError: If a dotted loss path cannot be resolved.
            ValueError: If a registered loss name cannot be resolved.
        """
        loss_cfg = self.config["loss"]
        raw_name = loss_cfg.get("name")
        if raw_name is None or str(raw_name).strip() == "":
            raw_name = self._default_loss_name_for_model()
            loss_cfg["name"] = raw_name

        name = str(raw_name).strip().lower()
        params = dict(loss_cfg.get("params", {}))

        if "." in name:
            module_name, class_name = name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)(**params)

        return BaseLoss.create_loss(name, **params).to(self.device)

    def _default_loss_name_for_model(self) -> str:
        """Infer a default loss name from the model class.

        Returns:
            Registered loss name when matched, otherwise ``"charbonnier"``.
        """
        model_name = self.model.__class__.__name__.strip()
        registered_losses = BaseLoss.list_registered_losses()
        registered_lookup = {name.lower(): name for name in registered_losses}

        candidates = [
            model_name,
            f"{model_name}_loss",
            f"{model_name}Loss",
        ]
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in registered_lookup:
                return registered_lookup[key]

        compact_model_name = "".join(ch for ch in model_name.lower() if ch.isalnum())
        for loss_name in registered_losses:
            compact_loss_name = "".join(ch for ch in loss_name.lower() if ch.isalnum())
            if compact_loss_name in {
                compact_model_name,
                f"{compact_model_name}loss",
            }:
                return loss_name

        return "charbonnier"

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the configured optimizer.

        Returns:
            Optimizer instance.

        Raises:
            ValueError: If the optimizer name is unsupported.
        """
        opt_cfg = self.config["optimizer"]
        name = str(opt_cfg.get("name", "adam")).lower()
        lr = opt_cfg.get("lr", 1e-4)
        params = dict(opt_cfg.get("params", {}))
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if name == "adam":
            return torch.optim.Adam(trainable_params, lr=lr, **params)
        if name == "adamw":
            return torch.optim.AdamW(trainable_params, lr=lr, **params)
        if name == "sgd":
            return torch.optim.SGD(trainable_params, lr=lr, **params)
        if name == "rmsprop":
            return torch.optim.RMSprop(trainable_params, lr=lr, **params)

        raise ValueError(f"Unsupported optimizer '{opt_cfg.get('name')}'.")

    def _build_scheduler(self):
        """Build the configured learning-rate scheduler.

        Returns:
            Scheduler instance or None.

        Raises:
            ValueError: If the scheduler name is unsupported.
        """
        sched_cfg = self.config["scheduler"]
        name = sched_cfg.get("name")
        if not name:
            return None

        params = dict(sched_cfg.get("params", {}))
        name = str(name).lower()

        if name == "steplr":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, **params)
        if name == "multisteplr":
            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **params)
        if name == "cosineannealinglr":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **params)
        if name == "reducelronplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **params)

        raise ValueError(f"Unsupported scheduler '{sched_cfg.get('name')}'.")

    def _move_batch_to_device(self, batch, require_target: bool = True):
        """Move a dataloader batch to the training device.

        Args:
            batch: Batch returned by a dataset.
            require_target: Whether a paired normal-light target is required.

        Returns:
            Tuple containing low-light tensor, optional target tensor, and
            optional filenames.

        Raises:
            ValueError: If the batch format is unsupported or a required target
                is missing.
        """
        if len(batch) == 3:
            low, high, filenames = batch
        elif len(batch) == 2:
            low, high = batch
            filenames = None
        else:
            raise ValueError(f"Unsupported batch format with {len(batch)} items.")

        if require_target and high is None:
            raise ValueError("Supervised training requires paired normal-light images.")

        low = low.to(self.device, non_blocking=True)
        if high is not None:
            high = high.to(self.device, non_blocking=True)

        return low, high, filenames

    def _extract_prediction(self, output: Any, target: torch.Tensor) -> torch.Tensor:
        """Extract a prediction tensor from a model output.

        Args:
            output: Raw model output.
            target: Target tensor used to select a compatible prediction.

        Returns:
            Prediction tensor.

        Raises:
            KeyError: If a configured ``loss.output_key`` is missing.
            TypeError: If no prediction tensor can be extracted.
        """
        loss_cfg = self.config["loss"]
        output_key = loss_cfg.get("output_key")
        output_index = loss_cfg.get("output_index")

        if output_key is not None and isinstance(output, dict):
            if output_key not in output:
                raise KeyError(f"Configured loss.output_key '{output_key}' not found in model output.")
            selected = output[output_key]
            if torch.is_tensor(selected):
                return selected
            tensors = self._flatten_tensors(selected)
            if tensors:
                return self._select_tensor(tensors, target)

        if output_index is not None and isinstance(output, (tuple, list)):
            selected = output[int(output_index)]
            if torch.is_tensor(selected):
                return selected
            tensors = self._flatten_tensors(selected)
            if tensors:
                return self._select_tensor(tensors, target)

        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            for key in ("pred", "enhanced", "output", "prediction", "result", "image"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            tensors = [value for value in output.values() if torch.is_tensor(value)]
            if tensors:
                return self._select_tensor(tensors, target)

        if isinstance(output, (tuple, list)):
            if self._looks_like_grouped_training_output(output):
                tensors = self._flatten_tensors(output[1])
                if tensors:
                    return self._select_tensor(tensors, target)

            tensors = self._flatten_tensors(output)
            if tensors:
                return self._select_tensor(tensors, target)

        raise TypeError(f"Cannot extract prediction tensor from model output type: {type(output).__name__}")

    @staticmethod
    def _looks_like_grouped_training_output(output: Any) -> bool:
        """Check whether output looks like grouped training output.

        Args:
            output: Raw model output.

        Returns:
            True if the output matches the grouped tuple/list pattern.
        """
        return (
            isinstance(output, (tuple, list))
            and len(output) >= 2
            and isinstance(output[0], (tuple, list))
            and isinstance(output[1], (tuple, list))
            and any(torch.is_tensor(item) for item in output[1])
        )

    def _flatten_tensors(self, value: Any):
        """Collect tensors from nested model output structures.

        Args:
            value: Tensor, dictionary, tuple, list, or arbitrary object.

        Returns:
            Flat list of tensors found in ``value``.
        """
        tensors = []
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, dict):
            for item in value.values():
                tensors.extend(self._flatten_tensors(item))
        elif isinstance(value, (tuple, list)):
            for item in value:
                tensors.extend(self._flatten_tensors(item))
        return tensors

    @staticmethod
    def _select_tensor(tensors, target: torch.Tensor) -> torch.Tensor:
        """Select the prediction tensor most compatible with a target.

        Args:
            tensors: Candidate tensors.
            target: Target tensor.

        Returns:
            Selected tensor.
        """
        for tensor in reversed(tensors):
            if tensor.dim() == target.dim() and tensor.shape[:2] == target.shape[:2]:
                return tensor
        for tensor in reversed(tensors):
            if tensor.dim() == target.dim():
                return tensor
        return tensors[-1]

    def _align_prediction(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Align prediction tensor spatial size and channel count to target.

        Args:
            prediction: Prediction tensor.
            target: Target tensor.

        Returns:
            Aligned prediction tensor.

        Raises:
            ValueError: If prediction and target channel counts differ.
        """
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if prediction.shape[1] != target.shape[1]:
            raise ValueError(
                f"Prediction and target channel counts do not match: "
                f"{prediction.shape[1]} vs {target.shape[1]}"
            )
        return prediction

    def _compute_loss(
            self,
            output: Any,
            target: Optional[torch.Tensor],
            input_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute loss from a raw model output.

        Args:
            output: Raw model output.
            target: Optional paired target tensor.
            input_tensor: Low-light input tensor required by ``BaseLoss``.

        Returns:
            Tuple containing scalar loss tensor and optional prediction tensor.

        Raises:
            ValueError: If required input or target tensors are missing.
        """
        if isinstance(self.criterion, BaseLoss):
            if input_tensor is None:
                raise ValueError("input_tensor is required when using BaseLoss.")

            return self.criterion.compute(
                input_tensor=input_tensor,
                model_output=output,
                target=target,
                extract_prediction=self._extract_prediction,
                align_prediction=self._align_prediction,
            )

        if target is None:
            raise ValueError(f"{self.criterion.__class__.__name__} requires a target tensor.")
        prediction = self._extract_prediction(output, target)
        prediction = self._align_prediction(prediction, target)
        return self.criterion(prediction, target), prediction

    def _compute_batch_loss(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one batch forward pass and compute its loss.

        Args:
            batch: Batch returned by the dataloader.

        Returns:
            Tuple containing scalar loss tensor and optional prediction tensor.
        """
        requires_target = not isinstance(self.criterion, BaseLoss) or self.criterion.requires_target
        low, high, _ = self._move_batch_to_device(batch, require_target=requires_target)

        original_mode = None
        paired_forward = bool(getattr(self.model, "requires_paired_forward", False))
        needs_training_output = isinstance(self.criterion, BaseLoss) and (
            not self.criterion.requires_target or paired_forward
        )
        if needs_training_output and hasattr(self.model, "config"):
            original_mode = self.model.config.get("mode")
            self.model.config["mode"] = "train"

        try:
            if paired_forward:
                output = self.model(low, paired_image=high)
            else:
                output = self.model(low)
        finally:
            if original_mode is not None:
                self.model.config["mode"] = original_mode

        loss, prediction = self._compute_loss(output, high, input_tensor=low)
        return loss, prediction

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary containing training history, best validation loss, and
            checkpoint directory.
        """
        epochs = int(self.config["train"]["epochs"])
        if self.training_started_at is None:
            self.training_started_at = datetime.now().isoformat(timespec="seconds")
            self.training_ended_at = None
            self._save_training_config()
            print(f"begin Training at {self.training_started_at}")

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                print('')
                epoch_start = time.time()
                train_loss = self.train_one_epoch(epoch)

                val_loss = None
                if self.val_loader is not None and epoch % int(self.config["train"]["validate_every"]) == 0:
                    val_loss = self.validate(epoch)

                self._step_scheduler(val_loss, train_loss)

                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "seconds": time.time() - epoch_start,
                }
                self.history.append(record)
                self._save_history()

                if epoch % int(self.config["train"]["save_every"]) == 0:
                    self.save_checkpoint("last.pt", epoch, val_loss)

                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt", epoch, val_loss)

                print(
                    f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} "
                    f"| val_loss={val_loss:.6f}" if val_loss is not None
                    else f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f}"
                )
        finally:
            self.training_ended_at = datetime.now().isoformat(timespec="seconds")
            self._save_training_config()

        print("\n"
            f"Finished Training at {self.training_started_at}. \n"
            f"Training results saved at {self.output_dir} \n"
        )

        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "checkpoint_dir": str(self.checkpoint_dir),
        }

    def train_one_epoch(self, epoch: int) -> float:
        """Train the model for_teach one epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Mean training loss for_teach the epoch.
        """
        self.model.train_mode()
        total_loss = 0.0
        total_samples = 0
        log_every = int(self.config["train"]["log_every"])
        grad_clip = self.config["train"].get("grad_clip")

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", unit="batch")
        for step, batch in enumerate(pbar, start=1):
            batch_size = batch[0].size(0)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                loss, _ = self._compute_batch_loss(batch)

            self.scaler.scale(loss).backward()

            if grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if step % log_every == 0 or step == 1:
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
                pbar.set_postfix(postfix)

        return total_loss / max(1, total_samples)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Run validation for_teach one epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Mean validation loss.
        """
        self.model.eval_mode()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc=f"Val {epoch}", unit="batch")
        for step, batch in enumerate(pbar, start=1):
            loss, _ = self._compute_batch_loss(batch)

            batch_size = batch[0].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            postfix = {"loss": f"{loss.item():.6f}"}
            pbar.set_postfix(postfix)

        return total_loss / max(1, total_samples)

    def _step_scheduler(self, val_loss: Optional[float], train_loss: float) -> None:
        """Advance the configured learning-rate scheduler.

        Args:
            val_loss: Optional validation loss.
            train_loss: Training loss for_teach the current epoch.
        """
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric = val_loss if val_loss is not None else train_loss
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def save_checkpoint(self, filename: str, epoch: int,
                        val_loss: Optional[float] = None) -> Path:
        """Save a training checkpoint.

        Args:
            filename: Checkpoint filename.
            epoch: Current epoch index.
            val_loss: Optional validation loss.

        Returns:
            Path to the saved checkpoint.
        """
        save_path = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": {
                key: value for key, value in self.model.state_dict().items()
                if key not in {"total_ops", "total_params"}
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
            "trainer_config": self.config,
            "model_class": self.model.__class__.__name__,
            "model_module": self.model.__class__.__module__,
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler.is_enabled():
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        self.model.save_config(save_path=self.output_dir)
        torch.save(checkpoint, save_path)
        return save_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Checkpoint file path.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler.is_enabled() and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"Resumed training from {checkpoint_path}, start_epoch={self.start_epoch}")

    def _save_history(self) -> None:
        """Save training history as JSON."""
        history_path = self.log_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def _save_training_config(self) -> Path:
        """Save the current training configuration as YAML.

        Returns:
            Path to the saved configuration file.
        """
        model_name = self.model.__class__.__name__ if hasattr(self, "model") else self._safe_name(
            self.config["model"].get("name", "model")
        )
        config_path = self.output_dir / f"{model_name}.yaml"
        payload = {
            "model_name": model_name,
            "dataset_name": self.config["data"].get("dataset"),
            "training_started_at": self.training_started_at,
            "training_ended_at": self.training_ended_at,
            "config": self._to_yaml_safe(self.config),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        return config_path

    @classmethod
    def _to_yaml_safe(cls, value: Any):
        """Convert values to YAML-serializable objects.

        Args:
            value: Value to convert.

        Returns:
            YAML-safe representation of ``value``.
        """
        if isinstance(value, dict):
            return {str(k): cls._to_yaml_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_yaml_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, LLIEModel):
            return value.__class__.__name__
        return str(value)
