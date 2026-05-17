"""Base model class and registry utilities for_teach low-light enhancement models."""

import torch
import torch.nn as nn
import abc
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import yaml
import json

from libllie.deepLearning.loss import BaseLoss


class LLIEModel(abc.ABC, nn.Module):
    """Base class for_teach low-light image enhancement models.

    The class provides automatic model registration, configuration merging,
    checkpoint serialization, checkpoint loading, and a standardized training
    output contract for_teach custom losses.
    """

    name: str = "basemodel"
    aliases: List[str] = []

    _model_registry = {}

    def __init_subclass__(cls, **kwargs):
        """Register model subclasses automatically.

        Args:
            **kwargs: Keyword arguments forwarded to ``nn.Module`` subclass
                initialization.
        """
        super().__init_subclass__(**kwargs)
        LLIEModel._register_model(cls)

    @classmethod
    def _normalize_key(cls, name: str) -> str:
        """Normalize a registry key.

        Args:
            name: Model name or alias.

        Returns:
            Lowercase key without leading or trailing whitespace.
        """
        return name.strip().lower()

    @classmethod
    def _register_model(cls, model_class):
        """Register a model class and its aliases.

        Args:
            model_class: Model class to register.

        Returns:
            The registered model class.

        Raises:
            TypeError: If ``model_class`` does not inherit ``LLIEModel``.
        """
        if not issubclass(model_class, LLIEModel):
            raise TypeError(f"model_class must inherit LLIEModel, got {model_class!r}.")

        candidate_names = [model_class.__name__]
        if "name" in model_class.__dict__:
            candidate_names.append(getattr(model_class, "name"))
        if "aliases" in model_class.__dict__:
            candidate_names.extend(getattr(model_class, "aliases"))

        for name in candidate_names:
            if isinstance(name, str) and name.strip():
                cls._model_registry[cls._normalize_key(name)] = model_class

        return model_class

    @classmethod
    def register(cls, model_class):
        """Register a model class manually.

        Args:
            model_class: Model class to register.

        Returns:
            The registered model class.
        """
        return cls._register_model(model_class)

    @classmethod
    def list_registered_models(cls) -> List[str]:
        """List registered model names and aliases.

        Returns:
            Sorted registry keys.
        """
        return sorted(cls._model_registry.keys())

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize a low-light enhancement model.

        Args:
            config: Configuration dictionary containing model parameters.
            **kwargs: Additional parameters that override same-named values in
                ``config``.
        """
        super().__init__()
        self.config = self._merge_configs(config, kwargs)
        self._validate_config()

        self.device = torch.device(
            self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        self.baseloss = BaseLoss

        self._init_model()
        self.to(self.device)

    @abc.abstractmethod
    def _init_model(self):
        """Initialize model architecture.

        Subclasses must implement this method to define their network layers.
        """
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run a model forward pass.

        Args:
            x: Input tensor.
            **kwargs: Additional forward-pass parameters.

        Returns:
            Training mode: standardized output dict.
            Inference mode: enhanced image tensor.
        """
        pass

    def _format_output(
            self,
            pred: torch.Tensor,
            loss_inputs: Optional[Dict[str, Any]] = None,
            meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format a standardized training-mode model output.

        Args:
            pred: Enhanced image tensor with shape ``[B, C, H, W]``.
            loss_inputs: Optional model-specific tensors required by custom
                losses.
            meta: Optional debug or inference metadata.

        Returns:
            Dictionary with ``pred``, ``loss_inputs``, and ``meta`` keys.
        """
        return {
            "pred": pred,
            "loss_inputs": loss_inputs or {},
            "meta": meta or {},
        }

    def _merge_configs(self, config: Optional[Dict[str, Any]],
                       kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge default, explicit, and keyword configuration values.

        Args:
            config: Base configuration dictionary.
            kwargs: Additional parameter overrides.

        Returns:
            Merged configuration dictionary.
        """
        default_config = self._get_default_config()
        merged_config = default_config.copy()

        if config:
            merged_config.update(config)

        merged_config.update(kwargs)

        return merged_config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get the model's default configuration.

        Returns:
            Default configuration dictionary.
        """
        return {
            'model_name': self.__class__.__name__,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'input_channels': 3,
            'save_dir': f'./checkpoints/{self.__class__.__name__}',
        }

    def _validate_config(self):
        """Validate configuration parameters.

        Subclasses can override this method to validate model-specific
        parameters. The default implementation checks shared required keys.

        Raises:
            ValueError: If a required configuration key is missing.
        """
        required_params = ['input_channels', ]

        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Required parameter '{param}' not found in config")

    def save_model(self, save_path: Union[str, Path]=None,
                   save_optimizer: bool = False,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[Any] = None,
                   epoch: int = 0,
                   metrics: Optional[Dict] = None):
        """Save a model checkpoint.

        Args:
            save_path: Directory where the checkpoint is saved. If None, the
                configured ``save_dir`` is used.
            save_optimizer: Whether to save optimizer state.
            optimizer: Optional optimizer instance.
            scheduler: Optional learning-rate scheduler.
            epoch: Current training epoch.
            metrics: Optional evaluation metrics to store.
        """
        if save_path:
            save_path = Path(save_path)
        else:
            save_path = Path(self.config['save_dir'])
        save_path.mkdir(parents=True, exist_ok=True)

        model_state_dict = {}
        for key, value in self.state_dict().items():
            if not key in ['total_ops', 'total_params']:
                model_state_dict[key] = value

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'config': self.config,
            'model_class': self.__class__.__name__,
            'model_module': self.__class__.__module__,
        }

        if save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if metrics:
            checkpoint['metrics'] = metrics

        save_path = save_path / f'{self.__class__.__name__}.pt'
        torch.save(checkpoint, save_path)

        print(f"✅ Model saved to: {save_path}")


    def save_config(self, save_path: Union[str, Path]=None,):
        """Save model configuration as a YAML file.

        Args:
            save_path: Directory where the configuration file is saved. If None,
                the configured ``save_dir`` is used.
        """
        if save_path:
            save_path = Path(save_path)
        else:
            save_path = Path(self.config['save_dir'])
        save_path.mkdir(parents=True, exist_ok=True)

        config_path = save_path / f'{self.__class__.__name__}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

    @classmethod
    def load_model(cls, checkpoint_path: Union[str, Path]=None,
                   config_overrides: Optional[Dict] = None,
                   device = None,
                   strict: bool = True) -> 'LLIEModel':
        """Load a model checkpoint and reconstruct the saved model class.

        Args:
            checkpoint_path: Checkpoint file path.
            config_overrides: Optional configuration parameter overrides.
            device: Device used to load model weights.
            strict: Whether to strictly match checkpoint keys.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If ``checkpoint_path`` is None.
            ValueError: If the checkpoint model class is not registered.
        """

        if checkpoint_path is None:
            raise FileNotFoundError(f"❌ Checkpoint file not found: {checkpoint_path}")
        else:
            checkpoint_path = Path(checkpoint_path)

        print(f"Loading checkpoint from {checkpoint_path}")
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_class_name = checkpoint.get('model_class', cls.__name__)
        model_module = checkpoint.get('model_module', __name__)

        print(f" Model class name in checkpoint: {model_class_name}")
        print(f" Module path in checkpoint: {model_module}")

        model_key = cls._normalize_key(model_class_name)
        if model_key in cls._model_registry:
            model_class = cls._model_registry[model_key]
            print(f"✅ Found {model_class_name} in registry list")
        else:
            raise ValueError(f"⚠️  Model class {model_class_name} not found in registry")

        config = checkpoint.get('config', {})
        config['device'] = device
        print('=' * 70)
        print(f"Original configuration parameters:")
        print('-' * 70)
        for key, value in config.items():
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            print(f"    {key}: {value_str}")
        print('=' * 70)

        if config_overrides:
            config.update(config_overrides)
            print(f"Applied configuration overrides: {list(config_overrides.keys())}")

        model = model_class(config=config)

        print(f"Loading model weights...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        device = torch.device(config.get('device', 'cpu'))
        model.to(device)

        print(f"✅ {model_class.__name__} model has been loaded into {device}")

        return model

    @classmethod
    def create_model(cls, model_name: str, config: Any = None, **kwargs) -> 'LLIEModel':
        """Create a registered model instance by name.

        Args:
            model_name: Registered model class name.
            config: Configuration dictionary or path to a config file.
            **kwargs: Additional configuration overrides.

        Returns:
            Model instance.

        Raises:
            ValueError: If ``model_name`` is not registered.
            TypeError: If ``config`` has an unsupported type.
        """
        model_key = cls._normalize_key(model_name)
        model_class = cls._model_registry.get(model_key)
        if model_class is None:
            available_models = cls.list_registered_models()
            error_msg = (
                f"❌ Model '{model_name}' is not registered.\n"
                f"   Available models: {available_models}\n"
                f"   Did you mean: {cls._get_similar_model_name(model_name, available_models)}"
            )
            raise ValueError(error_msg)

        if config is None:
            config = {}
        elif isinstance(config, dict):
            pass
        elif isinstance(config, (str, Path)):
            config = cls.load_config(config_path=config)
        else:
            raise TypeError(
                f"❌ Invalid config type: {type(config).__name__}. "
                f"Expected None, dict, str, or Path, but got {type(config).__name__}"
            )

        return model_class(config=config, **kwargs)

    @classmethod
    def _get_similar_model_name(cls, model_name: str, available_models: List[str],
                                max_suggestions: int = 3) -> str:
        """Find close model-name suggestions.

        Args:
            model_name: Requested model name.
            available_models: Registered model names.
            max_suggestions: Maximum number of suggestions to return.

        Returns:
            Suggested model names, or ``"No similar models found"``.
        """
        from difflib import get_close_matches

        suggestions = get_close_matches(
            model_name,
            available_models,
            n=max_suggestions,
            cutoff=0.4
        )

        if suggestions:
            return ", ".join(suggestions)
        else:
            return "No similar models found"

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from a YAML file.

        Args:
            config_path: Configuration file path.

        Returns:
            Loaded configuration dictionary.
        """
        config_path = Path(config_path)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config


    def summary(self):
        """Print model summary information."""
        try:
            params = sum(p.numel() for p in self.parameters())

        except Exception as e:
            print(f"⚠️  Params calculation failed: {e}")
            params = 0

        lines = [
            "=" * 70,
            f"Model Name: {self.__class__.__name__}",
            "-" * 70,
            f"Basic Information",
            f"├─ Device: {self.device}",
            f"└─  Mode: {self.config.get('mode', 'inference')}",
            "",
            f"Parameter Statistics",
            f"├─ Total Parameters: {int(params):,}",
            f"├─ Parameters : {params / 1e3:.2f} Kb",
            f"├─ Parameters : {params / 1e6:.2f} Mb",
            f"├─ Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}",
            "",
            f"Configuration Summary",
        ]

        config_lines = []
        for key, value in self.config.items():
            if key in ['device', 'mode', 'input_channels']:
                continue
            if isinstance(value, (int, float, str, bool)):
                config_lines.append(f"├─ {key}: {value}")
            elif isinstance(value, (list, tuple)) and len(str(value)) < 50:
                config_lines.append(f"├─ {key}: {value}")

        if config_lines:
            if len(config_lines) > 0:
                config_lines[-1] = config_lines[-1].replace('├─', '└─')
            lines.extend(config_lines)
        else:
            lines.append("└─ No additional configuration parameters")

        lines.append("=" * 70)

        print("\n".join(lines))

    def train_mode(self):
        """Switch the model to training mode.

        Returns:
            The model itself.
        """
        self.config['mode'] = 'train'
        self.train()
        return self

    def eval_mode(self):
        """Switch the model to evaluation/inference mode.

        Returns:
            The model itself.
        """
        self.config['mode'] = 'inference'
        self.eval()
        return self

