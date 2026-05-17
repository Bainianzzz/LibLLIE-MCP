"""Base loss abstractions and common supervised losses."""

from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Base class for_teach all libllie training losses.

    It provides:
    - automatic subclass registration
    - case-insensitive factory creation
    - a unified Trainer-facing compute() interface

    Supervised losses keep the standard forward(prediction, target) signature.
    Reference-free losses should set ``requires_target = False`` and implement
    forward(input_tensor, model_output).
    """

    name: str = "baseloss"
    aliases: List[str] = []
    requires_target: bool = True

    _loss_registry: Dict[str, Type["BaseLoss"]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register loss subclasses automatically.

        Args:
            **kwargs: Keyword arguments forwarded to ``nn.Module`` subclass
                initialization.
        """
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "BaseLoss":
            return
        BaseLoss._register_loss(cls)

    @classmethod
    def _normalize_key(cls, name: str) -> str:
        """Normalize a registry key.

        Args:
            name: Loss name or alias.

        Returns:
            Lowercase key without leading or trailing whitespace.
        """
        return name.strip().lower()

    @classmethod
    def _register_loss(cls, loss_class: Type["BaseLoss"]) -> Type["BaseLoss"]:
        """Register a loss class and its aliases.

        Args:
            loss_class: Loss class to register.

        Returns:
            The registered loss class.

        Raises:
            TypeError: If ``loss_class`` does not inherit from ``BaseLoss``.
        """
        if not issubclass(loss_class, BaseLoss):
            raise TypeError(f"loss_class must inherit BaseLoss, got {loss_class!r}.")

        candidate_names = [
            loss_class.__name__,
            getattr(loss_class, "name", loss_class.__name__),
            *getattr(loss_class, "aliases", []),
        ]

        for name in candidate_names:
            if isinstance(name, str) and name.strip():
                cls._loss_registry[cls._normalize_key(name)] = loss_class

        return loss_class

    @classmethod
    def register(cls, loss_class: Type["BaseLoss"]) -> Type["BaseLoss"]:
        """Register a loss class manually.

        Args:
            loss_class: Loss class to register.

        Returns:
            The registered loss class.
        """
        return cls._register_loss(loss_class)

    @classmethod
    def create_loss(cls, loss_name: str, **kwargs: Any) -> "BaseLoss":
        """Create a registered loss instance.

        Args:
            loss_name: Registered loss name or alias.
            **kwargs: Keyword arguments passed to the loss constructor.

        Returns:
            Instantiated loss object.

        Raises:
            ValueError: If ``loss_name`` is empty or not registered.
        """
        if not isinstance(loss_name, str) or not loss_name.strip():
            raise ValueError("loss_name must be a non-empty string.")

        key = cls._normalize_key(loss_name)
        loss_class = cls._loss_registry.get(key)
        if loss_class is None:
            available = cls.list_registered_losses()
            suggestion = cls._get_similar_loss_name(key, available)
            raise ValueError(
                f"Loss '{loss_name}' is not registered.\n"
                f"Available losses: {available}\n"
                f"Did you mean: {suggestion}"
            )

        return loss_class(**kwargs)

    @classmethod
    def list_registered_losses(cls) -> List[str]:
        """List registered loss names and aliases.

        Returns:
            Sorted registry keys.
        """
        return sorted(cls._loss_registry.keys())

    @staticmethod
    def _get_similar_loss_name(loss_name: str, available_losses: List[str]) -> str:
        """Find close loss-name suggestions.

        Args:
            loss_name: Requested loss name.
            available_losses: Registered loss names and aliases.

        Returns:
            Comma-separated suggestion string, or a fallback message.
        """
        from difflib import get_close_matches

        suggestions = get_close_matches(loss_name, available_losses, n=3, cutoff=0.4)
        return ", ".join(suggestions) if suggestions else "No similar losses found"

    def compute(
        self,
        *,
        input_tensor: torch.Tensor,
        model_output: Any,
        target: Optional[torch.Tensor] = None,
        extract_prediction: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        align_prediction: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute loss through the unified Trainer-facing interface.

        Args:
            input_tensor: Low-light input tensor.
            model_output: Raw model output.
            target: Optional paired target tensor.
            extract_prediction: Optional callback used to extract prediction
                tensors from structured model outputs.
            align_prediction: Optional callback used to align prediction shape
                with the target tensor.

        Returns:
            A tuple containing the scalar loss tensor and optional prediction
            tensor.

        Raises:
            ValueError: If the loss requires a target but no target is given.
            TypeError: If a supervised non-tensor model output cannot be
                converted to a prediction.
        """
        if self.requires_target:
            if target is None:
                raise ValueError(f"{self.__class__.__name__} requires a target tensor.")
            if extract_prediction is None:
                if not torch.is_tensor(model_output):
                    raise TypeError(
                        "extract_prediction is required when model_output is not a tensor."
                    )
                prediction = model_output
            else:
                prediction = extract_prediction(model_output, target)
            if align_prediction is not None:
                prediction = align_prediction(prediction, target)
            return self(prediction, target), prediction

        loss = self(input_tensor, model_output)
        prediction = None
        if extract_prediction is not None:
            try:
                prediction = extract_prediction(model_output, input_tensor)
                if align_prediction is not None:
                    prediction = align_prediction(prediction, input_tensor)
            except Exception:
                prediction = None
        return loss, prediction


class L1Loss(BaseLoss):
    """Registered wrapper around ``torch.nn.L1Loss``."""

    name = "l1"
    aliases = ["mae", "l1_loss"]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize an L1 loss.

        Args:
            **kwargs: Keyword arguments passed to ``torch.nn.L1Loss``.
        """
        super().__init__()
        self.loss = nn.L1Loss(**kwargs)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss.

        Args:
            prediction: Predicted tensor.
            target: Target tensor.

        Returns:
            Scalar loss tensor.
        """
        return self.loss(prediction, target)


class MSELoss(BaseLoss):
    """Registered wrapper around ``torch.nn.MSELoss``."""

    name = "mse"
    aliases = ["l2", "mse_loss"]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize an MSE loss.

        Args:
            **kwargs: Keyword arguments passed to ``torch.nn.MSELoss``.
        """
        super().__init__()
        self.loss = nn.MSELoss(**kwargs)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss.

        Args:
            prediction: Predicted tensor.
            target: Target tensor.

        Returns:
            Scalar loss tensor.
        """
        return self.loss(prediction, target)


class SmoothL1Loss(BaseLoss):
    """Registered wrapper around ``torch.nn.SmoothL1Loss``."""

    name = "smooth_l1"
    aliases = ["smoothl1", "huber", "smooth_l1_loss"]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a smooth L1 loss.

        Args:
            **kwargs: Keyword arguments passed to ``torch.nn.SmoothL1Loss``.
        """
        super().__init__()
        self.loss = nn.SmoothL1Loss(**kwargs)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute smooth L1 loss.

        Args:
            prediction: Predicted tensor.
            target: Target tensor.

        Returns:
            Scalar loss tensor.
        """
        return self.loss(prediction, target)


class CharbonnierLoss(BaseLoss):
    """Charbonnier loss for_teach paired image restoration."""

    name = "charbonnier"
    aliases = ["charbonnier_loss"]

    def __init__(self, eps: float = 1e-3) -> None:
        """Initialize Charbonnier loss.

        Args:
            eps: Small constant used for_teach numerical stability.
        """
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier loss.

        Args:
            prediction: Predicted tensor.
            target: Target tensor.

        Returns:
            Scalar loss tensor.
        """
        return torch.mean(torch.sqrt((prediction - target) ** 2 + self.eps ** 2))
