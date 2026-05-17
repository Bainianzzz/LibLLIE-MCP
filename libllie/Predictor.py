"""Unified prediction interface for_teach LibLLIE."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from libllie.deepLearning.models import LLIEModel
from libllie.deepLearning.predictor import Predictor as DeepLearningPredictor
from libllie.traditional.algorithms import LLIEnhancer
from libllie.traditional.predictor import Predictor as TraditionalPredictor


class Predictor:
    """Unified predictor for_teach LibLLIE.

    This wrapper dispatches to either the deep-learning predictor or the
    traditional-algorithm predictor. The backend can be selected explicitly or
    inferred from the requested model/method name.
    """

    DEEP_BACKENDS = {"deep", "deeplearning", "deep_learning", "dl", "model"}
    TRADITIONAL_BACKENDS = {"traditional", "traditionalalgorithm", "traditional_algorithm", "ta", "method"}

    def __init__(
        self,
        target: Optional[Union[str, Path, LLIEModel, LLIEnhancer]] = None,
        *,
        model: Optional[Union[str, Path, LLIEModel]] = None,
        method: Optional[Union[str, LLIEnhancer]] = None,
        backend: str = "auto",
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Any] = None,
        transform: Optional[Any] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the unified predictor.

        Args:
            target: Model name, checkpoint path, traditional method name, model
                instance, or enhancer instance.
            model: Explicit deep-learning model name, checkpoint path, or model
                instance.
            method: Explicit traditional method name or enhancer instance.
            backend: Backend selector. Use ``"auto"`` to infer from the target.
            output_dir: Default output directory used by the selected backend.
            config: Optional backend configuration dictionary.
            device: Device used by deep-learning prediction.
            transform: Optional transform used by deep-learning prediction.
            batch_size: Batch size used by deep-learning batch prediction.
            num_workers: Number of dataloader workers used by deep-learning
                prediction.
            **kwargs: Additional keyword arguments forwarded to the selected
                backend.

        Raises:
            ValueError: If the backend cannot be resolved or conflicting target
                arguments are provided.
            TypeError: If an instance target is incompatible with the selected
                backend.
        """
        self.backend = self._resolve_backend(
            target=target,
            model=model,
            method=method,
            backend=backend,
        )

        if self.backend == "deep":
            model_input = self._resolve_model_input(target=target, model=model, method=method)
            model_config = dict(config or {})
            model_config.update(kwargs)
            self.predictor = DeepLearningPredictor(
                model=model_input,
                output_dir=output_dir,
                config=model_config or None,
                device=device,
                transform=transform,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:
            method_input = self._resolve_method_input(target=target, model=model, method=method)
            self.predictor = TraditionalPredictor(
                method=method_input,
                output_dir=output_dir,
                config=config,
                **kwargs,
            )

    def __call__(self, source, output: Optional[Union[str, Path]] = None, **kwargs: Any):
        """Run prediction through the selected backend.

        Args:
            source: Image source or directory accepted by the backend predictor.
            output: Optional output file path or directory path.
            **kwargs: Additional keyword arguments forwarded to the backend
                predictor.

        Returns:
            Prediction result returned by the backend predictor.
        """
        return self.predictor(source, output=output, **kwargs)

    def predict(self, source, output: Optional[Union[str, Path]] = None, **kwargs: Any):
        """Run prediction through the unified call interface.

        Args:
            source: Image source or directory accepted by the backend predictor.
            output: Optional output file path or directory path.
            **kwargs: Additional keyword arguments forwarded to ``__call__``.

        Returns:
            Prediction result returned by ``__call__``.
        """
        return self(source, output=output, **kwargs)

    def predict_single(self, *args: Any, **kwargs: Any):
        """Enhance a single image through the selected backend.

        Args:
            *args: Positional arguments forwarded to the backend
                ``predict_single`` method.
            **kwargs: Keyword arguments forwarded to the backend
                ``predict_single`` method.

        Returns:
            Single-image prediction result returned by the backend predictor.
        """
        return self.predictor.predict_single(*args, **kwargs)

    def predict_batch(self, *args: Any, **kwargs: Any):
        """Enhance a batch or directory through the selected backend.

        Args:
            *args: Positional arguments forwarded to the backend
                ``predict_batch`` method.
            **kwargs: Keyword arguments forwarded to the backend
                ``predict_batch`` method.

        Returns:
            Batch prediction result returned by the backend predictor.
        """
        return self.predictor.predict_batch(*args, **kwargs)

    def get_params(self) -> Dict[str, Any]:
        """Get unified predictor parameters.

        Returns:
            Dictionary containing backend name and backend predictor parameters.
        """
        return {
            "backend": self.backend,
            "predictor": self.predictor.get_params(),
        }

    @classmethod
    def list_available_models(cls) -> List[str]:
        """List available deep-learning model names.

        Returns:
            List of model names supported by the deep-learning predictor.
        """
        return DeepLearningPredictor.list_available_models()

    @classmethod
    def list_available_methods(cls) -> List[str]:
        """List available traditional method names.

        Returns:
            List of method names supported by the traditional predictor.
        """
        return TraditionalPredictor.list_available_methods()

    @classmethod
    def list_available(cls) -> Dict[str, List[str]]:
        """List available models and traditional algorithms.

        Returns:
            Dictionary containing available ``models`` and ``algorithms``.
        """
        return {
            "models": cls.list_available_models(),
            "algorithms": cls.list_available_methods(),
        }

    @classmethod
    def _resolve_backend(
        cls,
        *,
        target: Optional[Union[str, Path, LLIEModel, LLIEnhancer]],
        model: Optional[Union[str, Path, LLIEModel]],
        method: Optional[Union[str, LLIEnhancer]],
        backend: str,
    ) -> str:
        """Resolve which predictor backend should be used.

        Args:
            target: Positional target passed to the unified predictor.
            model: Explicit deep-learning model selector.
            method: Explicit traditional method selector.
            backend: Backend selector requested by the caller.

        Returns:
            Normalized backend name, either ``"deep"`` or ``"traditional"``.

        Raises:
            ValueError: If selector arguments conflict, the backend name is
                unsupported, or automatic inference fails.
        """
        if model is not None and method is not None:
            raise ValueError("Pass only one of 'model' or 'method'.")
        if target is not None and (model is not None or method is not None):
            raise ValueError("Pass either positional 'target' or keyword 'model'/'method', not both.")

        normalized_backend = str(backend).strip().lower()
        if normalized_backend in cls.DEEP_BACKENDS:
            return "deep"
        if normalized_backend in cls.TRADITIONAL_BACKENDS:
            return "traditional"
        if normalized_backend != "auto":
            raise ValueError(
                f"Unsupported backend '{backend}'. Use 'auto', 'deep', or 'traditional'."
            )

        candidate = model if model is not None else method if method is not None else target
        if isinstance(candidate, LLIEModel):
            return "deep"
        if isinstance(candidate, LLIEnhancer):
            return "traditional"
        if method is not None:
            return "traditional"
        if model is not None:
            return "deep"
        if candidate is None:
            raise ValueError("A model, method, or target name is required.")

        if isinstance(candidate, Path):
            candidate = str(candidate)

        if isinstance(candidate, str):
            path = Path(candidate)
            if path.suffix.lower() in {".pt", ".pth"}:
                return "deep"

            deep_lookup = {name.lower() for name in DeepLearningPredictor.list_available_models()}
            method_lookup = {name.lower() for name in TraditionalPredictor.list_available_methods()}
            key = candidate.lower()

            if key in deep_lookup and key in method_lookup:
                raise ValueError(
                    f"'{candidate}' is both a deep-learning model and a traditional method. "
                    "Pass backend='deep' or backend='traditional'."
                )
            if key in deep_lookup:
                return "deep"
            if key in method_lookup:
                return "traditional"

        raise ValueError(
            f"Cannot infer predictor backend from {candidate!r}. \n"
            f"Available models: {DeepLearningPredictor.list_available_models()}; \n"
            f"available algorithms: {TraditionalPredictor.list_available_methods()}.\n"
        )

    @staticmethod
    def _resolve_model_input(
        *,
        target: Optional[Union[str, Path, LLIEModel, LLIEnhancer]],
        model: Optional[Union[str, Path, LLIEModel]],
        method: Optional[Union[str, LLIEnhancer]],
    ) -> Union[str, Path, LLIEModel]:
        """Resolve the model argument for_teach the deep-learning backend.

        Args:
            target: Positional target passed to the unified predictor.
            model: Explicit deep-learning model selector.
            method: Explicit traditional method selector.

        Returns:
            Deep-learning model name, checkpoint path, or model instance.

        Raises:
            ValueError: If no deep-learning model selector is available.
            TypeError: If a traditional enhancer instance is used as the model
                input.
        """
        model_input = model if model is not None else target
        if model_input is None or method is not None:
            raise ValueError("A deep-learning model name, checkpoint path, or LLIEModel instance is required.")
        if isinstance(model_input, LLIEnhancer):
            raise TypeError("LLIEnhancer instances must use the traditional backend.")
        return model_input

    @staticmethod
    def _resolve_method_input(
        *,
        target: Optional[Union[str, Path, LLIEModel, LLIEnhancer]],
        model: Optional[Union[str, Path, LLIEModel]],
        method: Optional[Union[str, LLIEnhancer]],
    ) -> Union[str, LLIEnhancer]:
        """Resolve the method argument for_teach the traditional backend.

        Args:
            target: Positional target passed to the unified predictor.
            model: Explicit deep-learning model selector.
            method: Explicit traditional method selector.

        Returns:
            Traditional method name or enhancer instance.

        Raises:
            ValueError: If no traditional method selector is available.
            TypeError: If a deep-learning model instance is used as the method
                input.
        """
        method_input = method if method is not None else target
        if method_input is None or model is not None:
            raise ValueError("A traditional method name or LLIEnhancer instance is required.")
        if isinstance(method_input, LLIEModel):
            raise TypeError("LLIEModel instances must use the deep backend.")
        if isinstance(method_input, Path):
            method_input = str(method_input)
        return method_input
