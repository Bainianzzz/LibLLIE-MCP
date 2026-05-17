"""Deep-learning predictor for_teach low-light image enhancement models."""

import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image

if hasattr(torchvision, "disable_beta_transforms_warning"):
    torchvision.disable_beta_transforms_warning()

from torchvision.transforms import v2
from tqdm import tqdm

from libllie.data.basetransform import predict_Trans
from libllie.data.image_io import ImageReader
from libllie.deepLearning.models import LLIEModel
from libllie.utils import log_info_env


ImageInput = Union[str, Path, bytes, bytearray, np.ndarray, Image.Image]


class Predictor:
    """Predictor for_teach deep-learning low-light image enhancement models.

    It supports every single-image input accepted by ImageReader. Directory
    input is also supported and will process all image files recursively.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        ".webp", ".gif", ".ppm", ".pgm", ".pbm", ".sr", ".ras",
    }

    def __init__(
        self,
        model: Union[str, Path, LLIEModel],
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        transform: Optional[Any] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        """Initialize a deep-learning predictor.

        Args:
            model: Registered model name, checkpoint path, or model instance.
            output_dir: Default output directory. If None, results are saved to
                ``results/{model_name}``.
            config: Optional model configuration overrides.
            device: Optional torch device.
            transform: Optional image transform used before prediction.
            batch_size: Batch size metadata kept for_teach API compatibility.
            num_workers: Worker count metadata kept for_teach API compatibility.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model(model, config=config)
        self.model.config["device"] = str(self.device)
        self.model.eval_mode()
        self.model.to(self.device)

        self.model_name = self._get_model_name(self.model)
        self.output_dir = Path(output_dir) if output_dir is not None else Path("results") / self.model_name
        self.transform = self._build_transform(transform)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_reader = ImageReader()

    def __call__(
        self,
        source: ImageInput,
        output: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        """Enhance a single image or every image under a directory.

        Directory input is routed to predict_batch(). All other ImageReader
        compatible inputs are routed to predict_single().

        Args:
            source: Single image input or directory path.
            output: Optional output file path or output directory.
            **kwargs: Extra keyword arguments forwarded to the selected
                prediction method.

        Returns:
            Single-image input returns ``(PIL.Image, saved_path)``. Directory
            input returns a list of saved paths.
        """
        log_info_env(self.device)
        if self._is_directory_source(source):
            return self.predict_batch(source, output_dir=output, **kwargs)
        return self.predict_single(source, save_path=output, **kwargs)

    def predict(self, source, output: Optional[Union[str, Path]] = None, **kwargs: Any):
        """Enhance a source image or directory.

        Args:
            source: Single image input or directory path.
            output: Optional output file path or output directory.
            **kwargs: Extra keyword arguments forwarded to ``__call__``.

        Returns:
            Prediction result returned by ``__call__``.
        """
        return self(source, output=output, **kwargs)


    def predict_single(
        self,
        image: ImageInput,
        save_path: Optional[Union[str, Path]] = None,
        *,
        output_name: Optional[str] = None,
        output_ext: Optional[str] = None,
        save: bool = True,
        transform: Optional[Any] = None,
        **reader_kwargs: Any,
    ) -> Tuple[Image.Image, Optional[Path]]:
        """Enhance one image.

        If save_path is a directory, the original image name is preserved. If it
        is a file path, that exact path is used; its suffix controls the saved
        image format.

        Args:
            image: Single-image input supported by ``ImageReader``.
            save_path: Optional output file path or output directory.
            output_name: Optional output filename.
            output_ext: Optional output suffix or image format.
            save: Whether to save the enhanced image to disk.
            transform: Optional transform overriding the predictor transform.
            **reader_kwargs: Extra keyword arguments passed to ``ImageReader``.

        Returns:
            A tuple containing the enhanced PIL image and optional saved path.

        Raises:
            TypeError: If transform or model output returns an unsupported type.
            ValueError: If transformed input or prediction tensor has an
                unsupported shape.
        """
        tensor = self._load_image_tensor(image, transform=transform, **reader_kwargs)
        enhanced_tensor = self._predict_tensor(tensor)
        enhanced_image = self._tensor_to_pil(enhanced_tensor)

        if not save:
            return enhanced_image, None

        target_path = self._resolve_single_output_path(
            image=image,
            save_path=save_path,
            output_name=output_name,
            output_ext=output_ext,
        )
        self._save_pil_image(enhanced_image, target_path)
        return enhanced_image, target_path

    def predict_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        *,
        progress_bar: bool = True,
        transform: Optional[Any] = None,
        **reader_kwargs: Any,
    ) -> List[Path]:
        """Enhance all image files under a directory recursively.

        Output filenames and suffixes are kept identical to the source images.
        Relative subdirectories are preserved to avoid filename collisions.

        Args:
            input_dir: Directory containing input images.
            output_dir: Optional output directory. If None, the predictor
                default output directory is used.
            progress_bar: Whether to show a tqdm progress bar.
            transform: Optional transform overriding the predictor transform.
            **reader_kwargs: Extra keyword arguments passed to ``ImageReader``.

        Returns:
            List of saved image paths.

        Raises:
            NotADirectoryError: If ``input_dir`` is not a directory.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"input_dir must be a directory, got {input_dir}.")

        image_files = self._list_images(input_dir)
        if not image_files:
            return []

        output_root = Path(output_dir) if output_dir is not None else self.output_dir
        saved_paths: List[Path] = []
        iterator = tqdm(image_files, desc=f"Enhancing with {self.model_name}") if progress_bar else image_files

        for image_path in iterator:
            relative_path = image_path.relative_to(input_dir)
            target_path = output_root / relative_path
            _, saved_path = self.predict_single(
                str(image_path),
                save_path=target_path,
                save=True,
                transform=transform,
                **reader_kwargs,
            )
            saved_paths.append(saved_path)

            if progress_bar:
                iterator.set_postfix({"current": image_path.name})

        return saved_paths

    def get_params(self) -> Dict[str, Any]:
        """Get predictor parameters and model configuration.

        Returns:
            Dictionary containing model name, device, output directory, and
            model configuration.
        """
        return {
            "model": self.model_name,
            "device": str(self.device),
            "output_dir": str(self.output_dir),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "config": dict(getattr(self.model, "config", {})),
        }

    @staticmethod
    def list_available_models() -> List[str]:
        """List registered deep-learning model names.

        Returns:
            List of registered model names.
        """
        return LLIEModel.list_registered_models()

    @staticmethod
    def _get_model_name(model: LLIEModel) -> str:
        """Get a model class name.

        Args:
            model: Model instance.

        Returns:
            Model class name.
        """
        return model.__class__.__name__

    def _load_model(
        self,
        model: Union[str, Path, LLIEModel],
        config: Optional[Dict[str, Any]] = None,
    ) -> LLIEModel:
        """Load a model from name, checkpoint, or model instance.

        Args:
            model: Registered model name, checkpoint path, or model instance.
            config: Optional model configuration overrides.

        Returns:
            Loaded model instance.

        Raises:
            TypeError: If ``model`` has an unsupported type.
            FileNotFoundError: If a checkpoint path does not exist.
            ValueError: If the model name is not registered.
        """
        if isinstance(model, LLIEModel):
            if config:
                model.config.update(config)
            return model

        if isinstance(model, Path):
            model = str(model)

        if not isinstance(model, str):
            raise TypeError(f"model must be str, Path, or LLIEModel, got {type(model)!r}.")

        model_path = Path(model)
        if model_path.suffix.lower() in {".pt", ".pth"}:
            if not model_path.is_file():
                raise FileNotFoundError(f"Checkpoint file does not exist: {model_path}")
            return self._load_checkpoint_model(model_path, config=config)

        model_name = self._resolve_model_name(model)
        model_config = dict(config or {})
        model_config["device"] = str(self.device)
        model_config.setdefault("mode", "inference")
        return LLIEModel.create_model(model_name, config=model_config)

    def _load_checkpoint_model(
        self,
        checkpoint_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> LLIEModel:
        """Load a model from a checkpoint file.

        Args:
            checkpoint_path: Checkpoint file path.
            config: Optional model configuration overrides.

        Returns:
            Loaded model instance.

        Raises:
            ValueError: If checkpoint content is not a dictionary.
            KeyError: If the checkpoint does not contain ``model_class``.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}.")

        model_class = checkpoint.get("model_class")
        if not model_class:
            raise KeyError(f"Checkpoint '{checkpoint_path}' does not contain key 'model_class'.")

        model_name = self._resolve_model_name(model_class)
        model_config = dict(checkpoint.get("config", {}))
        model_config["device"] = str(self.device)
        model_config["mode"] = "inference"
        if config:
            model_config.update(config)

        loaded_model = LLIEModel.create_model(model_name, config=model_config)
        loaded_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        loaded_model.to(self.device)
        return loaded_model

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

    def _load_image_tensor(
        self,
        image: ImageInput,
        transform: Optional[Any] = None,
        **reader_kwargs: Any,
    ) -> torch.Tensor:
        """Load an image input and convert it to a batched tensor.

        Args:
            image: Single-image input supported by ``ImageReader``.
            transform: Optional transform overriding the predictor transform.
            **reader_kwargs: Extra keyword arguments passed to ``ImageReader``.

        Returns:
            Batched image tensor on the predictor device.

        Raises:
            TypeError: If the transform does not return a tensor.
            ValueError: If the transformed tensor shape is unsupported.
        """
        pil_image = self.image_reader(image, output_format="pil", **reader_kwargs)
        transformer = self._build_transform(transform) if transform is not None else self.transform
        tensor = transformer(pil_image)
        if not torch.is_tensor(tensor):
            raise TypeError(f"transform must return a torch.Tensor, got {type(tensor)!r}.")
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError(f"Expected transformed image shape [C,H,W] or [N,C,H,W], got {tuple(tensor.shape)}.")
        return tensor.to(self.device)

    def _predict_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run model prediction for_teach a tensor batch.

        Args:
            tensor: Batched input tensor.

        Returns:
            Prediction tensor.

        Raises:
            TypeError: If the model output cannot be converted to a tensor.
        """
        with torch.no_grad():
            output = self.model(tensor)
        prediction = self._extract_prediction(output)
        if not torch.is_tensor(prediction):
            raise TypeError(f"Model prediction must be a torch.Tensor, got {type(prediction)!r}.")
        return prediction

    @classmethod
    def _extract_prediction(cls, output: Any) -> torch.Tensor:
        """Extract prediction tensor from a model output.

        Args:
            output: Raw model output.

        Returns:
            Prediction tensor.

        Raises:
            KeyError: If a dictionary output contains no tensor prediction.
            TypeError: If prediction cannot be extracted from ``output``.
        """
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            if "pred" in output:
                return output["pred"]
            tensors = cls._flatten_tensors(output)
            if tensors:
                return tensors[-1]
            raise KeyError("Model output dict must contain a tensor prediction.")
        if isinstance(output, (tuple, list)):
            tensors = cls._flatten_tensors(output)
            if tensors:
                return tensors[-1]
        raise TypeError(f"Cannot extract prediction tensor from model output type: {type(output).__name__}.")

    @classmethod
    def _flatten_tensors(cls, value: Any) -> List[torch.Tensor]:
        """Collect tensors from nested output structures.

        Args:
            value: Tensor, dictionary, tuple, list, or arbitrary object.

        Returns:
            Flat list of tensors found in ``value``.
        """
        tensors: List[torch.Tensor] = []
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, dict):
            for item in value.values():
                tensors.extend(cls._flatten_tensors(item))
        elif isinstance(value, (tuple, list)):
            for item in value:
                tensors.extend(cls._flatten_tensors(item))
        return tensors

    @staticmethod
    def _build_transform(transform: Optional[Any]):
        """Build an image transform callable.

        Args:
            transform: None, callable transform, or list of v2 transforms.

        Returns:
            Transform callable.

        Raises:
            TypeError: If ``transform`` has an unsupported type.
        """
        if transform is None:
            return predict_Trans
        if isinstance(transform, list):
            return v2.Compose(transform)
        if callable(transform):
            return transform
        raise TypeError(f"transform must be callable, a list of callables, or None, got {type(transform)!r}.")

    @classmethod
    def _list_images(cls, directory: Path) -> List[Path]:
        """List supported image files under a directory recursively.

        Args:
            directory: Directory to scan.

        Returns:
            Sorted list of image paths.
        """
        return sorted(
            [
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in cls.SUPPORTED_EXTENSIONS
            ],
            key=lambda path: str(path).lower(),
        )

    @staticmethod
    def _is_directory_source(source: ImageInput) -> bool:
        """Check whether a source is a directory path.

        Args:
            source: Prediction source.

        Returns:
            True if ``source`` is an existing directory path.
        """
        return isinstance(source, (str, Path)) and Path(source).is_dir()

    def _resolve_single_output_path(
        self,
        *,
        image: ImageInput,
        save_path: Optional[Union[str, Path]],
        output_name: Optional[str],
        output_ext: Optional[str],
    ) -> Path:
        """Resolve output path for_teach a single-image prediction.

        Args:
            image: Original image input.
            save_path: Optional output file path or directory.
            output_name: Optional output filename.
            output_ext: Optional output suffix or format.

        Returns:
            Resolved output path.

        Raises:
            ValueError: If ``output_ext`` is empty.
        """
        original_name = output_name or self._infer_source_name(image)
        original_path = Path(original_name)

        if output_ext is not None:
            suffix = self._normalize_suffix(output_ext)
            original_path = original_path.with_suffix(suffix)

        if save_path is None:
            return self.output_dir / original_path.name

        save_path = Path(save_path)
        if self._looks_like_file_path(save_path):
            return save_path
        return save_path / original_path.name

    @classmethod
    def _looks_like_file_path(cls, path: Path) -> bool:
        """Check whether a path should be treated as a file path.

        Args:
            path: Candidate path.

        Returns:
            True if the path exists as a file or has a suffix.
        """
        if path.exists():
            return path.is_file()
        return bool(path.suffix)

    @classmethod
    def _infer_source_name(cls, source: ImageInput) -> str:
        """Infer output filename from an image source.

        Args:
            source: Original image source.

        Returns:
            Inferred filename with suffix.
        """
        if isinstance(source, Path):
            return source.name

        if isinstance(source, str):
            parsed = urllib.parse.urlparse(source)
            if parsed.scheme in {"http", "https", "ftp", "file"}:
                name = Path(urllib.parse.unquote(parsed.path)).name
                if name:
                    return name

            path = Path(source)
            if path.name and path.suffix:
                return path.name

        return "image.jpg"

    @staticmethod
    def _normalize_suffix(ext: str) -> str:
        """Normalize an output suffix.

        Args:
            ext: Suffix or format name.

        Returns:
            Suffix with a leading dot.

        Raises:
            ValueError: If ``ext`` is empty.
        """
        ext = ext.strip()
        if not ext:
            raise ValueError("output_ext must not be empty.")
        return ext if ext.startswith(".") else f".{ext}"

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert a prediction tensor to a PIL image.

        Args:
            tensor: Prediction tensor with shape ``[C, H, W]`` or
                ``[N, C, H, W]``.

        Returns:
            PIL image in ``L`` or ``RGB`` mode.

        Raises:
            ValueError: If the tensor shape is unsupported.
        """
        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() != 3:
            raise ValueError(f"Expected prediction tensor shape [C,H,W] or [N,C,H,W], got {tuple(tensor.shape)}.")

        tensor = tensor.detach().cpu().float().clamp(0.0, 1.0)
        array = tensor.numpy()

        if array.shape[0] == 1:
            image = np.squeeze(array, axis=0)
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(image, mode="L")

        image = np.transpose(array[:3], (1, 2, 0))
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(image, mode="RGB")

    @staticmethod
    def _save_pil_image(image: Image.Image, save_path: Path) -> None:
        """Save a PIL image to disk.

        Args:
            image: PIL image to save.
            save_path: Output image path.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix.lower()
        output_image = image
        if suffix in {".jpg", ".jpeg"} and image.mode not in {"RGB", "L"}:
            output_image = image.convert("RGB")
        output_image.save(save_path)
