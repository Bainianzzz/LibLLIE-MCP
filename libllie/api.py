"""Top-level convenience API for_teach LibLLIE."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union


_PREDICT_CALL_KWARGS = {
    "progress_bar",
    "output_name",
    "output_ext",
    "save",
    "ext",
    "timeout",
    "headers",
    "verify_ssl",
}


def _split_predict_kwargs(kwargs: Dict[str, Any]):
    """Split predictor construction kwargs from prediction-call kwargs.

    Args:
        kwargs: Keyword arguments passed to the top-level ``predict`` API.

    Returns:
        Tuple of ``(predictor_kwargs, call_kwargs)``. ``predictor_kwargs`` are
        used to construct ``Predictor`` and ``call_kwargs`` are forwarded to
        the predictor call.
    """
    predictor_kwargs = {}
    call_kwargs = {}

    for key, value in kwargs.items():
        if key in _PREDICT_CALL_KWARGS:
            call_kwargs[key] = value
        else:
            predictor_kwargs[key] = value

    return predictor_kwargs, call_kwargs


def predict(
    target,
    source,
    output: Optional[Union[str, Path]] = None,
    **kwargs: Any,
):
    """Enhance a single image or folder with a deep-learning model or
    traditional method.

    Args:
        target: Deep-learning model name or traditional method name.
        source: Image source or folder path.
        output: Optional output file path or directory path.
        **kwargs: Keyword arguments used by ``Predictor`` construction or the
            prediction call.

    Returns:
        Prediction result returned by ``Predictor``.

    Examples:
        libllie.predict("ZeroDCE", "input.jpg", output="out.png")
        libllie.predict("he", "images", output="results/he")
    """
    from libllie.Predictor import Predictor

    predictor_kwargs, call_kwargs = _split_predict_kwargs(kwargs)
    predictor = Predictor(target, **predictor_kwargs)
    return predictor(source, output=output, **call_kwargs)

def enhance(target, source, output: Optional[Union[str, Path]] = None, **kwargs: Any,):
    """Enhance a source image or folder.

    Args:
        target: Deep-learning model name or traditional method name.
        source: Image source or folder path.
        output: Optional output file path or directory path.
        **kwargs: Keyword arguments forwarded to ``predict``.

    Returns:
        Prediction result returned by ``predict``.
    """
    return predict(target, source, output=output, **kwargs)

def train(config: Optional[Union[str, Path, Dict[str, Any]]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Train a model through the unified top-level API.

    Args:
        config: YAML path or config dictionary. Keyword arguments are forwarded
            to Trainer and can override config values.
        **kwargs: Additional keyword arguments forwarded to ``Trainer``.

    Returns:
        Training result dictionary returned by ``Trainer.train``.
    """
    from libllie.deepLearning import Trainer

    trainer = Trainer(config, **kwargs)
    return trainer.train()


def evaluate(
    en: Union[str, Path],
    ref: Optional[Union[str, Path]] = None,
    metrics: Optional[Union[str, List[str]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    return_evaluator: bool = False,
    **kwargs: Any,
):
    """Evaluate enhanced images through the unified top-level API.

    Args:
        en: Directory containing enhanced images.
        ref: Optional reference image directory.
        metrics: Metric name or list of metric names.
        save_path: Optional JSON result path.
        return_evaluator: Return the Evaluator instance instead of results.
        **kwargs: Additional keyword arguments forwarded to ``Evaluator``.

    Returns:
        Evaluation results, or the evaluator instance when
        ``return_evaluator`` is ``True``.
    """
    from libllie.evaluation import Evaluator
    import libllie.evaluation.metrics  # noqa: F401

    evaluator = Evaluator(
        en_img_dir=str(en),
        ref_img_dir=str(ref) if ref is not None else None,
        metrics=metrics,
        save_path=save_path,
        **kwargs,
    )
    return evaluator if return_evaluator else evaluator.results


def eval(*args: Any, **kwargs: Any):
    """Evaluate enhanced images through the alias API.

    Args:
        *args: Positional arguments forwarded to ``evaluate``.
        **kwargs: Keyword arguments forwarded to ``evaluate``.

    Returns:
        Evaluation result returned by ``evaluate``.
    """
    return evaluate(*args, **kwargs)


def imread(source: Any, output_format: str = "pil", **kwargs: Any) -> Any:
    """Read an image through the unified top-level API.

    Args:
        source: Image source accepted by ``read_image``.
        output_format: Desired output format.
        **kwargs: Additional keyword arguments forwarded to ``read_image``.

    Returns:
        Image object in the requested output format.
    """
    from libllie.data.image_io import read_image

    return read_image(source, output_format=output_format, **kwargs)


def imwrite(
    image: Any,
    output: Optional[Union[str, Path]] = None,
    *,
    save_format: Optional[str] = None,
    output_name: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """Write an image through the unified top-level API.

    Args:
        image: Image object accepted by ``write_image``.
        output: Optional output file path or directory path.
        save_format: Optional output format override.
        output_name: Optional output filename used when saving to a directory.
        **kwargs: Additional keyword arguments forwarded to ``write_image``.

    Returns:
        Saved image path.
    """
    from libllie.data.image_io import write_image

    return write_image(
        image,
        output=output,
        save_format=save_format,
        output_name=output_name,
        **kwargs,
    )


read_image = imread
write_image = imwrite


def list_models() -> List[str]:
    """List available deep-learning model names.

    Returns:
        List of model names registered in the unified predictor.
    """
    from libllie.Predictor import Predictor

    return Predictor.list_available_models()


def list_algorithms() -> List[str]:
    """List available traditional enhancement algorithm names.

    Returns:
        List of traditional algorithm names registered in the unified predictor.
    """
    from libllie.Predictor import Predictor

    return Predictor.list_available_methods()


def list_metrics() -> List[str]:
    """List available evaluation metric names.

    Returns:
        List of registered evaluation metric names.
    """
    import libllie.evaluation.metrics  # noqa: F401
    from libllie.evaluation import Evaluator

    return Evaluator.list_available_metrics()

def list_losses() -> List[str]:
    """List available deep-learning loss names.

    Returns:
        List of registered loss names.
    """
    from libllie.deepLearning import BaseLoss

    return BaseLoss.list_registered_losses()

def list_datasets() -> List[str]:
    """List available dataset names.

    Returns:
        List of registered dataset names.
    """
    from libllie.data import BaseDataset

    return BaseDataset.list_registered_datasets()


def list_available() -> Dict[str, List[str]]:
    """List available public components grouped by category.

    Returns:
        Dictionary containing available models, algorithms, metrics, losses, and
        datasets.
    """
    available_sources =  {
        "models": list_models(),
        "algorithms": list_algorithms(),
        "metrics": list_metrics(),
        "losses": list_losses(),
        "datasets": list_datasets(),
    }

    print("Available components:")
    for k, v in available_sources.items():
        print(f"{k}:\n {v} \n")

    return available_sources
