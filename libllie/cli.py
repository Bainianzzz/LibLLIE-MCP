"""Command-line interface for_teach LibLLIE."""

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import libllie as llie


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Run the LibLLIE command-line interface.

    Args:
        argv: Optional command-line arguments. When ``None``, arguments are
            read from ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    _print_result(result)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the root CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="libllie",
        description="Command-line tools for_teach LibLLIE.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_predict_parser(subparsers)
    _add_train_parser(subparsers)
    _add_evaluate_parser(subparsers, name="evaluate")
    _add_evaluate_parser(subparsers, name="eval")
    _add_imwrite_parser(subparsers)
    _add_list_parser(subparsers)

    return parser


def _add_predict_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the predict subcommand parser.

    Args:
        subparsers: Root subparser container.
    """
    parser = subparsers.add_parser(
        "predict",
        help="Enhance an image or folder with a model or traditional algorithm.",
    )
    parser.add_argument("target", help="Model name, checkpoint path, or algorithm name.")
    parser.add_argument("source", help="Input image path or image folder.")
    parser.add_argument("-o", "--output", default=None, help="Output file path or folder.")
    parser.add_argument("--backend", default="auto", help="Predictor backend: auto, deep, or traditional.")
    parser.add_argument("--device", default=None, help="Device for_teach deep-learning prediction.")
    parser.add_argument("--output-dir", default=None, help="Default predictor output directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size metadata for_teach predictor.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count metadata.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar for_teach folder prediction.")
    parser.add_argument("--no-save", action="store_true", help="Do not save single-image prediction output.")
    parser.add_argument("--output-name", default=None, help="Output file name when saving to a folder.")
    parser.add_argument("--output-ext", default=None, help="Output extension override.")
    parser.add_argument(
        "--kwargs",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra keyword arguments forwarded to llie.predict().",
    )
    parser.set_defaults(func=_cmd_predict)


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the train subcommand parser.

    Args:
        subparsers: Root subparser container.
    """
    parser = subparsers.add_parser("train", help="Train a model.")
    parser.add_argument("config", nargs="?", default=None, help="YAML config path.")
    parser.add_argument(
        "--kwargs",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra keyword arguments forwarded to llie.train().",
    )
    parser.set_defaults(func=_cmd_train)


def _add_evaluate_parser(subparsers: argparse._SubParsersAction, name: str = "evaluate") -> None:
    """Add the evaluate subcommand parser.

    Args:
        subparsers: Root subparser container.
        name: Subcommand name to register.
    """
    parser = subparsers.add_parser(name, help="Evaluate enhanced image folders.")
    parser.add_argument(
        "--en",
        "--en-img-dir",
        dest="en_img_dir",
        required=True,
        help="Directory containing enhanced images.",
    )
    parser.add_argument(
        "--ref",
        "--ref-img-dir",
        dest="ref_img_dir",
        default=None,
        help="Optional reference image directory.",
    )
    parser.add_argument("--metrics", nargs="*", default=None, help="Metric names.")
    parser.add_argument("--save-path", default=None, help="Path to save evaluation results.")
    parser.add_argument(
        "--return-evaluator",
        action="store_true",
        help="Return evaluator object instead of only results.",
    )
    parser.add_argument(
        "--kwargs",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra keyword arguments forwarded to llie.evaluate().",
    )
    parser.set_defaults(func=_cmd_evaluate)


def _add_imwrite_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the imwrite subcommand parser.

    Args:
        subparsers: Root subparser container.
    """
    parser = subparsers.add_parser("imwrite", help="Write or convert an image.")
    parser.add_argument("image", help="Input image path or supported string source.")
    parser.add_argument("-o", "--output", default=None, help="Output file path or folder.")
    parser.add_argument("--save-format", default=None, help="Output image format override.")
    parser.add_argument("--output-name", default=None, help="Output file name when saving to a folder.")
    parser.add_argument(
        "--kwargs",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra keyword arguments forwarded to llie.imwrite().",
    )
    parser.set_defaults(func=_cmd_imwrite)


def _add_list_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the list subcommand parser.

    Args:
        subparsers: Root subparser container.
    """
    parser = subparsers.add_parser("list", help="List available LibLLIE components.")
    parser.set_defaults(func=_cmd_list_available)


def _cmd_predict(args: argparse.Namespace) -> Any:
    """Run prediction from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Result returned by ``llie.predict``.
    """
    kwargs = _parse_key_value_args(args.kwargs)
    kwargs.update(
        _drop_none(
            {
                "backend": args.backend,
                "device": args.device,
                "output_dir": args.output_dir,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "progress_bar": not args.no_progress,
                "save": not args.no_save,
                "output_name": args.output_name,
                "output_ext": args.output_ext,
            }
        )
    )
    return llie.predict(args.target, args.source, output=args.output, **kwargs)


def _cmd_train(args: argparse.Namespace) -> Any:
    """Run training from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Result returned by ``llie.train``.
    """
    kwargs = _parse_key_value_args(args.kwargs)
    return llie.train(args.config, **kwargs)


def _cmd_evaluate(args: argparse.Namespace) -> Any:
    """Run evaluation from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Result returned by ``llie.evaluate``.
    """
    kwargs = _parse_key_value_args(args.kwargs)
    return llie.evaluate(
        en=args.en_img_dir,
        ref=args.ref_img_dir,
        metrics=args.metrics,
        save_path=args.save_path,
        return_evaluator=args.return_evaluator,
        **kwargs,
    )


def _cmd_imwrite(args: argparse.Namespace) -> Any:
    """Run image writing from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Result returned by ``llie.imwrite``.
    """
    kwargs = _parse_key_value_args(args.kwargs)
    return llie.imwrite(
        args.image,
        output=args.output,
        save_format=args.save_format,
        output_name=args.output_name,
        **kwargs,
    )


def _cmd_list_available(args: argparse.Namespace) -> Any:
    """List available components.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Available component dictionary.
    """
    return llie.list_available()


def _parse_key_value_args(items: Iterable[str]) -> Dict[str, Any]:
    """Parse ``KEY=VALUE`` CLI arguments into keyword arguments.

    Args:
        items: Iterable of ``KEY=VALUE`` strings.

    Returns:
        Parsed keyword argument dictionary.

    Raises:
        ValueError: If an item is not in ``KEY=VALUE`` format.
    """
    parsed: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE format, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Keyword name must not be empty: {item!r}.")
        parsed[key] = _parse_value(value)
    return parsed


def _parse_value(value: str) -> Any:
    """Parse a CLI value string.

    Args:
        value: Raw CLI value.

    Returns:
        Parsed Python value when possible, otherwise the original string.
    """
    value = value.strip()
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _drop_none(values: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys whose values are ``None``.

    Args:
        values: Input dictionary.

    Returns:
        Dictionary without ``None`` values.
    """
    return {key: value for key, value in values.items() if value is not None}


def _print_result(result: Any) -> None:
    """Print command results in a compact form.

    Args:
        result: Command result.
    """
    if result is None:
        return

    if _is_single_prediction_result(result):
        print(result[1])
        return

    # try:
    #     print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    # except TypeError:
    #     print(result)


def _is_single_prediction_result(value: Any) -> bool:
    """Check whether a command result is a single-image prediction result.

    Args:
        value: Command result.

    Returns:
        Whether the result follows ``(enhanced_image, saved_path)``.
    """
    if not isinstance(value, tuple) or len(value) != 2:
        return False

    enhanced, saved_path = value
    return _is_image_like(enhanced) and _is_path_like(saved_path)


def _is_image_like(value: Any) -> bool:
    """Check whether a value looks like an in-memory image object.

    Args:
        value: Candidate image value.

    Returns:
        Whether the value is likely an enhanced image returned by prediction.
    """
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return True
    return hasattr(value, "__class__") and value.__class__.__module__.startswith("PIL")


def _is_path_like(value: Any) -> bool:
    """Check whether a value looks like a saved path.

    Args:
        value: Candidate path value.

    Returns:
        Whether the value is a string or ``Path`` object.
    """
    return isinstance(value, (str, Path))


def _json_safe(value: Any) -> Any:
    """Convert common command results to JSON-safe objects.

    Args:
        value: Arbitrary result value.

    Returns:
        JSON-safe representation.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__class__") and value.__class__.__module__.startswith("PIL"):
        return repr(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
