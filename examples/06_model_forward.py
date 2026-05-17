from _utils import ROOT  # noqa: F401

import torch

from libllie.deepLearning.models import LLIEModel


def _flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def main():
    image = torch.rand(1, 3, 64, 64)
    model_names = LLIEModel.list_registered_models()

    print(f"Registered models: {model_names} \n")

    for model_name in model_names:
        model = LLIEModel.create_model(
            model_name,
            config={"device": "cpu", "mode": "inference", "input_channels": 3},
        ).eval_mode()

        with torch.no_grad():
            output = model(image)

        tensors = _flatten_tensors(output)
        shapes = [tuple(tensor.shape) for tensor in tensors]
        shapes_str = f"output tensors={shapes}"
        print(f"{model_name:<20}:", f"{shapes_str:<20}")


if __name__ == "__main__":
    main()
