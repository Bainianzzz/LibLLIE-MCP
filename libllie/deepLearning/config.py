import copy
from typing import Any, Dict

import torch


DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "model": {
        "name": None,
        "params": {},
    },
    "data": {
        "dataset": "LOLv1Dataset",
        "root_dir": None,
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,
        "drop_last": False,
        "train_split": "train",
        "val_split": "_test",
        "return_filename": True,
        "params": {},
        "train_params": {},
        "val_params": {},
    },
    "loss": {
        "name": None,
        "params": {},
        "output_index": None,
        "output_key": None,
    },
    "optimizer": {
        "name": "adam",
        "lr": 1e-4,
        "params": {},
    },
    "scheduler": {
        "name": None,
        "params": {},
    },
    "train": {
        "epochs": 100,
        "output_dir": None,
        "save_every": 1,
        "validate_every": 1,
        "log_every": 10,
        "grad_clip": None,
        "amp": False,
        "resume": None,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}


def get_default_train_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_TRAIN_CONFIG)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base
