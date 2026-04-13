from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "PyYAML is required to load config.yaml. Please install it with `pip install PyYAML`."
    ) from exc


CLI_TO_CONFIG_PATH = {
    "cuda": "runtime.cuda",
    "dataset_dir": "paths.dataset_dir",
    "data_name": "experiment.data_name",
    "model_name": "experiment.model_name",
    "batch_size": "train.batch_size",
    "lr": "optimizer.lr",
    "max_epoch": "train.max_epoch",
    "seed": "runtime.seed",
    "num_workers": "runtime.num_workers",
    "mix_test": "experiment.mix_test",
    "learnable_emb": "experiment.learnable_emb",
    "manual": "experiment.manual",
    "pri": "experiment.pri",
}


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {config_path}")
    return config


def get_nested(config: Dict[str, Any], dotted_path: str) -> Any:
    current = config
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing config path: {dotted_path}")
        current = current[key]
    return current


def set_nested(config: Dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def apply_cli_overrides(
    config: Dict[str, Any], cli_args: Any
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    merged = copy.deepcopy(config)
    allowed_paths = set(merged.get("cli_overrides", []))
    applied: Dict[str, Any] = {}

    for arg_name, config_path in CLI_TO_CONFIG_PATH.items():
        value = getattr(cli_args, arg_name, None)
        if value is None:
            continue
        if config_path not in allowed_paths:
            raise ValueError(
                f"CLI override `{arg_name}` is not allowed because `{config_path}` is not listed in `cli_overrides`."
            )
        set_nested(merged, config_path, value)
        applied[config_path] = value

    return merged, applied


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value
