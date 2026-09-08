"""YAML-in, dot-accessible-object-out. Nothing about an experiment should
live in Python code — it lives in a `configs/*.yaml` file, and this module
is the only place that reads YAML.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin dict wrapper giving `cfg.model.hidden_dim`-style access while
    still behaving like a dict (`cfg['model']['hidden_dim']`,
    `dict(cfg)`, `cfg.get(...)`) so it drops into existing dict-based code.
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to a plain dict — used when we save the
        config alongside a run's results, so results are never separated
        from the exact settings that produced them (see README rule 3).
        """
        out = {}
        for key, value in self._data.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return copy.deepcopy(out)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a `Config` object.

    Args:
        path: path to a `.yaml` file, typically under `configs/`.

    Returns:
        Config instance with dot-access to every key.

    Raises:
        FileNotFoundError: if `path` doesn't exist — fails loudly rather
            than silently falling back to defaults, since a mistyped
            `--config` flag should not silently run with wrong settings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Available configs: "
            f"{sorted(p.name for p in path.parent.glob('*.yaml'))}"
            if path.parent.exists()
            else f"Config not found: {path}"
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data)


def save_config(cfg: Config, path: str | Path) -> None:
    """Write a config back out to YAML — used to snapshot the exact config
    a run used, into that run's results directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
