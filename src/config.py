"""Configuration loading and global seeding.

Every script in this project takes a YAML config path. Nothing is hardcoded.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


class Config(dict):
    """A dict that also supports attribute access and dotted lookup."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Look up a nested key such as 'retrieval.dense.model'."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path) -> Config:
    """Read a YAML config file into a Config object."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    return Config(raw)


def seed_everything(seed: int) -> int:
    """Seed python, numpy and torch. Returns the seed so callers can log it."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def device() -> str:
    """Return the torch device string this machine should use."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def report_vram(tag: str) -> float:
    """Print and return peak allocated VRAM in GB. Required after every GPU run."""
    if not torch.cuda.is_available():
        print(f"[vram] {tag}: no CUDA device")
        return 0.0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[vram] {tag}: peak {peak_gb:.2f} GB / {total_gb:.2f} GB total")
    return peak_gb
