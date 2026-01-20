from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dicts (used for config inheritance)."""
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(cfg_path: str | os.PathLike) -> Dict[str, Any]:
    """Load config YAML with optional `inherit:` key."""
    cfg_path = Path(cfg_path)
    cfg = load_yaml(cfg_path)

    inherit = cfg.get("inherit")
    if inherit:
        base = load_yaml(cfg_path.parent.parent / inherit if str(inherit).startswith("configs/") else cfg_path.parent / inherit)
        cfg = deep_update(base, {k: v for k, v in cfg.items() if k != "inherit"})
    return cfg
