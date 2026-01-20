from __future__ import annotations

from typing import Any, Dict, List

from monai.data import CacheDataset, PersistentDataset, Dataset


def make_dataset(
    items: List[Dict[str, Any]],
    transform,
    cache_cfg: Dict[str, Any],
):
    kind = cache_cfg.get("kind", "cache")
    if kind == "none":
        return Dataset(data=items, transform=transform)
    if kind == "persistent":
        cache_dir = cache_cfg.get("cache_dir", "./data/processed/_persistent_cache")
        return PersistentDataset(data=items, transform=transform, cache_dir=cache_dir)
    # default cache
    return CacheDataset(
        data=items,
        transform=transform,
        cache_rate=float(cache_cfg.get("cache_rate", 0.3)),
        num_workers=int(cache_cfg.get("num_workers", 2)),
    )
