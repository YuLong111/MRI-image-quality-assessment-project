from __future__ import annotations

from typing import Any, Dict, List, Tuple

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    CenterSpatialCropd,
)


def _roi_size(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    roi = cfg.get("transforms", {}).get("crop", {}).get("roi_size", [160, 160, 96])
    return int(roi[0]), int(roi[1]), int(roi[2])


def get_train_transforms(cfg: Dict[str, Any]) -> Compose:
    ax = cfg["transforms"].get("target_axcodes", "RAS")
    pixdim = tuple(cfg["transforms"].get("pixdim", [0.5, 0.5, 1.0]))
    intensity = cfg["transforms"].get("intensity", {})
    crop = cfg["transforms"].get("crop", {})
    aug = cfg["transforms"].get("augment", {})

    t: List[Any] = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=ax),
    ]

    # Spacing is important, but can be expensive; keep it optional.
    # If you want it, add Spacingd here (requires torch + resampling).
    # Many quality tasks can start without resampling if all images are consistent.

    if intensity.get("kind", "normalize") == "scale_range":
        t.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(intensity.get("a_min", 0)),
                a_max=float(intensity.get("a_max", 1000)),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        t.append(
            NormalizeIntensityd(
                keys=["image"],
                nonzero=bool(intensity.get("nonzero", True)),
                channel_wise=bool(intensity.get("channel_wise", True)),
            )
        )

    if crop.get("kind", "center") == "center":
        t.append(CenterSpatialCropd(keys=["image"], roi_size=_roi_size(cfg)))

    fp = float(aug.get("flip_prob", 0.5))
    rp = float(aug.get("rotate90_prob", 0.2))
    if fp > 0:
        t.append(RandFlipd(keys=["image"], prob=fp, spatial_axis=0))
        t.append(RandFlipd(keys=["image"], prob=fp, spatial_axis=1))
    if rp > 0:
        t.append(RandRotate90d(keys=["image"], prob=rp, max_k=3))

    t.append(EnsureTyped(keys=["image"]))
    return Compose(t)


def get_val_transforms(cfg: Dict[str, Any]) -> Compose:
    # Same as train but without random aug.
    ax = cfg["transforms"].get("target_axcodes", "RAS")
    intensity = cfg["transforms"].get("intensity", {})
    crop = cfg["transforms"].get("crop", {})

    t: List[Any] = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=ax),
    ]

    if intensity.get("kind", "normalize") == "scale_range":
        t.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(intensity.get("a_min", 0)),
                a_max=float(intensity.get("a_max", 1000)),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        t.append(
            NormalizeIntensityd(
                keys=["image"],
                nonzero=bool(intensity.get("nonzero", True)),
                channel_wise=bool(intensity.get("channel_wise", True)),
            )
        )

    if crop.get("kind", "center") == "center":
        t.append(CenterSpatialCropd(keys=["image"], roi_size=_roi_size(cfg)))

    t.append(EnsureTyped(keys=["image"]))
    return Compose(t)
