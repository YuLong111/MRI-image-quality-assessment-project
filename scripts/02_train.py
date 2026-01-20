from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from monai.data import DataLoader

from pqmonai.datasets import make_dataset
from pqmonai.engine import train_one_epoch_binary, eval_binary
from pqmonai.models.cnn3d import DenseNet1213DClassifier
from pqmonai.transforms import get_train_transforms, get_val_transforms
from pqmonai.utils import resolve_config, seed_all


def _device_from_cfg(cfg):
    d = cfg.get("device", "auto")
    if d == "cuda":
        return torch.device("cuda")
    if d == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_items(json_path: str | Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = resolve_config(args.config)
    seed = int(cfg.get("seed", 0))
    seed_all(seed)

    device = _device_from_cfg(cfg)
    print("Device:", device)

    run_dir = Path(cfg["logging"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    # save resolved config for reproducibility
    import yaml

    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_items = load_items(cfg["paths"]["datalist_train"])
    val_items = load_items(cfg["paths"]["datalist_val"]) if Path(cfg["paths"]["datalist_val"]).exists() else []

    # Expect label_bin present.
    for it in train_items[:3]:
        if "label_bin" not in it:
            raise ValueError("Datalist items must include 'label_bin'. Run scripts/00_make_datalist.py first.")

    t_train = get_train_transforms(cfg)
    t_val = get_val_transforms(cfg)

    ds_train = make_dataset(train_items, t_train, cfg["data"]["cache"])
    ds_val = make_dataset(val_items, t_val, cfg["data"]["cache"]) if val_items else None

    train_loader = DataLoader(ds_train, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=int(cfg["training"]["num_workers"]))
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=int(cfg["training"]["num_workers"])) if ds_val else None

    model = DenseNet1213DClassifier(
        in_channels=int(cfg["model"].get("in_channels", 1)),
        num_classes=int(cfg["model"].get("num_classes", 2)),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    best_auc = -1.0
    max_epochs = int(cfg["training"]["max_epochs"])

    for epoch in range(1, max_epochs + 1):
        tr = train_one_epoch_binary(model, train_loader, optimizer, device)
        print(f"Epoch {epoch}/{max_epochs} | train loss={tr['loss']:.4f} acc={tr['acc']:.3f} f1={tr['f1']:.3f}")

        if val_loader and (epoch % int(cfg["training"]["val_every"]) == 0):
            va = eval_binary(model, val_loader, device)
            print(f"                val acc={va['acc']:.3f} f1={va['f1']:.3f} auc={va['auc']:.3f}")

            auc = va["auc"]
            if auc == auc and auc > best_auc:  # auc==auc filters nan
                best_auc = auc
                ckpt = {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "val_auc": best_auc,
                }
                torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
                print("Saved best checkpoint.")

    # save last
    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": max_epochs}, run_dir / "checkpoints" / "last.pt")
    print("Saved last checkpoint.")


if __name__ == "__main__":
    main()
