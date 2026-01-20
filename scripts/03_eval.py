from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from monai.data import DataLoader

from pqmonai.datasets import make_dataset
from pqmonai.infer import predict_binary
from pqmonai.models.cnn3d import DenseNet1213DClassifier
from pqmonai.transforms import get_val_transforms
from pqmonai.utils import resolve_config


def load_items(json_path: str | Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--datalist", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    cfg = resolve_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = DenseNet1213DClassifier(in_channels=int(cfg["model"].get("in_channels", 1)), num_classes=2)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    items = load_items(args.datalist)
    t = get_val_transforms(cfg)
    ds = make_dataset(items, t, cfg["data"]["cache"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label_bin", "pred", "prob_pos"])
        for batch in loader:
            pred = predict_binary(model, batch, device)
            case_id = batch.get("id", ["unknown"])[0]
            y = int(batch.get("label_bin", torch.tensor([0])).item())
            w.writerow([case_id, y, int(pred["pred"].item()), float(pred["prob_pos"].item())])

    print(f"Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
