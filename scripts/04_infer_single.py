from __future__ import annotations

import argparse
from pathlib import Path

import torch
from monai.transforms import Compose

from pqmonai.models.cnn3d import DenseNet1213DClassifier
from pqmonai.transforms import get_val_transforms
from pqmonai.utils import resolve_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--nifti", required=True)
    args = p.parse_args()

    cfg = resolve_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = DenseNet1213DClassifier(in_channels=int(cfg["model"].get("in_channels", 1)), num_classes=2)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Reuse val transforms; apply to dict with 'image'
    t = get_val_transforms(cfg)
    sample = {"image": str(Path(args.nifti).resolve())}
    sample = t(sample)

    x = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        print("Pred class:", int(torch.argmax(probs).item()))
        print("Prob low, high:", float(probs[0].item()), float(probs[1].item()))


if __name__ == "__main__":
    main()
