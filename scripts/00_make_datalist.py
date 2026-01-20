from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pqmonai.datalist import build_datalist_from_xlsx


def split_items(items, seed: int = 0, train: float = 0.8, val: float = 0.1):
    """Simple deterministic split: train/val/test.

    - test gets the remainder after train+val.
    - For small datasets, consider cross-validation instead.
    """
    if not (0 < train < 1) or not (0 <= val < 1) or train + val >= 1:
        raise ValueError("Require 0<train<1, 0<=val<1, and train+val<1")

    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)

    n = len(items)
    n_train = int(round(train * n))
    n_val = int(round(val * n))

    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val :]
    return train_items, val_items, test_items


def write_json(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Build MONAI datalist JSONs from your labels spreadsheet (e.g., sampled_file_list.xlsx) "
            "and split into train/val/test JSON files."
        )
    )
    p.add_argument("--xlsx", required=True, help="Path to sampled_file_list.xlsx or labels table")
    p.add_argument("--image_dir", required=True, help="Directory containing NIfTI images")
    p.add_argument("--out_dir", default="data/splits", help="Output directory for datalist_*.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    args = p.parse_args()

    items = build_datalist_from_xlsx(args.xlsx, args.image_dir, out_json=Path(args.out_dir) / "_tmp_all.json")
    print(f"Matched {len(items)} images from spreadsheet.")

    if len(items) == 0:
        print("No items were matched. Check your filenames and --image_dir.")
        return

    train_items, val_items, test_items = split_items(items, seed=args.seed, train=args.train, val=args.val)

    out_dir = Path(args.out_dir)
    write_json(out_dir / "datalist_train.json", train_items)
    write_json(out_dir / "datalist_val.json", val_items)
    write_json(out_dir / "datalist_test.json", test_items)

    # remove temp file
    try:
        (out_dir / "_tmp_all.json").unlink()
    except Exception:
        pass

    print(f"Wrote train/val/test: {len(train_items)}/{len(val_items)}/{len(test_items)} -> {out_dir}")


if __name__ == "__main__":
    main()
