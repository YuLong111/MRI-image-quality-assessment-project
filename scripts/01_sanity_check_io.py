from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def save_mid_slices(vol: np.ndarray, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    x, y, z = (s // 2 for s in vol.shape)

    # axial: z fixed
    ax = vol[:, :, z]
    # coronal: y fixed
    cor = vol[:, y, :]
    # sagittal: x fixed
    sag = vol[x, :, :]

    for name, img in [("axial", ax), ("coronal", cor), ("sagittal", sag)]:
        plt.figure(figsize=(5, 5))
        plt.imshow(img.T, cmap="gray", origin="lower")
        plt.title(f"{stem} - {name} (mid-slice)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(outdir / f"{stem}_{name}.png", dpi=150)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nifti", required=True, help="Path to a .nii or .nii.gz file")
    p.add_argument("--outdir", default="runs/_debug", help="Where to save PNG outputs")
    args = p.parse_args()

    path = Path(args.nifti)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)

    zooms = img.header.get_zooms()[:3]
    print("File:", path)
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Voxel spacing (zooms):", zooms)
    print("Affine:\n", img.affine)
    print("Intensity min/max/mean:", float(np.min(data)), float(np.max(data)), float(np.mean(data)))

    stem = path.name.replace(".nii.gz", "").replace(".nii", "")
    save_mid_slices(data, Path(args.outdir), stem)
    print(f"Saved mid-slice PNGs to: {args.outdir}")


if __name__ == "__main__":
    main()
