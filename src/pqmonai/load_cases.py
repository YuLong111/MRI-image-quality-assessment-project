import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header.get_zooms()

def show_mid_slices(vol, title):
    z = vol.shape[2] // 2
    plt.imshow(vol[:, :, z], cmap="gray")
    plt.title(title)
    plt.axis("off")

def main():
    # EDIT these paths to one example from each folder
    dwi_path  = r"D:\1\杂物\学校\ucl\year3\project\project\OneDrive_1_2026-1-15\dwi\Patient001061633_study_0.nii.gz"
    t2_path   = r"D:\1\杂物\学校\ucl\year3\project\project\OneDrive_1_2026-1-15\t2\Patient001061633_study_0.nii.gz"
    mask_path = r"D:\1\杂物\学校\ucl\year3\project\project\OneDrive_1_2026-1-15\prostate_mask\Patient001061633_study_0.nii.gz"

    dwi, dwi_aff, dwi_zooms = load_nifti(dwi_path)
    t2,  t2_aff,  t2_zooms  = load_nifti(t2_path)
    msk, msk_aff, msk_zooms = load_nifti(mask_path)

    print("DWI shape/zooms:", dwi.shape, dwi_zooms)
    print("T2  shape/zooms:", t2.shape,  t2_zooms)
    print("MSK shape/zooms:", msk.shape, msk_zooms)

    # crude checks
    print("Affine close (DWI vs T2):", np.allclose(dwi_aff, t2_aff, atol=1e-3))
    print("Affine close (MSK vs T2):", np.allclose(msk_aff, t2_aff, atol=1e-3))

    # visual check: overlay mask on T2 mid-slice
    z = t2.shape[2] // 2
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); show_mid_slices(dwi, "DWI mid-slice")
    plt.subplot(1,3,2); show_mid_slices(t2,  "T2 mid-slice")
    plt.subplot(1,3,3)
    plt.imshow(t2[:, :, z], cmap="gray")
    plt.imshow(msk[:, :, z] > 0, alpha=0.3)  # mask overlay
    plt.title("T2 + mask overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
