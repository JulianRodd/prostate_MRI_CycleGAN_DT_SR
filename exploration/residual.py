import os
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


def calculate_residual(
    img1_path, img2_path, output_dir, abs_diff=False, normalize=False
):
    """
    Calculate the residual between two 3D NIfTI images and generate visualizations.

    Parameters:
    -----------
    img1_path : str
        Path to the first NIfTI image (.nii.gz)
    img2_path : str
        Path to the second NIfTI image (.nii.gz)
    output_dir : str
        Directory to save outputs
    abs_diff : bool, optional
        If True, computes absolute difference
    normalize : bool, optional
        If True, normalizes the residual
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img1_nib = nib.load(img1_path)
    img2_nib = nib.load(img2_path)

    img1_data = img1_nib.get_fdata()
    img2_data = img2_nib.get_fdata()

    if img1_data.shape != img2_data.shape:
        raise ValueError(
            f"Image dimensions do not match: {img1_data.shape} vs {img2_data.shape}"
        )

    if abs_diff:
        residual = np.abs(img1_data - img2_data)
        prefix = "abs_diff"
    else:
        residual = img1_data - img2_data
        prefix = "diff"

    img1_name = Path(img1_path).stem.split(".")[0]
    img2_name = Path(img2_path).stem.split(".")[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{prefix}_{img1_name}_vs_{img2_name}_{timestamp}"

    nii_path = os.path.join(output_dir, f"{base_filename}.nii.gz")
    residual_img = nib.Nifti1Image(residual, img1_nib.affine, img1_nib.header)
    nib.save(residual_img, nii_path)

    metrics = {}
    metrics["mean_residual"] = float(np.mean(residual))
    metrics["max_residual"] = float(np.max(residual))
    metrics["min_residual"] = float(np.min(residual))

    if not abs_diff:
        mse = np.mean((img1_data - img2_data) ** 2)
        metrics["rmse"] = float(np.sqrt(mse))

    metrics_path = os.path.join(output_dir, f"{base_filename}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Residual Analysis: {img1_name} vs {img2_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Difference type: {'Absolute' if abs_diff else 'Signed'}\n\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

    z_mid = img1_data.shape[2] // 2
    y_mid = img1_data.shape[1] // 2
    x_mid = img1_data.shape[0] // 2

    if abs_diff:
        cmap = "hot"
        vmin = 0
        vmax = np.percentile(residual, 99)
    else:
        cmap = "coolwarm"
        abs_max = max(abs(np.percentile(residual, 1)), abs(np.percentile(residual, 99)))
        vmin = -abs_max
        vmax = abs_max

    def normalize_for_display(img_slice):
        p1, p99 = np.percentile(img_slice, (1, 99))
        img_norm = np.clip(img_slice, p1, p99)
        img_norm = (img_norm - p1) / (p99 - p1)
        return img_norm

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(normalize_for_display(img1_data[:, :, z_mid]), cmap="gray")
    axes[0, 0].set_title(f"Image 1: {Path(img1_path).name}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(normalize_for_display(img2_data[:, :, z_mid]), cmap="gray")
    axes[0, 1].set_title(f"Image 2: {Path(img2_path).name}")
    axes[0, 1].axis("off")

    im = axes[1, 0].imshow(residual[:, :, z_mid], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Residual Map (Axial)")
    axes[1, 0].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    axes[1, 1].hist(residual.flatten(), bins=50, color="darkblue", alpha=0.7)
    axes[1, 1].set_title("Residual Histogram")
    axes[1, 1].set_xlabel("Residual Value")
    axes[1, 1].set_ylabel("Frequency")

    metrics_text = "Metrics:\n"
    for key, value in metrics.items():
        metrics_text += f"{key}: {value:.4f}\n"
    axes[1, 1].text(
        0.05,
        0.6,
        metrics_text,
        transform=axes[1, 1].transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.suptitle(f"Residual Analysis: {img1_name} vs {img2_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    viz_path = os.path.join(output_dir, f"{base_filename}_viz.png")
    plt.savefig(viz_path, dpi=200)
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    z_slices = [
        img1_data.shape[2] // 4,
        img1_data.shape[2] // 2,
        3 * img1_data.shape[2] // 4,
    ]

    for i, z in enumerate(z_slices):
        axes[0, i].imshow(normalize_for_display(img1_data[:, :, z]), cmap="gray")
        axes[0, i].set_title(f"Image 1: Slice {z}")
        axes[0, i].axis("off")

        axes[1, i].imshow(normalize_for_display(img2_data[:, :, z]), cmap="gray")
        axes[1, i].set_title(f"Image 2: Slice {z}")
        axes[1, i].axis("off")

        im = axes[2, i].imshow(residual[:, :, z], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2, i].set_title(f"Residual: Slice {z}")
        axes[2, i].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.suptitle(f"Multi-slice Comparison: {img1_name} vs {img2_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    multi_viz_path = os.path.join(output_dir, f"{base_filename}_multi_slice.png")
    plt.savefig(multi_viz_path, dpi=200)
    plt.close()

    print(f"Analysis complete. Files saved to {output_dir}:")
    print(f"  - NIfTI residual: {Path(nii_path).name}")
    print(f"  - Metrics: {Path(metrics_path).name}")
    print(f"  - Visualization: {Path(viz_path).name}")
    print(f"  - Multi-slice view: {Path(multi_viz_path).name}")

    return metrics
