from pathlib import Path
from typing import Dict, Any

import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt

from preprocessing.preprocessing_actions.io import save_intermediate, save_comparison_visualization


def plot_mri_slices(
        images,
        titles,
        output_path,
        slice_idx=None,
        cmap="viridis",
        share_colorbar=False,
        vmin=None,
        vmax=None,
):
    """Create a diagnostic plot that safely handles both 2D and 3D images."""
    try:
        num_images = len(images)
        if num_images == 0:
            return

        # Determine slice to use and ensure images are 2D
        processed_images = []
        for img in images:
            if img is None:
                processed_images.append(np.zeros((10, 10)))
                continue

            if not isinstance(img, np.ndarray):
                img = np.array(img)

            if img.ndim == 3:
                if slice_idx is None:
                    slice_idx = img.shape[2] // 2
                slice_idx = min(slice_idx, img.shape[2] - 1)
                processed_images.append(img[:, :, slice_idx])
            elif img.ndim == 2:
                processed_images.append(img)
            else:
                processed_images.append(np.zeros((10, 10)))

        # Determine common colorbar range if needed
        if share_colorbar and (vmin is None or vmax is None):
            vmin = min(np.min(img) for img in processed_images if img.size > 0)
            vmax = max(np.max(img) for img in processed_images if img.size > 0)

        # Create figure
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        if num_images == 1:
            axes = [axes]

        # Plot each image
        for i, (img, title, ax) in enumerate(zip(processed_images, titles, axes)):
            im = ax.imshow(
                img.T,
                cmap=cmap,
                origin="lower",
                vmin=vmin if share_colorbar else None,
                vmax=vmax if share_colorbar else None,
            )
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Warning: Diagnostic plot failed: {e}")
        plt.close("all")


def plot_histograms(fixed_np, moving_np, title_prefix, output_path, logger=None):
    """Plot histograms and intensity distributions for image comparison."""
    try:
        plt.figure(figsize=(10, 4))

        # Histogram comparison
        plt.subplot(1, 2, 1)
        plt.hist(fixed_np.flatten(), bins=50, alpha=0.5, label="Fixed")
        plt.hist(moving_np.flatten(), bins=50, alpha=0.5, label="Moving")
        plt.legend()
        plt.title(f"{title_prefix} Histograms")

        # Intensity distribution
        plt.subplot(1, 2, 2)
        fixed_masked = fixed_np[fixed_np > 0]
        moving_masked = moving_np[moving_np > 0]

        # Sample points if too many
        if len(fixed_masked) > 1000:
            fixed_indices = np.random.choice(len(fixed_masked), 1000, replace=False)
            fixed_masked = fixed_masked[fixed_indices]
        if len(moving_masked) > 1000:
            moving_indices = np.random.choice(len(moving_masked), 1000, replace=False)
            moving_masked = moving_masked[moving_indices]

        plt.scatter(fixed_masked, fixed_masked, alpha=0.3, label="Fixed (identity)")
        plt.scatter(moving_masked, np.zeros_like(moving_masked) if "Before" in title_prefix
        else moving_masked, alpha=0.3, label="Moving")
        plt.legend()
        plt.title(f"Intensity Distribution {title_prefix}")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save histogram visualization: {e}")


def find_best_content_slice(data):
    """Find the slice with the most content."""
    try:
        if data.ndim < 3:
            return 0
        non_zero_count = [np.sum(data[:, :, i] > 0) for i in range(data.shape[2])]
        return np.argmax(non_zero_count)
    except Exception:
        return 0


def save_debug_visualizations(
        images: Dict[str, sitk.Image],
        stage: str,
        output_dir: Path,
        pair_id: str,
        logger: Any,
        save_comparison: bool = True,
        make_nifti: bool = False,  # New parameter for saving NIfTI files
) -> None:
    """
    Save debug visualizations for a given processing stage.

    Args:
        images: Dictionary of named images to visualize
        stage: Processing stage identifier
        output_dir: Base output directory
        pair_id: ID of the image pair being processed
        logger: Logger instance
        save_comparison: Whether to save a comparison visualization
        make_nifti: Whether to also save NIfTI (.nii.gz) files
    """
    temp_dir = output_dir / "temp" / pair_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Save individual images
    for name, image in images.items():
        # Save PNG visualization
        save_intermediate(
            image,
            temp_dir / f"{stage}_{name}.png",
            save_middle_slice_only=True,
        )

        # Also save NIfTI file if requested
        if make_nifti:
            try:
                nifti_path = temp_dir / f"{stage}_{name}.nii.gz"
                sitk.WriteImage(image, str(nifti_path), True)  # True for compressed
                if logger:
                    logger.debug(f"Saved NIfTI debug file: {nifti_path}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save NIfTI debug file {name}: {e}")

    # Save comparison if requested
    if save_comparison and len(images) > 1:
        image_tuples = [(img, name) for name, img in images.items()]
        save_comparison_visualization(
            image_tuples,
            temp_dir / f"{stage}_comparison.png",
            logger=logger,
        )
