import argparse
import os

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from skimage import exposure, filters, transform

from metrics.prostateMRIFeatureMetrics import ProstateMRIFeatureMetrics


def find_best_patch(image, patch_size=100):
    """
    Find the most informative patch in the image (with highest variance/intensity).

    Args:
        image: Input image (2D array)
        patch_size: Size of the square patch to extract

    Returns:
        start_row, start_col: Starting coordinates of the optimal patch
    """
    max_row = image.shape[0] - patch_size
    max_col = image.shape[1] - patch_size

    if max_row <= 0 or max_col <= 0:
        center_row = image.shape[0] // 2
        center_col = image.shape[1] // 2
        start_row = max(0, center_row - patch_size // 2)
        start_col = max(0, center_col - patch_size // 2)
        return start_row, start_col

    best_score = -1
    best_pos = (0, 0)

    for row in range(0, max_row, 10):
        for col in range(0, max_col, 10):
            patch = image[row : row + patch_size, col : col + patch_size]
            score = np.mean(patch) + np.std(patch)

            if score > best_score:
                best_score = score
                best_pos = (row, col)

    return best_pos


def create_gaussian_taper(shape, sigma_factor=0.3):
    """
    Create a Gaussian tapering window to reduce edge effects.

    Args:
        shape: (height, width) of the window
        sigma_factor: Controls the width of the Gaussian relative to image size

    Returns:
        window: 2D Gaussian window with given shape
    """
    h, w = shape
    center_y, center_x = h // 2, w // 2

    y, x = np.ogrid[:h, :w]

    sigma_y = h * sigma_factor
    sigma_x = w * sigma_factor

    window = np.exp(
        -(
            (y - center_y) ** 2 / (2 * sigma_y**2)
            + (x - center_x) ** 2 / (2 * sigma_x**2)
        )
    )

    return window


def normalize_feature_map(
    feature_map, foreground_mask, min_percentile=2, max_percentile=98, smooth_factor=0.5
):
    """
    Enhanced normalization function that ensures smooth feature maps without holes.

    Args:
        feature_map: Input feature map to normalize
        foreground_mask: Binary mask of foreground regions
        min_percentile: Lower percentile for normalization (default: 2)
        max_percentile: Upper percentile for normalization (default: 98)
        smooth_factor: Factor controlling additional Gaussian smoothing to fill holes (default: 0.5)

    Returns:
        normalized_map: Normalized feature map with values in [0, 1] range, with holes filled
    """
    from scipy import ndimage

    # Apply initial smoothing to reduce potential holes
    feature_map = ndimage.gaussian_filter(feature_map, sigma=smooth_factor)

    # Extract only foreground values for percentile calculation
    foreground_values = feature_map[foreground_mask > 0]

    if len(foreground_values) > 0:
        # Calculate percentiles from foreground values only
        p_low = np.percentile(foreground_values, min_percentile)
        p_high = np.percentile(foreground_values, max_percentile)

        # Ensure we don't divide by zero
        if p_high > p_low:
            # Apply percentile-based normalization
            normalized_map = np.clip((feature_map - p_low) / (p_high - p_low), 0, 1)
        else:
            # Fallback if percentiles are too close
            normalized_map = (feature_map - feature_map.min()) / (
                feature_map.max() - feature_map.min() + 1e-8
            )
    else:
        # Fallback if no foreground values
        normalized_map = np.zeros_like(feature_map)

    # Apply the foreground mask after normalization
    normalized_map = normalized_map * foreground_mask

    # Fill small holes in the feature map using morphological closing
    if np.any(normalized_map > 0):
        # Create a binary threshold of the feature map to identify potentially active regions
        active_regions = normalized_map > 0.1  # Adjust threshold as needed

        # Apply morphological closing to fill small holes
        filled_regions = ndimage.binary_closing(
            active_regions, structure=np.ones((3, 3))
        )

        # Identify the holes that were filled
        holes = filled_regions & ~active_regions

        # For the identified holes, assign interpolated values
        if np.any(holes):
            # Use distance transform to assign interpolated values to holes
            dist = ndimage.distance_transform_edt(holes)
            max_dist = dist.max()
            if max_dist > 0:
                hole_values = normalized_map.copy()

                # Get a dilated version of active regions to capture boundary values
                dilated = ndimage.binary_dilation(
                    active_regions, structure=np.ones((5, 5))
                )
                boundary = dilated & ~active_regions & ~holes

                # Calculate average value of nearby active regions
                avg_value = (
                    np.mean(normalized_map[boundary]) if np.any(boundary) else 0.3
                )

                # Fill holes with values that decrease with distance from boundary
                # This creates a smooth interpolation within holes
                normalized_map[holes] = avg_value * (
                    1 - dist[holes] / (max_dist * 1.5 + 1e-8)
                )

    return normalized_map


def create_foreground_mask(image, bg_threshold=0.05):
    """
    Create a binary mask of the foreground (non-background) regions with improved
    hole-filling for more consistent masks.

    Args:
        image: Input image (2D array)
        bg_threshold: Threshold value below which pixels are considered background

    Returns:
        mask: Binary mask where 1 indicates foreground and 0 indicates background
    """
    from scipy import ndimage

    # Create initial binary mask
    binary_mask = image > bg_threshold

    # Apply morphological operations to clean up the mask
    # Fill holes more aggressively
    binary_mask = ndimage.binary_fill_holes(binary_mask)

    # Apply morphological closing to merge nearby foreground regions
    struct = ndimage.generate_binary_structure(2, 2)  # More aggressive connectivity
    binary_mask = ndimage.binary_closing(binary_mask, structure=struct, iterations=2)

    # Fill holes again after closing
    binary_mask = ndimage.binary_fill_holes(binary_mask)

    # Remove small isolated objects
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features > 1:
        sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_features + 1))
        mask_size = sizes < 100  # Adjust threshold as needed
        remove_pixel = mask_size[labeled_mask - 1]
        remove_pixel[labeled_mask == 0] = False
        binary_mask[remove_pixel] = 0

    return binary_mask


def visualize_prostate_features(
    input_file,
    output_prefix,
    slice_idx=None,
    patch_size=200,
    colormap="viridis",
    min_opacity=0.05,
    max_opacity=0.8,
    threshold_pct=0,
    apply_taper=True,
    bg_threshold=0.1,
    balance_levels=True,
    taper_sigmas=None,
    start_row=None,
    start_col=None,
):
    """
    Extracts and visualizes features from a patch of a prostate MRI with
    GradCAM-style overlays where opacity varies with feature intensity.
    Each feature level uses its own specific edge taper sigma.

    Args:
        input_file: Path to the input prostate MRI volume (NIfTI format)
        output_prefix: Prefix for output visualization files
        slice_idx: Index of the slice to visualize. If None, uses the middle slice
        patch_size: Size of the square patch to extract
        colormap: Colormap to use for visualization
        min_opacity: Minimum opacity for low-activation regions (0.0-1.0)
        max_opacity: Maximum opacity for high-activation regions (0.0-1.0)
        threshold_pct: Percentile threshold below which activations are not shown (0.0-1.0)
        apply_taper: Whether to apply Gaussian tapering to reduce edge effects
        bg_threshold: Threshold below which pixels are considered background
        balance_levels: Whether to independently normalize each feature level for balanced visualization
        taper_sigmas: Dictionary mapping feature level names to sigma values for tapering
        start_row: Starting row for patch extraction (if None, automatically determined)
        start_col: Starting column for patch extraction (if None, automatically determined)
    """

    if taper_sigmas is None:
        taper_sigmas = {
            "lowlevel": 0.5,
            "midlevel": 0.4,
            "highlevel": 0.25,
        }

    print(f"Loading prostate MRI from {input_file}")

    nii_img = nib.load(input_file)
    volume = nii_img.get_fdata()

    has_negatives = np.any(volume < 0)

    if has_negatives:
        print("Found negative values (likely background)")
        background_mask = volume < 0
        volume[background_mask] = 0
    else:
        background_mask = None

    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

    if slice_idx is None:
        slice_idx = volume.shape[2] // 2

    print(f"Processing slice {slice_idx} of {volume.shape[2]}")

    original_slice = volume[:, :, slice_idx]

    if start_row is None or start_col is None:
        start_row, start_col = find_best_patch(original_slice, patch_size)

    patch = original_slice[
        start_row : start_row + patch_size, start_col : start_col + patch_size
    ]
    print(
        f"Extracted {patch_size}x{patch_size} patch at position ({start_row}, {start_col})"
    )

    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        pad_rows = max(0, patch_size - patch.shape[0])
        pad_cols = max(0, patch_size - patch.shape[1])
        patch = np.pad(patch, ((0, pad_rows), (0, pad_cols)), mode="constant")
        print(f"Padded patch to {patch_size}x{patch_size}")

    if background_mask is not None:
        bg_mask_slice = background_mask[:, :, slice_idx]
        patch_bg_mask = bg_mask_slice[
            start_row : start_row + patch_size, start_col : start_col + patch_size
        ]
        if hasattr(locals(), "pad_rows") and hasattr(locals(), "pad_cols"):
            if pad_rows > 0 or pad_cols > 0:
                patch_bg_mask = np.pad(
                    patch_bg_mask,
                    ((0, pad_rows), (0, pad_cols)),
                    mode="constant",
                    constant_values=True,
                )
        foreground_mask = ~patch_bg_mask
    else:
        foreground_mask = create_foreground_mask(patch, bg_threshold)

    print(f"Created foreground mask to prevent signal overlay on background")

    taper_windows = {}
    if apply_taper:
        for level, sigma in taper_sigmas.items():
            taper_windows[level] = create_gaussian_taper(
                patch.shape, sigma_factor=sigma
            )
            print(
                f"Created Gaussian taper window for {level} features with sigma={sigma}"
            )

    tensor_patch = (
        torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    feature_extractor = ProstateMRIFeatureMetrics(
        device=device,
        use_layers=[
            "model.submodule.encoder1",
            "model.submodule.encoder3",
            "model.submodule.encoder5",
        ],
        layer_weights={
            "model.submodule.encoder1": 0.33,
            "model.submodule.encoder3": 0.33,
            "model.submodule.encoder5": 0.34,
        },
    )

    with torch.no_grad():
        features = feature_extractor.extract_features(tensor_patch)

    plt.style.use("default")
    mpl.rcParams.update(
        {
            "axes.linewidth": 0.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.transparent": True,
        }
    )

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(original_slice, cmap="gray")
    rect = patches.Rectangle(
        (start_col, start_row),
        patch_size,
        patch_size,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    plt.gca().add_patch(rect)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"{output_prefix}_full_slice.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(patch, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"{output_prefix}_patch.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    feature_configs = [
        {
            "layer": "model.submodule.encoder1",
            "name": "lowlevel",
            "sigma": 0.5,
            "description": "Low-level features (texture, edges)",
        },
        {
            "layer": "model.submodule.encoder3",
            "name": "midlevel",
            "sigma": 1.0,
            "description": "Mid-level features (anatomical boundaries)",
        },
        {
            "layer": "model.submodule.encoder5",
            "name": "highlevel",
            "sigma": 1.5,
            "description": "High-level features (zonal architecture)",
        },
    ]

    for config in feature_configs:
        print(f"Processing {config['description']}...")

        feature_map = features[config["layer"]].squeeze().cpu().numpy()

        if len(feature_map.shape) == 3:
            vis_map = np.mean(feature_map, axis=0)
        else:
            vis_map = feature_map

        vis_map = exposure.equalize_adapthist(vis_map, clip_limit=0.03)

        vis_map = filters.gaussian(vis_map, sigma=config["sigma"])

        if vis_map.shape != patch.shape:
            vis_map = transform.resize

    if vis_map.shape != patch.shape:
        vis_map = transform.resize(
            vis_map, patch.shape, order=3, mode="reflect", anti_aliasing=True
        )

    if apply_taper and config["name"] in taper_windows:
        vis_map = vis_map * taper_windows[config["name"]]
        print(
            f"  Applied {config['name']}-specific taper with sigma={taper_sigmas[config['name']]}"
        )

    vis_map = vis_map * foreground_mask

    if balance_levels:
        vis_map = normalize_feature_map(
            vis_map, foreground_mask, min_percentile=1, max_percentile=99
        )
        print(f"  Applied level-specific normalization for balanced visualization")
    else:
        if np.max(vis_map) > 0:
            vis_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min() + 1e-8)

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(vis_map, cmap=colormap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(
        f"{output_prefix}_{config['name']}.png", bbox_inches="tight", pad_inches=0
    )
    plt.close()

    fig = plt.figure(figsize=(5, 5))

    plt.imshow(patch, cmap="gray")

    nonzero_values = vis_map[vis_map > 0]
    if len(nonzero_values) > 0:
        threshold = np.percentile(nonzero_values, threshold_pct * 100)
    else:
        threshold = 0

    masked_vis_map = np.copy(vis_map)
    masked_vis_map[masked_vis_map < threshold] = 0

    alpha_channel = np.zeros_like(masked_vis_map)
    nonzero_mask = masked_vis_map > 0
    if np.any(nonzero_mask):
        alpha_channel[nonzero_mask] = min_opacity + masked_vis_map[nonzero_mask] * (
            max_opacity - min_opacity
        )

    cmap = plt.get_cmap(colormap)

    rgba_data = np.zeros((*masked_vis_map.shape, 4))
    for i in range(3):
        rgba_data[:, :, i] = cmap(masked_vis_map)[:, :, i]
    rgba_data[:, :, 3] = alpha_channel

    plt.imshow(rgba_data)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(
        f"{output_prefix}_{config['name']}_overlay.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    print(f"  Saved {config['name']} feature visualization")


def process_paired_mri_scans(
    invivo_path, exvivo_path, output_dir, slice_idx=None, patch_size=200
):
    """
    Process paired in-vivo and ex-vivo scans with the same patch location.

    Args:
        invivo_path: Path to the in-vivo MRI scan (.nii.gz)
        exvivo_path: Path to the ex-vivo MRI scan (.nii.gz)
        output_dir: Directory to save outputs
        slice_idx: Slice index to use (if None, middle slice is used)
        patch_size: Size of patches to extract
    """
    invivo_output_dir = os.path.join(output_dir, "invivo")
    exvivo_output_dir = os.path.join(output_dir, "exvivo")

    os.makedirs(invivo_output_dir, exist_ok=True)
    os.makedirs(exvivo_output_dir, exist_ok=True)

    print(
        f"Processing paired scans:\n  In-vivo: {invivo_path}\n  Ex-vivo: {exvivo_path}"
    )

    invivo_nii = nib.load(invivo_path)
    invivo_volume = invivo_nii.get_fdata()

    invivo_volume = (invivo_volume - invivo_volume.min()) / (
        invivo_volume.max() - invivo_volume.min() + 1e-8
    )

    if slice_idx is None:
        slice_idx = invivo_volume.shape[2] // 2
        print(f"Using middle slice: {slice_idx}")

    invivo_slice = invivo_volume[:, :, slice_idx]

    start_row, start_col = find_best_patch(invivo_slice, patch_size)
    print(
        f"Selected patch location: ({start_row}, {start_col}) with size {patch_size}x{patch_size}"
    )

    print("\n--- Processing in-vivo scan ---")
    invivo_output_prefix = os.path.join(invivo_output_dir, "features")

    visualize_prostate_features_at_location(
        invivo_path,
        invivo_output_prefix,
        slice_idx,
        patch_size,
        start_row=start_row,
        start_col=start_col,
        colormap="viridis",
        min_opacity=0.2,
        max_opacity=0.8,
        threshold_pct=0,
        apply_taper=True,
        bg_threshold=0.1,
        balance_levels=True,
        taper_sigmas={
            "lowlevel": 0.5,
            "midlevel": 0.4,
            "highlevel": 0.25,
        },
    )

    print("\n--- Processing ex-vivo scan ---")
    exvivo_output_prefix = os.path.join(exvivo_output_dir, "features")

    visualize_prostate_features_at_location(
        exvivo_path,
        exvivo_output_prefix,
        slice_idx,
        patch_size,
        start_row=start_row,
        start_col=start_col,
        colormap="viridis",
        min_opacity=0.2,
        max_opacity=0.8,
        threshold_pct=0,
        apply_taper=True,
        bg_threshold=0.1,
        balance_levels=True,
        taper_sigmas={
            "lowlevel": 0.5,
            "midlevel": 0.4,
            "highlevel": 0.25,
        },
    )

    print(
        f"\nProcessing complete. Results saved to:\n  {invivo_output_dir}\n  {exvivo_output_dir}"
    )


def visualize_prostate_features_at_location(
    input_file,
    output_prefix,
    slice_idx=None,
    patch_size=200,
    start_row=None,
    start_col=None,
    colormap="viridis",
    min_opacity=0.05,
    max_opacity=0.8,
    threshold_pct=0,
    apply_taper=True,
    bg_threshold=0.1,
    balance_levels=True,
    taper_sigmas=None,
):
    """
    Modified version of visualize_prostate_features that accepts a predefined patch location.

    Args:
        input_file: Path to the input prostate MRI volume
        output_prefix: Prefix for output visualization files
        slice_idx: Index of the slice to visualize
        patch_size: Size of the square patch to extract
        start_row: Starting row for patch extraction (if None, automatically determined)
        start_col: Starting column for patch extraction (if None, automatically determined)
        colormap: Colormap to use for visualization
        min_opacity: Minimum opacity for low-activation regions (0.0-1.0)
        max_opacity: Maximum opacity for high-activation regions (0.0-1.0)
        threshold_pct: Percentile threshold below which activations are not shown (0.0-1.0)
        apply_taper: Whether to apply Gaussian tapering to reduce edge effects
        bg_threshold: Threshold below which pixels are considered background
        balance_levels: Whether to independently normalize each feature level
        taper_sigmas: Dictionary mapping feature level names to sigma values for tapering
    """
    visualize_prostate_features(
        input_file,
        output_prefix,
        slice_idx,
        patch_size,
        colormap,
        min_opacity,
        max_opacity,
        threshold_pct,
        apply_taper,
        bg_threshold,
        balance_levels,
        taper_sigmas,
        start_row,
        start_col,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process paired in-vivo and ex-vivo prostate MRI scans with matched patch locations"
    )
    parser.add_argument(
        "--invivo", required=True, help="Path to in-vivo prostate MRI (NIfTI)"
    )
    parser.add_argument(
        "--exvivo", required=True, help="Path to ex-vivo prostate MRI (NIfTI)"
    )
    parser.add_argument(
        "--output", required=True, help="Directory to save visualization results"
    )
    parser.add_argument(
        "--slice", type=int, help="Slice index to visualize (default: middle slice)"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=200,
        help="Size of the square patch to extract (default: 200)",
    )

    args = parser.parse_args()

    process_paired_mri_scans(
        args.invivo, args.exvivo, args.output, args.slice, args.patch_size
    )
