"""
Input/output operations for the preprocessing pipeline.
Handles file loading, saving, and finding paired images.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def save_intermediate(
    image: sitk.Image,
    path: Union[str, Path],
    compress: bool = False,
    save_middle_slice_only: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Save an intermediate result to disk.
    If save_middle_slice_only is True, saves only a PNG of the middle slice instead of the full image.

    Args:
        image: SimpleITK image to save
        path: Path to save the image
        compress: Whether to compress the image (ignored if save_middle_slice_only is True)
        save_middle_slice_only: Whether to save only a PNG of the middle slice
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if save_middle_slice_only:
        # Convert path to PNG
        png_path = path.with_suffix(".png")

        # Get middle slice
        array = sitk.GetArrayFromImage(image)
        middle_idx = array.shape[0] // 2
        middle_slice = array[middle_idx]

        # Normalize if needed
        if np.any(middle_slice > 0):
            mask = middle_slice > 0
            min_val = np.min(middle_slice[mask])
            max_val = np.max(middle_slice[mask])
            if max_val > min_val:
                normalized = middle_slice.copy()
                normalized[mask] = (middle_slice[mask] - min_val) / (max_val - min_val)
            else:
                normalized = middle_slice
        else:
            normalized = middle_slice

        # Create figure with a readable size
        plt.figure(figsize=(10, 10))
        plt.imshow(normalized, cmap="gray")
        plt.colorbar(label="Normalized Intensity")
        plt.title(f"Slice {middle_idx} of {path.stem}")
        plt.axis("off")

        # Save as PNG
        plt.savefig(png_path, bbox_inches="tight", dpi=150)
        plt.close()

        logger.debug(f"Saved middle slice visualization to {png_path}")
    else:
        # Save full image
        sitk.WriteImage(image, str(path), compress)
        logger.debug(f"Saved full image to {path}")


def save_comparison_visualization(
    images: List[Tuple[sitk.Image, str]],
    output_path: Union[str, Path],
    slice_idx: Optional[int] = None,
    cmap: str = "gray",
    figsize: Tuple[int, int] = (15, 10),
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Save a multi-image comparison visualization showing the same slice from multiple images.

    Args:
        images: List of (image, title) tuples
        output_path: Path to save the visualization
        slice_idx: Slice index to visualize (None = middle slice)
        cmap: Colormap to use
        figsize: Figure size
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Number of images to display
    n_images = len(images)
    if n_images == 0:
        logger.warning("No images to visualize")
        return

    # Create figure
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    if n_images == 1:
        axes = [axes]  # Make it iterable

    # Process each image
    for i, (image, title) in enumerate(images):
        # Convert to numpy array
        array = sitk.GetArrayFromImage(image)

        # Determine slice to show
        if slice_idx is None:
            slice_idx = array.shape[0] // 2

        # Get the slice
        if slice_idx < array.shape[0]:
            slice_data = array[slice_idx]
        else:
            logger.warning(
                f"Slice index {slice_idx} out of bounds for image with {array.shape[0]} slices"
            )
            slice_data = array[0]

        # Normalize the slice for visualization
        if np.any(slice_data > 0):
            mask = slice_data > 0
            min_val = np.min(slice_data[mask])
            max_val = np.max(slice_data[mask])
            if max_val > min_val:
                normalized = slice_data.copy()
                normalized[mask] = (slice_data[mask] - min_val) / (max_val - min_val)
            else:
                normalized = slice_data
        else:
            normalized = slice_data

        # Display the slice
        im = axes[i].imshow(normalized, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    logger.debug(f"Saved comparison visualization to {output_path}")


def save_final_results(
    image_pair: "ImagePair",
    output_dir: Union[str, Path],
    split: str = "train",
    compress: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """
    Save final processing results to disk.

    Args:
        image_pair: Processed image pair
        output_dir: Base output directory
        split: Dataset split ('train' or 'test')
        compress: Whether to compress the images
        logger: Logger instance

    Returns:
        Dict mapping output types to file paths
    """
    from preprocessing.models import ImagePair

    if not isinstance(image_pair, ImagePair):
        raise TypeError("Expected ImagePair object")

    if logger is None:
        logger = logging.getLogger(__name__)

    return image_pair.save_results(output_dir, split, compress)


def find_paired_images(
    input_dir_invivo: Union[str, Path],
    input_dir_exvivo: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    skip_list: Optional[List[str]] = None,
    only_run: Optional[List[str]] = None,
) -> Tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    """
    Find and match paired in-vivo and ex-vivo images.

    Args:
        input_dir_invivo: Directory containing in-vivo MRI scans
        input_dir_exvivo: Directory containing ex-vivo MRI scans
        logger: Logger instance
        skip_list: List of IDs to skip
        only_run: List of IDs to exclusively process

    Returns:
        Tuple of (pairs, invivo_only, exvivo_only)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    skip_list = skip_list or []
    only_run = only_run or []

    # Find all valid image files
    invivo_files = sorted(
        [
            f
            for f in os.listdir(input_dir_invivo)
            if f.endswith((".nii", ".nii.gz", ".mha"))
        ]
    )
    exvivo_files = sorted(
        [
            f
            for f in os.listdir(input_dir_exvivo)
            if f.endswith((".nii", ".nii.gz", ".mha"))
        ]
    )

    # Create dictionaries mapping base names to full filenames
    invivo_bases = {os.path.splitext(f)[0].split(".")[0]: f for f in invivo_files}
    exvivo_bases = {os.path.splitext(f)[0].split(".")[0]: f for f in exvivo_files}

    # Find common IDs
    common_ids = set(invivo_bases.keys()) & set(exvivo_bases.keys())
    common_ids = common_ids - set(skip_list)

    # Filter to only specified IDs if provided
    if only_run:
        common_ids = common_ids & set(only_run)
        logger.info(f"Filtering to only process {len(common_ids)} specified cases")

    # Create pairs
    pairs = [(id, invivo_bases[id], exvivo_bases[id]) for id in sorted(common_ids)]

    # Find unpaired images
    invivo_only = set(invivo_bases.keys()) - common_ids - set(skip_list)
    exvivo_only = set(exvivo_bases.keys()) - common_ids - set(skip_list)

    return pairs, list(invivo_only), list(exvivo_only)


def report_processing_stats(
    pairs: List[Tuple[str, str, str]],
    invivo_only: List[str],
    exvivo_only: List[str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Report statistics about the processing job.

    Args:
        pairs: List of (pair_id, invivo_file, exvivo_file) tuples
        invivo_only: List of IDs with only in-vivo images
        exvivo_only: List of IDs with only ex-vivo images
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Found {len(pairs)} matching in-vivo/ex-vivo pairs")

    if invivo_only:
        logger.info(
            f"Skipping {len(invivo_only)} in-vivo files without matching ex-vivo"
        )
        for id in sorted(invivo_only[:4]):
            logger.info(f"  - {id}")
        if len(invivo_only) > 4:
            logger.info(f"  - and {len(invivo_only) - 4} more...")

    if exvivo_only:
        logger.info(
            f"Skipping {len(exvivo_only)} ex-vivo files without matching in-vivo"
        )
        for id in sorted(exvivo_only[:4]):
            logger.info(f"  - {id}")
        if len(exvivo_only) > 4:
            logger.info(f"  - and {len(exvivo_only) - 4} more...")


def report_special_cases(
    pairs: List[Tuple[str, str, str]],
    flip_cases: List[str],
    skip_cases: List[str],
    only_run: List[str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Report special cases that require specific handling.

    Args:
        pairs: List of (pair_id, invivo_file, exvivo_file) tuples
        flip_cases: List of IDs requiring anterior-posterior flipping
        skip_cases: List of IDs to skip
        only_run: List of IDs to exclusively process
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    common_ids = [pair[0] for pair in pairs]
    ap_flip_cases = [id for id in common_ids if id in flip_cases]

    if ap_flip_cases:
        logger.info(
            f"The following {len(ap_flip_cases)} cases will have anterior-posterior flipping (in-vivo only):"
        )
        for id in ap_flip_cases:
            logger.info(f"  - {id}")

    if skip_cases:
        logger.info(f"The following {len(skip_cases)} cases will be skipped:")
        for id in skip_cases:
            logger.info(f"  - {id}")

    if only_run:
        logger.info(f"Processing only the following {len(only_run)} specified cases:")
        for id in only_run:
            if id in common_ids:
                logger.info(f"  - {id}")
            else:
                logger.info(f"  - {id} (not found or skipped)")


def standardize_dimensions(
    image_dir: Union[str, Path],
    padding_value: float = 0,
    logger: Optional[logging.Logger] = None,
) -> Optional[Tuple[int, int, int]]:
    """
    Standardize all images in the directory to have the same dimensions.

    Args:
        image_dir: Directory containing images to standardize
        padding_value: Value to use for padding
        logger: Logger instance

    Returns:
        The standard dimensions used or None if no images were processed
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Analyze image dimensions
    max_dims, _ = analyze_image_dimensions(image_dir, logger)

    if not max_dims:
        return None

    # Find all images
    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        if "temp" in dirs:
            dirs.remove("temp")
        for file in files:
            if file.endswith((".nii", ".nii.gz", ".mha")):
                image_paths.append(os.path.join(root, file))

    logger.info(f"\nStandardizing all images to dimensions: {max_dims}")

    # Process each image
    for path in tqdm(image_paths, desc="Standardizing images"):
        try:
            img = sitk.ReadImage(path)
            current_size = img.GetSize()

            if current_size != tuple(max_dims):
                padded_img = pad_image_centered(img, max_dims, padding_value)
                sitk.WriteImage(padded_img, path)
                logger.debug(
                    f"Padded {os.path.basename(path)} from {current_size} to {max_dims}"
                )
            else:
                logger.debug(
                    f"Image {os.path.basename(path)} already at target dimensions"
                )

        except Exception as e:
            logger.error(f"Error standardizing {path}: {e}")

    logger.info("\nStandardization complete!")
    logger.info(f"All images now have dimensions: {max_dims}")

    return max_dims


def analyze_image_dimensions(
    image_dir: Union[str, Path], logger: Optional[logging.Logger] = None
) -> Tuple[Optional[List[int]], List[Dict]]:
    """
    Analyze dimensions of all images in a directory.

    Args:
        image_dir: Directory containing images
        logger: Logger instance

    Returns:
        Tuple of (max_dims, dimension_stats)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Find all images
    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        if "temp" in dirs:
            dirs.remove("temp")
        for file in files:
            if file.endswith((".nii", ".nii.gz", ".mha")):
                image_paths.append(os.path.join(root, file))

    logger.info(f"Found {len(image_paths)} images to analyze")

    # Get dimensions
    dimensions = []
    for path in tqdm(image_paths, desc="Analyzing dimensions"):
        try:
            img = sitk.ReadImage(path)
            dimensions.append(img.GetSize())
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")

    if not dimensions:
        logger.warning("No valid images found")
        return None, []

    # Calculate maximum dimensions
    max_dims = [max(dim[i] for dim in dimensions) for i in range(3)]

    # Calculate statistics
    logger.info("\nImage Dimension Analysis:")
    logger.info(f"Total images analyzed: {len(dimensions)}")
    logger.info(f"Maximum dimensions found: {max_dims}")

    dimension_stats = []
    for i, axis in enumerate(["X", "Y", "Z"]):
        values = [dim[i] for dim in dimensions]
        min_val, max_val = min(values), max(values)
        mean_val = sum(values) / len(values)
        dimension_stats.append(
            {"axis": axis, "min": min_val, "max": max_val, "mean": mean_val}
        )
        logger.info(f"  {axis}-axis: min={min_val}, max={max_val}, mean={mean_val:.1f}")

    return max_dims, dimension_stats


def pad_image_centered(
    image: sitk.Image, target_size: List[int], padding_value: float = 0
) -> sitk.Image:
    """
    Pad an image to target dimensions with the original content centered.

    Args:
        image: SimpleITK image to pad
        target_size: Target dimensions [x, y, z]
        padding_value: Value to use for padding

    Returns:
        Padded SimpleITK image
    """
    current_size = image.GetSize()
    pad_lower = [(target_size[i] - current_size[i]) // 2 for i in range(3)]
    pad_upper = [target_size[i] - current_size[i] - pad_lower[i] for i in range(3)]

    return sitk.ConstantPad(image, pad_lower, pad_upper, padding_value)


def cleanup_temp_directory(
    output_dir: Union[str, Path], logger: Optional[logging.Logger] = None
) -> None:
    """
    Clean up the temporary directory after processing.

    Args:
        output_dir: The main output directory
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    temp_dir = Path(output_dir) / "temp"

    if temp_dir.exists():
        logger.info("\nCleaning up temporary files...")
        try:
            # shutil.rmtree(temp_dir)
            logger.info("Temporary directory removed")
        except Exception as e:
            logger.error(f"Error removing temporary directory: {e}")


def set_image_metadata(
    image: sitk.Image,
    metadata: Dict[str, Any],
    image_type: str,
    flip_direction: bool,
) -> sitk.Image:
    """Apply consistent metadata to a SimpleITK image."""
    # Always use invivo direction for consistency
    image.SetDirection(metadata["invivo_direction"])
    image.SetOrigin(metadata["invivo_origin"])

    # Apply spacing based on pipeline mode
    if "invivo_spacing" in metadata and "exvivo_spacing" in metadata:
        if image_type == "invivo":
            if flip_direction:
                image.SetSpacing(metadata["exvivo_spacing"])
            else:
                image.SetSpacing(metadata["invivo_spacing"])
        else:  # exvivo
            image.SetSpacing(metadata["exvivo_spacing"])

    return image
