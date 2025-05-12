"""
Contrast Limited Adaptive Histogram Equalization (CLAHE) for the preprocessing pipeline.
Enhances local contrast in images while limiting noise amplification.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from skimage import exposure

from preprocessing.utils.error_handling import handle_exception


@handle_exception(reraise=True)
def apply_clahe(
    invivo_np: np.ndarray,
    exvivo_np: np.ndarray,
    clip_limit: float = 0.03,
    tile_grid_size: Tuple[int, int] = (8, 8),
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to both images.

    Args:
        invivo_np: In-vivo image as NumPy array
        exvivo_np: Ex-vivo image as NumPy array
        clip_limit: Clipping limit for contrast enhancement
        tile_grid_size: Size of grid tiles for adaptive equalization
        logger: Logger instance

    Returns:
        Tuple of (clahe_invivo, clahe_exvivo) enhanced images
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Applying CLAHE for contrast enhancement")
    logger.debug(
        f"Parameters: clip_limit={clip_limit}, tile_grid_size={tile_grid_size}"
    )

    # Check inputs
    if invivo_np is None or exvivo_np is None:
        logger.warning("One or both input images are None, skipping CLAHE")
        return invivo_np, exvivo_np

    # Ensure inputs are float32 (important for skimage functions)
    invivo_np = invivo_np.astype(np.float32)
    exvivo_np = exvivo_np.astype(np.float32)

    # Process in-vivo image
    clahe_invivo = process_volume_with_clahe(
        invivo_np, clip_limit, tile_grid_size, "in-vivo", logger
    )

    # Process ex-vivo image
    clahe_exvivo = process_volume_with_clahe(
        exvivo_np, clip_limit, tile_grid_size, "ex-vivo", logger
    )

    return clahe_invivo, clahe_exvivo


@handle_exception(reraise=False)
def apply_clahe_to_slice(
    slice_data: np.ndarray,
    clip_limit: float = 0.03,
    tile_grid_size: Tuple[int, int] = (8, 8),
    mask: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Apply CLAHE to a single 2D slice.

    Args:
        slice_data: 2D slice as NumPy array
        clip_limit: Clipping limit for contrast enhancement
        tile_grid_size: Size of grid tiles for adaptive equalization
        mask: Optional mask of voxels to process (None = non-zero)
        logger: Logger instance

    Returns:
        CLAHE-enhanced slice
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create mask if not provided
    if mask is None:
        mask = slice_data > 0

    # Skip processing if no foreground
    if not np.any(mask):
        return slice_data

    # Create a copy of the input slice
    slice_copy = slice_data.copy()

    # Normalize to 0-1 range
    slice_min, slice_max = np.min(slice_copy[mask]), np.max(slice_copy[mask])

    if slice_max <= slice_min:
        return slice_copy

    # Normalize to [0,1] range
    slice_norm = slice_copy.copy()
    if mask.any():
        slice_norm[mask] = (slice_copy[mask] - slice_min) / (slice_max - slice_min)

    # Ensure the array is float32 (to avoid object dtype issues)
    slice_norm = slice_norm.astype(np.float32)

    # Apply CLAHE
    try:
        # Make sure all values are in [0,1] range to avoid scikit-image errors
        slice_norm = np.clip(slice_norm, 0.0, 1.0)

        enhanced = exposure.equalize_adapthist(
            slice_norm, kernel_size=tile_grid_size, clip_limit=clip_limit, nbins=256
        )

        # Preserve background
        enhanced[~mask] = 0

        return enhanced

    except Exception as e:
        logger.warning(f"CLAHE failed for slice: {e}")
        return slice_copy


@handle_exception(reraise=False)
def process_volume_with_clahe(
    volume: np.ndarray,
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
    label: str,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Process a 3D volume with CLAHE, applying it slice by slice.

    Args:
        volume: 3D volume as NumPy array
        clip_limit: Clipping limit for contrast enhancement
        tile_grid_size: Size of grid tiles for adaptive equalization
        label: Label for logging
        logger: Logger instance

    Returns:
        CLAHE-enhanced volume
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.debug(f"Processing {label} with CLAHE")

    # Ensure the input is float32 to avoid data type issues
    volume = volume.astype(np.float32)

    # Create output volume
    clahe_volume = np.zeros_like(volume, dtype=np.float32)

    # Process each slice
    for z in range(volume.shape[0]):
        # Skip empty slices
        if not np.any(volume[z] > 0):
            clahe_volume[z] = volume[z]
            continue

        # Get current slice
        slice_data = volume[z].copy()

        # Create mask for non-zero voxels
        mask = slice_data > 0

        # Apply CLAHE to this slice
        enhanced_slice = apply_clahe_to_slice(
            slice_data, clip_limit, tile_grid_size, mask, logger
        )

        clahe_volume[z] = enhanced_slice

    logger.debug(f"CLAHE applied to {label}, slice-by-slice")

    return clahe_volume
