"""
Image normalization functionality for the preprocessing pipeline.
Handles intensity normalization, standardization, and contrast enhancement.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from preprocessing.utils.error_handling import handle_exception


@handle_exception(reraise=True)
def normalize_images(
    invivo_np: np.ndarray,
    exvivo_np: np.ndarray,
    use_zscore: bool = True,
    p_low: float = 1,
    p_high: float = 99,
    scale_to_range: List[float] = [-1, 1],
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize both in-vivo and ex-vivo images to a standardized intensity range.

    Args:
        invivo_np: In-vivo image as NumPy array
        exvivo_np: Ex-vivo image as NumPy array
        use_zscore: Whether to apply z-score normalization after min-max
        p_low: Lower percentile for intensity clipping
        p_high: Upper percentile for intensity clipping
        scale_to_range: Target range for normalized values
        logger: Logger instance

    Returns:
        Tuple of (norm_invivo, norm_exvivo) normalized images
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Normalizing images")

    # Check for None inputs
    if invivo_np is None or exvivo_np is None:
        logger.warning("One or both input images are None, skipping normalization")
        return invivo_np, exvivo_np

    # Create copies to avoid modifying originals
    norm_invivo = invivo_np.copy().astype(np.float32)
    norm_exvivo = exvivo_np.copy().astype(np.float32)

    # Create masks for non-zero voxels
    invivo_mask = norm_invivo > 0
    exvivo_mask = norm_exvivo > 0

    # Normalize in-vivo image
    if np.any(invivo_mask):
        # Min-max normalization
        invivo_min = np.min(norm_invivo[invivo_mask])
        invivo_max = np.max(norm_invivo[invivo_mask])

        if invivo_max > invivo_min:
            norm_invivo[invivo_mask] = (norm_invivo[invivo_mask] - invivo_min) / (
                invivo_max - invivo_min
            )
            logger.debug(
                f"Min-max normalized in-vivo: range [{invivo_min}, {invivo_max}] -> [0, 1]"
            )
        else:
            logger.warning(
                "In-vivo image has constant intensity, skipping min-max normalization"
            )
    else:
        logger.warning("In-vivo image is empty, skipping normalization")

    # Normalize ex-vivo image
    if np.any(exvivo_mask):
        # Min-max normalization
        exvivo_min = np.min(norm_exvivo[exvivo_mask])
        exvivo_max = np.max(norm_exvivo[exvivo_mask])

        if exvivo_max > exvivo_min:
            norm_exvivo[exvivo_mask] = (norm_exvivo[exvivo_mask] - exvivo_min) / (
                exvivo_max - exvivo_min
            )
            logger.debug(
                f"Min-max normalized ex-vivo: range [{exvivo_min}, {exvivo_max}] -> [0, 1]"
            )
        else:
            logger.warning(
                "Ex-vivo image has constant intensity, skipping min-max normalization"
            )
    else:
        logger.warning("Ex-vivo image is empty, skipping normalization")

    # Apply z-score normalization if requested
    if use_zscore:
        logger.debug("Applying z-score normalization")

        # Z-score normalize in-vivo
        if np.any(invivo_mask):
            norm_invivo = apply_zscore_normalization(
                norm_invivo,
                invivo_mask,
                p_low,
                p_high,
                scale_to_range,
                "in-vivo",
                logger,
            )

        # Z-score normalize ex-vivo
        if np.any(exvivo_mask):
            norm_exvivo = apply_zscore_normalization(
                norm_exvivo,
                exvivo_mask,
                p_low,
                p_high,
                scale_to_range,
                "ex-vivo",
                logger,
            )

    return norm_invivo, norm_exvivo


@handle_exception(reraise=False)
def apply_zscore_normalization(
    image: np.ndarray,
    mask: np.ndarray,
    p_low: float,
    p_high: float,
    scale_to_range: List[float],
    label: str,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Apply z-score normalization to an image.

    Args:
        image: Image to normalize
        mask: Mask of voxels to include
        p_low: Lower percentile for clipping
        p_high: Upper percentile for clipping
        scale_to_range: Target range for normalized values
        label: Label for logging
        logger: Logger instance

    Returns:
        Normalized image
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Get masked image
    masked_image = image[mask]

    # Calculate percentile values
    p_low_val = np.percentile(masked_image, p_low)
    p_high_val = np.percentile(masked_image, p_high)

    # Clip intensities
    clipped = np.clip(masked_image, p_low_val, p_high_val)

    # Calculate mean and standard deviation
    mean = np.mean(clipped)
    std = np.std(clipped)

    # Apply z-score normalization
    if std > 0:
        z_scores = (clipped - mean) / std

        # Scale to target range
        range_min, range_max = scale_to_range
        z_min, z_max = np.min(z_scores), np.max(z_scores)

        if z_max > z_min:
            normalized = (z_scores - z_min) / (z_max - z_min) * (
                range_max - range_min
            ) + range_min

            # Create copy of input image
            result = image.copy()

            # Replace values in mask
            result[mask] = normalized

            logger.debug(
                f"Z-score normalized {label}: mean={mean:.4f}, std={std:.4f}, range=[{range_min}, {range_max}]"
            )

            return result
        else:
            logger.warning(
                f"Z-scores have constant value for {label}, skipping scaling"
            )
            return image
    else:
        logger.warning(
            f"Standard deviation is zero for {label}, skipping z-score normalization"
        )
        return image
