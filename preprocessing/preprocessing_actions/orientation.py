"""
Image orientation functionality for the preprocessing pipeline.
Handles flipping, reorientation, and metadata extraction.
"""

import logging
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from preprocessing.utils.error_handling import handle_exception


@handle_exception(reraise=True)
def extract_metadata(
    invivo_sitk: sitk.Image,
    exvivo_sitk: sitk.Image,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Union[Tuple, float]]:
    """
    Extract metadata from in-vivo and ex-vivo images.

    Args:
        invivo_sitk: In-vivo SimpleITK image
        exvivo_sitk: Ex-vivo SimpleITK image
        logger: Logger instance

    Returns:
        Dictionary containing metadata from both images
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Extracting image metadata")

    try:
        # Extract metadata
        metadata = {
            "invivo_spacing": invivo_sitk.GetSpacing(),
            "exvivo_spacing": exvivo_sitk.GetSpacing(),
            "invivo_origin": invivo_sitk.GetOrigin(),
            "exvivo_origin": exvivo_sitk.GetOrigin(),
            "invivo_direction": invivo_sitk.GetDirection(),
            "exvivo_direction": exvivo_sitk.GetDirection(),
            "invivo_size": invivo_sitk.GetSize(),
            "exvivo_size": exvivo_sitk.GetSize(),
        }

        # Calculate resolution ratios
        inv_voxel_vol = np.prod(metadata["invivo_spacing"])
        exv_voxel_vol = np.prod(metadata["exvivo_spacing"])
        metadata["resolution_ratio"] = (
            exv_voxel_vol / inv_voxel_vol if inv_voxel_vol > 0 else 1.0
        )

        # Log metadata
        logger.debug(f"In-vivo spacing: {metadata['invivo_spacing']}")
        logger.debug(f"Ex-vivo spacing: {metadata['exvivo_spacing']}")
        logger.debug(f"Resolution ratio: {metadata['resolution_ratio']:.2f}x")

        return metadata

    except Exception as e:
        logger.error(f"Failed to extract metadata: {e}")
        logger.warning("Creating minimal fallback metadata")

        # Create minimal metadata as fallback
        return {
            "invivo_spacing": (1.0, 1.0, 1.0),
            "exvivo_spacing": (1.0, 1.0, 1.0),
            "invivo_origin": (0.0, 0.0, 0.0),
            "exvivo_origin": (0.0, 0.0, 0.0),
            "invivo_direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            "exvivo_direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            "resolution_ratio": 1.0,
        }


@handle_exception(reraise=False)
def flip_anterior_posterior(
    image_np: np.ndarray, logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Flip image along the anterior-posterior axis (axis 1 in NumPy array).

    Args:
        image_np: NumPy array to flip
        logger: Logger instance

    Returns:
        Flipped NumPy array
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Flipping image along anterior-posterior axis")

    try:
        flipped_array = np.flip(image_np, axis=1)
        logger.debug(f"Flipped array shape: {flipped_array.shape}")
        return flipped_array
    except Exception as e:
        logger.error(f"Failed to flip image: {e}")
        return image_np


@handle_exception(reraise=False)
def flip_left_right(
    image_np: np.ndarray, logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Flip image along the left-right axis (axis 2 in NumPy array).

    Args:
        image_np: NumPy array to flip
        logger: Logger instance

    Returns:
        Flipped NumPy array
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Flipping image along left-right axis")

    try:
        flipped_array = np.flip(image_np, axis=2)
        logger.debug(f"Flipped array shape: {flipped_array.shape}")
        return flipped_array
    except Exception as e:
        logger.error(f"Failed to flip image: {e}")
        return image_np


def standardize_orientation(
    invivo_sitk: sitk.Image,
    exvivo_sitk: sitk.Image,
    metadata: Optional[Dict] = None,  # Make metadata optional
    logger: Optional[Union[logging.Logger, Callable]] = None,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Standardize orientation of both images to ensure consistency.

    Args:
        invivo_sitk: In-vivo SimpleITK image
        exvivo_sitk: Ex-vivo SimpleITK image
        metadata: Dictionary containing metadata (optional)
        logger: Logger instance

    Returns:
        Tuple of (standardized_invivo, standardized_exvivo)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Standardizing image orientations")

    # Get original metadata
    invivo_direction = invivo_sitk.GetDirection()
    invivo_origin = invivo_sitk.GetOrigin()

    # Store in metadata if available
    if metadata is not None:
        metadata["original_invivo_direction"] = invivo_direction
        metadata["original_invivo_origin"] = invivo_origin
        metadata["original_exvivo_direction"] = exvivo_sitk.GetDirection()
        metadata["original_exvivo_origin"] = exvivo_sitk.GetOrigin()

    # Set standardized orientation on exvivo image
    standardized_exvivo = sitk.Image(exvivo_sitk)
    standardized_exvivo.SetDirection(invivo_direction)
    standardized_exvivo.SetOrigin(invivo_origin)

    logger.debug(f"Standardized orientation using direction: {invivo_direction}")
    logger.debug(f"Standardized origin: {invivo_origin}")

    return invivo_sitk, standardized_exvivo
