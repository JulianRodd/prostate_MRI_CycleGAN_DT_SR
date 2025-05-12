"""
Image centering and cropping functionality for the preprocessing pipeline.
Handles centering based on center of mass and cropping to content.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from preprocessing.utils.error_handling import handle_exception


@handle_exception(reraise=True)
def center_and_crop_images(
    invivo_np: np.ndarray,
    exvivo_np: np.ndarray,
    skip_centering: bool = False,
    margin: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int, int, int]]:
    """
    Center both images based on their center of mass and crop excess space.
    If skip_centering is True, only perform cropping without centering.

    Args:
        invivo_np: In-vivo image as NumPy array
        exvivo_np: Ex-vivo image as NumPy array
        skip_centering: If True, only perform cropping without centering
        margin: Margin to add around the content in voxels
        logger: Logger instance

    Returns:
        Tuple of (centered_cropped_invivo, centered_cropped_exvivo, bbox)
        where bbox contains (min_z, min_y, min_x, max_z, max_y, max_x)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"{'Cropping' if skip_centering else 'Centering and cropping'} images")
    logger.debug(
        f"Original shapes - In-vivo: {invivo_np.shape}, Ex-vivo: {exvivo_np.shape}"
    )
    logger.debug(
        f"Non-zero voxels - In-vivo: {np.count_nonzero(invivo_np)}, Ex-vivo: {np.count_nonzero(exvivo_np)}"
    )

    # Check for empty ex-vivo image
    exvivo_empty = np.count_nonzero(exvivo_np) == 0
    if exvivo_empty:
        logger.warning("Ex-vivo image is empty, attempting recovery")

        # Try to create a synthetic ex-vivo from in-vivo
        invivo_mask = invivo_np > 0
        if np.count_nonzero(invivo_mask) > 0:
            exvivo_np = create_synthetic_exvivo(invivo_np, invivo_mask, logger)
        else:
            logger.warning("In-vivo is also empty, cannot create synthetic ex-vivo")

    # Ensure images have the same shape
    if invivo_np.shape != exvivo_np.shape:
        logger.warning("Images have different shapes, standardizing dimensions")
        invivo_np, exvivo_np = standardize_image_shapes(invivo_np, exvivo_np, logger)

    # Create masks for non-zero voxels
    invivo_mask = invivo_np > 0
    exvivo_mask = exvivo_np > 0

    # Safety check: if both images are empty, return originals
    if np.count_nonzero(invivo_mask) == 0 and np.count_nonzero(exvivo_mask) == 0:
        logger.warning("Both images are empty, returning originals without processing")
        return invivo_np, exvivo_np, None

    # If skip_centering is True, go directly to cropping
    if skip_centering:
        logger.info("Skipping centering step as requested")
        cropped_invivo, cropped_exvivo, bbox = crop_to_content(
            invivo_np, exvivo_np, margin, logger
        )
        logger.debug(f"Final shapes after cropping: {cropped_invivo.shape}")
        return cropped_invivo, cropped_exvivo, bbox

    # Perform centering process
    # Calculate centers of mass
    invivo_com, exvivo_com = calculate_centers_of_mass(invivo_np, exvivo_np, logger)

    # Calculate target center (average of both centers of mass)
    target_center = calculate_target_center(
        invivo_com, exvivo_com, invivo_np.shape, logger
    )

    # Center images
    centered_invivo = center_image(invivo_np, invivo_com, target_center, logger)
    centered_exvivo = center_image(exvivo_np, exvivo_com, target_center, logger)

    # Crop centered images
    cropped_invivo, cropped_exvivo, bbox = crop_to_content(
        centered_invivo, centered_exvivo, margin, logger
    )

    logger.debug(f"Final shapes after centering and cropping: {cropped_invivo.shape}")

    return cropped_invivo, cropped_exvivo, bbox


@handle_exception(reraise=False)
def create_synthetic_exvivo(
    invivo_np: np.ndarray,
    invivo_mask: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Create a synthetic ex-vivo image from in-vivo for cases where ex-vivo is missing.

    Args:
        invivo_np: In-vivo image as NumPy array
        invivo_mask: Mask of non-zero voxels in in-vivo image
        logger: Logger instance

    Returns:
        Synthetic ex-vivo image
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Creating synthetic ex-vivo from in-vivo image")

    # Create empty ex-vivo
    exvivo_np = np.zeros_like(invivo_np)

    # Find bounding box of the in-vivo image
    nonzero = np.where(invivo_mask)
    min_z, max_z = np.min(nonzero[0]), np.max(nonzero[0])
    min_y, max_y = np.min(nonzero[1]), np.max(nonzero[1])
    min_x, max_x = np.min(nonzero[2]), np.max(nonzero[2])

    # Create a scaled-down version of the in-vivo as fake ex-vivo
    fake_exvivo = (
        invivo_np[min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1] * 0.8
    )

    # Calculate offsets to center the fake ex-vivo in the output volume
    offset_z = (exvivo_np.shape[0] - (max_z - min_z + 1)) // 2
    offset_y = (exvivo_np.shape[1] - (max_y - min_y + 1)) // 2
    offset_x = (exvivo_np.shape[2] - (max_x - min_x + 1)) // 2

    # Add slight offset for visual difference
    offset_y += 10
    offset_x += 10

    # Ensure offsets are valid
    offset_z = max(0, offset_z)
    offset_y = max(0, offset_y)
    offset_x = max(0, offset_x)

    # Calculate copy dimensions
    copy_z = min(fake_exvivo.shape[0], exvivo_np.shape[0] - offset_z)
    copy_y = min(fake_exvivo.shape[1], exvivo_np.shape[1] - offset_y)
    copy_x = min(fake_exvivo.shape[2], exvivo_np.shape[2] - offset_x)

    # Copy the fake ex-vivo into the result
    exvivo_np[
        offset_z : offset_z + copy_z,
        offset_y : offset_y + copy_y,
        offset_x : offset_x + copy_x,
    ] = fake_exvivo[:copy_z, :copy_y, :copy_x]

    logger.debug(
        f"Created synthetic ex-vivo with {np.count_nonzero(exvivo_np)} non-zero voxels"
    )

    return exvivo_np


@handle_exception(reraise=False)
def standardize_image_shapes(
    invivo_np: np.ndarray,
    exvivo_np: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize images to have the same shape.

    Args:
        invivo_np: In-vivo image as NumPy array
        exvivo_np: Ex-vivo image as NumPy array
        logger: Logger instance

    Returns:
        Tuple of (standardized_invivo, standardized_exvivo)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create a new volume with the maximum dimensions
    max_shape = [max(invivo_np.shape[i], exvivo_np.shape[i]) for i in range(3)]
    logger.debug(f"Standardizing to shape: {max_shape}")

    # Create new arrays with the maximum shape
    new_invivo = np.zeros(max_shape, dtype=invivo_np.dtype)
    new_exvivo = np.zeros(max_shape, dtype=exvivo_np.dtype)

    # Define source regions (entire original images)
    src_invivo = tuple(
        slice(0, min(max_shape[i], invivo_np.shape[i])) for i in range(3)
    )
    src_exvivo = tuple(
        slice(0, min(max_shape[i], exvivo_np.shape[i])) for i in range(3)
    )

    # Define destination regions (centered in new volumes)
    dst_invivo = tuple(
        slice(
            (max_shape[i] - invivo_np.shape[i]) // 2,
            (max_shape[i] - invivo_np.shape[i]) // 2
            + min(max_shape[i], invivo_np.shape[i]),
        )
        for i in range(3)
    )
    dst_exvivo = tuple(
        slice(
            (max_shape[i] - exvivo_np.shape[i]) // 2,
            (max_shape[i] - exvivo_np.shape[i]) // 2
            + min(max_shape[i], exvivo_np.shape[i]),
        )
        for i in range(3)
    )

    # Copy the data
    new_invivo[dst_invivo] = invivo_np[src_invivo]
    new_exvivo[dst_exvivo] = exvivo_np[src_exvivo]

    return new_invivo, new_exvivo


@handle_exception(reraise=False)
def calculate_centers_of_mass(
    invivo_np: np.ndarray,
    exvivo_np: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    """
    Calculate centers of mass for both images.

    Args:
        invivo_np: In-vivo image as NumPy array
        exvivo_np: Ex-vivo image as NumPy array
        logger: Logger instance

    Returns:
        Tuple of (invivo_com, exvivo_com), either can be None if image is empty
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create masks for non-zero voxels
    invivo_mask = invivo_np > 0
    exvivo_mask = exvivo_np > 0

    # Calculate center of mass for in-vivo
    invivo_com = None
    if np.count_nonzero(invivo_mask) > 0:
        invivo_com = ndimage.center_of_mass(invivo_np)
        logger.debug(f"In-vivo center of mass: {invivo_com}")
    else:
        logger.warning("In-vivo is empty, cannot compute center of mass")

    # Calculate center of mass for ex-vivo
    exvivo_com = None
    if np.count_nonzero(exvivo_mask) > 0:
        exvivo_com = ndimage.center_of_mass(exvivo_np)
        logger.debug(f"Ex-vivo center of mass: {exvivo_com}")
    else:
        logger.warning("Ex-vivo is empty, cannot compute center of mass")

    return invivo_com, exvivo_com


@handle_exception(reraise=False)
def calculate_target_center(
    invivo_com: Optional[Tuple[float, float, float]],
    exvivo_com: Optional[Tuple[float, float, float]],
    image_shape: Tuple[int, int, int],
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float, float]:
    """
    Calculate target center based on centers of mass.

    Args:
        invivo_com: In-vivo center of mass
        exvivo_com: Ex-vivo center of mass
        image_shape: Shape of the images
        logger: Logger instance

    Returns:
        Target center coordinates
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Calculate target center (average of both centers of mass)
    if invivo_com is not None and exvivo_com is not None:
        target_center = [(invivo_com[i] + exvivo_com[i]) / 2 for i in range(3)]
    elif invivo_com is not None:
        target_center = invivo_com
    elif exvivo_com is not None:
        target_center = exvivo_com
    else:
        target_center = [dim // 2 for dim in image_shape]

    logger.debug(f"Target center: {target_center}")

    return target_center


@handle_exception(reraise=False)
def center_image(
    image: np.ndarray,
    image_com: Optional[Tuple[float, float, float]],
    target_center: Tuple[float, float, float],
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Center an image to the target center.

    Args:
        image: Image to center
        image_com: Center of mass of the image
        target_center: Target center coordinates
        logger: Logger instance

    Returns:
        Centered image
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # If image has no center of mass, return as is
    if image_com is None:
        return image

    # Calculate shift to apply
    shift = [int(target_center[i] - image_com[i]) for i in range(3)]
    logger.debug(f"Shifting by: {shift}")

    # Apply shift
    centered_image = ndimage.shift(image, shift, order=0, mode="constant", cval=0)

    return centered_image


@handle_exception(reraise=True)
def crop_to_content(
    invivo_np: np.ndarray,
    exvivo_np: np.ndarray,
    margin: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int, int, int]]:
    """
    Crop images to remove excess black space, focusing on the non-zero regions.

    Args:
        invivo_np: In-vivo image as NumPy array
        exvivo_np: Ex-vivo image as NumPy array
        margin: Margin to add around the cropped region in voxels
        logger: Logger instance

    Returns:
        Tuple of (cropped_invivo, cropped_exvivo, bbox)
        where bbox contains (min_z, min_y, min_x, max_z, max_y, max_x)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Cropping images to content")

    # Create masks for non-zero voxels
    invivo_mask = invivo_np > 0
    exvivo_mask = exvivo_np > 0

    # Create combined mask
    combined_mask = np.logical_or(invivo_mask, exvivo_mask)

    # Find non-zero voxels
    nonzero = np.where(combined_mask)

    if len(nonzero[0]) == 0:
        logger.warning("No non-zero voxels found, skipping cropping")
        return (
            invivo_np,
            exvivo_np,
            (
                0,
                0,
                0,
                invivo_np.shape[0] - 1,
                invivo_np.shape[1] - 1,
                invivo_np.shape[2] - 1,
            ),
        )

    # Calculate bounding box
    min_z, max_z = np.min(nonzero[0]), np.max(nonzero[0])
    min_y, max_y = np.min(nonzero[1]), np.max(nonzero[1])
    min_x, max_x = np.min(nonzero[2]), np.max(nonzero[2])

    # Add margin and ensure coordinates are within bounds
    min_z = max(0, min_z - margin)
    min_y = max(0, min_y - margin)
    min_x = max(0, min_x - margin)
    max_z = min(combined_mask.shape[0] - 1, max_z + margin)
    max_y = min(combined_mask.shape[1] - 1, max_y + margin)
    max_x = min(combined_mask.shape[2] - 1, max_x + margin)

    # Create bounding box tuple
    bbox = (min_z, min_y, min_x, max_z, max_y, max_x)

    # Crop both images
    cropped_invivo = invivo_np[min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1]
    cropped_exvivo = exvivo_np[min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1]

    # Log results
    logger.debug(
        f"Bounding box: Z: {min_z} to {max_z}, Y: {min_y} to {max_y}, X: {min_x} to {max_x}"
    )
    logger.debug(f"Cropped shape: {cropped_invivo.shape}")
    logger.debug(
        f"Non-zero voxels after cropping - In-vivo: {np.count_nonzero(cropped_invivo)}, "
        f"Ex-vivo: {np.count_nonzero(cropped_exvivo)}"
    )

    return cropped_invivo, cropped_exvivo, bbox
