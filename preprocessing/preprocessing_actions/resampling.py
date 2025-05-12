"""
Image resampling functionality for the preprocessing pipeline.
Handles resampling between different resolutions and physical spaces.
"""

import logging
from typing import Dict, Optional, Tuple, Any

import SimpleITK as sitk
import numpy as np

from preprocessing.utils.error_handling import handle_exception


@handle_exception(reraise=True)
def resample_with_physical_space(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolation: int = sitk.sitkLinear,
    logger: Optional[logging.Logger] = None,
) -> sitk.Image:
    """
    Resample an image to a target physical spacing while preserving physical space.

    Args:
        image: SimpleITK image to resample
        target_spacing: Target spacing in physical units
        interpolation: Interpolation method
        logger: Logger instance

    Returns:
        Resampled SimpleITK image
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Get current properties
    current_spacing = image.GetSpacing()
    current_size = image.GetSize()

    # set z dimension of target spacing to 1 if it is less than 1

    # Calculate output size
    output_size = [
        int(round(current_size[i] * current_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    logger.debug(f"Resampling from spacing {current_spacing} to {target_spacing}")
    logger.debug(f"Size will change from {current_size} to {output_size}")

    # Create resampling filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetSize(output_size)
    resampler.SetInterpolator(interpolation)
    resampler.SetDefaultPixelValue(0)

    # Perform resampling
    try:
        resampled_image = resampler.Execute(image)
        logger.debug("Resampling completed successfully")
        return resampled_image
    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        raise ValueError(f"Resampling failed: {e}")


@handle_exception(reraise=True)
def resample_invivo_to_exvivo(
    invivo_sitk: sitk.Image,
    metadata: Dict,
    apply_smoothing: bool = False,
    logger: Optional[logging.Logger] = None,
) -> sitk.Image:
    """
    Upsample in-vivo image to match ex-vivo resolution.

    Args:
        invivo_sitk: In-vivo SimpleITK image
        metadata: Dictionary containing metadata
        apply_smoothing: Whether to apply smoothing before upsampling
        logger: Logger instance

    Returns:
        Upsampled in-vivo SimpleITK image
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Upsampling in-vivo to ex-vivo resolution")

    # Calculate original intensity statistics for later use
    stats = sitk.StatisticsImageFilter()
    stats.Execute(invivo_sitk)
    original_min = stats.GetMinimum()
    original_max = stats.GetMaximum()
    original_mean = stats.GetMean()
    original_std = stats.GetVariance() ** 0.5

    logger.debug(
        f"Original intensity range: [{original_min}, {original_max}], mean: {original_mean}, std: {original_std}"
    )

    # Apply smoothing if requested
    if apply_smoothing:
        logger.debug("Applying smoothing before upsampling")
        smoothing_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        sigma = [
            0.5 * s for s in invivo_sitk.GetSpacing()
        ]  # Reduced sigma for less blurring
        smoothing_filter.SetSigma(sigma)
        invivo_sitk = smoothing_filter.Execute(invivo_sitk)

    # Get target spacing
    target_spacing = metadata["exvivo_spacing"]

    if target_spacing[2] < 0.8:
        logger.warning(
            "Ex-vivo z-spacing is less than 0.8mm, which may cause memory issues"
        )
        target_spacing = (target_spacing[0], target_spacing[1], 0.8)
    # Perform resampling with linear interpolation instead of B-spline
    # Linear produces fewer artifacts for medical images with high contrast
    upsampled_sitk = resample_with_physical_space(
        invivo_sitk,
        target_spacing,
        interpolation=sitk.sitkLinear,  # Changed from BSpline to Linear
        logger=logger,
    )

    # Check if upsampling created extreme values
    stats.Execute(upsampled_sitk)
    upsampled_min = stats.GetMinimum()
    upsampled_max = stats.GetMaximum()

    logger.debug(f"Upsampled intensity range: [{upsampled_min}, {upsampled_max}]")

    # If the upsampling created outliers, apply intensity clamping
    if upsampled_max > original_max * 1.2 or upsampled_min < original_min * 0.8:
        logger.warning(
            "Extreme intensity values detected after upsampling, applying clipping"
        )

        # Apply robust intensity clipping to handle outliers
        # Allow slightly higher than original max to preserve genuine detail
        intensity_window_max = original_max * 1.1
        intensity_window_min = max(0, original_min * 0.9)

        # Use threshold filter for clamping
        thresholdFilter = sitk.ClampImageFilter()
        thresholdFilter.SetLowerBound(intensity_window_min)
        thresholdFilter.SetUpperBound(intensity_window_max)
        upsampled_sitk = thresholdFilter.Execute(upsampled_sitk)

        logger.debug(
            f"Intensity range after clamping: [{intensity_window_min}, {intensity_window_max}]"
        )

    return upsampled_sitk


@handle_exception(reraise=True)
def resample_exvivo_to_invivo(
    exvivo_sitk: sitk.Image,
    metadata: Dict,
    apply_smoothing: bool = True,
    logger: Optional[logging.Logger] = None,
) -> sitk.Image:
    """
    Downsample ex-vivo image to match in-vivo resolution.

    Args:
        exvivo_sitk: Ex-vivo SimpleITK image
        metadata: Dictionary containing metadata
        apply_smoothing: Whether to apply smoothing before downsampling
        logger: Logger instance

    Returns:
        Downsampled ex-vivo SimpleITK image
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Downsampling ex-vivo to in-vivo resolution")

    # Apply smoothing if requested (recommended for downsampling to prevent aliasing)
    if apply_smoothing:
        logger.debug("Applying smoothing before downsampling")
        smoothing_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        sigma = [0.7 * s for s in exvivo_sitk.GetSpacing()]
        smoothing_filter.SetSigma(sigma)
        exvivo_sitk = smoothing_filter.Execute(exvivo_sitk)

    # Perform resampling
    target_spacing = metadata["invivo_spacing"]
    downsampled_sitk = resample_with_physical_space(
        exvivo_sitk,
        target_spacing,
        interpolation=sitk.sitkLinear,  # Linear interpolation is usually sufficient for downsampling
        logger=logger,
    )

    return downsampled_sitk


def resample_for_resolution_matching(
    invivo_np: np.ndarray,
    exvivo_np: np.ndarray,
    metadata: Dict[str, Any],
    flip_direction: bool,
    logger: Any,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Match resolution between images based on flip_direction."""
    hr_version = None
    lr_version = None

    if flip_direction:
        logger.info("Upsampling in-vivo to ex-vivo resolution...")
        invivo_lr_np = invivo_np.copy()

        # Create sitk image with proper metadata
        temp_sitk = sitk.GetImageFromArray(invivo_np)
        temp_sitk.SetSpacing(metadata["invivo_spacing"])
        temp_sitk.SetOrigin(metadata["invivo_origin"])
        temp_sitk.SetDirection(metadata["invivo_direction"])

        # Upsample
        hr_sitk = resample_invivo_to_exvivo(temp_sitk, metadata, logger=logger)
        invivo_hr_np = sitk.GetArrayFromImage(hr_sitk)

        invivo_np = invivo_hr_np
        hr_version = invivo_hr_np
        lr_version = invivo_lr_np
    else:
        logger.info("Downsampling ex-vivo to in-vivo resolution...")
        exvivo_hr_np = exvivo_np.copy()

        # Create sitk image with proper metadata
        temp_sitk = sitk.GetImageFromArray(exvivo_np)
        temp_sitk.SetSpacing(metadata["exvivo_spacing"])
        temp_sitk.SetOrigin(metadata["exvivo_origin"])
        temp_sitk.SetDirection(metadata["invivo_direction"])

        # Downsample
        lr_sitk = resample_exvivo_to_invivo(temp_sitk, metadata, logger=logger)
        exvivo_lr_np = sitk.GetArrayFromImage(lr_sitk)

        exvivo_np = exvivo_lr_np
        hr_version = exvivo_hr_np
        lr_version = exvivo_lr_np

    return invivo_np, exvivo_np, hr_version, lr_version


def match_z_axis_resolution(
    source_img: sitk.Image,
    target_z_size: int,
    is_invivo: bool,
    metadata: Dict[str, Any],
    logger: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resample an image to match a target z-dimension."""
    current_size = source_img.GetSize()
    current_spacing = source_img.GetSpacing()

    # Calculate new spacing to maintain physical size
    physical_z_size = current_size[2] * current_spacing[2]
    new_z_spacing = physical_z_size / target_z_size
    new_spacing = (current_spacing[0], current_spacing[1], new_z_spacing)

    logger.debug(f"Resampling with new spacing: {new_spacing}")

    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(source_img.GetOrigin())
    resampler.SetOutputDirection(source_img.GetDirection())
    resampler.SetSize((current_size[0], current_size[1], target_z_size))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    # Resample
    resampled = resampler.Execute(source_img)

    # Remove low-intensity artifacts
    threshold = 0.01
    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(threshold)
    threshold_filter.SetUpperThreshold(float("inf"))
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)

    mask = threshold_filter.Execute(resampled)
    masked = sitk.Multiply(resampled, sitk.Cast(mask, resampled.GetPixelID()))

    # Update metadata
    key = "invivo_spacing" if is_invivo else "exvivo_spacing"
    metadata[key] = new_spacing

    return sitk.GetArrayFromImage(masked), metadata


def ensure_matching_dimensions(
    invivo_np: np.ndarray,
    exvivo_np: np.ndarray,
    metadata: Dict[str, Any],
    flip_direction: bool,
    logger: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Ensure both images have the same dimensions."""
    logger.debug(
        f"Before standardization - In-vivo: {invivo_np.shape}, Ex-vivo: {exvivo_np.shape}"
    )

    # Create SimpleITK images for resampling
    invivo_sitk = sitk.GetImageFromArray(invivo_np)
    exvivo_sitk = sitk.GetImageFromArray(exvivo_np)

    # Set spacing based on pipeline mode
    if flip_direction:
        invivo_sitk.SetSpacing(metadata["exvivo_spacing"])
    else:
        invivo_sitk.SetSpacing(metadata["invivo_spacing"])
    exvivo_sitk.SetSpacing(metadata["exvivo_spacing"])

    # Use consistent direction
    invivo_sitk.SetDirection(metadata["invivo_direction"])
    exvivo_sitk.SetDirection(metadata["invivo_direction"])

    # Check if dimensions mismatch in z-direction
    if invivo_np.shape[0] != exvivo_np.shape[0]:
        logger.warning(
            f"Z-dimension mismatch: in-vivo={invivo_np.shape[0]}, ex-vivo={exvivo_np.shape[0]}"
        )

        # If in-vivo has more slices than ex-vivo, resample ex-vivo
        if invivo_np.shape[0] > exvivo_np.shape[0]:
            logger.info(
                f"Resampling ex-vivo to match in-vivo z-dimension ({invivo_np.shape[0]} slices)"
            )
            exvivo_np, metadata = match_z_axis_resolution(
                source_img=exvivo_sitk,
                target_z_size=invivo_sitk.GetSize()[2],
                is_invivo=False,
                metadata=metadata,
                logger=logger,
            )
        # If ex-vivo has more slices than in-vivo, resample in-vivo
        else:
            logger.info(
                f"Resampling in-vivo to match ex-vivo z-dimension ({exvivo_np.shape[0]} slices)"
            )
            invivo_np, metadata = match_z_axis_resolution(
                source_img=invivo_sitk,
                target_z_size=exvivo_sitk.GetSize()[2],
                is_invivo=True,
                metadata=metadata,
                logger=logger,
            )

    logger.debug(
        f"After standardization - In-vivo: {invivo_np.shape}, Ex-vivo: {exvivo_np.shape}"
    )

    return invivo_np, exvivo_np, metadata
