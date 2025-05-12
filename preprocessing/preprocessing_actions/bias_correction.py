"""
Bias field correction functionality for the preprocessing pipeline.
Implements N4 bias field correction for MRI images.
"""

import logging
import time
from typing import Optional, Tuple

import SimpleITK as sitk
import numpy as np

from preprocessing.utils.error_handling import handle_exception


@handle_exception(reraise=True)
def apply_n4_correction(
    invivo_sitk: sitk.Image,
    exvivo_sitk: sitk.Image,
    apply_correction: bool = True,
    performance_mode: str = "accurate",
    logger: Optional[logging.Logger] = None,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Apply N4 bias field correction to both in-vivo and ex-vivo images.

    Args:
        invivo_sitk: In-vivo SimpleITK image
        exvivo_sitk: Ex-vivo SimpleITK image
        apply_correction: Whether to apply correction or return original images
        performance_mode: "fast", "balanced", or "accurate"
        logger: Logger instance

    Returns:
        Tuple of (invivo_corrected, exvivo_corrected) SimpleITK images
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not apply_correction:
        logger.info("Skipping N4 bias field correction (disabled)")
        return invivo_sitk, exvivo_sitk

    logger.info(f"Applying N4 bias field correction (mode: {performance_mode})")

    # Use all available threads for computation
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(0)

    # Apply correction to in-vivo image
    invivo_corrected = correct_single_image(
        invivo_sitk, "in-vivo", performance_mode, logger
    )

    # Apply correction to ex-vivo image
    exvivo_corrected = correct_single_image(
        exvivo_sitk, "ex-vivo", performance_mode, logger
    )

    return invivo_corrected, exvivo_corrected


@handle_exception(reraise=False)
def correct_single_image(
    image: sitk.Image,
    label: str,
    performance_mode: str = "balanced",
    logger: Optional[logging.Logger] = None,
) -> sitk.Image:
    """
    Apply N4 bias field correction to a single image.

    Args:
        image: SimpleITK image to correct
        label: Label for logging
        performance_mode: "fast", "balanced", or "accurate"
        logger: Logger instance

    Returns:
        Bias-corrected SimpleITK image
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    start_time = time.time()

    # Check if the image has any significant bias that needs correcting
    # Quick test - see if there's significant intensity variation
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    original_variance = stats.GetVariance()
    original_mean = stats.GetMean()

    # Calculate coefficient of variation
    cv = (original_variance**0.5) / original_mean if original_mean > 0 else 0

    # Skip correction if CV is very low
    if cv < 0.1:
        logger.info(
            f"Skipping {label} correction - low intensity variation (CV={cv:.4f})"
        )
        return image

    try:
        # Cast to float if needed
        if image.GetPixelID() != sitk.sitkFloat32:
            image = sitk.Cast(image, sitk.sitkFloat32)

        # Create mask from image - use a lower threshold for ex-vivo images
        threshold = 0.005 if "ex-vivo" in label.lower() else 0.01
        mask = sitk.BinaryThreshold(image, threshold, float("inf"))
        mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])

        # Configure N4 correction with performance mode
        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        # Set parameters based on performance mode
        if performance_mode == "fast":
            logger.debug(f"Using fast mode for {label}")
            corrector.SetMaximumNumberOfIterations([20, 10])
            corrector.SetNumberOfHistogramBins(100)
            corrector.SetBiasFieldFullWidthAtHalfMaximum(0.3)
            corrector.SetConvergenceThreshold(0.002)

            # Use shrink factor for faster processing
            shrink_filter = sitk.ShrinkImageFilter()
            shrink_factor = [2, 2, 2]
            shrink_filter.SetShrinkFactors(shrink_factor)

            image_small = shrink_filter.Execute(image)
            mask_small = shrink_filter.Execute(mask)

            # Apply correction to downsampled image
            logger.debug(f"Applying fast N4 correction to downsampled {label}...")
            corrected_small = corrector.Execute(image_small, mask_small)

            # Compute bias field
            epsilon = 1e-6
            safe_corrected_small = sitk.Add(corrected_small, epsilon)
            bias_small = sitk.Divide(image_small, safe_corrected_small)

            # Smooth the bias field
            bias_small = sitk.DiscreteGaussian(bias_small, 1.0)

            # Upsample bias field to original size
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkLinear)
            bias_full = resampler.Execute(bias_small)

            # Apply upsampled bias field to original image
            corrected = sitk.Divide(image, sitk.Add(bias_full, epsilon))

        elif performance_mode == "balanced":
            logger.debug(f"Using balanced mode for {label}")
            corrector.SetMaximumNumberOfIterations([30, 20, 10])
            corrector.SetNumberOfHistogramBins(150)
            corrector.SetBiasFieldFullWidthAtHalfMaximum(0.2)
            corrector.SetConvergenceThreshold(0.0015)

            logger.debug(f"Applying balanced N4 correction to {label}...")
            corrected = corrector.Execute(image, mask)

        else:  # accurate mode
            logger.debug(f"Using accurate mode for {label}")
            corrector.SetMaximumNumberOfIterations([50, 40, 30])
            corrector.SetNumberOfHistogramBins(200)
            corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
            corrector.SetConvergenceThreshold(0.001)

            logger.debug(f"Applying high-quality N4 correction to {label}...")
            corrected = corrector.Execute(image, mask)

        # Safety check for numerical stability
        corrected_stats = sitk.StatisticsImageFilter()
        corrected_stats.Execute(corrected)

        corrected_variance = corrected_stats.GetVariance()
        corrected_max = corrected_stats.GetMaximum()
        corrected_min = corrected_stats.GetMinimum()

        # Check for problematic values
        if (
            not np.isfinite(corrected_variance)
            or corrected_variance > 1e10
            or corrected_max > 1e10
            or not np.isfinite(corrected_max)
            or not np.isfinite(corrected_min)
            or corrected_min < 0  # Check for negative values
        ):
            logger.warning(
                f"N4 correction produced unstable results for {label}, using original image"
            )
            return image

        # Calculate percent change in variance - if it's very small, maybe correction wasn't needed
        variance_change_percent = (
            abs(corrected_variance - original_variance) / original_variance * 100
        )
        if variance_change_percent < 1.0:
            logger.info(
                f"N4 correction had minimal effect on {label} (variance change: {variance_change_percent:.2f}%)"
            )

        # Fix any remaining black holes by clipping very small values
        if "ex-vivo" in label.lower():
            # Find the 1st percentile of non-zero values as the minimum threshold
            array = sitk.GetArrayFromImage(corrected)
            non_zero_values = array[array > 0]
            if len(non_zero_values) > 0:
                min_threshold = max(np.percentile(non_zero_values, 1), 1e-5)

                # Create a mask of areas to fix (very small but non-zero values)
                fix_mask = sitk.BinaryThreshold(
                    corrected, 0.0, float(min_threshold), 1, 0
                )
                fix_mask = sitk.Cast(fix_mask, sitk.sitkFloat32)

                # Replace these areas with the minimum threshold value
                min_value_image = sitk.Image(corrected.GetSize(), sitk.sitkFloat32)
                min_value_image.CopyInformation(corrected)
                min_value_image = sitk.Add(min_value_image, float(min_threshold))

                # Apply the fix only to the problematic areas
                corrected = sitk.Multiply(corrected, sitk.Subtract(1, fix_mask))
                corrected = sitk.Add(
                    corrected, sitk.Multiply(min_value_image, fix_mask)
                )

                logger.debug(f"Fixed potential black holes in {label} image")

        # Log correction results
        end_time = time.time()
        logger.debug(
            f"{label} correction completed in {end_time - start_time:.2f} seconds"
        )
        logger.debug(f"Original variance: {original_variance:.2f}")
        logger.debug(f"Corrected variance: {corrected_variance:.2f}")
        logger.debug(f"Variance change: {variance_change_percent:.2f}%")

        return corrected

    except RuntimeError as e:
        logger.error(f"N4 correction failed for {label}: {e}")
        logger.info(f"Using original image instead for {label}")
        return image
