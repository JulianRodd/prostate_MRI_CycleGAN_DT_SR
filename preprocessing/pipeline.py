"""
Processing pipeline for MRI preprocessing.
Defines procedures for processing individual pairs of image pairs.
"""

import time
from pathlib import Path
from typing import Tuple, Union

import SimpleITK as sitk

from preprocessing.config import (
    QUICK_RUN,
    SKIP_CLAHE,
    ZSCORE_NORMALIZATION,
    PERCENTILE_LOW,
    PERCENTILE_HIGH,
    NORMALIZATION_RANGE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    GAUSSIAN_SIGMA,
    INITIAL_CROP_MARGIN,
    FINAL_CROP_MARGIN,
    REGISTRATION_THRESHOLD_FLIP,
    REGISTRATION_THRESHOLD_NORMAL,
)
from preprocessing.models import ImagePair, ProcessingResult
from preprocessing.preprocessing_actions.bias_correction import apply_n4_correction
from preprocessing.preprocessing_actions.centering import (
    center_and_crop_images,
    crop_to_content,
)
from preprocessing.preprocessing_actions.clahe import apply_clahe
from preprocessing.preprocessing_actions.io import set_image_metadata
from preprocessing.preprocessing_actions.normalization import normalize_images
from preprocessing.preprocessing_actions.orientation import (
    extract_metadata,
    flip_anterior_posterior,
    flip_left_right,
    standardize_orientation,
)
from preprocessing.preprocessing_actions.registration import register_mri_images_ants
from preprocessing.preprocessing_actions.resampling import (
    resample_for_resolution_matching,
    ensure_matching_dimensions,
)
from preprocessing.utils.error_handling import log_error
from preprocessing.utils.logging import get_logger
from preprocessing.utils.plotting import save_debug_visualizations


def preprocess_mri_pair(
    invivo_path: Union[str, Path],
    exvivo_path: Union[str, Path],
    output_dir: Union[str, Path],
    debug: bool = False,
    flip_invivo_anterior_posterior: bool = False,
    flip_invivo_left_right: bool = False,
    flip_direction: bool = False,
    make_nifti: bool = False,  # Added make_nifti parameter
) -> Union[Tuple[sitk.Image, sitk.Image, sitk.Image, sitk.Image], ProcessingResult]:
    """
    Process a pair of in-vivo and ex-vivo images.

    Args:
        invivo_path: Path to in-vivo image file
        exvivo_path: Path to ex-vivo image file
        output_dir: Directory to save output
        debug: Whether to print detailed debug information
        flip_invivo_anterior_posterior: Whether to flip the in-vivo image along AP axis
        flip_invivo_left_right: Whether to flip the in-vivo image along LR axis
        flip_direction: Whether to upsample in-vivo instead of downsampling ex-vivo
        make_nifti: Whether to save NIfTI files for each debug visualization step

    Returns:
        Either a tuple of processed SimpleITK images or a ProcessingResult object:
        - If flip_direction=True: (invivo, exvivo, invivo_hr, invivo_lr)
        - If flip_direction=False: (invivo, exvivo, exvivo_hr, exvivo_lr)
    """
    # --- STEP 0: Setup and initialization ---
    start_time = time.time()
    pair_id = Path(invivo_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug" if debug else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = get_logger(f"process_{pair_id}", debug=debug)

    # Create ImagePair object
    image_pair = ImagePair(
        pair_id=pair_id, flip_anteroposterior=flip_invivo_anterior_posterior
    )
    result = ProcessingResult(image_pair=image_pair, start_time=start_time)

    try:
        # Log processing start
        logger.info(f"Processing scan pair: {pair_id}")
        if flip_invivo_anterior_posterior:
            logger.info("AP flipping will be applied to in-vivo image")
        if flip_invivo_left_right:
            logger.info("LR flipping will be applied to in-vivo image")
        logger.info(
            f"Processing mode: {'Upsampling in-vivo' if flip_direction else 'Downsampling ex-vivo'}"
        )
        if make_nifti:
            logger.info("NIfTI debug files will be saved for each visualization step")

        # --- STEP 1: Load images ---
        logger.info("Step 1: Loading images...")
        invivo_sitk = sitk.ReadImage(str(invivo_path))
        exvivo_sitk = sitk.ReadImage(str(exvivo_path))

        if debug:
            save_debug_visualizations(
                {"original_invivo": invivo_sitk, "original_exvivo": exvivo_sitk},
                "1",
                output_dir,
                pair_id,
                logger,
                make_nifti=make_nifti,
            )

        # --- STEP 2: Apply flips to in-vivo image if needed ---
        if flip_invivo_anterior_posterior:
            logger.info("Step 2a: Applying anterior-posterior flip to in-vivo image...")
            invivo_array = sitk.GetArrayFromImage(invivo_sitk)
            flipped_array = flip_anterior_posterior(invivo_array, logger)
            flipped_sitk = sitk.GetImageFromArray(flipped_array)
            flipped_sitk.CopyInformation(invivo_sitk)
            invivo_sitk = flipped_sitk
            if debug:
                save_debug_visualizations(
                    {"flipped_invivo": invivo_sitk, "original_exvivo": exvivo_sitk},
                    "2a",
                    output_dir,
                    pair_id,
                    logger,
                    make_nifti=make_nifti,
                )

        if flip_invivo_left_right:
            logger.info("Step 2b: Applying left-right flip to in-vivo image...")
            invivo_array = sitk.GetArrayFromImage(invivo_sitk)
            flipped_array = flip_left_right(invivo_array, logger)
            flipped_sitk = sitk.GetImageFromArray(flipped_array)
            flipped_sitk.CopyInformation(invivo_sitk)
            invivo_sitk = flipped_sitk
            if debug:
                save_debug_visualizations(
                    {"flipped_invivo": invivo_sitk, "original_exvivo": exvivo_sitk},
                    "2b",
                    output_dir,
                    pair_id,
                    logger,
                    make_nifti=make_nifti,
                )

        # --- STEP 3: Apply bias field correction ---
        if QUICK_RUN:
            logger.info("Step 3: Skipping bias field correction (quick run)")
            invivo_corrected = invivo_sitk
            exvivo_corrected = exvivo_sitk
        else:
            logger.info("Step 3: Applying bias field correction...")
            invivo_corrected, exvivo_corrected = apply_n4_correction(
                invivo_sitk, exvivo_sitk, logger=logger
            )

        # --- STEP 3.5-3.6: Extract metadata and standardize orientation ---
        logger.info("Step 3.5: Extracting metadata...")
        metadata = extract_metadata(invivo_corrected, exvivo_corrected, logger)

        logger.info("Step 3.6: Standardizing image orientations...")
        invivo_corrected, exvivo_corrected = standardize_orientation(
            invivo_corrected, exvivo_corrected, metadata, logger
        )

        if debug:
            save_debug_visualizations(
                {
                    "bias_corrected_invivo": invivo_corrected,
                    "bias_corrected_exvivo": exvivo_corrected,
                },
                "3",
                output_dir,
                pair_id,
                logger,
                make_nifti=make_nifti,
            )

        # --- STEP 4: Convert to NumPy arrays ---
        logger.info("Step 4: Converting to NumPy arrays...")
        invivo_np = sitk.GetArrayFromImage(invivo_corrected)
        exvivo_np = sitk.GetArrayFromImage(exvivo_corrected)

        # --- STEP 5: Center and crop images ---
        logger.info("Step 5: Centering and cropping images...")
        invivo_np, exvivo_np, bbox = center_and_crop_images(
            invivo_np, exvivo_np, margin=INITIAL_CROP_MARGIN, logger=logger
        )

        if debug:
            # Convert numpy arrays to SimpleITK for visualization
            centered_invivo_sitk = sitk.GetImageFromArray(invivo_np)
            centered_exvivo_sitk = sitk.GetImageFromArray(exvivo_np)

            # Apply metadata
            set_image_metadata(centered_invivo_sitk, metadata, "invivo", flip_direction)
            set_image_metadata(centered_exvivo_sitk, metadata, "exvivo", flip_direction)

            save_debug_visualizations(
                {
                    "centered_invivo": centered_invivo_sitk,
                    "centered_exvivo": centered_exvivo_sitk,
                },
                "4",
                output_dir,
                pair_id,
                logger,
                make_nifti=make_nifti,
            )

        # --- STEP 6: Perform resolution matching ---
        logger.info("Step 6: Matching resolution between images...")
        invivo_np, exvivo_np, hr_version, lr_version = resample_for_resolution_matching(
            invivo_np, exvivo_np, metadata, flip_direction, logger
        )

        if debug:
            if flip_direction:
                upsampled_invivo_sitk = sitk.GetImageFromArray(invivo_np)
                set_image_metadata(
                    upsampled_invivo_sitk, metadata, "invivo", flip_direction
                )
                save_debug_visualizations(
                    {"upsampled_invivo": upsampled_invivo_sitk},
                    "5",
                    output_dir,
                    pair_id,
                    logger,
                    save_comparison=False,
                    make_nifti=make_nifti,
                )
            else:
                downsampled_exvivo_sitk = sitk.GetImageFromArray(exvivo_np)
                set_image_metadata(
                    downsampled_exvivo_sitk, metadata, "exvivo", flip_direction
                )
                save_debug_visualizations(
                    {"downsampled_exvivo": downsampled_exvivo_sitk},
                    "5",
                    output_dir,
                    pair_id,
                    logger,
                    save_comparison=False,
                    make_nifti=make_nifti,
                )

        # --- STEP 7: Standardize dimensions ---
        logger.info("Step 7: Ensuring matching dimensions between images...")
        invivo_np, exvivo_np, metadata = ensure_matching_dimensions(
            invivo_np, exvivo_np, metadata, flip_direction, logger
        )

        # --- STEP 7b: Reapply centering and cropping ---
        logger.info("Step 7b: Reapplying centering and cropping...")
        invivo_np, exvivo_np, bbox = center_and_crop_images(
            invivo_np, exvivo_np, margin=INITIAL_CROP_MARGIN, logger=logger
        )

        if debug:
            temp_invivo_sitk = sitk.GetImageFromArray(invivo_np)
            temp_exvivo_sitk = sitk.GetImageFromArray(exvivo_np)

            set_image_metadata(temp_invivo_sitk, metadata, "invivo", flip_direction)
            set_image_metadata(temp_exvivo_sitk, metadata, "exvivo", flip_direction)

            save_debug_visualizations(
                {
                    "recentered_invivo": temp_invivo_sitk,
                    "recentered_exvivo": temp_exvivo_sitk,
                },
                "5b",
                output_dir,
                pair_id,
                logger,
                make_nifti=make_nifti,
            )

        # --- STEP 8: Apply image enhancement (normalization and optionally CLAHE) ---
        # Normalize images
        logger.info("Step 8: Normalizing images...")
        norm_invivo, norm_exvivo = normalize_images(
            invivo_np,
            exvivo_np,
            use_zscore=ZSCORE_NORMALIZATION,
            p_low=PERCENTILE_LOW,
            p_high=PERCENTILE_HIGH,
            scale_to_range=NORMALIZATION_RANGE,
            logger=logger,
        )

        if debug:
            norm_invivo_sitk = sitk.GetImageFromArray(norm_invivo)
            norm_exvivo_sitk = sitk.GetImageFromArray(norm_exvivo)
            norm_invivo_sitk.SetDirection(metadata["invivo_direction"])
            norm_exvivo_sitk.SetDirection(metadata["invivo_direction"])

            save_debug_visualizations(
                {
                    "normalized_invivo": norm_invivo_sitk,
                    "normalized_exvivo": norm_exvivo_sitk,
                },
                "6",
                output_dir,
                pair_id,
                logger,
                make_nifti=make_nifti,
            )

        # Skip CLAHE if requested or in quick run
        if SKIP_CLAHE or QUICK_RUN:
            logger.info(
                f"Step 9: Skipping CLAHE ({('as requested' if SKIP_CLAHE else 'quick run')})"
            )
            enhanced_invivo = norm_invivo
            enhanced_exvivo = norm_exvivo
        else:
            # Apply CLAHE
            logger.info("Step 9: Applying CLAHE for contrast enhancement...")
            clahe_invivo, clahe_exvivo = apply_clahe(
                norm_invivo,
                norm_exvivo,
                clip_limit=CLAHE_CLIP_LIMIT,
                tile_grid_size=CLAHE_TILE_GRID_SIZE,
                logger=logger,
            )

            if debug:
                clahe_invivo_sitk = sitk.GetImageFromArray(clahe_invivo)
                clahe_exvivo_sitk = sitk.GetImageFromArray(clahe_exvivo)
                clahe_invivo_sitk.SetDirection(metadata["invivo_direction"])
                clahe_exvivo_sitk.SetDirection(metadata["invivo_direction"])

                save_debug_visualizations(
                    {
                        "clahe_invivo": clahe_invivo_sitk,
                        "clahe_exvivo": clahe_exvivo_sitk,
                    },
                    "7",
                    output_dir,
                    pair_id,
                    logger,
                    make_nifti=make_nifti,
                )

            # Apply Gaussian smoothing
            logger.info("Step 9.5: Applying Gaussian smoothing after CLAHE...")

            # Create SimpleITK images with proper orientation
            clahe_invivo_sitk = sitk.GetImageFromArray(clahe_invivo)
            clahe_exvivo_sitk = sitk.GetImageFromArray(clahe_exvivo)
            clahe_invivo_sitk.SetDirection(metadata["invivo_direction"])
            clahe_exvivo_sitk.SetDirection(metadata["invivo_direction"])

            # Apply smoothing
            clahe_invivo_smoothed = sitk.DiscreteGaussian(
                clahe_invivo_sitk, GAUSSIAN_SIGMA
            )
            clahe_exvivo_smoothed = sitk.DiscreteGaussian(
                clahe_exvivo_sitk, GAUSSIAN_SIGMA
            )

            # Convert back to numpy arrays
            enhanced_invivo = sitk.GetArrayFromImage(clahe_invivo_smoothed)
            enhanced_exvivo = sitk.GetArrayFromImage(clahe_exvivo_smoothed)

            if debug:
                # Save visualizations
                clahe_invivo_sitk = sitk.GetImageFromArray(enhanced_invivo)
                clahe_exvivo_sitk = sitk.GetImageFromArray(enhanced_exvivo)
                clahe_invivo_sitk.SetDirection(metadata["invivo_direction"])
                clahe_exvivo_sitk.SetDirection(metadata["invivo_direction"])

                save_debug_visualizations(
                    {
                        "clahe_smoothed_invivo": clahe_invivo_sitk,
                        "clahe_smoothed_exvivo": clahe_exvivo_sitk,
                    },
                    "7.5",
                    output_dir,
                    pair_id,
                    logger,
                    make_nifti=make_nifti,
                )

        # --- STEP 10: Apply registration ---
        logger.info("Step 10: Applying shape registration between images...")
        debug_path = output_dir / "temp" if debug else None

        try:
            logger.info(
                f"Registration direction: {'ex-vivo into in-vivo' if flip_direction else 'in-vivo into ex-vivo'}"
            )

            # Select appropriate threshold based on direction
            threshold = (
                REGISTRATION_THRESHOLD_FLIP
                if flip_direction
                else REGISTRATION_THRESHOLD_NORMAL
            )

            final_invivo, final_exvivo = register_mri_images_ants(
                enhanced_invivo,
                enhanced_exvivo,
                flip_direction=flip_direction,
                threshold=threshold,
                logger=logger,
                debug_dir=debug_path,
                pair_id=pair_id,
            )

            if debug_path is not None:
                diag_path = debug_path / pair_id / "diagnostics"
                logger.info(f"Diagnostic visualizations saved to: {diag_path}")
        except Exception as e:
            # Continue with original data if registration fails
            logger.error(f"Registration failed but continuing with original data: {e}")
            final_invivo = enhanced_invivo.copy()
            final_exvivo = enhanced_exvivo.copy()

        # --- STEP 11: Apply final center and crop ---
        logger.info("Step 11: Applying final crop to remove excess black space...")
        final_invivo, final_exvivo, final_bbox = center_and_crop_images(
            final_invivo,
            final_exvivo,
            skip_centering=True,
            margin=FINAL_CROP_MARGIN,
            logger=logger,
        )

        if debug:
            final_invivo_sitk = sitk.GetImageFromArray(final_invivo)
            final_exvivo_sitk = sitk.GetImageFromArray(final_exvivo)

            set_image_metadata(final_invivo_sitk, metadata, "invivo", flip_direction)
            set_image_metadata(final_exvivo_sitk, metadata, "exvivo", flip_direction)

            save_debug_visualizations(
                {
                    "final_cropped_invivo": final_invivo_sitk,
                    "final_cropped_exvivo": final_exvivo_sitk,
                },
                "10.5",
                output_dir,
                pair_id,
                logger,
                make_nifti=make_nifti,
            )

        # --- STEP 12: Perform final content-based crop ---
        logger.info("Step 12: Performing final content-based crop...")
        final_invivo, final_exvivo, _ = crop_to_content(
            final_invivo, final_exvivo, 0, logger
        )

        # --- STEP 13: Create final SimpleITK images with metadata ---
        logger.info("Step 13: Creating final SimpleITK images with metadata...")
        final_invivo_sitk = sitk.GetImageFromArray(final_invivo)
        final_exvivo_sitk = sitk.GetImageFromArray(final_exvivo)

        set_image_metadata(final_invivo_sitk, metadata, "invivo", flip_direction)
        set_image_metadata(final_exvivo_sitk, metadata, "exvivo", flip_direction)

        # Final debug visualization
        if debug:
            save_debug_visualizations(
                {"final_invivo": final_invivo_sitk, "final_exvivo": final_exvivo_sitk},
                "final",
                output_dir,
                pair_id,
                logger,
                make_nifti=make_nifti,
            )

        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")

        # --- STEP 14: Prepare return values based on processing direction ---
        logger.info("Step 14: Preparing return values...")
        if flip_direction:
            # For upsampling in-vivo
            if hr_version is not None and lr_version is not None:
                invivo_hr_sitk = sitk.GetImageFromArray(hr_version)
                invivo_lr_sitk = sitk.GetImageFromArray(lr_version)
                set_image_metadata(invivo_hr_sitk, metadata, "invivo", True)
                set_image_metadata(invivo_lr_sitk, metadata, "invivo", False)
                return (
                    final_invivo_sitk,
                    final_exvivo_sitk,
                    invivo_hr_sitk,
                    invivo_lr_sitk,
                )
            else:
                return (final_invivo_sitk, final_exvivo_sitk, None, None)
        else:
            # For downsampling ex-vivo
            if hr_version is not None:
                exvivo_hr_sitk = sitk.GetImageFromArray(hr_version)
                set_image_metadata(exvivo_hr_sitk, metadata, "exvivo", False)
                return (
                    final_invivo_sitk,
                    final_exvivo_sitk,
                    exvivo_hr_sitk,
                    final_exvivo_sitk,
                )
            else:
                return (final_invivo_sitk, final_exvivo_sitk, None, None)

    except Exception as e:
        # --- Error handling ---
        logger.error(f"Failed to process pair {pair_id}: {e}")
        log_error(pair_id, str(e))

        import traceback

        logger.debug(traceback.format_exc())

        result.success = False
        result.error_message = str(e)
        result.end_time = time.time()

        raise RuntimeError(f"Failed to process pair {pair_id}: {e}")
