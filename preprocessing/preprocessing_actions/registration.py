import os
from pathlib import Path
from typing import Tuple, Optional, Callable, Union

import ants
import numpy as np
from scipy import ndimage

from preprocessing.config import REGISTRATION_CONFIG
from preprocessing.preprocessing_actions.histogram_matching import (
    apply_histogram_matching,
)
from preprocessing.preprocessing_actions.masking import (
    create_tissue_mask,
    clean_binary_mask,
)
from preprocessing.utils.plotting import (
    plot_mri_slices,
    plot_histograms,
    find_best_content_slice,
)


def register_mri_images_ants(
    invivo_data: np.ndarray,
    exvivo_data: np.ndarray,
    flip_direction: bool = REGISTRATION_CONFIG["general"]["flip_direction"],
    threshold: float = REGISTRATION_CONFIG["general"]["threshold"],
    logger: Optional[Callable] = None,
    debug_dir: Optional[Union[str, Path]] = None,
    pair_id: Optional[str] = None,
    histogram_match: bool = REGISTRATION_CONFIG["general"]["histogram_match"],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ANTs implementation of the warping function with mask-based processing for consistent
    representation across images.
    """
    try:
        if logger:
            logger.info(
                f"Warping {'ex-vivo into in-vivo' if flip_direction else 'in-vivo into ex-vivo'} using ANTs"
            )

        # Create diagnostic directory
        diag_dir = None
        if debug_dir is not None and pair_id is not None:
            diag_dir = Path(debug_dir) / pair_id / "diagnostics"
            os.makedirs(diag_dir, exist_ok=True)

        # Find representative slices for visualization
        inv_slice_idx = find_best_content_slice(invivo_data)
        ex_slice_idx = find_best_content_slice(exvivo_data)

        # Log data shapes
        if logger:
            logger.debug(f"In-vivo data shape: {invivo_data.shape}")
            logger.debug(f"Ex-vivo data shape: {exvivo_data.shape}")

        # Plot original images if debug enabled
        if diag_dir is not None:
            try:
                inv_slice = (
                    invivo_data[:, :, inv_slice_idx]
                    if invivo_data.ndim == 3
                    else invivo_data
                )
                ex_slice = (
                    exvivo_data[:, :, ex_slice_idx]
                    if exvivo_data.ndim == 3
                    else exvivo_data
                )

                plot_mri_slices(
                    [inv_slice, ex_slice],
                    ["In-vivo Original", "Ex-vivo Original"],
                    diag_dir / "01_original_images.png",
                    cmap="gray",
                )
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save diagnostic plot: {e}")

        # Create masks for both images with optimized parameters
        invivo_mask = create_tissue_mask(
            invivo_data,
            threshold=threshold
            * REGISTRATION_CONFIG["invivo_mask"]["threshold_factor"],
            fill_holes=REGISTRATION_CONFIG["invivo_mask"]["fill_holes"],
            smooth_sigma=REGISTRATION_CONFIG["invivo_mask"]["smooth_sigma"],
            iterations=REGISTRATION_CONFIG["invivo_mask"]["iterations"],
        )

        exvivo_mask = create_tissue_mask(
            exvivo_data,
            threshold=threshold
            * REGISTRATION_CONFIG["exvivo_mask"]["threshold_factor"],
            fill_holes=REGISTRATION_CONFIG["exvivo_mask"]["fill_holes"],
            smooth_sigma=REGISTRATION_CONFIG["exvivo_mask"]["smooth_sigma"],
            iterations=REGISTRATION_CONFIG["exvivo_mask"]["iterations"],
        )

        # Visualize masks if in debug mode
        if diag_dir is not None:
            try:
                inv_mask_slice = (
                    invivo_mask[:, :, inv_slice_idx]
                    if invivo_mask.ndim == 3
                    else invivo_mask
                )
                ex_mask_slice = (
                    exvivo_mask[:, :, ex_slice_idx]
                    if exvivo_mask.ndim == 3
                    else exvivo_mask
                )

                plot_mri_slices(
                    [inv_mask_slice.astype(float), ex_mask_slice.astype(float)],
                    ["In-vivo Mask", "Ex-vivo Mask"],
                    diag_dir / "02_masks.png",
                )
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save mask visualization: {e}")

        # Prepare data for ANTs
        # Convert to float32 and normalize to 0-1 range
        invivo_data_float = invivo_data.astype(np.float32)
        exvivo_data_float = exvivo_data.astype(np.float32)

        if np.max(invivo_data_float) > 0:
            invivo_data_float = invivo_data_float / np.max(invivo_data_float)
        if np.max(exvivo_data_float) > 0:
            exvivo_data_float = exvivo_data_float / np.max(exvivo_data_float)

        # Convert masks to float for ANTs
        invivo_mask_float = invivo_mask.astype(np.float32)
        exvivo_mask_float = exvivo_mask.astype(np.float32)

        # Convert to ANTs format with proper spacing
        if invivo_data.ndim == 3:
            fixed_image = ants.from_numpy(invivo_data_float)
            moving_image = ants.from_numpy(exvivo_data_float)
            fixed_mask = ants.from_numpy(invivo_mask_float)
            moving_mask = ants.from_numpy(exvivo_mask_float)

            # Set image spacing to ensure proper alignment
            for img in [fixed_image, moving_image, fixed_mask, moving_mask]:
                img.set_spacing([1.0] * 3)
        else:
            # Handle 2D images - add a dimension for ANTs
            fixed_image = ants.from_numpy(invivo_data_float[:, :, np.newaxis])
            moving_image = ants.from_numpy(exvivo_data_float[:, :, np.newaxis])
            fixed_mask = ants.from_numpy(invivo_mask_float[:, :, np.newaxis])
            moving_mask = ants.from_numpy(exvivo_mask_float[:, :, np.newaxis])

            # Set image spacing for 2D+1 data
            for img in [fixed_image, moving_image, fixed_mask, moving_mask]:
                img.set_spacing([1.0, 1.0, 1.0])

        if flip_direction:
            # WARP EX-VIVO TO IN-VIVO SHAPE
            if logger:
                logger.info("Performing ANTs registration: ex-vivo to in-vivo")

            try:
                # Apply histogram matching if enabled
                if histogram_match:
                    if logger:
                        logger.info(
                            "Applying histogram matching: in-vivo to ex-vivo (better contrast preservation)"
                        )

                    # Create visualization of histograms before matching
                    if diag_dir is not None:
                        try:
                            # Extract central slice for histogram
                            fixed_slice = (
                                fixed_image[:, :, fixed_image.shape[2] // 2]
                                if fixed_image.dimension == 3
                                else fixed_image
                            )
                            moving_slice = (
                                moving_image[:, :, moving_image.shape[2] // 2]
                                if moving_image.dimension == 3
                                else moving_image
                            )

                            # Visualize histograms before matching
                            plot_histograms(
                                fixed_slice.numpy(),
                                moving_slice.numpy(),
                                "Before Matching",
                                diag_dir / "02b_before_histogram_matching.png",
                                logger,
                            )
                        except Exception as e:
                            if logger:
                                logger.warning(
                                    f"Failed to save histogram visualization: {e}"
                                )

                    fixed_image_matched = apply_histogram_matching(
                        fixed_image,  # in-vivo to be matched
                        moving_image,  # ex-vivo reference
                        source_mask=fixed_mask,
                        reference_mask=moving_mask,
                        logger=logger,
                    )

                    if fixed_image_matched is not None:
                        fixed_image = fixed_image_matched
                        if logger:
                            logger.info(
                                "Histogram matching successful (in-vivo matched to ex-vivo)"
                            )

                        # Create visualization of histograms after matching
                        if diag_dir is not None:
                            try:
                                # Extract central slice for histogram
                                fixed_slice = (
                                    fixed_image[:, :, fixed_image.shape[2] // 2]
                                    if fixed_image.dimension == 3
                                    else fixed_image
                                )
                                moving_slice = (
                                    moving_image[:, :, moving_image.shape[2] // 2]
                                    if moving_image.dimension == 3
                                    else moving_image
                                )

                                # Visualize histograms after matching
                                plot_histograms(
                                    fixed_slice.numpy(),
                                    moving_slice.numpy(),
                                    "After Matching",
                                    diag_dir / "02c_after_histogram_matching.png",
                                    logger,
                                )
                            except Exception as e:
                                if logger:
                                    logger.warning(
                                        f"Failed to save histogram visualization: {e}"
                                    )

                # Set up registration parameters
                reg_params = {
                    "type_of_transform": REGISTRATION_CONFIG["registration"][
                        "type_of_transform"
                    ],
                    "aff_metric": REGISTRATION_CONFIG["registration"]["aff_metric"],
                    "syn_metric": REGISTRATION_CONFIG["registration"]["syn_metric"],
                    "aff_iterations": REGISTRATION_CONFIG["registration"][
                        "aff_iterations"
                    ],
                    "aff_shrink_factors": REGISTRATION_CONFIG["registration"][
                        "aff_shrink_factors"
                    ],
                    "aff_smoothing_sigmas": REGISTRATION_CONFIG["registration"][
                        "aff_smoothing_sigmas"
                    ],
                    "syn_iterations": REGISTRATION_CONFIG["registration"][
                        "syn_iterations"
                    ],
                    "syn_shrink_factors": REGISTRATION_CONFIG["registration"][
                        "syn_shrink_factors"
                    ],
                    "syn_smoothing_sigmas": REGISTRATION_CONFIG["registration"][
                        "syn_smoothing_sigmas"
                    ],
                    "grad_step": REGISTRATION_CONFIG["registration"]["grad_step"],
                    "verbose": True if logger else False,
                }

                # Perform registration
                registration = ants.registration(
                    fixed=fixed_image,
                    moving=moving_image,
                    mask=fixed_mask,  # Use in-vivo mask to force warping within region
                    type_of_transform=reg_params["type_of_transform"],
                    aff_metric=reg_params["aff_metric"],
                    syn_metric=reg_params["syn_metric"],
                    aff_iterations=reg_params["aff_iterations"],
                    aff_shrink_factors=reg_params["aff_shrink_factors"],
                    aff_smoothing_sigmas=reg_params["aff_smoothing_sigmas"],
                    reg_iterations=reg_params["syn_iterations"],
                    reg_shrink_factors=reg_params["syn_shrink_factors"],
                    reg_smoothing_sigmas=reg_params["syn_smoothing_sigmas"],
                    gradient_step=reg_params["grad_step"],
                    verbose=reg_params["verbose"],
                )

                if logger:
                    logger.info("ANTs registration completed successfully")

                # Extract the warped image
                warped_exvivo_ants = registration["warpedmovout"]

                # Convert back to numpy
                warped_exvivo = warped_exvivo_ants.numpy()
                if invivo_data.ndim == 2:
                    warped_exvivo = warped_exvivo[
                        :, :, 0
                    ]  # Remove added dimension for 2D case

                # Post-processing steps

                # 1. Remove light gray values in edge regions only (not within tissue)
                bite_threshold = REGISTRATION_CONFIG["general"]["bite_threshold"]

                # Create a copy of the warped mask to identify tissue regions
                tissue_mask = create_tissue_mask(
                    warped_exvivo,
                    threshold=threshold
                    * REGISTRATION_CONFIG["exvivo_mask"]["threshold_factor"],
                    fill_holes=REGISTRATION_CONFIG["exvivo_mask"]["fill_holes"],
                    smooth_sigma=REGISTRATION_CONFIG["exvivo_mask"]["smooth_sigma"],
                    iterations=REGISTRATION_CONFIG["exvivo_mask"]["iterations"],
                )

                # Identify edge regions by eroding the mask and finding difference
                inner_region = ndimage.binary_erosion(
                    tissue_mask,
                    iterations=REGISTRATION_CONFIG["post_processing"].get(
                        "edge_detection_erosion", 3
                    ),
                )
                edge_region = (
                    tissue_mask & ~inner_region
                )  # Edge is the difference between mask and eroded mask

                # Apply threshold only to edge regions, preserving interior tissue values
                edge_pixels = edge_region > 0
                low_intensity_pixels = warped_exvivo < bite_threshold
                # Only apply thresholding where both conditions are true: edge region AND low intensity
                pixels_to_zero = edge_pixels & low_intensity_pixels
                warped_exvivo[pixels_to_zero] = 0.0

                # Optional: Log information about the process
                if logger:
                    edge_pixel_count = np.sum(edge_pixels)
                    thresholded_pixel_count = np.sum(pixels_to_zero)
                    logger.debug(f"Edge pixels identified: {edge_pixel_count}")
                    logger.debug(f"Thresholded edge pixels: {thresholded_pixel_count}")

                # 2. Apply mask to ensure no data outside target region
                invivo_mask_dilated = np.zeros_like(invivo_mask)
                if invivo_data.ndim == 3:
                    for z in range(invivo_mask.shape[2]):
                        invivo_mask_dilated[:, :, z] = ndimage.binary_dilation(
                            invivo_mask[:, :, z],
                            iterations=REGISTRATION_CONFIG["post_processing"][
                                "mask_dilation_iterations"
                            ],
                        )
                else:
                    invivo_mask_dilated = ndimage.binary_dilation(
                        invivo_mask,
                        iterations=REGISTRATION_CONFIG["post_processing"][
                            "mask_dilation_iterations"
                        ],
                    )

                warped_exvivo = warped_exvivo * invivo_mask_dilated

                # 3. Create clean mask from warped ex-vivo
                warped_exvivo_mask = create_tissue_mask(
                    warped_exvivo,
                    threshold=threshold
                    * REGISTRATION_CONFIG["exvivo_mask"]["threshold_factor"],
                    fill_holes=REGISTRATION_CONFIG["exvivo_mask"]["fill_holes"],
                    smooth_sigma=REGISTRATION_CONFIG["exvivo_mask"]["smooth_sigma"],
                    iterations=REGISTRATION_CONFIG["exvivo_mask"]["iterations"],
                )

                # 4. Process mask for clean edges and consistent bites
                warped_exvivo_mask = clean_binary_mask(
                    warped_exvivo_mask,
                    REGISTRATION_CONFIG["post_processing"]["clean_mask_iterations"],
                    REGISTRATION_CONFIG["post_processing"]["clean_mask_dilate"],
                    REGISTRATION_CONFIG["post_processing"]["clean_mask_erode"],
                )

                # 5. Apply mask to both images for consistency
                warped_exvivo = warped_exvivo * warped_exvivo_mask
                invivo_result = invivo_data.copy() * warped_exvivo_mask

                # 6. Apply light smoothing for artifact removal
                warped_exvivo = ndimage.gaussian_filter(
                    warped_exvivo,
                    sigma=REGISTRATION_CONFIG["post_processing"][
                        "gaussian_filter_sigma"
                    ],
                )

                # 7. Rescale back to original intensity range
                if np.max(exvivo_data) > 0:
                    warped_exvivo = warped_exvivo * np.max(exvivo_data)

                # Visualize results if in debug mode
                if diag_dir is not None:
                    try:
                        # Visualize mask
                        mask_slice = (
                            warped_exvivo_mask[:, :, inv_slice_idx]
                            if invivo_data.ndim == 3
                            else warped_exvivo_mask
                        )
                        plot_mri_slices(
                            [mask_slice.astype(float)],
                            ["Warped Ex-vivo Mask"],
                            diag_dir / "03_warped_exvivo_mask.png",
                            cmap="gray",
                        )

                        # Visualize final results
                        inv_slice = (
                            invivo_result[:, :, inv_slice_idx]
                            if invivo_data.ndim == 3
                            else invivo_result
                        )
                        warped_slice = (
                            warped_exvivo[:, :, inv_slice_idx]
                            if invivo_data.ndim == 3
                            else warped_exvivo
                        )

                        # Normalize for visualization
                        inv_slice = (
                            inv_slice / np.max(inv_slice)
                            if np.max(inv_slice) > 0
                            else inv_slice
                        )
                        warped_slice = (
                            warped_slice / np.max(warped_slice)
                            if np.max(warped_slice) > 0
                            else warped_slice
                        )

                        # Plot warped result
                        plot_mri_slices(
                            [warped_slice],
                            ["Warped Ex-vivo (ANTs)"],
                            diag_dir / "03_warped_exvivo.png",
                            cmap="gray",
                        )

                        # Plot comparison
                        plot_mri_slices(
                            [inv_slice, warped_slice],
                            ["Target In-vivo", "Warped Ex-vivo (ANTs)"],
                            diag_dir / "03_result.png",
                            cmap="gray",
                        )
                    except Exception as e:
                        if logger:
                            logger.warning(f"Failed to save result visualization: {e}")

                return invivo_result, warped_exvivo

            except Exception as e:
                if logger:
                    logger.error(f"ANTs registration failed: {e}")
                # Fall back to original data
                return invivo_data, exvivo_data

        else:
            # WARP IN-VIVO TO EX-VIVO SHAPE
            if logger:
                logger.info("Performing ANTs registration: in-vivo to ex-vivo")

            try:
                # For this direction, fixed=ex-vivo and moving=in-vivo
                fixed_image_here = moving_image  # ex-vivo is fixed
                moving_image_here = fixed_image  # in-vivo is moving
                fixed_mask_here = moving_mask  # ex-vivo mask
                moving_mask_here = fixed_mask  # in-vivo mask

                # Apply histogram matching if enabled
                if histogram_match:
                    if logger:
                        logger.info("Applying histogram matching: in-vivo to ex-vivo")

                    # Create visualization of histograms before matching
                    if diag_dir is not None:
                        try:
                            # Extract central slice for histogram
                            fixed_slice = (
                                fixed_image_here[:, :, fixed_image_here.shape[2] // 2]
                                if fixed_image_here.dimension == 3
                                else fixed_image_here
                            )
                            moving_slice = (
                                moving_image_here[:, :, moving_image_here.shape[2] // 2]
                                if moving_image_here.dimension == 3
                                else moving_image_here
                            )

                            # Visualize histograms before matching
                            plot_histograms(
                                fixed_slice.numpy(),
                                moving_slice.numpy(),
                                "Before Matching",
                                diag_dir / "02b_before_histogram_matching.png",
                                logger,
                            )
                        except Exception as e:
                            if logger:
                                logger.warning(
                                    f"Failed to save histogram visualization: {e}"
                                )

                    # Apply histogram matching
                    moving_image_matched = apply_histogram_matching(
                        moving_image_here,  # in-vivo to be matched
                        fixed_image_here,  # ex-vivo reference
                        source_mask=moving_mask_here,
                        reference_mask=fixed_mask_here,
                        logger=logger,
                    )

                    if moving_image_matched is not None:
                        moving_image_here = moving_image_matched
                        if logger:
                            logger.info("Histogram matching successful")

                        # Create visualization of histograms after matching
                        if diag_dir is not None:
                            try:
                                # Extract central slice for histogram
                                fixed_slice = (
                                    fixed_image_here[
                                        :, :, fixed_image_here.shape[2] // 2
                                    ]
                                    if fixed_image_here.dimension == 3
                                    else fixed_image_here
                                )
                                moving_slice = (
                                    moving_image_here[
                                        :, :, moving_image_here.shape[2] // 2
                                    ]
                                    if moving_image_here.dimension == 3
                                    else moving_image_here
                                )

                                # Visualize histograms after matching
                                plot_histograms(
                                    fixed_slice.numpy(),
                                    moving_slice.numpy(),
                                    "After Matching",
                                    diag_dir / "02c_after_histogram_matching.png",
                                    logger,
                                )
                            except Exception as e:
                                if logger:
                                    logger.warning(
                                        f"Failed to save histogram visualization: {e}"
                                    )

                # Set up registration parameters (same as above)
                reg_params = {
                    "type_of_transform": REGISTRATION_CONFIG["registration"][
                        "type_of_transform"
                    ],
                    "aff_metric": REGISTRATION_CONFIG["registration"]["aff_metric"],
                    "syn_metric": REGISTRATION_CONFIG["registration"]["syn_metric"],
                    "aff_iterations": REGISTRATION_CONFIG["registration"][
                        "aff_iterations"
                    ],
                    "aff_shrink_factors": REGISTRATION_CONFIG["registration"][
                        "aff_shrink_factors"
                    ],
                    "aff_smoothing_sigmas": REGISTRATION_CONFIG["registration"][
                        "aff_smoothing_sigmas"
                    ],
                    "syn_iterations": REGISTRATION_CONFIG["registration"][
                        "syn_iterations"
                    ],
                    "syn_shrink_factors": REGISTRATION_CONFIG["registration"][
                        "syn_shrink_factors"
                    ],
                    "syn_smoothing_sigmas": REGISTRATION_CONFIG["registration"][
                        "syn_smoothing_sigmas"
                    ],
                    "grad_step": REGISTRATION_CONFIG["registration"]["grad_step"],
                    "verbose": True if logger else False,
                }

                # Perform registration
                registration = ants.registration(
                    fixed=fixed_image_here,  # ex-vivo is fixed
                    moving=moving_image_here,  # in-vivo is moving
                    mask=fixed_mask_here,  # ex-vivo mask
                    type_of_transform=reg_params["type_of_transform"],
                    aff_metric=reg_params["aff_metric"],
                    syn_metric=reg_params["syn_metric"],
                    aff_iterations=reg_params["aff_iterations"],
                    aff_shrink_factors=reg_params["aff_shrink_factors"],
                    aff_smoothing_sigmas=reg_params["aff_smoothing_sigmas"],
                    reg_iterations=reg_params["syn_iterations"],
                    reg_shrink_factors=reg_params["syn_shrink_factors"],
                    reg_smoothing_sigmas=reg_params["syn_smoothing_sigmas"],
                    gradient_step=reg_params["grad_step"],
                    verbose=reg_params["verbose"],
                )

                if logger:
                    logger.info("ANTs registration completed successfully")

                # Extract the warped image
                warped_invivo_ants = registration["warpedmovout"]

                # Convert back to numpy
                warped_invivo = warped_invivo_ants.numpy()
                if invivo_data.ndim == 2:
                    warped_invivo = warped_invivo[
                        :, :, 0
                    ]  # Remove added dimension for 2D case

                # Post-processing steps

                # 1. Remove light gray values in regions that should be black
                bite_threshold = REGISTRATION_CONFIG["general"]["bite_threshold"]
                warped_invivo[warped_invivo < bite_threshold] = 0.0

                # 2. Process ex-vivo mask for clean edges and consistent bites
                refined_exvivo_mask = clean_binary_mask(
                    exvivo_mask,
                    REGISTRATION_CONFIG["post_processing"]["clean_mask_iterations"],
                    REGISTRATION_CONFIG["post_processing"]["clean_mask_dilate"],
                    REGISTRATION_CONFIG["post_processing"]["clean_mask_erode"],
                )

                # 3. Apply mask to ensure consistent bites
                warped_invivo = warped_invivo * refined_exvivo_mask

                # 4. Apply light smoothing for artifact removal
                warped_invivo = ndimage.gaussian_filter(
                    warped_invivo,
                    sigma=REGISTRATION_CONFIG["post_processing"][
                        "gaussian_filter_sigma"
                    ],
                )

                # 5. Rescale back to original intensity range
                if np.max(invivo_data) > 0:
                    warped_invivo = warped_invivo * np.max(invivo_data)

                # Visualize results if in debug mode
                if diag_dir is not None:
                    try:
                        # Visualize mask
                        mask_slice = (
                            refined_exvivo_mask[:, :, ex_slice_idx]
                            if exvivo_data.ndim == 3
                            else refined_exvivo_mask
                        )
                        plot_mri_slices(
                            [mask_slice.astype(float)],
                            ["Refined Ex-vivo Mask"],
                            diag_dir / "03_refined_exvivo_mask.png",
                            cmap="gray",
                        )

                        # Visualize final results
                        ex_slice = (
                            exvivo_data[:, :, ex_slice_idx]
                            if exvivo_data.ndim == 3
                            else exvivo_data
                        )
                        warped_slice = (
                            warped_invivo[:, :, ex_slice_idx]
                            if exvivo_data.ndim == 3
                            else warped_invivo
                        )

                        # Normalize for visualization
                        ex_slice = (
                            ex_slice / np.max(ex_slice)
                            if np.max(ex_slice) > 0
                            else ex_slice
                        )
                        warped_slice = (
                            warped_slice / np.max(warped_slice)
                            if np.max(warped_slice) > 0
                            else warped_slice
                        )

                        # Plot warped result
                        plot_mri_slices(
                            [warped_slice],
                            ["Warped In-vivo (ANTs)"],
                            diag_dir / "03_warped_invivo.png",
                            cmap="gray",
                        )

                        # Plot comparison
                        plot_mri_slices(
                            [ex_slice, warped_slice],
                            ["Target Ex-vivo", "Warped In-vivo (ANTs)"],
                            diag_dir / "03_result.png",
                            cmap="gray",
                        )
                    except Exception as e:
                        if logger:
                            logger.warning(f"Failed to save result visualization: {e}")

                return warped_invivo, exvivo_data

            except Exception as e:
                if logger:
                    logger.error(f"ANTs registration failed: {e}")
                # Fall back to original data
                return invivo_data, exvivo_data

    except Exception as e:
        # Catch all exceptions to prevent crashing
        if logger:
            logger.error(f"Warping failed: {e}")
        # Return original data if warping fails
        return invivo_data, exvivo_data
