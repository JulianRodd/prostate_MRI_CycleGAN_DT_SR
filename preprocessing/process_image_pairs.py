import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Union

import SimpleITK as sitk
from tqdm import tqdm

from preprocessing.models import ProcessingResult
from preprocessing.preprocessing_actions.io import save_final_results
from preprocessing.utils.error_handling import log_error
from preprocessing.utils.logging import get_logger


def process_image_pairs(
    input_dir_invivo: Union[str, Path],
    input_dir_exvivo: Union[str, Path],
    output_dir: Union[str, Path],
    test_count: int = 5,
    debug: bool = False,
    flip_direction: bool = False,
    skip_list: Optional[List[str]] = None,
    only_run: Optional[List[str]] = None,
    flip_cases_ap: Optional[List[str]] = None,
    flip_cases_lr: Optional[List[str]] = None,
    padding_value: float = 0.0,
    num_threads: int = 1,  # Parameter for multi-threading
    make_nifti: bool = False,  # New parameter for saving NIfTI files
) -> List[Path]:
    """
    Process all paired in-vivo and ex-vivo MRI scans in the input directories.
    Also generates HR/LR pairs for super-resolution training.
    Now supports multi-threading for parallel processing.

    Args:
        input_dir_invivo: Directory containing in-vivo MRI scans
        input_dir_exvivo: Directory containing ex-vivo MRI scans
        output_dir: Directory where processed pairs will be saved
        test_count: Number of pairs to include in test set
        debug: Whether to print detailed debug information
        flip_direction: Whether to flip the processing direction
        skip_list: List of IDs to skip
        only_run: List of IDs to exclusively process
        flip_cases_ap: List of IDs requiring anterior-posterior flipping
        flip_cases_lr: List of IDs requiring left-right flipping
        padding_value: Value to use for padding when standardizing dimensions
        num_threads: Number of threads to use for parallel processing
        make_nifti: Whether to save NIfTI files for each debug visualization step

    Returns:
        List of paths to all processed image pairs
    """
    from preprocessing.preprocessing_actions.io import (
        find_paired_images,
        report_processing_stats,
        standardize_dimensions,
    )

    # Set default values for lists
    skip_list = skip_list or []
    only_run = only_run or []
    # Set default values for lists
    flip_cases_ap = flip_cases_ap or []
    flip_cases_lr = flip_cases_lr or []

    # Setup logger
    logger = get_logger("batch_processing", debug=debug)

    # Limit number of threads to a reasonable value
    num_threads = min(num_threads, os.cpu_count() or 1)

    # Log job information
    logger.info("Starting batch processing job")
    logger.info(f"Input in-vivo directory: {input_dir_invivo}")
    logger.info(f"Input ex-vivo directory: {input_dir_exvivo}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Processing direction: {'Upsampling in-vivo' if flip_direction else 'Downsampling ex-vivo'}"
    )
    logger.info(f"Using {num_threads} threads for processing")

    # Create output directories
    output_path = Path(output_dir)

    for split in ["train", "test"]:
        os.makedirs(output_path / split / "invivo", exist_ok=True)
        os.makedirs(output_path / split / "exvivo", exist_ok=True)
        os.makedirs(output_path / split / "sr_hr", exist_ok=True)
        os.makedirs(output_path / split / "sr_lr", exist_ok=True)

    temp_dir = output_path / "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Find paired images
    pairs, invivo_only, exvivo_only = find_paired_images(
        input_dir_invivo, input_dir_exvivo, logger, skip_list, only_run
    )

    # Report statistics
    report_processing_stats(pairs, invivo_only, exvivo_only, logger)

    def process_pair_task(task_index, pair_data):
        pair_id, invivo_file, exvivo_file = pair_data

        # Create a thread-specific logger
        thread_logger = get_logger(f"thread_{pair_id}", debug=debug)

        # Set the split for this pair
        pair_split = "test" if task_index < test_count else "train"

        thread_logger.info(
            f"Processing pair {task_index + 1}/{len(pairs)}: {pair_id} (split: {pair_split})"
        )

        # Setup paths
        invivo_path = os.path.join(input_dir_invivo, invivo_file)
        exvivo_path = os.path.join(input_dir_exvivo, exvivo_file)

        # Create directories (thread-safe)
        pair_output_dir = temp_dir / pair_id
        os.makedirs(pair_output_dir, exist_ok=True)

        # Check if this pair needs AP flipping
        flip_ap = pair_id in flip_cases_ap
        flip_lr = pair_id in flip_cases_lr
        if flip_ap:
            thread_logger.info(
                f"Special case: applying anterior-posterior flipping for {pair_id}"
            )
        if flip_lr:
            thread_logger.info(
                f"Special case: applying left-right flipping for {pair_id}"
            )

        try:
            # Process the pair
            from preprocessing.pipeline import preprocess_mri_pair

            processing_result = preprocess_mri_pair(
                invivo_path=invivo_path,
                exvivo_path=exvivo_path,
                output_dir=output_path,
                debug=debug,
                flip_invivo_anterior_posterior=flip_ap,
                flip_invivo_left_right=flip_lr,
                flip_direction=flip_direction,
                make_nifti=make_nifti,  # Pass the make_nifti flag
            )

            # Create output directories for this split (thread-safe)
            for subdir in ["invivo", "exvivo", "sr_hr", "sr_lr"]:
                os.makedirs(output_path / pair_split / subdir, exist_ok=True)

            # Define output file paths
            invivo_output = output_path / pair_split / "invivo" / f"{pair_id}.nii.gz"
            exvivo_output = output_path / pair_split / "exvivo" / f"{pair_id}.nii.gz"

            # Initialize list for paths to return
            result_paths = []

            # Handle different result types
            if isinstance(processing_result, tuple) and len(processing_result) == 4:
                if flip_direction:
                    # Upsampling in-vivo
                    final_invivo, final_exvivo, invivo_hr, invivo_lr = processing_result

                    # Save image pair
                    sitk.WriteImage(final_invivo, str(invivo_output), False)
                    sitk.WriteImage(final_exvivo, str(exvivo_output), False)

                    result_paths.extend([invivo_output, exvivo_output])
                    thread_logger.info(
                        f"Saved to {pair_split} set: {invivo_output.name} and {exvivo_output.name}"
                    )

                else:
                    # Downsampling ex-vivo
                    final_invivo, final_exvivo, exvivo_hr, exvivo_lr = processing_result

                    # Save image pair
                    sitk.WriteImage(final_invivo, str(invivo_output), True)
                    sitk.WriteImage(final_exvivo, str(exvivo_output), True)

                    result_paths.extend([invivo_output, exvivo_output])
                    thread_logger.info(
                        f"Saved to {pair_split} set: {invivo_output.name} and {exvivo_output.name}"
                    )

            elif isinstance(processing_result, ProcessingResult):
                # Handle ProcessingResult objects
                if processing_result.success:
                    # Save results using the existing function
                    output_paths = save_final_results(
                        processing_result.image_pair,
                        output_path,
                        pair_split,
                        compress=False,
                    )
                    result_paths.extend(list(output_paths.values()))
                    thread_logger.info(f"Saved processing results to {pair_split} set")
                else:
                    # If processing failed, raise an exception to be handled
                    raise RuntimeError(processing_result.error_message)

            # Return success with the paths
            return {
                "success": True,
                "pair_id": pair_id,
                "paths": result_paths,
                "split": pair_split,
            }

        except Exception as e:
            # Log errors and return failure
            thread_logger.error(f"Failed to process pair {pair_id}: {e}")
            log_error(pair_id, str(e))

            import traceback

            thread_logger.debug(traceback.format_exc())

            return {"success": False, "pair_id": pair_id, "error": str(e)}

    # Process pairs based on threading mode
    processed_paths = []
    failed_pairs = []

    if num_threads > 1 and len(pairs) > 1:
        # Parallel processing using ThreadPoolExecutor
        logger.info(f"Using {num_threads} threads for parallel processing")
        logger.info(f"Processing {len(pairs)} image pairs...")

        # Create a lock for thread-safe operations
        results_lock = threading.Lock()

        # Create and configure progress bar
        progress_bar = tqdm(total=len(pairs), desc="Processing pairs", ncols=100)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(process_pair_task, idx, pair_data): idx
                for idx, pair_data in enumerate(pairs)
            }

            # Process results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                pair_id = pairs[idx][0]

                try:
                    result = future.result()

                    # Thread-safe update of shared state
                    with results_lock:
                        if result["success"]:
                            processed_paths.extend(result["paths"])
                            logger.info(f"Successfully processed {pair_id} (✓)")
                        else:
                            failed_pairs.append(pair_id)
                            logger.error(
                                f"Failed to process {pair_id}: {result.get('error', 'Unknown error')} (✗)"
                            )

                except Exception as e:
                    logger.error(f"Unexpected error processing {pair_id}: {e} (✗)")
                    with results_lock:
                        failed_pairs.append(pair_id)

                # Update progress
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Processed {progress_bar.n}/{len(pairs)} pairs"
                )

            progress_bar.close()

    else:
        # Sequential processing (original approach)
        logger.info(f"Sequential processing of {len(pairs)} image pairs...")
        progress_bar = tqdm(total=len(pairs), desc="Processing pairs", ncols=100)

        for idx, pair_data in enumerate(pairs):
            pair_id = pair_data[0]
            progress_bar.set_description(f"Processing {pair_id}")

            # Process this pair
            result = process_pair_task(idx, pair_data)

            if result["success"]:
                processed_paths.extend(result["paths"])
                logger.info(f"Successfully processed {pair_id} (✓)")
            else:
                failed_pairs.append(pair_id)
                logger.error(
                    f"Failed to process {pair_id}: {result.get('error', 'Unknown error')} (✗)"
                )

            progress_bar.update(1)

        progress_bar.close()

    # Log summary
    main_pairs = len([p for p in processed_paths if "/invivo/" in str(p)])
    sr_pairs = len([p for p in processed_paths if "/sr_hr/" in str(p)])

    logger.info(f"\nProcessed {main_pairs} image pairs successfully")
    logger.info(f"Generated {sr_pairs} super-resolution HR/LR pairs")

    if failed_pairs:
        logger.warning(f"Failed to process {len(failed_pairs)} pairs")
        for failed_id in failed_pairs:
            logger.warning(f"  - {failed_id}")
        log_error(
            "SUMMARY",
            f"Failed to process {len(failed_pairs)} pairs: {', '.join(failed_pairs)}",
        )
    else:
        logger.info("All pairs processed successfully")

    # Standardize dimensions if needed
    logger.info("\n=== Standardizing Image Dimensions ===")
    std_dims = standardize_dimensions(output_dir, padding_value, logger)

    if std_dims:
        logger.info("\n=== Final Standardized Dimensions ===")
        logger.info(f"X: {std_dims[0]}")
        logger.info(f"Y: {std_dims[1]}")
        logger.info(f"Z: {std_dims[2]}")
        logger.info(f"Total voxels: {std_dims[0] * std_dims[1] * std_dims[2]}")

    return processed_paths
