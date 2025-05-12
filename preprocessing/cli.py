import argparse
import sys
import time
from pathlib import Path
from preprocessing.config import (
    SKIP_ICARUS_NUMBERS,
    ONLY_RUN_FOR,
    FLIP_ANTERIOR_POSTERIOR,
    FLIP_LEFT_RIGHT,
)
from preprocessing.process_image_pairs import process_image_pairs


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Process and standardize MRI scan pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required input/output arguments
    parser.add_argument(
        "--invivo", required=True, help="Path to in-vivo images directory"
    )
    parser.add_argument(
        "--exvivo", required=True, help="Path to ex-vivo images directory"
    )
    parser.add_argument("--output", default="organized_data", help="Output directory")

    # Processing options
    parser.add_argument(
        "--test-count",
        type=int,
        default=5,
        help="Number of pairs to include in test set",
    )
    parser.add_argument(
        "--flip-direction",
        action="store_true",
        help="Flip processing direction: upsample in-vivo instead of downsampling ex-vivo",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip processing pairs and only standardize dimensions",
    )

    # Filtering options
    parser.add_argument(
        "--only-process",
        nargs="+",
        help="List of case IDs to process (e.g., 'icarus_001 icarus_002')",
    )
    parser.add_argument(
        "--skip", nargs="+", help="List of case IDs to skip (e.g., 'icarus_021')"
    )

    # Configuration options
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument(
        "--padding-value",
        type=float,
        default=0,
        help="Value to use for padding when standardizing dimensions",
    )

    # Multi-threading option (NEW)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for parallel processing (default: 1)",
    )

    # Cleanup options
    parser.add_argument(
        "--keep-temp", action="store_true", help="Keep temporary files after processing"
    )

    # Logging options
    parser.add_argument(
        "--debug", action="store_true", help="Print detailed debug information"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress non-error console output"
    )
    parser.add_argument(
        "--log-dir", help="Directory for log files (defaults to OUTPUT/logs)"
    )

    # Debug visualization options
    parser.add_argument(
        "--make-nifti",
        action="store_true",
        help="Save NIfTI files for each debug visualization step",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the command-line interface.
    Now supports multi-threading.

    Returns:
        Exit code (0 for success, non-zero for error)
    """

    # Parse and validate arguments
    args = parse_arguments()
    if not validate_arguments(args):
        return 1

    # Set up logging
    main_logger, error_logger = setup_logging(args)

    # Process image pairs
    try:
        main_logger.info("=== MRI Preprocessing Pipeline ===")
        main_logger.info(f"In-vivo directory: {args.invivo}")
        main_logger.info(f"Ex-vivo directory: {args.exvivo}")
        main_logger.info(f"Output directory: {args.output}")
        main_logger.info(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
        main_logger.info(f"Make NIfTI: {'Enabled' if args.make_nifti else 'Disabled'}")
        main_logger.info(f"Multi-threading: {args.threads} threads")

        # Start timing
        start_time = time.time()

        # Process image pairs
        if not args.skip_processing:
            main_logger.info("\n=== Processing Image Pairs ===")

            # Set up configuration for processing
            process_config = {
                "input_dir_invivo": args.invivo,
                "input_dir_exvivo": args.exvivo,
                "output_dir": args.output,
                "test_count": args.test_count,
                "debug": args.debug,
                "flip_direction": args.flip_direction,
                "skip_list": SKIP_ICARUS_NUMBERS,
                "only_run": ONLY_RUN_FOR,
                "flip_cases_ap": FLIP_ANTERIOR_POSTERIOR,
                "flip_cases_lr": FLIP_LEFT_RIGHT,
                "padding_value": args.padding_value,
                "num_threads": args.threads,  # Pass thread count to processing function
                "make_nifti": args.make_nifti,  # Pass make_nifti flag to processing function
            }

            # Override with command-line options if provided
            if args.only_process:
                process_config["only_run"] = args.only_process

            if args.skip:
                process_config["skip_list"] = args.skip

            # Process image pairs with multi-threading support
            processed_paths = process_image_pairs(**process_config)

            processing_time = time.time() - start_time
            main_logger.info(
                f"\nProcessed {len(processed_paths) // 2} pairs in {processing_time:.2f} seconds"
            )

        # Clean up if requested
        if not args.keep_temp:
            from preprocessing.preprocessing_actions.io import cleanup_temp_directory

            main_logger.info("\n=== Cleaning Up ===")
            cleanup_temp_directory(args.output, main_logger)
        else:
            main_logger.info("\nKeeping temporary files as requested")

        # Report total time
        total_time = time.time() - start_time
        main_logger.info(f"\nTotal processing time: {total_time:.2f} seconds")
        main_logger.info("\nPreprocessing completed successfully")
        main_logger.info(
            f"Check {Path(args.output) / 'logs' / 'errors.log'} for any failed tasks"
        )

        return 0

    except Exception as e:
        main_logger.error(f"Preprocessing failed: {e}")
        error_logger.error(f"CRITICAL: Pipeline failed - {str(e)}")
        import traceback

        main_logger.debug(traceback.format_exc())
        return 1


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        True if arguments are valid, False otherwise
    """
    # Check input directories
    if not Path(args.invivo).exists():
        print(f"Error: In-vivo directory does not exist: {args.invivo}")
        return False

    if not Path(args.exvivo).exists():
        print(f"Error: Ex-vivo directory does not exist: {args.exvivo}")
        return False

    # Check config file if specified
    if args.config and not Path(args.config).exists():
        print(f"Error: Configuration file does not exist: {args.config}")
        return False

    # Check thread count is reasonable
    if args.threads < 1:
        print(f"Error: Thread count must be at least 1, got {args.threads}")
        return False

    # Warn if thread count is very high
    if args.threads > 32:
        print(
            f"Warning: Using a very high thread count ({args.threads}). This may cause resource issues."
        )

    # Check other options
    if args.padding_value < 0:
        print(f"Error: Padding value must be non-negative: {args.padding_value}")
        return False

    return True


def setup_logging(args: argparse.Namespace) -> tuple:
    """
    Set up logging based on command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Tuple of (main_logger, error_logger)
    """
    import logging
    from preprocessing.utils.logging import setup_main_logger, setup_error_logger

    output_dir = Path(args.output)
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set log level based on command-line arguments
    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.quiet:
        console_level = logging.WARNING
    else:
        console_level = log_level

    # Set up loggers
    main_logger = setup_main_logger(log_dir, log_level, console_level)
    error_logger = setup_error_logger(log_dir)

    return main_logger, error_logger


if __name__ == "__main__":
    sys.exit(main())


# python -m preprocessing.cli --invivo /Users/julianroddeman/Desktop/invivos --exvivo /Users/julianroddeman/Desktop/exvivo_to_check/masked_scans --output ./organized_data/ --flip-direction
# python3 -m preprocessing.cli --invivo /data/temporary/julian/organized_mri/invivo/masked_scans/ --exvivo /data/temporary/julian/organized_mri/exvivo/masked_scans/ --output /data/temporary/julian/organized_data_latest/ --flip-direction
