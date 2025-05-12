"""
Configuration parameters for MRI preprocessing pipeline.

This module contains all configuration parameters used throughout the MRI
preprocessing pipeline, organized by processing stage and functionality.
"""

# ------------------------------------------------------------------------------
# CASE SELECTION CONFIGURATION
# ------------------------------------------------------------------------------

# Cases requiring anterior-posterior flipping during preprocessing
# These cases have inverted anterior-posterior orientation that needs correction
FLIP_ANTERIOR_POSTERIOR = [
    "icarus_001",
    "icarus_031",
    "icarus_032",
    "icarus_004",
    "icarus_008",
    "icarus_010",
    "icarus_011",
    "icarus_020",
    "icarus_027",
    "icarus_029",
    "icarus_035",
    "icarus_045",
    "icarus_046",
    "icarus_049",
    "icarus_050",
    "icarus_059",
    "icarus_061",
    "icarus_063",
    "icarus_067",
    "icarus_068",
    "icarus_070",
    "icarus_072",
    "icarus_074",
    "icarus_077",
    "icarus_078",
    "icarus_037",
]

# Cases requiring left-right flipping during preprocessing
FLIP_LEFT_RIGHT = [
    "icarus_017",
    "icarus_032",
    "icarus_026",
    "icarus_013",
    "icarus_014",
    "icarus_051",
    "icarus_055",
    "icarus_058",
    "icarus_004",
    "icarus_024",
    "icarus_019",
    "icarus_022",
    "icarus_034",
    "icarus_039",
    "icarus_060",
    "icarus_066",
    "icarus_071",
    "icarus_041",
    "icarus_025",
    "icarus_045",
]
# Cases to skip entirely due to quality issues or incompatibility
SKIP_ICARUS_NUMBERS = [
    "icarus_021",  # wrong orientation in vivo
    "icarus_036",  # catheter in in vivo
    "icarus_082",  # Missing slices in ex vivo
]

# If not empty, only these cases will be processed (for testing or focused analysis)
ONLY_RUN_FOR = []

# ------------------------------------------------------------------------------
# PROCESSING CONTROL OPTIONS
# ------------------------------------------------------------------------------

# Quick run mode - skips computationally intensive steps for faster testing
QUICK_RUN = False

# Direction for resolution matching
# False = downsample ex-vivo to match in-vivo (default)
# True = upsample in-vivo to match ex-vivo
FLIP_DIRECTION = True

# Registration thresholds for different processing directions
# These control tissue segmentation sensitivity during registration
REGISTRATION_THRESHOLD_FLIP = 0.2  # For ex-vivo to in-vivo registration
REGISTRATION_THRESHOLD_NORMAL = 0.05  # For in-vivo to ex-vivo registration

# ------------------------------------------------------------------------------
# IMAGE PROCESSING PARAMETERS
# ------------------------------------------------------------------------------

# Contrast Limited Adaptive Histogram Equalization (CLAHE) settings
SKIP_CLAHE = True  # Skip CLAHE processing completely
CLAHE_CLIP_LIMIT = 0.03  # Limits contrast enhancement to reduce noise
CLAHE_TILE_GRID_SIZE = (8, 8)  # Grid size for local histogram equalization

# Normalization parameters
ZSCORE_NORMALIZATION = True  # Use Z-score normalization (vs. min-max)
PERCENTILE_LOW = 1  # Lower percentile for intensity normalization
PERCENTILE_HIGH = 99  # Upper percentile for intensity normalization
NORMALIZATION_RANGE = [0, 1]  # Target range for normalized values

# Image processing parameters
GAUSSIAN_SIGMA = 0.5  # Gaussian smoothing sigma for noise reduction
INITIAL_CROP_MARGIN = 20  # Margin (in voxels) for initial cropping
FINAL_CROP_MARGIN = 0  # Margin (in voxels) for final cropping

# ------------------------------------------------------------------------------
# MAIN PIPELINE CONFIGURATION
# ------------------------------------------------------------------------------

# Complete pipeline configuration with all parameters
DEFAULT_CONFIG = {
    # Case selection
    "flip_anteroposterior": FLIP_ANTERIOR_POSTERIOR,
    "skip_list": SKIP_ICARUS_NUMBERS,
    "only_run": ONLY_RUN_FOR,
    # Processing options
    "flip_direction": FLIP_DIRECTION,  # False = downsample ex-vivo, True = upsample in-vivo
    "test_count": 5,  # Number of pairs to include in test set
    "padding_value": 0.0,  # Value for padding when standardizing dimensions
    # Normalization options
    "normalization": {
        "use_zscore": ZSCORE_NORMALIZATION,
        "p_low": PERCENTILE_LOW,  # Lower percentile for intensities
        "p_high": PERCENTILE_HIGH,  # Upper percentile for intensities
        "target_range": [-1, 1],  # Target range for normalized values
    },
    # CLAHE options
    "clahe": {
        "clip_limit": CLAHE_CLIP_LIMIT,
        "tile_grid_size": CLAHE_TILE_GRID_SIZE,
        "apply_per_slice": True,  # Apply CLAHE to each 2D slice individually
    },
    # Bias correction options
    "bias_correction": {
        "fast_mode": False,  # Use faster but slightly less accurate mode
        "downsample_large_images": False,  # Downsample large images for faster processing
    },
    # Centering and cropping options
    "centering": {
        "margin": INITIAL_CROP_MARGIN,  # Margin to add around the content in voxels
        "force_same_shape": True,  # Force both images to have the same shape
    },
}

# ------------------------------------------------------------------------------
# REGISTRATION CONFIGURATION
# ------------------------------------------------------------------------------

# Advanced registration parameters for ANTs-based image registration
REGISTRATION_CONFIG = {
    "general": {
        # Whether to warp ex-vivo into in-vivo (True) or in-vivo into ex-vivo (False)
        "flip_direction": FLIP_DIRECTION,
        # Base threshold for tissue masking during registration
        "threshold": 0.1,
        # Whether to apply histogram matching before registration
        "histogram_match": True,
        # Threshold for removing light gray values that should be black
        "bite_threshold": 0.05,
    },
    # Parameters for in-vivo image masking
    "invivo_mask": {
        "threshold_factor": 1.0,  # Multiplied by general threshold
        "fill_holes": True,  # Fill holes in the binary mask
        "smooth_sigma": 1.5,  # Gaussian smoothing sigma for the mask
        "iterations": 7,  # Number of morphological iterations
    },
    # Parameters for ex-vivo image masking
    "exvivo_mask": {
        "threshold_factor": 0.8,  # Multiplied by general threshold
        "fill_holes": True,  # Fill holes in the binary mask
        "smooth_sigma": 1.5,  # Gaussian smoothing sigma for the mask
        "iterations": 7,  # Number of morphological iterations
    },
    # ANTs registration parameters
    "registration": {
        "type_of_transform": "SyN",  # Symmetric Normalization (non-rigid)
        "aff_metric": "mattes",  # Mutual information metric for affine
        "syn_metric": "mattes",  # Mutual information metric for SyN
        # Multi-resolution strategy for affine registration
        "aff_iterations": [150, 150, 20],  # Iterations at each level for affine
        "aff_shrink_factors": [8, 4, 2],  # Image shrink factors for each level
        "aff_smoothing_sigmas": [3, 2, 1],  # Smoothing sigmas for each level
        # Multi-resolution strategy for SyN registration
        "syn_iterations": [150, 100, 20],  # Iterations at each level for SyN
        "syn_shrink_factors": [8, 4, 2],  # Image shrink factors for each level
        "syn_smoothing_sigmas": [3, 2, 1],  # Smoothing sigmas for each level
        "grad_step": 0.1,  # Gradient step size for optimization
    },
    # Post-processing after registration
    "post_processing": {
        "mask_dilation_iterations": 2,  # For dilating invivo mask
        "clean_mask_iterations": 3,  # For clean_binary_mask operation
        "clean_mask_dilate": 2,  # For clean_binary_mask dilation
        "clean_mask_erode": 2,  # For clean_binary_mask erosion
        "gaussian_filter_sigma": 0.5,  # Sigma for final smoothing
    },
}

# ------------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------------------------------

# Configuration for logging throughout the pipeline
LOGGING_CONFIG = {
    "debug": False,  # Enable debug-level logging
    "log_to_console": True,  # Output logs to console
    "log_to_file": True,  # Output logs to file
    "console_level": "INFO",  # Console logging level
    "file_level": "DEBUG",  # File logging level
    "max_log_files": 10,  # Maximum number of log files to keep
    # Format for log messages
    "log_file_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "console_format": "%(levelname)s - %(message)s",
}
