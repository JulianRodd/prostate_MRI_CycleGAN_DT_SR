"""
Configuration file for the FID evaluation script.
Contains global variables and constants used across modules.
"""

# Global cache for training data
CACHED_TRAIN_EXVIVO_DATA = None

# Default patch sizes and thresholds
DEFAULT_PATCH_SIZE = [64, 64, 32]
DEFAULT_MIN_PATCH_SIZE = [16, 16, 8]
DEFAULT_MASK_THRESHOLD = -0.95

# Memory optimization parameters
SLICE_BATCH_SIZE = 32
BATCH_SLICE_PROCESSING = True

# Batch processing parameters for large datasets
MAX_SLICES_PER_BATCH = 100
MAX_VOLUMES_TO_CACHE = 50

# Visualization parameters
VISUALIZATION_DPI = 150
PERCENTILE_NORM_LOW = 1.0
PERCENTILE_NORM_HIGH = 99.5
