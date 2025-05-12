"""
Preprocessing actions for MRI image processing.
Each module handles a specific aspect of the preprocessing pipeline.
"""

# Image enhancement
from preprocessing.preprocessing_actions.bias_correction import (
    apply_n4_correction,
    correct_single_image,
)

# Centering and cropping
from preprocessing.preprocessing_actions.centering import (
    center_and_crop_images,
    crop_to_content,
    standardize_image_shapes,
)
from preprocessing.preprocessing_actions.clahe import (
    apply_clahe,
    process_volume_with_clahe,
    apply_clahe_to_slice,
)
from preprocessing.preprocessing_actions.normalization import (
    normalize_images,
    apply_zscore_normalization,
)

# Image orientation and geometric operations
from preprocessing.preprocessing_actions.orientation import (
    extract_metadata,
    flip_anterior_posterior,
    standardize_orientation,
)

# Resampling
from preprocessing.preprocessing_actions.resampling import (
    resample_with_physical_space,
    resample_invivo_to_exvivo,
    resample_exvivo_to_invivo,
)
