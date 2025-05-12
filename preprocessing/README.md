# Preprocessing

This directory contains a comprehensive preprocessing pipeline for MRI scans, specifically designed for prostate MRI
domain translation between in-vivo and ex-vivo images.

## Key Components

- `__init__.py`: Module initialization
- `cli.py`: Command-line interface for the preprocessing pipeline
- `config.py`: Configuration parameters and constants
- `models.py`: Data models for images, metadata, and processing results
- `pipeline.py`: Core preprocessing pipeline implementation
- `process_image_pairs.py`: Batch processing for multiple image pairs

### Processing Modules

- `preprocessing_actions/`: Individual preprocessing operations
    - `__init__.py`: Module initialization
    - `bias_correction.py`: N4 bias field correction
    - `centering.py`: Image centering and content-aware cropping
    - `clahe.py`: Contrast Limited Adaptive Histogram Equalization
    - `histogram_matching.py`: Intensity histogram matching between paired scans
    - `io.py`: Input/output operations and file handling
    - `masking.py`: Tissue mask generation and processing
    - `normalization.py`: Image intensity normalization
    - `orientation.py`: Image orientation standardization
    - `registration.py`: ANTs-based image registration
    - `resampling.py`: Resolution matching and resampling

### Utility Modules

- `utils/`: Helper functions for preprocessing
    - `__init__.py`: Module initialization
    - `error_handling.py`: Error handling utilities
    - `logging.py`: Logging configuration
    - `plotting.py`: Visualization utilities

## Preprocessing Pipeline

The preprocessing pipeline consists of eight sequential steps:

1. **Orientation Correction**: Flipping in-vivo scans to match ex-vivo orientation
2. **N4 Bias Field Correction**: Addressing intensity non-uniformity
3. **Centering and Initial Alignment**: Establishing spatial correspondence between paired volumes
4. **Resolution Matching**: Upsampling in-vivo images to match ex-vivo resolution
5. **Dimension Standardization**: Ensuring consistent input size
6. **Intensity Normalization**: Z-score normalization and scaling to [-1, 1]
7. **Deformable Registration**: ANTs-based registration to handle tissue deformations
8. **Final Cropping**: Removing excess background to focus computational resources
