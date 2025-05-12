# Exploration

A toolkit for analyzing paired in-vivo and ex-vivo MRI scans to support machine learning model development for
simultaneous domain translation and super-resolution tasks.

## Key Components

- `exploration.py`: Main entry point script for data exploration
- `analysis.py`: Core analysis workflows and pipelines
- `data_loading.py`: Utilities for loading different MRI data formats
- `statistics.py`: Statistical feature extraction from MRI volumes
- `structural.py`: Analysis of structural properties (volume, surface area, centroid)
- `texture.py`: Texture feature extraction using GLCM-based methods
- `visualization.py`: Tools for visualizing and comparing MRI scans
- `registration.py`: Analysis of registration quality between paired scans
- `residual.py`: Calculation and visualization of residual differences

## Features

### Basic Analysis

- Intensity statistics calculation
- Structural properties analysis
- Texture feature extraction
- Paired slice visualization
- Support for .mha and .nii/.nii.gz file formats

### Advanced Analysis

- Average volume creation and visualization
- Intensity correlation heatmap generation
- Edge detection and comparison
- PCA-based feature analysis
- Registration analysis with deformation fields
- Residual image calculation

## Usage Examples

```bash
# Basic analysis
python exploration.py --base-dir organized_data --output-dir results

# Advanced analysis
python exploration.py --base-dir organized_data --output-dir results --advanced

# Residual analysis
python exploration.py --residual --img1 image1.nii.gz --img2 image2.nii.gz --abs-diff
```
