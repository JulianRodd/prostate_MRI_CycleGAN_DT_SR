# Dataset

This directory contains data loading, patch extraction, and preprocessing utilities for handling MRI volumes in the
domain translation framework.

## Key Components

- `data_loader.py`: Core functions for loading and preprocessing MRI data
- `niftiDataset.py`: Dataset class for handling NIfTI format MRI volumes
- `precomputed_dataset.py`: Dataset class for using precomputed patches
- `preprocess_patches.py`: Script for generating and caching patches for efficient training
- `cleanup_script.py`: Utility for removing temporary files and caches

## Processing Modules

- `processing/augmentation/`: MRI-specific data augmentation techniques
    - `augmentation.py`: Base augmentation framework
    - `mri_augmentation.py`: Specialized transformations for MRI data (geometric, intensity, MRI-specific artifacts)
- `processing/padding.py`: Utilities for padding patches for network inputs
- `processing/random_crop.py`: Content-aware patch extraction from MRI volumes

## Features

- Support for NIfTI (.nii/.nii.gz) and MHA (.mha) format medical images
- Content-aware patch extraction focusing on relevant tissue regions
- Efficient caching mechanism for precomputed patches
- Patch uniqueness scoring for diverse training examples
- MRI-specific augmentation preserving anatomical structures:
    - Geometric transformations constrained to biologically plausible ranges
    - Anisotropic resolution simulation
    - Intensity transformations
    - MRI-specific artifact simulations
- Memory-efficient data loading for large volumetric datasets
