# Models

This directory contains the core model architecture for MRI domain translation between in-vivo and ex-vivo prostate
scans.

## Key Components

- `__init__.py`: Module initialization and model imports
- `base_model.py`: Abstract base class for all models
- `cycle_gan_model.py`: Implementation of the CycleGAN model for domain translation
- `test_model.py`: Specialized model for inference and evaluation

### Generator and Discriminator

- `generator/unet.py`: Encoder-decoder structure with skip connections, featuring downsampling blocks, a bottleneck
  containing residual blocks, and upsampling blocks with fusion modules
- `discriminator/patchgan.py`: PatchGAN discriminator for 3D volumes with spectral normalization

### Utility Functions

- `utils/`: Helper functions for model operations
    - `__init__.py`: Utility module initialization
    - `cycle_gan_utils.py`: Specific utilities for CycleGAN functionality
    - `model_utils.py`: General model utilities including initialization and normalization

### Pre-trained Model

- `prostate_mri_anatomy/`: Domain-specific pre-trained model for feature extraction
    - `docs/README.md`: Documentation for the pre-trained prostate MRI anatomy model
    - `scripts/center_crop.py`: Utilities for preprocessing inputs for the pre-trained model

## Features

- 3D medical image support with volumetric processing
- Encoder-decoder generator with residual blocks in the bottleneck
- PatchGAN discriminators with spectral normalization for local domain evaluation
- Memory-optimized implementation for large 3D volumes
- Optional spatial transformer networks for handling deformations
- Support for both training and inference modes
