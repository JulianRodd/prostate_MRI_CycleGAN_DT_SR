# Utils

This directory contains common utility functions used across the MRI domain translation framework.

## Key Components

- `image_pool.py`: Image buffer mechanism to stabilize GAN training
    - Implements a history of generated images
    - Helps prevent discriminator overfitting
    - Reduces oscillations during training

- `model_utils.py`: Common utilities for model operations
    - Network weight initialization
    - Spectral normalization implementation
    - Tensor manipulation utilities

- `utils.py`: General utility functions
    - Configuration file handling
    - Directory management
    - Logging utilities
    - Image conversion and normalization

## Features

- Efficient implementation of image buffer for GAN training stability
- Well-tested initialization routines for different network architectures
- Comprehensive logging and file management utilities
- Common tensor operations optimized for MRI data
