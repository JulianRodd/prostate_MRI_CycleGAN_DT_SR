# Training

This directory contains memory-optimized training infrastructure for handling large 3D MRI volumes in domain translation
models.

## Key Components

- `schedulerers.py`: Learning rate and loss weight scheduling implementations
    - Cyclical learning rate scheduler with periodic restarts
    - Lambda weight schedulers for dynamic loss weighting
- `trainer.py`: Memory-optimized training loop with customizable parameters
    - Adaptive gradient accumulation
    - Mixed precision training
    - Configurable discriminator update frequency
    - Checkpoint management
- `validation_handler.py`: Sliding window validation for large volumes
    - Memory-efficient validation on full 3D volumes
    - Comprehensive metrics calculation
- `memory_utils.py`: Memory management utilities for efficient training
    - CUDA memory monitoring
    - Automatic garbage collection
    - Patch size adaptation
- `optimization_utils.py`: Optimization utilities for training
    - Gradient clipping
    - Optimizer state management
    - Parameter group configuration
- `metrics.py`: Implementation of metrics for training monitoring
    - Loss tracking
    - Performance metric calculation
    - Moving averages for stability

## Features

- Memory-optimized training for large 3D medical datasets
- Adaptive gradient accumulation and mixed precision
- Sliding window validation for large volumes
- Customizable loss weight scheduling
- Comprehensive validation metrics
- Automatic black region cropping for efficient processing
