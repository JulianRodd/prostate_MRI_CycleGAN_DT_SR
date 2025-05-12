# Metrics

This directory contains a comprehensive metrics toolkit for evaluating domain translation tasks in prostate MRI images,
with a focus on simultaneously evaluating both domain translation (DT) and super-resolution (SR) capabilities.

## Key Components

- `do_metrics.py`: Implementation of domain translation metrics
    - `NCC (Normalized Cross-Correlation)`: Measures structural similarity between images
    - `FID (Fréchet Inception Distance)`: Evaluates statistical similarity between image distributions
    - `DomainMetricsCalculator`: Unified interface for domain translation quality metrics

- `prostateMRIFeatureMetrics.py`: Prostate-specific feature metrics
    - Leverages anatomy-aware pre-trained models for feature extraction
    - Implements perceptual loss functions optimized for prostate MRI data
    - Provides LPIPS calculations for perceptual similarity

- `sr_metrics.py`: Super-resolution and image quality metrics
    - `SSIM (Structural Similarity Index)`: Evaluates structural and perceptual similarity
    - `PSNR (Peak Signal-to-Noise Ratio)`: Measures signal fidelity and noise levels
    - `LPIPS`: Calculates perceptual similarity using deep features

- `val_metrics.py`: Unified evaluation framework for validation
    - Combines all metrics into a comprehensive evaluation pipeline
    - Provides normalization and standardization across different metrics
    - Includes combined scoring mechanisms for overall model quality assessment
    - Manages memory efficiently for processing large volumetric datasets

## Evaluation Framework

The evaluation framework employs complementary metrics to assess different aspects of the simultaneous DT and SR tasks:

### Super Resolution Metrics

- SSIM: Evaluates structural similarity based on patterns of luminance, contrast, and structure
- PSNR: Provides a pixel-level fidelity measure
- LPIPS: Extends evaluation into feature space using perceptual representations

### Domain Translation Metrics

- FID: Measures the distance between feature distributions of real and generated images
- NCC: Evaluates how well intensity relationships are preserved

## Features

- GPU-optimized implementations for all metrics
- Memory-efficient processing of large 3D volumes
- Automatic handling of invalid values (NaN/Inf)
- Specialized for prostate MRI data characteristics
- Supports both slice-based and volume-based evaluation
