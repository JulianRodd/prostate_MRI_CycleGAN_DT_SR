# Loss Functions

This directory contains specialized loss functions for deep learning-based MRI domain translation between in-vivo and
ex-vivo prostate scans.

## Key Components

- `combined_loss.py`: Utilities for computing and combining multiple loss types
- `discriminator_loss.py`: Enhanced discriminator loss with relativistic formulation and gradient penalty
- `domain_adaptation_loss.py`: Specialized loss for medical image domain adaptation
- `feature_matching_loss.py`: Feature-level matching between real and generated images
- `GANLoss.py`: Implementation of various GAN loss types (LSGAN, vanilla, hinge, Wasserstein)
- `identity_loss.py`: Identity preservation loss using L1 distance
- `mmd_loss.py`: Maximum Mean Discrepancy loss for domain adaptation
- `PerceptualLoss.py`: Perceptual loss using domain-specific pre-trained prostate MRI anatomy model
- `ssim_loss.py`: Structural Similarity Index-based loss

## Features

- Robust implementation with extensive error handling for medical imaging data
- Support for both 2D and 3D medical volumes
- Multiple complementary loss components for domain adaptation
- Enhanced gradient penalty and relativistic discriminator formulations
- Specialized handling for MRI-specific features and challenges

## Loss Framework

The framework implements a weighted combination of five primary loss categories:

- **Adversarial Loss**: Multiple formulations (LSGAN, Hinge, Relativistic Hinge, Wasserstein)
- **Cycle Consistency Loss**: Ensures bidirectional translation fidelity (L1, L2, SSIM, Perceptual, and combinations)
- **Identity Loss**: Maintains identity when translating same-domain images
- **Feature Matching Loss**: Reduces mode collapse by matching discriminator feature statistics
- **Domain Adaptation Loss**: Custom loss with multiple components:
    - Histogram similarity
    - Texture similarity
    - Contrast similarity
    - Structural similarity
    - Gradient similarity
    - NCC (Normalized Cross-Correlation)
