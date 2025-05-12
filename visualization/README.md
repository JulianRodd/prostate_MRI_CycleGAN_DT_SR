# Visualization

This directory contains tools for visualizing model outputs, feature maps, and training progress for the MRI domain
translation framework.

## Key Components

- `feature_visualizer.py`: Utilities for visualizing feature maps in networks
    - `FeatureVisualizer`: Extracts and visualizes feature maps from network layers
    - `SliceFeatureVisualizer`: Extends visualization capabilities to full 2D slices
    - Support for visualizing gradient flow and attention maps

- `plot_model_images.py`: Tools for visualizing model inputs and outputs
    - `plot_full_validation_images`: Plots validation images with consistent dimensions
    - `plot_model_images`: Creates visualizations comparing original, generated, and reconstructed images

- `visualize_monai.py`: MONAI-based visualization techniques
    - GradCAM-style feature visualizations with controllable opacity
    - Support for paired in-vivo and ex-vivo scan comparisons
    - Automatic detection of informative image patches

- `visualizer.py`: Training progress tracking and metric visualization
    - Loss curve plotting
    - Metric aggregation
    - Automatic tracking of best model performance
    - Weights & Biases integration

## Features

- Comprehensive tools for feature map visualization
- Model behavior analysis through attention map visualization
- Training progress monitoring with customizable metrics
- Comparison utilities for generated images showing:
    - Original in-vivo MRI
    - Generated ex-vivo MRI
    - Original ex-vivo MRI
    - Reconstructed in-vivo MRI
- Integration with Weights & Biases for experiment tracking
