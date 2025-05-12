# Evaluation

This directory contains a comprehensive evaluation framework for assessing MRI domain translation models that
simultaneously perform domain translation (DT) and super-resolution (SR) between in-vivo and ex-vivo prostate scans.

## Key Components

- `config.py`: Global configuration parameters for evaluation
- `data_processing.py`: Data loading, preprocessing, and batch processing functions
- `evaluation.py`: Core evaluation implementation with FID calculation
- `main.py`: Entry point for running evaluation on models
- `masking.py`: Functions for creating and applying masks to focus on relevant tissue
- `metrics.py`: Implementation of evaluation metrics
- `sliding_window.py`: Memory-efficient processing for large volumes
- `visualization.py`: Functions for visualizing and saving MRI slices

## Features

- Simultaneous evaluation of domain translation and super-resolution quality
- Slice-based FID (Fréchet Inception Distance) evaluation
- Memory-efficient sliding window processing for large 3D volumes
- Consistent slice extraction for fair comparison
- Automatic foreground masking to focus evaluation on relevant tissue
- Multiple complementary metrics:
    - SR metrics: PSNR, SSIM, LPIPS (between real in-vivo and generated ex-vivo)
    - DT metrics: FID, NCC (between generated ex-vivo and real ex-vivo)
- Visualization of model outputs

## Usage Example

```bash
python main.py --models model1 model2 --which_epoch latest --patches_per_image 10 --output_file results.csv
```
