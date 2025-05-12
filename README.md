# MRI Domain Translation for Prostate Imaging

This repository contains code for my master's thesis research on MRI domain translation between in-vivo and ex-vivo
prostate scans using deep learning approaches. The framework implements a CycleGAN architecture that simultaneously
performs domain translation (DT) and super-resolution (SR), addressing the substantial resolution gap from 0.3 × 0.3 ×
3.0 mm to 0.15 × 0.15 × 1.0 mm.

## Project Overview

This research addresses a critical challenge in prostate cancer diagnosis: the limited availability of ex-vivo MRI data
for effective registration between in-vivo MRI and histopathology. Our approach translates standard clinical in-vivo T2W
MRI to synthetic ex-vivo quality MRI, maintaining anatomical structures while transforming domain appearance
characteristics.

## Repository Structure

| Directory                                    | Description                                                                  |
|----------------------------------------------|------------------------------------------------------------------------------|
| [commands](./commands/README.md)             | Execution utilities and environment setup scripts                            |
| [dataset](./dataset/README.md)               | Data loading, patch extraction, and preprocessing utilities                  |
| [evaluation](./evaluation/README.md)         | Quantitative assessment framework for domain translation quality             |
| [exploration](./exploration/README.md)       | Analysis tools for investigating MRI characteristics                         |
| [loss_functions](./loss_functions/README.md) | Specialized loss components for MRI domain adaptation                        |
| [metrics](./metrics/README.md)               | Implementation of domain translation and structural similarity metrics       |
| [models](./models/README.md)                 | CycleGAN architecture with generators, discriminators, and utility functions |
| [preprocessing](./preprocessing/README.md)   | Pipeline for preparing and standardizing MRI volumes                         |
| [run_sweep](./run_sweep/README.md)           | Automated hyperparameter optimization using W&B sweeps                       |
| [training](./training/README.md)             | Memory-optimized training infrastructure for large 3D volumes                |
| [utils](./utils/README.md)                   | Common utility functions for the entire project                              |
| [visualization](./visualization/README.md)   | Tools for visualizing model outputs and feature maps                         |
| [configurations](./configurations/README.md) | YAML configuration files for all experimental runs                           |
| [inference](./inference/README.md)           | Code for performing inference with trained models                            |
| [options](./options/README.md)               | Configurable options for training, testing and base model parameters         |

## Key Features

- Simultaneous domain translation and super-resolution in a single CycleGAN model
- 3D patch-based processing of volumetric MRI data
- Generator with encoder-decoder architecture and residual blocks in the bottleneck
- PatchGAN discriminator for evaluating local domain characteristics
- Multi-component loss framework balancing adversarial, cycle consistency, identity, and domain adaptation components
- Comprehensive configuration system for experiment management
- Integrated evaluation metrics (FID, PSNR, SSIM, LPIPS, NCC)
- WandB integration for experiment tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prostate-SR_domain-correction.git
cd prostate-SR_domain-correction

# Install dependencies
pip install -r requirements.txt

# Install specialized dependencies for preprocessing
pip install antspy torchio monai
```

## Usage

The framework provides a unified command-line interface through `run.sh`:

```bash
# Training
./commands/run.sh train-local|train-server <config> [continue_run_id] [which_epoch]

# Testing
./commands/run.sh test-local|test-server <config> <prefix> [which_epoch]

# Visualization
./commands/run.sh vis-local|vis-server <config>

# Evaluation
./commands/utils/fid_eval.sh local|server <config> <prefix> [which_epoch]
```

For batch evaluation of multiple models:

```bash
./commands/utils/fid_eval_batch.sh local|server <models.csv>
```

## Data Structure

The framework expects data organized as:

```
organized_data/
  ├── train/
  │   ├── invivo/
  │   │   ├── sample_001.nii.gz
  │   └── exvivo/
  │       ├── sample_001.nii.gz
  └── test/
      ├── invivo/
      └── exvivo/
```

## Requirements

The framework requires:

- Python 3.8+
- PyTorch 2.0+
- MONAI 1.4.0+
- TorchIO 0.20.5+
- ANTsPy for registration
- Weights & Biases for experiment tracking

See [requirements.txt](./requirements.txt) for a complete list of dependencies.

## Acknowledgments

This project was conducted as part of a data science master's thesis at Radboudumc's Computational Pathology Group (
CPG).
