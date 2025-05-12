# Run Sweep

Automated hyperparameter optimization system for MRI domain translation models that simultaneously perform domain
translation (DT) and super-resolution (SR) using Weights & Biases (W&B) sweeps with Bayesian optimization.

## Key Components

- `sweep.py`: Main entry point for initializing and running hyperparameter sweeps
- `config.py`: Configuration of hyperparameter search spaces and default values
- `metrics.py`: Custom metric implementations for evaluating model performance
- `agent.py`: Agent implementation for executing sweep runs
- `cleanup.py`: Resource management for preventing memory/disk issues

## Features

- Parallel agents for distributed hyperparameter search
- Kaggle integration for cloud execution
- Comprehensive loss function exploration
- Automatic failed run detection and resource reclamation
- Custom weighted metric combining FID (60%), SSIM/PSNR/LPIPS/NCC (10% each)

## Core Hyperparameters

1. **Loss Functions**
    - Adversarial losses: LSGAN, Hinge, Wasserstein, Relativistic variants
    - Cycle/Identity losses: L1, L2, SSIM, Perceptual (and combinations)

2. **Loss Weights**
    - Cycle consistency path A/B
    - Identity mapping
    - Adversarial
    - Feature matching

3. **Domain Adaptation**
    - Master weight and component weights (histogram, contrast, texture, structure, gradient, NCC)

4. **Architecture**
    - Residual learning
    - Spatial transformer network

5. **Optimization**
    - Discriminator learning rate factors
    - Discriminator update frequency

## Usage Examples

```bash
# Initialize a new sweep
python sweep.py --init --project your_project_name

# Join existing sweep
python sweep.py --sweep_id <sweep_id> --count 20

# Run on Kaggle
python sweep.py --init --kaggle --project your_project_name
```
