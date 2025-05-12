# Commands

This directory contains execution utilities and environment setup scripts for running the MRI domain translation
framework.

## Contents

- `run.sh`: Main entry point for training, testing, and visualization
- `setup/`: Environment and configuration utilities
    - `common.sh`: Common functions used across scripts
    - `config_loader.py`: Loads and parses YAML configuration files
    - `environment.sh`: Sets up environment variables and paths
- `utils/`: Execution utilities
    - `fid_eval.sh`: Script for calculating FID metrics on models
    - `fid_eval_batch.sh`: Batch evaluation of multiple models
    - `organize_data.sh`: Data organization utilities

## Usage Examples

```bash
# Start training on local machine
./run.sh train-local configs/my_config.yaml

# Continue training from specific epoch
./run.sh train-local configs/my_config.yaml run_20230501 50

# Run testing on specific epoch
./run.sh test-local configs/my_config.yaml model_prefix 100

# Calculate FID metrics
./utils/fid_eval.sh local configs/my_config.yaml model_prefix latest
```
