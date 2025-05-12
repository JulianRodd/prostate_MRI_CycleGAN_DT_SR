#!/bin/bash

# Add fid-eval command to run.sh
# Adds FID evaluation capability to work with the existing project structure

# Directory structure
SCRIPT_DIR="$(dirname "$0")/../setup"
ROOT_DIR="$(dirname "$0")/../../"
CONFIG_DIR="$ROOT_DIR/configurations"

# Function to show usage
show_usage() {
    echo "Usage: ./utils/fid_eval.sh <environment> <config> <prefix> [which_epoch]"
    echo ""
    echo "Options:"
    echo "  environment    'local' or 'server'"
    echo "  config         Configuration name"
    echo "  prefix         3-character prefix for the model (required)"
    echo "  which_epoch    Optional epoch to evaluate (default: 'latest')"
    echo ""
    echo "Example:"
    echo "  ./utils/fid_eval.sh local baseline XYZ latest"
    echo ""
    echo "Available configurations:"
    ls $CONFIG_DIR | grep '\.yaml$' | grep -v 'base_config.yaml' | sed 's/\.yaml$//'
    exit 1
}

# Source environment settings with full path
source "$SCRIPT_DIR/environment.sh"

# Check if command is provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    show_usage
fi

# Parse command
ENV=$1
CONFIG=$2
PREFIX=$3
WHICH_EPOCH=${4:-"latest"}

# Validate prefix
if [ ${#PREFIX} -ne 3 ]; then
    echo "Error: Prefix must be exactly 3 characters"
    show_usage
fi

# Check if running on server or local
if [[ $ENV == "server" ]]; then
    PYTHON_CMD="python3"
    ENV_ARGS="$SERVER_ENV_ARGS"
else
    PYTHON_CMD="python"
    ENV_ARGS="$LOCAL_ENV_ARGS"
fi

# Load the Python config loader and get arguments and parameters
MODEL_PARAMS=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "
from config_loader import ConfigLoader
loader = ConfigLoader('$CONFIG_DIR')
config = loader.load_config('$CONFIG')
params = loader.get_model_params(config)
print(f'{params[\"ngf\"]} {params[\"ndf\"]} {params[\"patch_x\"]} {params[\"patch_y\"]} {params[\"patch_z\"]}')")

# Read the model parameters into variables
read NGF NDF PATCH_X PATCH_Y PATCH_Z <<< "$MODEL_PARAMS"

CONFIG_ARGS=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "
from config_loader import ConfigLoader
loader = ConfigLoader('$CONFIG_DIR')
config = loader.load_config('$CONFIG')
print(loader.build_args_string(config, is_training=False))")

# Extract config name without .yaml extension
CONFIG_NAME=$(basename "$CONFIG" .yaml)

# Build the full model name with prefix
MODEL_NAME="${PREFIX}_${CONFIG_NAME}_ngf${NGF}_ndf${NDF}_patch${PATCH_X}_${PATCH_Y}_${PATCH_Z}"

# Change to the root directory
cd "$ROOT_DIR"

echo "Running FID evaluation for model: $MODEL_NAME"
echo "Using checkpoint: $WHICH_EPOCH"

# Execute the FID evaluation script
$PYTHON_CMD "$ROOT_DIR/fid_evaluation.py" \
    $CONFIG_ARGS \
    $ENV_ARGS \
    --models "$MODEL_NAME" \
    --which_epoch "$WHICH_EPOCH" \

    --use_full_validation

echo "FID evaluation completed"