#!/bin/bash

# Directory structure
SCRIPT_DIR="setup"
CONFIG_DIR="configurations"

# Get absolute paths to important directories
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_PATH")"
SCRIPT_DIR="$SCRIPT_PATH/$SCRIPT_DIR"
CONFIG_DIR="$SCRIPT_PATH/../$CONFIG_DIR"

# Function to show usage
show_usage() {
    echo "Usage: ./run.sh <command> <config> [continue_run_id/prefix] [which_epoch]"
    echo ""
    echo "Commands:"
    echo "  train-local   Run local training"
    echo "  train-server  Run server training"
    echo "  train-kaggle  Run training on Kaggle"
    echo "  test-local    Run local testing (requires prefix)"
    echo "  test-server   Run server testing (requires prefix)"
    echo "  vis-local     Visualize patches locally"
    echo "  vis-server    Visualize patches on server"
    echo ""
    echo "Options:"
    echo "  For training:"
    echo "    continue_run_id  Optional 3-character string to continue a specific training run"
    echo "    which_epoch      Optional epoch to load (e.g., 'latest', '20', etc.)"
    echo "  For testing:"
    echo "    prefix           Required 3-character string prefix for test output"
    echo "    which_epoch      Optional epoch to load (defaults to 'latest')"
    echo ""
    echo "Available configurations:"
    ls $CONFIG_DIR | grep '\.yaml$' | grep -v 'base_config.yaml' | sed 's/\.yaml$//'
    exit 1
}

# Source environment settings with full path
source "$SCRIPT_DIR/environment.sh"

# Check if command is provided
if [ -z "$1" ]; then
    show_usage
fi

# Parse command
COMMAND=$1
CONFIG=$2
THIRD_ARG=$3    # This could be either continue_run_id or prefix
FOURTH_ARG=$4   # This would be which_epoch for both training and testing

# Validate command
case $COMMAND in
    train-local|train-server|train-kaggle|test-local|test-server|vis-local|vis-server)
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        show_usage
        ;;
esac


# Validate config is provided
if [ -z "$CONFIG" ]; then
    echo "Error: Configuration name is required"
    show_usage
fi

# Check if running on server, local, or kaggle
if [[ $COMMAND == *"-server" ]]; then
    PYTHON_CMD="python3"
    ENV="server"
elif [[ $COMMAND == *"-kaggle" ]]; then
    PYTHON_CMD="python"
    ENV="kaggle"
else
    PYTHON_CMD="python"
    ENV="local"
fi

# Load the Python config loader and get arguments and parameters
MODEL_PARAMS=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "
from config_loader import ConfigLoader
loader = ConfigLoader('$CONFIG_DIR')
config = loader.load_config('$CONFIG')
params = loader.get_model_params(config)
print(f'{params[\"ngf\"]} {params[\"ndf\"]} {params[\"patch_x\"]} {params[\"patch_y\"]} {params[\"patch_z\"]}')")

# Read the model parameters into variables
read NGF NDF PATCH_X PATCH_Y PATCH_Z NET_G NET_D <<< "$MODEL_PARAMS"

CONFIG_ARGS=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "
from config_loader import ConfigLoader
loader = ConfigLoader('$CONFIG_DIR')
config = loader.load_config('$CONFIG')
is_training = '$COMMAND'.startswith('train-')
print(loader.build_args_string(config, is_training))")

# Extract config name without .yaml extension
CONFIG_NAME=$(basename "$CONFIG" .yaml)

# Change to the root directory
cd "$ROOT_DIR"

# Execute command based on type
case $COMMAND in
    train-*)
        # Add continuation parameters if provided
        if [ ! -z "$THIRD_ARG" ]; then
            CONTINUE_ARGS="--continue_train --continue_training_run=$THIRD_ARG"
            if [ ! -z "$FOURTH_ARG" ]; then
                CONTINUE_ARGS="$CONTINUE_ARGS --which_epoch=$FOURTH_ARG"
            fi
        fi

        source "$SCRIPT_DIR/environment.sh"
        if [ "$ENV" = "server" ]; then
            ENV_ARGS="$SERVER_ENV_ARGS"
        elif [ "$ENV" = "kaggle" ]; then
            ENV_ARGS="$KAGGLE_ENV_ARGS"
        else
            ENV_ARGS="$LOCAL_ENV_ARGS"
        fi

        $PYTHON_CMD "$ROOT_DIR/train.py" $CONFIG_ARGS $ENV_ARGS $CONTINUE_ARGS --name "$CONFIG_NAME"
        ;;

    test-*)
        if [ -z "$THIRD_ARG" ]; then
            echo "Error: Three-letter prefix is required for test commands"
            show_usage
        fi

        if [ ${#THIRD_ARG} -ne 3 ]; then
            echo "Error: Prefix must be exactly 3 characters"
            show_usage
        fi

        # Parse additional test parameters
        IMAGE_NUM=4
        DIRECTION="AtoB"
        for arg in "$@"; do
            if [[ $arg == "--image_num="* ]]; then
                IMAGE_NUM="${arg#*=}"
            elif [[ $arg == "--which_direction="* ]]; then
                DIRECTION="${arg#*=}"
            fi
        done

        source "$SCRIPT_DIR/environment.sh"
        # Build experiment name with correct parameters
        EXPERIMENT_NAME="${THIRD_ARG}_${CONFIG_NAME}_ngf${NGF}_ndf${NDF}_patch${PATCH_X}_${PATCH_Y}_${PATCH_Z}"

        if [ "$ENV" = "server" ]; then
            ENV_ARGS=$(construct_server_test_args "$IMAGE_NUM" "$DIRECTION" "$EXPERIMENT_NAME")
        else
            ENV_ARGS=$(construct_local_test_args "$IMAGE_NUM" "$DIRECTION" "$EXPERIMENT_NAME")
        fi

        echo "Loading model from: checkpoints/$EXPERIMENT_NAME/${FOURTH_ARG:-latest}_net_G_A.pth"

        if [ ! -z "$FOURTH_ARG" ]; then
            $PYTHON_CMD "$ROOT_DIR/test.py" $CONFIG_ARGS $ENV_ARGS --name "$EXPERIMENT_NAME" --which_epoch "$FOURTH_ARG"
        else
            $PYTHON_CMD "$ROOT_DIR/test.py" $CONFIG_ARGS $ENV_ARGS --name "$EXPERIMENT_NAME"
        fi
        ;;

    vis-*)
        source "$SCRIPT_DIR/environment.sh"
        if [ "$ENV" = "server" ]; then
            DATA_PATH="$SERVER_DATA_PATH"
        else
            DATA_PATH="$LOCAL_DATA_PATH"
        fi

        # Get values from config
        MIN_PIXEL=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "from config_loader import ConfigLoader; loader = ConfigLoader('$CONFIG_DIR'); config = loader.load_config('$CONFIG'); print(config['optimization']['min_pixel'])")
        PATCHES_PER_IMAGE=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "from config_loader import ConfigLoader; loader = ConfigLoader('$CONFIG_DIR'); config = loader.load_config('$CONFIG'); print(config['training']['patches_per_image'])")
        REGISTRATION_TYPE=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "from config_loader import ConfigLoader; loader = ConfigLoader('$CONFIG_DIR'); config = loader.load_config('$CONFIG'); print(config['registration']['type'])")

        $PYTHON_CMD "$ROOT_DIR/check_loader_patches.py" \
            --data_path="$DATA_PATH" \
            --patch_size $PATCH_X $PATCH_Y $PATCH_Z \
            --min_pixel "$MIN_PIXEL" \
            --patches_per_image "$PATCHES_PER_IMAGE" \
            --registration_type "$REGISTRATION_TYPE"
        ;;
esac