#!/bin/bash

# Local paths (relative to root directory)
LOCAL_DATA_PATH="organized_data/train"
LOCAL_VAL_PATH="organized_data/test"

# Server paths
SERVER_DATA_PATH="/data/temporary/julian/organized_data/train"
SERVER_VAL_PATH="/data/temporary/julian/organized_data/test"

# Kaggle paths
KAGGLE_DATA_PATH="/kaggle/input/prostate-data/organized_data/train"
KAGGLE_VAL_PATH="/kaggle/input/prostate-data/organized_data/test"

construct_file_paths() {
    local val_path=$1
    local image_num=$(printf "%03d" ${2:-4})  # Ensure zero-padding (e.g., 004 instead of 4)
    local direction=${3:-"AtoB"}  # Default to AtoB if not provided
    local experiment_name=${4:-""}  # Default to empty if not provided

    # Create result directory path with experiment name
    local result_dir
    if [ -z "$experiment_name" ]; then
        result_dir="$val_path/../test_results"
    else
        result_dir="$val_path/../test_results/${experiment_name}"
    fi

    # Always use "icarus_xxx.nii" instead of "xxx.nii"
    local filename="icarus_${image_num}.nii"

    if [ "$direction" = "AtoB" ]; then
        echo "$val_path/invivo/${filename} $result_dir/invivo/${filename}"
    else
        echo "$val_path/exvivo/${filename} $result_dir/exvivo/${filename}"
    fi
}


# Get default paths (using AtoB as default)
read LOCAL_DEFAULT_IMAGE LOCAL_DEFAULT_RESULT <<< $(construct_file_paths "$LOCAL_VAL_PATH")
read SERVER_DEFAULT_IMAGE SERVER_DEFAULT_RESULT <<< $(construct_file_paths "$SERVER_VAL_PATH")
read KAGGLE_DEFAULT_IMAGE KAGGLE_DEFAULT_RESULT <<< $(construct_file_paths "$KAGGLE_VAL_PATH")

# Local environment settings
LOCAL_ENV_ARGS="
  --data_path=$LOCAL_DATA_PATH \
  --val_path=$LOCAL_VAL_PATH \
  --wandb_project=prostate_SR-domain_cor_LOCAL \
  --gpu_ids=-1"

# Local test-specific settings
construct_local_test_args() {
    local image_num=${1:-0}  # Default to 0 if not provided
    local direction=${2:-"AtoB"}  # Default to AtoB if not provided
    local experiment_name=${3:-""}  # Default to empty if not provided
    read local_img local_res <<< $(construct_file_paths "$LOCAL_VAL_PATH" "$image_num" "$direction" "$experiment_name")
    echo "
  --data_path=$LOCAL_DATA_PATH \
  --val_path=$LOCAL_VAL_PATH \
  --image=$local_img \
  --result=$local_res \
  --which_direction=$direction \
  --gpu_ids=-1"
}

# Server environment settings
SERVER_ENV_ARGS="
  --data_path=$SERVER_DATA_PATH \
  --val_path=$SERVER_VAL_PATH \
  --use_wandb \
  --wandb_project=prostate_SR-domain_cor \
  --gpu_ids=0"

# Server test-specific settings
construct_server_test_args() {
    local image_num=${1:-0}  # Default to 0 if not provided
    local direction=${2:-"AtoB"}  # Default to AtoB if not provided
    local experiment_name=${3:-""}  # Default to empty if not provided
    read server_img server_res <<< $(construct_file_paths "$SERVER_VAL_PATH" "$image_num" "$direction" "$experiment_name")
    echo "
  --data_path=$SERVER_DATA_PATH \
  --val_path=$SERVER_VAL_PATH \
  --image=$server_img \
  --result=$server_res \
  --which_direction=$direction \
  --gpu_ids=0"
}

# Kaggle environment settings
KAGGLE_ENV_ARGS="
  --data_path=$KAGGLE_DATA_PATH \
  --val_path=$KAGGLE_VAL_PATH \
  --use_wandb \
  --wandb_project=prostate_SR-domain_cor \
  --gpu_ids=0"

# Kaggle test-specific settings
construct_kaggle_test_args() {
    local image_num=${1:-0}  # Default to 0 if not provided
    local direction=${2:-"AtoB"}  # Default to AtoB if not provided
    local experiment_name=${3:-""}  # Default to empty if not provided
    read kaggle_img kaggle_res <<< $(construct_file_paths "$KAGGLE_VAL_PATH" "$image_num" "$direction" "$experiment_name")
    echo "
  --data_path=$KAGGLE_DATA_PATH \
  --val_path=$KAGGLE_VAL_PATH \
  --image=$kaggle_img \
  --result=$kaggle_res \
  --which_direction=$direction \
  --gpu_ids=0"
}

# Export functions so they can be used by other scripts
export -f construct_file_paths
export -f construct_local_test_args
export -f construct_server_test_args
export -f construct_kaggle_test_args