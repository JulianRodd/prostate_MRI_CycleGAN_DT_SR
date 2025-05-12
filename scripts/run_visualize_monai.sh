#!/bin/bash

# Define paths
INVIVO_PATH="/Users/julianroddeman/Desktop/organized_data/test/invivo/icarus_012.nii.gz"
EXVIVO_PATH="/Users/julianroddeman/Desktop/organized_data/test/exvivo/icarus_012.nii.gz"
OUTPUT_DIR="./prostate_features_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the script
python visualize_monai.py \
  --invivo "$INVIVO_PATH" \
  --exvivo "$EXVIVO_PATH" \
  --output "$OUTPUT_DIR" \
  --patch-size 200