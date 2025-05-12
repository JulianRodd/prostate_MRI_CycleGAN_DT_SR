#!/bin/bash

# Enhanced FID evaluation script that processes all models in the CSV file
# and logs comprehensive metrics to WandB

# Directory structure
SCRIPT_DIR="$(dirname "$0")/../setup"
ROOT_DIR="$(dirname "$0")/../../"
CONFIG_DIR="$ROOT_DIR/configurations"

# Get absolute paths
ROOT_ABS_PATH="$(cd "$ROOT_DIR" && pwd)"

# Function to show usage
show_usage() {
    echo "Usage: ./utils/fid_eval_batch.sh <environment> <csv_file>"
    echo ""
    echo "Options:"
    echo "  environment    'local' or 'server'"
    echo "  csv_file       Path to CSV file with model configurations to evaluate"
    echo ""
    exit 1
}

# Source environment settings
source "$SCRIPT_DIR/environment.sh"

# Check arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    show_usage
fi

# Parse command
ENV=$1
CSV_FILE=$2

# Get absolute path of CSV file
if [[ "$CSV_FILE" != /* ]]; then
    CSV_FILE_ABS="$(cd "$(dirname "$CSV_FILE")" && pwd)/$(basename "$CSV_FILE")"
else
    CSV_FILE_ABS="$CSV_FILE"
fi

# Check if CSV file exists
if [ ! -f "$CSV_FILE_ABS" ]; then
    echo "Error: CSV file not found: $CSV_FILE_ABS"
    exit 1
fi

# Set environment
if [[ $ENV == "server" ]]; then
    PYTHON_CMD="python3"
    ENV_ARGS="$SERVER_ENV_ARGS"
else
    PYTHON_CMD="python"
    ENV_ARGS="$LOCAL_ENV_ARGS"
fi

# Create timestamp and output directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$ROOT_ABS_PATH/results/fid_evaluation"
mkdir -p "$OUTPUT_DIR"

# Set output files
OUTPUT_FILE="$OUTPUT_DIR/evaluation_results_${TIMESTAMP}.csv"
TEMP_CSV="$OUTPUT_DIR/temp_models_${TIMESTAMP}.csv"

echo "Starting comprehensive model evaluation from CSV: $CSV_FILE_ABS"
echo "Results will be saved to: $OUTPUT_FILE"

# Create header for the output CSV file with all metrics
echo "model_name,fid_val,fid_train,fid_combined,psnr,ssim,lpips,ncc,config_name,run_name,epoch" > "$OUTPUT_FILE"

# Count the number of models (excluding header)
NUM_MODELS=$(( $(wc -l < "$CSV_FILE_ABS") - 1 ))
echo "Found $NUM_MODELS models to evaluate"

# Process each model in the CSV (skip header)
CURRENT_MODEL=1
tail -n +2 "$CSV_FILE_ABS" | while IFS=, read -r CONFIG_NAME RUN_NAME EPOCH || [[ -n "$CONFIG_NAME" ]]; do
    # Remove any whitespace or control characters
    CONFIG_NAME=$(echo "$CONFIG_NAME" | tr -d '\r\n\t ')
    RUN_NAME=$(echo "$RUN_NAME" | tr -d '\r\n\t ')
    EPOCH=$(echo "$EPOCH" | tr -d '\r\n\t -')  # Remove dashes too

    echo "======================================================="
    echo "Processing model $CURRENT_MODEL/$NUM_MODELS:"
    echo "  CONFIG_NAME: '$CONFIG_NAME'"
    echo "  RUN_NAME: '$RUN_NAME'"
    echo "  EPOCH: '$EPOCH'"
    echo "======================================================="

    # Default to latest if epoch is empty
    if [ -z "$EPOCH" ]; then
        EPOCH="latest"
        echo "Using default epoch: $EPOCH"
    fi

    # First check if the CONFIG_NAME.yaml exists directly
    if [ -f "$CONFIG_DIR/$CONFIG_NAME.yaml" ]; then
        CONFIG_FILE="$CONFIG_NAME"
        echo "Found matching configuration file: $CONFIG_DIR/$CONFIG_FILE.yaml"
    else
        # Fall back to baseline_mini if no matching config file
        CONFIG_FILE="baseline_mini"
        echo "No matching configuration file for $CONFIG_NAME, using fallback: $CONFIG_FILE"
    fi

    # Extract parameters from config
    echo "Loading parameters from $CONFIG_FILE.yaml"
    MODEL_PARAMS=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "from config_loader import ConfigLoader; loader = ConfigLoader('$CONFIG_DIR'); config = loader.load_config('$CONFIG_FILE'); params = loader.get_model_params(config); print('{} {} {} {} {} {}'.format(params['ngf'], params['ndf'], params['patch_x'], params['patch_y'], params['patch_z'], params['patches_per_image']))")
    echo "Config parameters: $MODEL_PARAMS"

    read NGF NDF PATCH_X PATCH_Y PATCH_Z PPI <<< "$MODEL_PARAMS"

    # Create model name - use CONFIG_NAME as is (numeric or otherwise)
    FULL_MODEL_NAME="${RUN_NAME}_${CONFIG_NAME}_ngf${NGF}_ndf${NDF}_patch${PATCH_X}_${PATCH_Y}_${PATCH_Z}"
    echo "Full model name: $FULL_MODEL_NAME"

    # Create temporary CSV for evaluation
    echo "model_name,epoch" > "$TEMP_CSV"
    echo "$FULL_MODEL_NAME,$EPOCH" >> "$TEMP_CSV"

    echo "Created temporary CSV with content:"
    cat "$TEMP_CSV"

    # Get config args from the correct configuration file
    echo "Getting configuration arguments from $CONFIG_FILE.yaml"
    CONFIG_ARGS=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "from config_loader import ConfigLoader; loader = ConfigLoader('$CONFIG_DIR'); config = loader.load_config('$CONFIG_FILE'); print(loader.build_args_string(config, is_training=False))")

    # Add group name from the config
    GROUP_NAME=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "from config_loader import ConfigLoader; loader = ConfigLoader('$CONFIG_DIR'); config = loader.load_config('$CONFIG_FILE'); print(config.get('group_name', 'experiments'))")

    # Check if residual mode is enabled in the config
    USE_RESIDUAL=$(PYTHONPATH="$SCRIPT_DIR" $PYTHON_CMD -c "from config_loader import ConfigLoader; loader = ConfigLoader('$CONFIG_DIR'); config = loader.load_config('$CONFIG_FILE'); model_config = config.get('model', {}); print('True' if model_config.get('use_residual', False) else 'False')")

    echo "Model using residual mode: $USE_RESIDUAL"
    echo "Configuration arguments: $CONFIG_ARGS"

    # Change to root directory
    cd "$ROOT_ABS_PATH"

    # Set a temporary result file for this model
    TEMP_RESULT_FILE="$OUTPUT_DIR/temp_result_${TIMESTAMP}_${CURRENT_MODEL}.csv"

    # Run comprehensive evaluation
    echo "Running evaluation for model $CURRENT_MODEL/$NUM_MODELS..."
    $PYTHON_CMD "$ROOT_ABS_PATH/fid_evaluation.py" \
        $CONFIG_ARGS \
        $ENV_ARGS \
        --csv_path "$TEMP_CSV" \
        --use_full_validation \
        --output_file "$TEMP_RESULT_FILE" \
        --patches_per_image $PPI \
        --use_wandb \
        --group_name "$GROUP_NAME" \
        --wandb_project "prostate_SR-domain_cor"

    EVAL_STATUS=$?

    # Process the result and append to the main output file
    if [ $EVAL_STATUS -eq 0 ] && [ -f "$TEMP_RESULT_FILE" ]; then
        # Extract all metrics from the temp result file (skip header)
        METRICS=$(tail -n +2 "$TEMP_RESULT_FILE" | head -n 1)

        # Append to the main output with additional metadata
        echo "$METRICS,$CONFIG_NAME,$RUN_NAME,$EPOCH" >> "$OUTPUT_FILE"

        echo "Model $CURRENT_MODEL/$NUM_MODELS evaluated successfully"
        echo "Result: $METRICS"
    else
        echo "Error evaluating model $CURRENT_MODEL/$NUM_MODELS (exit code: $EVAL_STATUS)"
        # Add a placeholder row with error indicators
        echo "$FULL_MODEL_NAME,999.999,999.999,999.999,0.0,0.0,1.0,0.0,$CONFIG_NAME,$RUN_NAME,$EPOCH" >> "$OUTPUT_FILE"
    fi

    # Clean up temporary files
    rm -f "$TEMP_CSV" "$TEMP_RESULT_FILE"

    # Force clean up of any GPU memory
    if [ "$ENV" == "server" ]; then
        nvidia-smi --gpu-reset 2>/dev/null || true
    fi

    # Increment the model counter for display purposes
    CURRENT_MODEL=$((CURRENT_MODEL + 1))

    echo "------------------------------------------------------"
done

echo "All models processed."
echo "Final results saved to: $OUTPUT_FILE"

# Print a summary of the results with all metrics
echo "Summary of Evaluation Metrics:"
echo "=========================================="
column -t -s, "$OUTPUT_FILE" | head -n 1
echo "------------------------------------------"
column -t -s, "$OUTPUT_FILE" | tail -n +2 | sort -n -k3,3
echo "=========================================="

# Generate a markdown report with results
REPORT_FILE="$OUTPUT_DIR/evaluation_report_${TIMESTAMP}.md"
echo "# Model Evaluation Report" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "## Results Summary" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| Model | FID | PSNR | SSIM | LPIPS | NCC | Config | Run | Epoch |" >> "$REPORT_FILE"
echo "|-------|-----|------|------|-------|-----|--------|-----|-------|" >> "$REPORT_FILE"

# Add sorted results to the report
tail -n +2 "$OUTPUT_FILE" | sort -t, -n -k3,3 | while IFS=, read -r MODEL FID_VAL FID_TRAIN FID_COMBINED PSNR SSIM LPIPS NCC CONFIG RUN EPOCH; do
    echo "| $MODEL | $FID_COMBINED | $PSNR | $SSIM | $LPIPS | $NCC | $CONFIG | $RUN | $EPOCH |" >> "$REPORT_FILE"
done

echo "" >> "$REPORT_FILE"
echo "Report generated by enhanced evaluation script" >> "$REPORT_FILE"

echo "Markdown report saved to: $REPORT_FILE"