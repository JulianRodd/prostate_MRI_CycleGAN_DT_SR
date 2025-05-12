#!/bin/bash

# Default paths (can be overridden by command line arguments)
IMAGES_PATH=""
LABELS_PATH=""
OUTPUT_PATH="organized_data"
SPLIT_RATIO=10

# Function to show usage
show_usage() {
    echo "Usage: ./organize_data.sh [options]"
    echo ""
    echo "Options:"
    echo "  -i, --images PATH     Path to images directory (required)"
    echo "  -l, --labels PATH     Path to labels directory (required)"
    echo "  -o, --output PATH     Output directory (default: organized_data)"
    echo "  -s, --split NUMBER    Split ratio for train/test (default: 10)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Example:"
    echo "  ./organize_data.sh -i /path/to/images -l /path/to/labels -o organized_data -s 10"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--images)
            IMAGES_PATH="$2"
            shift 2
            ;;
        -l|--labels)
            LABELS_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT_RATIO="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$IMAGES_PATH" ] || [ -z "$LABELS_PATH" ]; then
    echo "Error: Both images and labels paths are required"
    show_usage
fi

# Print configuration
echo "Configuration:"
echo "  Images path: $IMAGES_PATH"
echo "  Labels path: $LABELS_PATH"
echo "  Output path: $OUTPUT_PATH"
echo "  Split ratio: $SPLIT_RATIO"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")/../"

# Run the organization script
cd "$ROOT_DIR"
python organize.py \
    --images="$IMAGES_PATH" \
    --labels="$LABELS_PATH" \
    --split="$SPLIT_RATIO" \
    --output="$OUTPUT_PATH"