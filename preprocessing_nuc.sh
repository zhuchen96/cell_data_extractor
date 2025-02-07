#!/bin/bash

# Ensure script exits on error
set -e

# Prompt user for input path
read -p "Enter the path containing mask files (TIFF): " INPUT_MASK_PATH
read -p "Enter the path containing nuclei image files (TIFF): " INPUT_IMG_PATH
# Define output path
OUTPUT_MASK_PATH="$(pwd)/zarr_masks_nuc"
OUTPUT_IMG_PATH="$(pwd)/zarr_images_nuc"
OUTPUT_JSON_PATH="$(pwd)/json_files_nuc"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_MASK_PATH"
mkdir -p "$OUTPUT_IMG_PATH"
mkdir -p "$OUTPUT_JSON_PATH"

# Run the Python script
python3 process_mask_nuc.py --in_path "$INPUT_MASK_PATH" --out_path "$OUTPUT_MASK_PATH" --out_json_path "$OUTPUT_JSON_PATH"
python3 process_image.py --in_path "$INPUT_IMG_PATH" --out_path "$OUTPUT_IMG_PATH"

# Print completion message
echo "Processing complete! Zarr files saved"