#!/bin/bash

# Example configuration and checkpoint paths (edit to match your environment)
CONFIG="/path/to/your/config.json"
CKPT="/path/to/your/checkpoint.pth"

DATA_ROOT="/path/to/UAV123/data_seq/UAV123"
ANNO_ROOT="/path/to/UAV123/anno/UAV123"
OUTPUT_DIR="/path/to/output"
MATLAB_CONFIG="/path/to/UAV123/configSeqs.m"
WORKSPACE=$((1 << 30))  # 1GB workspace for TensorRT builder

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

python trt_pipeline.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --data_root "$DATA_ROOT" \
    --anno_root "$ANNO_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --matlab_config "$MATLAB_CONFIG" \
    --workspace $WORKSPACE

