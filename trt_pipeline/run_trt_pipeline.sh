# Example configuration and checkpoint paths (edit to match your environment)
CKPT="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/MixFormer_ep0805_removed_mlp_ScoreV425.pth"
CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/224_depth4_mlp1_score_feat_sz.json"

DATA_ROOT="/media/ilya/3C4C5BE24C5B958C/Dataset_UAV123/UAV123/data_seq/UAV123"
ANNO_ROOT="/media/ilya/3C4C5BE24C5B958C/Dataset_UAV123/UAV123/anno/UAV123"
OUTPUT_DIR="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/output"
MATLAB_CONFIG="/media/ilya/3C4C5BE24C5B958C/Dataset_UAV123/UAV123/configSeqs.m"
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

