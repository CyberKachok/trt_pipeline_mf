#!/bin/bash

#CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/224_depth12_mlp1.json"
#CKPT="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/MixFormer_ep0396_feat_sz.pth"
#CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/224_depth4_mlp1_score_feat_sz.json"

CKPT="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/MixFormer_ep0805_removed_mlp_ScoreV410.pth"
CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/224_depth4_mlp1_score_feat_sz.json"


#CKPT="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/MixFormer_ep0620_feat_sz_TeacherV2_Dense.pth"
#CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/224_depth4_mlp4_score_feat_sz.json"


DATA_ROOT="/media/ilya/Data/Datasets/input_data/test_by_size"
ANNO_ROOT="/media/ilya/Data/Downloads/Dataset_UAV123/UAV123/anno/UAV123"
OUTPUT_DIR="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/size_bench/MixFormer_ep1090_iou_scoreV2.csv"
MATLAB_CONFIG="/media/ilya/Data/Downloads/Dataset_UAV123/UAV123/configSeqs.m"


python compute_metrics.py \
    --cfg "$CONFIG" \
    --ckpt "$CKPT" \
    --data_root "$DATA_ROOT" \
    --output "$OUTPUT_DIR" \
    --vis \
  --vis_out_dir "/home/ilya/PycharmProjects/Trackers/UAV123Benchark/vis_out" \
  --vis_stride 1 \
  --reinit_after 300


#python compute_metricsNanotrack.py \
#    --cfg "$CONFIG" \
#    --ckpt "$CKPT" \
#    --data_root "$DATA_ROOT" \
#    --output "$OUTPUT_DIR" \
#    --vis \
#  --vis_out_dir "/home/ilya/PycharmProjects/Trackers/UAV123Benchark/vis_out" \
#  --vis_stride 1

