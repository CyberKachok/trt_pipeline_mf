#!/bin/bash

#CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/224_depth12_mlp1.json"
#CKPT="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/MixFormer_ep0210_feat_sz.pth"
#CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/224_depth4_mlp1_score_feat_sz.json"

#CKPT="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/mixformerv2_small.pth"
#CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/224_depth4_mlp1_score.json"


#CKPT="/media/ilya/Data/Downloads/mixformer2_release/mixformer2_base.pth"
#CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/288_depth8_mlp4_score.json"

CKPT="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/MixFormer_ep0805_removed_mlp_ScoreV425.pth"
CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/224_depth4_mlp1_score_feat_sz.json"

#CKPT="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/MixFormer_ep0050_feat_sz_TeacherV2.pth"
#CONFIG="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/cfg_weight/224_depth12_mlp4_score_feat_sz.json"


DATA_ROOT="/media/ilya/3C4C5BE24C5B958C/Dataset_UAV123/UAV123/data_seq/UAV123"
ANNO_ROOT="/media/ilya/3C4C5BE24C5B958C/Dataset_UAV123/UAV123/anno/UAV123"
OUTPUT_DIR="/home/ilya/PycharmProjects/Trackers/UAV123Benchark/output"
MATLAB_CONFIG="/media/ilya/3C4C5BE24C5B958C/Dataset_UAV123/UAV123/configSeqs.m"


python tracking_test_uav123V3.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --data_root "$DATA_ROOT" \
    --anno_root "$ANNO_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --matlab_config "$MATLAB_CONFIG" \
    #--visualize

#python tracking_test_uav123_Nanotrack.py \
#    --config "$CONFIG" \
#    --ckpt "$CKPT" \
#    --data_root "$DATA_ROOT" \
#    --anno_root "$ANNO_ROOT" \
#    --output_dir "$OUTPUT_DIR" \MixFormer_ep0805_removed_mlp_ScoreV425
#    --matlab_config "$MATLAB_CONFIG" \
#    #--visualize