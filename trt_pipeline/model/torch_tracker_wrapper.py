import os
import torch
import torch.nn as nn
from functools import partial

from mixformer_utils.processing_utils import sample_target, clip_box, Preprocessor_torch
from mixformer_utils.misc import is_main_process
from .mixformer2_vit import MixFormer, VisionTransformer
from .head import build_box_head, build_score_decoder
from .tracker_wrapper import TrackerWrapper


class TorchTrackerWrapper(TrackerWrapper):
    def __init__(self, cfg_path, ckpt_path):
        super().__init__(cfg_path)

        _, ext = os.path.splitext(ckpt_path)
        if ext == '.pth':
            self.network = self.build_mixformer_vit_online(self.cfg, ckpt_path)
            self.network.cuda()
            self.name = 'torch'
        elif ext == '.pt':
            self.network = torch.jit.load(ckpt_path)
            self.name = 'jit'
        else:
            self.network = None

        self.network.eval()
        self.preprocessor = Preprocessor_torch()

    def initialize(self, image, init_bbox: list):
        # forward the template once
        z_patch_arr, _, = sample_target(image, init_bbox, 
                                        self.template_factor,
                                        output_sz=self.template_size)
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template

        # save states
        self.state = init_bbox

    def track(self, image):
        frame_id: int = 1
        H, W, _ = image.shape
        x_patch_arr, resize_factor = sample_target(image, self.state, 
                                                   self.search_factor,
                                                   output_sz=self.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            pred_boxes, pred_score = self.network.forward(self.template, 
                                                          self.online_template, search)
        pred_score = pred_score.item()
        #pred_score = 0
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes * self.search_size / resize_factor)  # (cx, cy, w, h) [0,1]

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), 
                              H, W, margin=10)
        # update template
        if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr, _ = sample_target(image, self.state,
                                           self.template_factor,
                                           output_sz=self.template_size)  # (x1, y1, w, h)
            self.online_max_template = self.preprocessor.process(z_patch_arr)
            self.max_pred_score = pred_score
        if frame_id % self.update_interval == 0:
            self.online_template = self.online_max_template

            self.max_pred_score = -1
            self.online_max_template = self.template

        return self.state, pred_score

    # creating MixFormer backbone
    def get_mixformer_vit(self, config) -> VisionTransformer:
        img_size_s = config.DATA.SEARCH.SIZE
        img_size_t = config.DATA.TEMPLATE.SIZE

        if config.MODEL.VIT_TYPE == 'base_patch16':
            vit = VisionTransformer(
                img_size_s=img_size_s, img_size_t=img_size_t,
                patch_size=16, embed_dim=768, depth=config.MODEL.BACKBONE.DEPTH, 
                num_heads=12, mlp_ratio=config.MODEL.BACKBONE.MLP_RATIO, 
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                drop_path_rate=0.1)
        else:
            raise KeyError(f"[ERROR] VIT_TYPE shoule set to 'base_patch16'")

        return vit

    # creating final model
    def build_mixformer_vit_online(self, cfg, ckpt_path=None) -> MixFormer:
        backbone = self.get_mixformer_vit(cfg)          # backbone without positional encoding and attention mask
        box_head = build_box_head(cfg)                  # a simple corner head
        score_head = build_score_decoder(cfg)
        model = MixFormer(
            backbone,
            box_head,
            score_head,
            cfg.MODEL.HEAD_TYPE
        )

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)

            if is_main_process():
                print("[INFO] Loading pretrained mixformer weights from {}".format(ckpt_path))
                if missing_keys:
                    print("[WARNING] missing keys:", missing_keys)
                if unexpected_keys:
                    print("[WARNING] unexpected keys:", unexpected_keys)
                print("[INFO[ Loading pretrained mixformer weights done.")

        return model
