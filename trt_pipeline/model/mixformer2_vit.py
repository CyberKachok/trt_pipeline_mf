import torch
import torch.nn as nn
import collections.abc

from itertools import repeat
from timm.layers import DropPath, trunc_normal_

from mixformer_utils.processing_utils import get_2d_sincos_pos_embed, box_xyxy_to_cxcywh

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, 
                 norm_layer=None, flatten=True):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x

# used in Block of model backbone
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., 
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        # type: (Tensor, int, int, int, int) -> Tensor
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        input_lst = [(t_h * t_w * 2), (s_h * s_w + 4)]

        q_mt, q_s = torch.split(q, input_lst, dim=2)
        k_mt, k_s = torch.split(k, input_lst, dim=2)
        v_mt, v_s = torch.split(v, input_lst, dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h * t_w * 2, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h * s_w + 4, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# used in model backbone
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, 
                       hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, 
                       drop=drop)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dim = dim

    def forward(self, x, t_h, t_w, s_h, s_w):
        # type: (Tensor, int, int, int, int) -> Tensor
        x = x + self.drop_path1(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x

# model backbone
class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size_s=256, img_size_t=128, patch_size=16, 
                 in_chans=3, embed_dim=768, depth=12, num_heads=12, 
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None):
        super().__init__()

        self.patch_embed = embed_layer(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])

        self.feat_sz_s = img_size_s // patch_size
        self.feat_sz_t = img_size_t // patch_size
        self.num_patches_s = self.feat_sz_s ** 2
        self.num_patches_t = self.feat_sz_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim), requires_grad=False)

        self.reg_tokens = nn.Parameter(torch.randn(1, 4, embed_dim))
        self.pos_embed_reg = nn.Parameter(torch.zeros(1, 4, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.init_pos_embed()

        trunc_normal_(self.reg_tokens, std=.02)

    def init_pos_embed(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], 
                                              int(self.num_patches_t ** .5), 
                                              cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], 
                                              int(self.num_patches_s ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, x_t, x_ot, x_s):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        x_t = self.patch_embed(x_t)  # BCHW-->BNC
        x_ot = self.patch_embed(x_ot)
        x_s = self.patch_embed(x_s)
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.feat_sz_s
        H_t = W_t = self.feat_sz_t

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t

        reg_tokens = self.reg_tokens.expand(B, -1, -1)  # (b, 4, embed_dim)
        reg_tokens = reg_tokens + self.pos_embed_reg
        
        x = torch.cat([x_t, x_ot, x_s, reg_tokens], dim=1)  # (b, hw+hw+HW+4, embed_dim)
        x = self.pos_drop(x)

        distill_feat_list = []

        for blk in self.blocks:
            x = blk(x, H_t, W_t, H_s, W_s)
            distill_feat_list.append(x)

        x_t, x_ot, x_s, reg_tokens = torch.split(x, [H_t*W_t, H_t*W_t, H_s*W_s, 4], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_ot_2d, x_s_2d, reg_tokens, distill_feat_list


class MixFormer(nn.Module):
    """ Mixformer tracking with score prediction module, whcih jointly perform feature extraction and interaction. """
    def __init__(self, backbone, box_head, score_head, head_type):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.head_type = head_type
        self.score_head = score_head

    def forward(self, template, online_template, search):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, online_template, search, reg_tokens, _ = self.backbone(template, online_template, search)

        # Forward the corner head and score head
        pred_boxes, pred_score = self.forward_head(reg_tokens=reg_tokens)

        pred_boxes = pred_boxes.view(-1, 4).mean(dim=0)
        pred_score = pred_score.view(1).sigmoid()

        return pred_boxes, pred_score

    def forward_head(self, reg_tokens):
        """
        :param search: (b, c, h, w), reg_mask: (b, h, w)
        :return:
        """
        outputs_coord_new = self.forward_box_head(reg_tokens=reg_tokens)

        return outputs_coord_new, self.score_head(reg_tokens).view(-1)

    def forward_box_head(self, reg_tokens):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "MLP" in self.head_type:
            b = reg_tokens.size(0)
            pred_boxes = self.box_head(reg_tokens, softmax=True)[0]

            outputs_coord = box_xyxy_to_cxcywh(pred_boxes)
            outputs_coord_new = outputs_coord.view(b, 1, 4)

            return outputs_coord_new
        else:
            raise KeyError
