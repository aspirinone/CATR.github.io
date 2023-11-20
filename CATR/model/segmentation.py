from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import math
from einops import rearrange, repeat
import numpy as np



class FPNSpatialDecoder(nn.Module):
    """
    An FPN-like spatial decoder. Generates high-res, semantically rich features which serve as the base for creating
    instance segmentation masks.
    """
    def __init__(self, context_dim, fpn_dims, mask_kernels_dim=8):
        super().__init__()

        inter_dims = [context_dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16]
        self.lay1 = torch.nn.Conv2d(context_dim, inter_dims[0], 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, inter_dims[0])
        self.lay2 = torch.nn.Conv2d(inter_dims[0], inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.context_dim = context_dim

        self.add_extra_layer = len(fpn_dims) == 3
        if self.add_extra_layer:
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
            self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
            self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
            self.out_lay = torch.nn.Conv2d(inter_dims[4], mask_kernels_dim, 3, padding=1)
        else:
            self.out_lay = torch.nn.Conv2d(inter_dims[3], mask_kernels_dim, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, layer_features: List[Tensor]):
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(layer_features[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(layer_features[1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        if self.add_extra_layer:
            cur_fpn = self.adapter3(layer_features[2])
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
            x = self.lay5(x)
            x = self.gn5(x)
            x = F.relu(x)

        x = self.out_lay(x)
        
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def dice_loss(inputs, targets, num_masks):
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def make_gaussian(y_idx, x_idx, height, width, sigma=7):
    yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

    yv = yv.reshape(height*width).unsqueeze(0).float().cuda()
    xv = xv.reshape(height*width).unsqueeze(0).float().cuda()

    y_idx = y_idx.transpose(0, 1)
    x_idx = x_idx.transpose(0, 1)

    g = torch.exp(- ((yv-y_idx)**2 + (xv-x_idx)**2) / (2*sigma**2) )

    return g

def softmax_w_g_top(x, top=None, gauss=None):
    if top is not None:
        if gauss is not None:
            maxes = torch.max(x, dim=1, keepdim=True)[0]
            x_exp = torch.exp(x - maxes)*gauss
            x_exp, indices = torch.topk(x_exp, k=top, dim=1)
        else:
            values, indices = torch.topk(x, k=top, dim=1)
            x_exp = torch.exp(values - values[:,0])

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp = x_exp/x_exp_sum
        # The types should be the same already
        # some people report an error here so an additional guard is added
        x = torch.zeros_like(x).scatter(1, indices, x_exp.type(x.dtype)) # B * THW * HW

        output = x
    else:
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        if gauss is not None:
            x_exp = torch.exp(x-maxes)*gauss

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        output = x_exp

    return output

class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.vid_embed_proj = nn.Conv2d(5, 15680, kernel_size=1)
        self.top_k = 50
        self.km = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None): # tgt [3328, 1, 256] mem [14, 1, 256]
        tgt2,weight = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt * tgt2

        tgt = tgt * tgt2

        return tgt

       
