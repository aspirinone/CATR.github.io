import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def F1_IoU_BCELoss(pred_masks, first_gt_mask):
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks) # [bs*5, 1, 224, 224]
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1) # [bs, 1, 224, 224]
    first_bce_loss = nn.BCEWithLogitsLoss()(pred_masks, first_gt_mask)

    return first_bce_loss



def A_MaskedV_SimmLoss(pred_masks, a_fea_list, v_map_list, \
                        count_stages=[], \
                        mask_pooling_type='avg', norm_fea=True):
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks) # [B*5, 1, 224, 224]
    total_loss = 0
    for stage in count_stages:
        a_fea, v_map = a_fea_list[stage], v_map_list[stage]
        a_fea = a_fea.view(-1, a_fea.shape[-1]) # [B*5, C]
        C, H, W = v_map.shape[1], v_map.shape[-2], v_map.shape[-1]
        assert C == a_fea.shape[-1], 'Error: dimensions of audio and visual features are not equal'
        if mask_pooling_type == "avg":
            downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(pred_masks) # [bs*5, 1, H, W]
        elif mask_pooling_type == 'max':
            downsample_pred_masks = nn.AdaptiveMaxPool2d((H, W))(pred_masks) # [bs*5, 1, H, W]
        downsample_pred_masks = (downsample_pred_masks > 0.5).float() # [bs*5, 1, H, W]
        obj_pixel_num = downsample_pred_masks.sum(-1).sum(-1) # [bs*5, 1]
        masked_v_map = torch.mul(v_map, downsample_pred_masks)  # [bs*5, C, H, W]
        masked_v_fea = masked_v_map.sum(-1).sum(-1) / (obj_pixel_num + 1e-6)# [bs*5, C]

        if norm_fea:
            a_fea = F.normalize(a_fea, dim=-1)
            masked_v_fea = F.normalize(masked_v_fea, dim=-1)

        cos_simm_va = torch.sum(torch.mul(masked_v_fea, a_fea), dim=-1) # [bs*5]
        cos_simm_va = F.relu(cos_simm_va) + 1e-6
        cos_simm_va = (-1) * cos_simm_va.log()
        loss = cos_simm_va.mean()
        total_loss += loss

    total_loss /= len(count_stages)

    return total_loss



def IouSemanticAwareLoss(pred_masks, first_gt_mask, mask_pooling_type='avg'):
    total_loss = 0
    f1_iou_loss = F1_IoU_BCELoss(pred_masks, first_gt_mask)
    total_loss += f1_iou_loss
    return total_loss


if __name__ == "__main__":

    pdb.set_trace()
