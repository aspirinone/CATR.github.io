import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AVSPostProcess(nn.Module):
    def __init__(self):
        super(AVSPostProcess, self).__init__()

    @torch.inference_mode()
    def forward(self, outputs, resized_padded_sample_size, resized_sample_sizes, orig_sample_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            resized_padded_sample_size: size of samples (input to model) after size augmentation + padding.
            resized_sample_sizes: size of samples after size augmentation but without padding.
            orig_sample_sizes: original size of the samples (no augmentations or padding)
        """
        pred_is_referred = outputs['pred_is_referred'] # [5,50,2]
        prob = F.softmax(pred_is_referred, dim=-1) # [5,50,2]
        scores = prob[..., 0] 
        pred_masks = outputs['pred_masks'] # [5,50]
        pred_masks = F.interpolate(pred_masks, size=resized_padded_sample_size, mode="bilinear", align_corners=False)
        pred_masks = (pred_masks.sigmoid() > 0.5)
        processed_pred_masks, rle_masks = [], []
        for f_pred_masks, resized_size, orig_size in zip(pred_masks, resized_sample_sizes, orig_sample_sizes):
            f_mask_h, f_mask_w = resized_size  # resized shape without padding
            f_pred_masks_no_pad = f_pred_masks[:, :f_mask_h, :f_mask_w].unsqueeze(1)  # remove the samples' padding
            # resize the samples back to their original dataset (target) size for evaluation
            f_pred_masks_processed = F.interpolate(f_pred_masks_no_pad.float(), size=orig_size, mode="nearest")
            f_pred_rle_masks = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                                for mask in f_pred_masks_processed.cpu()]
            processed_pred_masks.append(f_pred_masks_processed)
            rle_masks.append(f_pred_rle_masks)
        predictions = [{'scores': s, 'masks': m, 'rle_masks': rle}
                       for s, m, rle in zip(scores, processed_pred_masks, rle_masks)]
        return predictions


