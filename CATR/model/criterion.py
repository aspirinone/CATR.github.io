import torch
from torch import nn
from misc import nested_tensor_from_tensor_list, get_world_size, interpolate, is_dist_avail_and_initialized
from .segmentation import dice_loss, sigmoid_focal_loss
from util import flatten_temporal_batch_dims
from loss import IouSemanticAwareLoss


class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, eos_coef):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        # make sure that only loss functions with non-zero weights are computed:
        losses_to_compute = []
        if weight_dict['loss_dice'] > 0 or weight_dict['loss_sigmoid_focal'] > 0:
            losses_to_compute.append('masks')
        if weight_dict['loss_is_referred'] > 0:
            losses_to_compute.append('is_referred')
        self.losses = losses_to_compute

    def forward(self, outputs, targets):
        aux_outputs_list = outputs.pop('aux_outputs', None)
        # compute the losses for the output of the last decoder layer:
        losses = self.compute_criterion(outputs, targets, losses_to_compute=self.losses)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate decoder layer.
        if aux_outputs_list is not None:
            aux_losses_to_compute = self.losses.copy()
            for i, aux_outputs in enumerate(aux_outputs_list):
                aux_outputs['pred_masks'] = aux_outputs['pred_masks'][0].unsqueeze(0)
                aux_outputs['pred_is_referred'] = aux_outputs['pred_is_referred'][0].unsqueeze(0)
                losses_dict = self.compute_criterion(aux_outputs, targets, aux_losses_to_compute)
                losses_dict = {k + f'_{i}': v for k, v in losses_dict.items()}
                losses.update(losses_dict)

        return losses

    def compute_criterion(self, outputs, targets, losses_to_compute):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # T & B dims are flattened so loss functions can be computed per frame (but with same indices per video).
        # also, indices are repeated so so the same indices can be used for frames of the same video.
        T = len(targets)
        outputs, targets = flatten_temporal_batch_dims(outputs, targets)
        # repeat the indices list T times so the same indices can be used for each video frame
        indices = T * indices

        # Compute the average number of target masks across all nodes, for normalization purposes
        num_masks = sum(len(t["masks"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=indices[0][0].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in losses_to_compute:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks=num_masks))
        return losses

    def loss_is_referred(self, outputs, targets, indices, **kwargs):
        device = outputs['pred_is_referred'].device
        bs = outputs['pred_is_referred'].shape[0]
        pred_is_referred = outputs['pred_is_referred'].log_softmax(dim=-1)  # note that log-softmax is used here
        target_is_referred = torch.zeros_like(pred_is_referred) # [4,50,2]
        # extract indices of object queries that where matched with text-referred target objects
        query_referred_indices = self._get_query_referred_indices(indices, targets)
        # by default penalize compared to the no-object class (last token)
        target_is_referred[:, :, :] = torch.tensor([0.0, 1.0], device=device)
        if 'is_ref_inst_visible' in targets[0]:  # visibility labels are available per-frame for the referred object:
            is_ref_inst_visible_per_frame = torch.stack([t['is_ref_inst_visible'] for t in targets])
            ref_inst_visible_frame_indices = is_ref_inst_visible_per_frame.nonzero().squeeze()
            # keep only the matched query indices of the frames in which the referred object is visible:
            visible_query_referred_indices = query_referred_indices[ref_inst_visible_frame_indices]
            target_is_referred[ref_inst_visible_frame_indices, visible_query_referred_indices] = torch.tensor([1.0, 0.0], device=device)
        else:  # assume that the referred object is visible in every frame:
            target_is_referred[torch.arange(bs), query_referred_indices] = torch.tensor([1.0, 0.0], device=device)
        loss = -(pred_is_referred * target_is_referred).sum(-1)
        # apply no-object class weights:
        eos_coef = torch.full(loss.shape, self.eos_coef, device=loss.device)
        eos_coef[torch.arange(bs), query_referred_indices] = 1.0
        loss = loss * eos_coef
        bs = len(indices)
        loss = loss.sum() / bs  # sum and normalize the loss by the batch size
        losses = {'loss_is_referred': loss}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, **kwargs):
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx] # 【2，50，80，142】
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose() #target_masks [2,3,320,568]
        target_masks = target_masks.to(src_masks)
        target_masks_1 = target_masks[tgt_idx] # [5,320,568]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks_1 = src_masks[:, 0].flatten(1)

        target_masks_1 = target_masks_1.flatten(1)
        target_masks_1 = target_masks_1.view(src_masks_1.shape)
        losses = {
            "loss_sigmoid_focal": sigmoid_focal_loss(src_masks_1, target_masks_1, num_masks),
            "loss_dice": dice_loss(src_masks_1, target_masks_1, num_masks),
            "bce_loss": IouSemanticAwareLoss(src_masks, target_masks, mask_pooling_type='avg')
        }
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @staticmethod
    def _get_query_referred_indices(indices, targets):
        """
        extract indices of object queries that where matched with text-referred target objects
        """
        query_referred_indices = []
        for (query_idxs, target_idxs), target in zip(indices, targets):
            ref_query_idx = query_idxs[torch.where(target_idxs == target['referred_instance_idx'])[0]]
            query_referred_indices.append(ref_query_idx)
        query_referred_indices = torch.cat(query_referred_indices)
        return query_referred_indices

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'masks': self.loss_masks,
            'is_referred': self.loss_is_referred,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)
