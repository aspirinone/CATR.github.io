import torch
import torch.nn as nn
from model.pvt import pvt_v2_b5
import pdb
from einops import rearrange
from misc import NestedTensor
import torch.nn.functional as F
from model.multimodal_transformer import MultimodalTransformer
from model.segmentation import FPNSpatialDecoder, VisionLanguageFusionModule
from model.matcher import build_matcher
from model.criterion import SetCriterion
from model.postprocessing import AVSPostProcess



class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x

class Pred_endecoder(nn.Module):
    # pvt-v2 based encoder decoder
    def __init__(self, channel=256, config=None, vis_dim=[64,128,320,512], **kwargs):
        super(Pred_endecoder, self).__init__() #96, 192, 384, 512 #64,128,320,512
        self.cfg = config
        self.vis_dim = vis_dim
        self.encoder_backbone = pvt_v2_b5()
        self.relu = nn.ReLU(inplace=True)
        mask_kernels_dim = 64

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[3])
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[2])
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[1])
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[0])

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)

        d_model = 256
        self.vid_embed_proj = nn.Conv2d(channel, d_model, kernel_size=1) # change 1
        self.obj_queries = nn.Embedding(50, d_model)  # pos embeddings for the object queries
        self.transformer = MultimodalTransformer(**kwargs)
        self.spatial_decoder = FPNSpatialDecoder(d_model, [256,256], mask_kernels_dim) # change 2 [256,256,256] or [320,128,64]
        self.is_referred_head = nn.Linear(d_model, 2)
        self.instance_kernels_head = MLP(d_model, d_model, output_dim=mask_kernels_dim, num_layers=2)
        self.aux_loss = True

        self.fusion_module = VisionLanguageFusionModule(d_model=256, nhead=8)
        if self.training:
            self.initialize_pvt_weights(**kwargs)


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, video_samples, audio_pad, x, audio_feature=None):
        x1, x2, x3, x4= self.encoder_backbone(x) # [3,5,448,448]
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        # shape for pvt-v2-b5
        # BF x  64 x 56 x 56
        # BF x 128 x 28 x 28
        # BF x 320 x 14 x 14
        # BF x 512 x  7 x  7

        conv1_feat = self.conv1(x1)    # BF x 256 x 56 x 56
        conv2_feat = self.conv2(x2)    # BF x 256 x 28 x 28
        conv3_feat = self.conv3(x3)    # BF x 256 x 14 x 14
        conv4_feat = self.conv4(x4)    # BF x 256 x  7 x  7

        T,B,_,_,_ = video_samples.tensors.shape
        layer_outputs = []
        layer_outputs.append(rearrange(conv1_feat, '(t b) c h w -> t b c h w', t=T, b=B))
        layer_outputs.append(rearrange(conv2_feat, '(t b) c h w -> t b c h w', t=T, b=B))
        layer_outputs.append(rearrange(conv3_feat, '(t b) c h w -> t b c h w', t=T, b=B))
        layer_outputs.append(rearrange(conv4_feat, '(t b) c h w -> t b c h w', t=T, b=B))

        backbone_out = []
        video_samples.mask = video_samples.mask.cuda()
        video_samples.tensors = video_samples.tensors.cuda()
        orig_pad_mask = video_samples.mask # [8,2,320,568]
        for l_out in layer_outputs:
            pad_mask = F.interpolate(orig_pad_mask.float(), size=l_out.shape[-2:]).to(torch.bool)
            backbone_out.append(NestedTensor(l_out, pad_mask)) # l_out is video feature

    
        bbone_final_layer_output = backbone_out[1]# 1201
        vid_embeds, vid_pad_mask = bbone_final_layer_output.decompose() #vid_embeds [20,256,112,112] #vid_pad_mask [5,4,112,112]
        T, B, _, H, W = vid_embeds.shape
        vid_embeds = rearrange(vid_embeds, 't b c h w -> (t b) c h w') # [20,256,112,112]
        vid_embeds = self.vid_embed_proj(vid_embeds)
        vid_embeds = rearrange(vid_embeds, '(t b) c h w -> t b c h w', t=T, b=B) # [5,4,256,112,112]

        davt_transformer = self.transformer(vid_embeds, vid_pad_mask, audio_feature, audio_pad, self.obj_queries.weight)
        hs, vid_memory= davt_transformer # vid [5,1,256,28,28] audio 5-layer [5,256]
        

        bbone_middle_layer_outputs = [rearrange(o.tensors, 't b d h w -> (t b) d h w') for o in backbone_out[0:][::-1]]
        decoded_frame_features = self.spatial_decoder(vid_memory, bbone_middle_layer_outputs) # [5,64,56,56]
        decoded_frame_features = rearrange(decoded_frame_features, '(t b) d h w -> t b d h w', t=T, b=B) # [5 1 16 56 56]
        instance_kernels = self.instance_kernels_head(hs)  # [L, T, B, N, C] [3,5,1,50,256]
        # output masks is: [L, T, B, N, H_mask, W_mask]
        output_masks = torch.einsum('ltbnc,tbchw->ltbnhw', instance_kernels, decoded_frame_features)#[3,1,2,50,80,142]
        outputs_is_referred = self.is_referred_head(hs)  # [L, T, B, N, 2] 

        layer_outputs = []
        for pm, pir in zip(output_masks, outputs_is_referred):
            layer_out = {'pred_masks': pm, # [1,2,50,80,142]
                         'pred_is_referred': pir} #[1,2,50,2]
            layer_outputs.append(layer_out)
        out = layer_outputs[-1]  # the output for the last decoder layer is used by default
        if self.aux_loss:
            out['aux_outputs'] = layer_outputs[:-1]
        return out


    def initialize_pvt_weights(self,**kwargs):
        pvt_model_dict = self.encoder_backbone.state_dict()
        pretrained_state_dicts = torch.load(kwargs['TRAIN'].PRETRAINED_PVTV2_PATH)
        state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in pvt_model_dict.keys()}
        pvt_model_dict.update(state_dict)
        self.encoder_backbone.load_state_dict(pvt_model_dict)
        t = kwargs['TRAIN'].PRETRAINED_PVTV2_PATH
        print(f'==> Load pvt-v2-b5 parameters pretrained on ImageNet from {t}')
        # pdb.set_trace()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = args.device
    # model = Pred_endecoder(**vars(args))
    matcher = build_matcher(args)
    weight_dict = {'loss_is_referred': 1, #2
                   'loss_dice': args.dice_loss_coef, #5
                   'loss_sigmoid_focal': 5, #2
                   'bce_loss':4}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_decoder_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef)
    criterion.to(device)
    postprocessor = AVSPostProcess()
    
    return criterion, postprocessor


if __name__ == "__main__":
    imgs = torch.randn(10, 3, 224, 224)
    audio = torch.randn(2, 5, 128)
    model = Pred_endecoder(channel=256)
    output = model(imgs, audio)
    pdb.set_trace()