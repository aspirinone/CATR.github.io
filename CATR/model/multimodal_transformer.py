import copy
import os
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat
from model.position_encoding_2d import PositionEmbeddingSine2D
from model.segmentation import VisionLanguageFusionModule
os.environ["TOKENIZERS_PARALLELISM"] = "false" 


class MultimodalTransformer(nn.Module):
    def __init__(self, num_encoder_layers=3, num_decoder_layers=3, **kwargs):
        super().__init__()
        self.d_model = kwargs['d_model']
        spatial_encoder_layer = SpatialEncoderLayer(**kwargs)
        temporal_encoder_layer = TemporalEncoderLayer(**kwargs)
        self.spatial_encoder = Spatial_Encoder(spatial_encoder_layer, num_encoder_layers)
        self.temporal_encoder = Temporal_Encoder(temporal_encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(**kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, norm=nn.LayerNorm(self.d_model),
                                          return_intermediate=True)
        self.pos_encoder_2d = PositionEmbeddingSine2D()
        self._reset_parameters()

        self.audio_proj = FeatureResizer(
            input_feat_size=128,
            output_feat_size=self.d_model,
            dropout=kwargs['dropout'],
        )
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, vid_embeds, vid_pad_mask, audio_feature, audio_pad, obj_queries): # vid_embeds [1,2,256,20,36] #vid_pad_mask [1,2,20,36]
        t, b, _, h, w = vid_embeds.shape
        vid_sh = vid_embeds

        audio_memory = repeat(audio_feature, '(b t) c -> t b c', b=b) # [5,4,128]
        audio_memory = self.audio_proj(audio_memory) #[5,4,256]
        audio_memory = repeat(audio_memory,'t b c -> t (repeat b) c', repeat=5)
        audio_pad_mask_ = audio_pad
        audio_pad_mask = repeat(audio_pad_mask_,'b t -> (repeat b) t', repeat=5)
        vid_embeds = rearrange(vid_embeds, 't b c h w -> (h w) (t b) c') # [49,20,256]
        encoder_src_seq = torch.cat((vid_embeds, audio_memory), dim=0) #[54,4,256] [49,20,256]
        seq_mask = torch.cat((rearrange(vid_pad_mask, 't b h w -> (t b) (h w)'), audio_pad_mask), dim=1) # [4,54]
        # vid_pos_embed is: [T*B, H, W, d_model]
        vid_pos_embed = self.pos_encoder_2d(rearrange(vid_pad_mask, 't b h w -> (t b) h w'), self.d_model)
        # use zeros in place of pos embeds for the text sequence:
        pos_embed = torch.cat((rearrange(vid_pos_embed, 't_b h w c -> (h w) t_b c'), torch.zeros_like(audio_memory)), dim=0)
        memory = self.spatial_encoder(encoder_src_seq, src_key_padding_mask=seq_mask, pos=pos_embed)  # [S, T*B, C] [730,2,256]
        vid_memory = self.temporal_encoder(vid_sh, audio_pad_mask_, vid_pad_mask, vid_pos_embed, encoder_src_seq, src_key_padding_mask=seq_mask, pos=pos_embed)

        # add T*B dims to query embeds (was: [N, C], where N is the number of object queries):
        obj_queries = repeat(obj_queries, 'n c -> n (t b) c', t=t, b=b) # [50,2,256] [50,1024]
        tgt = torch.zeros_like(obj_queries)  # [N, T*B, C]

        # hs is [L, N, T*B, C] where L is number of layers in the decoder
        hs = self.decoder(tgt, memory, memory_key_padding_mask=seq_mask, pos=pos_embed, query_pos=obj_queries)
        hs = rearrange(hs, 'l n (t b) c -> l t b n c', t=t, b=b) # [3,1,2,50,256]
        return hs, vid_memory

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Spatial_Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos) # [730,2,256]

        if self.norm is not None:
            output = self.norm(output)

        return output

class Temporal_Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.av_atten = VisionLanguageFusionModule(d_model=256, nhead=8)
        self.va_atten = VisionLanguageFusionModule(d_model=256, nhead=8)
        self.num_layers = num_layers
        self.norm = norm 
        self.attention_feature = nn.Sequential(nn.Conv2d(512, 2, kernel_size=3, padding=1))
        self.output1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())

    def forward(self, vid_sh, audio_pad_mask, vid_pad_mask, vid_pos_embed, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        gate = []

        for layer in self.layers:
            src, output = layer(vid_sh, audio_pad_mask, vid_pad_mask, vid_pos_embed, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos) # [730,2,256]
            gate.append(src)

        
 
        #########################################################
        t, b, _, h, w = vid_sh.shape
        E1 = rearrange(gate[0][:h*w, :, :], '(h w) (t b) c -> (t b) c h w', h=h, w=w, t=t, b=b)
        T1 = rearrange(gate[1][:h*w, :, :], '(h w) (t b) c -> (t b) c h w', h=h, w=w, t=t, b=b)
        G1 = self.attention_feature(torch.cat((E1, T1), 1))
        G1 = F.adaptive_avg_pool2d(F.sigmoid(G1),1)
        # D1 = G1*T1
        D1 = G1[:, 0, :, :].unsqueeze(1).repeat(1, 256, 1, 1) * T1
        D1 = self.output1(D1)

        E2 = rearrange(gate[1][:h*w, :, :], '(h w) (t b) c -> (t b) c h w', h=h, w=w, t=t, b=b)
        T2 = rearrange(gate[2][:h*w, :, :], '(h w) (t b) c -> (t b) c h w', h=h, w=w, t=t, b=b)
        G2 = self.attention_feature(torch.cat((E2, T2), 1))
        G2 = F.adaptive_avg_pool3d(F.sigmoid(G1),1)
        output = self.output2(F.upsample(D1, size=E1.size()[2:], mode='bilinear')+G2[:, 0, :, :].unsqueeze(1).repeat(1, 256, 1, 1) * T2)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class SpatialEncoderLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TemporalEncoderLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=256, dropout=0.1,
                 activation="relu", normalize_before=False, **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.av_atten = VisionLanguageFusionModule(d_model=256, nhead=8)
        self.va_atten = VisionLanguageFusionModule(d_model=256, nhead=8)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, 
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)  #[3141,5,256]
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, vid_sh, audio_pad_mask, vid_pad_mask, vid_pos_embed, src, 
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        else:
            memory = self.forward_post(src, src_mask, src_key_padding_mask, pos) #[3141,5,256]
            t, b, _, h, w = vid_sh.shape
            vid_memory = rearrange(memory[:h*w, :, :], '(h w) (t b) c -> t b c h w', h=h, w=w, t=t, b=b) # [1,2,256,20,36]
            audio_memory = memory[h*w:, :, :] # [10,2,256]
            audio_memory = rearrange(audio_memory, 's t_b c -> t_b s c')
            audio_memory = [t_mem[~pad_mask] for t_mem, pad_mask in zip(audio_memory, audio_pad_mask)]  # remove padding
            #vid [5,1,256,56,56] au[]
            audio_mem = []
            count = 0
            for mem in audio_memory:
                audio_mem.append(mem)
                count = count+1

            audio_mem = torch.stack(audio_mem,dim=0)
            audio_mem = rearrange(audio_mem, 'b t c -> t b c', b=b, t=t)
            src = rearrange(vid_memory, 't b c h w -> (t h w) b c', b=b, t=t)
            text_pos = torch.zeros_like(audio_mem) #[5,2,256]
            vid_pos_embed = rearrange(vid_pos_embed, '(t b) h w c -> (t h w) b c', b=b, t=t)
            vid_pad_mask = rearrange(vid_pad_mask, 't b h w -> b (t h w)', b=b, t=t)
            
            src = self.av_atten(tgt=src,
                                memory=audio_mem,
                                memory_key_padding_mask=audio_pad_mask,
                                pos=text_pos,
                                query_pos=None
                    ) # src [15680,1,256]
            aud = self.va_atten(tgt=audio_mem,
                                memory=src,
                                memory_key_padding_mask=vid_pad_mask,
                                pos=vid_pos_embed,
                                query_pos=None  
                    )  
            
            src = rearrange(src, '(t h w) b c -> (h w) (t b) c', b=b, t=t,h=h,w=w)
            aud = repeat(aud,'t b c -> t (repeat b) c', repeat=5, b=b, t=t)
            memory = torch.cat((src, aud), dim=0) #[54,4,256]

            return src, memory #vid [15680,1,256]  aud [5,1,256]


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, **kwargs):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
