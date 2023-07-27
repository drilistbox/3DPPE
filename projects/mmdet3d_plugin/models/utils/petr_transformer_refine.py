# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
import copy
import torch.utils.checkpoint as cp


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    """
    Args:
        pos: (N_query, 3)
        num_pos_feats:
        temperature:
    Returns:
        posemb: (N_query, num_feats * 3)
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)     # (num_feats, )
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)   # (num_feats, )   [10000^(0/128), 10000^(0/128), 10000^(2/128), 10000^(2/128), ...]
    pos_x = pos[..., 0, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_x/10000^(0/128), pos_x/10000^(0/128), pos_x/10000^(2/128), pos_x/10000^(2/128), ...]
    pos_y = pos[..., 1, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_y/10000^(0/128), pos_y/10000^(0/128), pos_y/10000^(2/128), pos_y/10000^(2/128), ...]
    pos_z = pos[..., 2, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_z/10000^(0/128), pos_z/10000^(0/128), pos_z/10000^(2/128), pos_z/10000^(2/128), ...]

    # (N_query, num_feats/2, 2) --> (N_query, num_feats)
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_x/10000^(0/128)), cos(pos_x/10000^(0/128)), sin(pos_x/10000^(2/128)), cos(pos_x/10000^(2/128)), ...]
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_y/10000^(0/128)), cos(pos_y/10000^(0/128)), sin(pos_y/10000^(2/128)), cos(pos_y/10000^(2/128)), ...]
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_z/10000^(0/128)), cos(pos_z/10000^(0/128)), sin(pos_z/10000^(2/128)), cos(pos_z/10000^(2/128)), ...]
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)   # (N_query, num_feats * 3)
    return posemb


@TRANSFORMER.register_module()
class PETRTransformer_Refine(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(PETRTransformer_Refine, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed, reference_points, query_embedding, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape (B, N_view, C, H, W) where
                C = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape (B, N_view, H, W).
            query_embed (Tensor): The query embedding for decoder, with shape
                (N_query, embed_dims).
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.    (B, N_view, embed_dims, H, W)
            reference_points:  (B, N_query, 3)
            query_embedding:
                self.query_embedding = nn.Sequential(
                    nn.Linear(self.embed_dims*3//2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                )

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, n, c, h, w = x.shape
        # 利用全部的image feature 做global attention.
        # (B, N_view, C, H, W) --> (N_view, H, W, B, C) --> (L=N_view*H*W, B, C)
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)
        # (B, N_view, C, H, W) --> (N_view, H, W, B, C) --> (L=N_view*H*W, B, C)
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)

        # (N_query, 1, C=embed_dims) --> (N_query, B, C=embed_dims)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)
        # 用于在做cross-attention时, 消除图像pad部分的影响.
        mask = mask.view(bs, -1)    # (B, N_view, H, W) --> (B, L=N_view*H*W)
        target = torch.zeros_like(query_embed)      # (N_query, B, C=embed_dims)
        init_reference_out = reference_points       # (B, N_query, 3)

        # out_dec: [num_layers, num_query, bs, dim]
        inter_states, inter_references = self.decoder(
            query=target,   # (N_query, B, C=embed_dims)
            key=memory,     # (L=N_view*H*W, B, C)
            value=memory,   # (L=N_view*H*W, B, C)
            key_pos=pos_embed,  # (L=N_view*H*W, B, C)
            query_pos=query_embed,  # (N_query, B, C=embed_dims)
            key_padding_mask=mask,  # (B, L=N_view*H*W)
            reference_points=reference_points,  # (B, N_query, 3)
            query_embedding=query_embedding,
            reg_branch=reg_branch,
            )
        # inter_states: (N_decoders, N_query, B, C=embed_dims)
        # inter_references: (N_decoders, B, N_query, 3)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerDecoder_Refine(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=None,
                 return_intermediate=False,
                 **kwargs):

        super(PETRTransformerDecoder_Refine, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, key, value, key_pos, query_pos, key_padding_mask, reference_points,
                query_embedding, reg_branch, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): # (N_query, B, C=embed_dims).
            key: # (L=N_view*H*W, B, C)
            value: # (L=N_view*H*W, B, C)
            key_pos: # (L=N_view*H*W, B, C)
            query_pos: # (N_query, B, C=embed_dims)
            key_padding_mask: # (B, L=N_view*H*W)
            reference_points: # (B, N_query, 3)
            query_embedding:
            reg_branch:
        Returns:
            output:   (N_decoders, N_query, B, C=embed_dims)
            reference_points:   (N_decoders, B, N_query, 3)
        """
        if not self.return_intermediate:
            x = super().forward(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        output = query  # (N_query, B, C=embed_dims)
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                query=output,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
            )   # (N_query, B, C=embed_dims)

            output = output.permute(1, 0, 2).contiguous()  # (B, N_query, C)
            if reg_branch is not None:
                # 说明是with_box_refine, 要对reference进行refine.
                tmp = reg_branch[lid](output)    # (B, N_query, code_size)
                assert reference_points.shape[-1] == 3
                new_reference_points = torch.zeros_like(reference_points)  # (B, N_query, 3)
                new_reference_points[..., :2] = tmp[
                                                ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                                                 ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()    # (B, N_query, 3)
                # (B, N_query, 3) --> (B, N_query, C)
                query_pos = query_embedding(pos2posemb3d(inverse_sigmoid(reference_points)))
                # (B, N_query, C) --> (N_query, B, C)
                query_pos = query_pos.permute(1, 0, 2).contiguous()

            output = output.permute(1, 0, 2).contiguous()  # (N_query, B, C)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            # (N_decoders, N_query, B, C=embed_dims), (N_decoders, B, N_query, 3)
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points
