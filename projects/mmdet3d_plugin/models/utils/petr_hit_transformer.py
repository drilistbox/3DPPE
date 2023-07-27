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
from mmdet.models.utils.transformer import inverse_sigmoid
import torch.utils.checkpoint as cp


@TRANSFORMER.register_module()
class PETRHitTransformer(BaseModule):
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
        super(PETRHitTransformer, self).__init__(init_cfg=init_cfg)
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

    def forward(self, x, mask, query_embed, pos_embed, reference_points, lidar2img, down_sample,
                reg_branch=None):
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
            reference_points (Tensor):  # (B, N_query, 3)
            lidar2img(Tensor):     # (B, N_view, 4, 4)
            down_sample: int
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

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,   # (N_query, B, C=embed_dims)
            key=memory,     # (L=N_view*H*W, B, C)
            value=memory,   # (L=N_view*H*W, B, C)
            key_pos=pos_embed,  # (L=N_view*H*W, B, C)
            query_pos=query_embed,  # (N_query, B, C=embed_dims)
            key_padding_mask=mask,  # (B, L=N_view*H*W)
            reference_points=reference_points,  # (B, N_query, 3)
            lidar2img=lidar2img,  # (B, N_view, 4, 4)
            reg_branch=reg_branch,
            h=h,
            w=w,
            down_sample=down_sample,
            )   # (num_layers, num_query, bs, embed_dims)
        out_dec = out_dec.transpose(1, 2)   # (num_layers, B, N_query, C=embed_dims)
        # (L=N_view*H*W, B, C)  --> (N_view, H, W, B, C) --> (B, N_view, C, H, W)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return out_dec, memory


@ATTENTION.register_module()
class PETRHitMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 aggregation_method='mean',
                 **kwargs):
        super(PETRHitMultiheadAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

        self.aggregation_method = aggregation_method

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                reference_points=None,
                lidar2img=None,
                h=None,
                w=None,
                down_sample=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].       # (N_query, B, C=embed_dims)
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .        # (L=N_view*H*W, B, C)
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.    # (L=N_view*H*W, B, C)
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.  # (N_query, B, C=embed_dims)
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.   # (L=N_view*H*W, B, C)
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.   # (B, L=N_view*H*W)
            reference_points (Tensor):  # (B, N_query, 3)
            lidar2img(Tensor):     # (B, N_view, 4, 4)

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]       # (N_query, B, C=embed_dims)
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos   # (N_query, B, C=embed_dims)
        if key_pos is not None:
            key = key + key_pos         # (L=N_view*H*W, B, C)

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = torch.zeros_like(query)   # (N_query, B, C=embed_dims)
        batch_size, n_query = reference_points.shape[:2]
        if self.aggregation_method == 'mean':
            hit_num = query.new_zeros(n_query, batch_size)      # (N_query, B)
        n_key = h*w
        N_view = lidar2img.shape[1]
        for batch_idx in range(batch_size):
            cur_reference_points = reference_points[batch_idx]      # (N_query, 3)  3: (x, y, z)
            cur_reference_points = torch.cat([cur_reference_points,
                                              torch.ones_like(cur_reference_points[:, :1])], dim=-1)     # (N_query, 4)   4: (x, y, z, 1)

            cur_lidar2img = lidar2img[batch_idx]     # (N_view, 4, 4)
            for view_idx in range(N_view):
                # (N_query, 4) @ (4, 4) --> (N_query, 4)
                pts_proj = cur_reference_points @ cur_lidar2img[view_idx].T

                eps = 1e-5
                on_the_image = pts_proj[:, 2] > eps
                pts_2d = pts_proj[:, :2] / torch.maximum(pts_proj[:, 2:3], torch.ones_like(pts_proj[:, 2:3]) * eps)     # (N_query, 2)
                pts_2d /= down_sample

                # 判断该参考点的投影点是否落在该图像上
                on_the_image = on_the_image & (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)    # (N_query, )
                if on_the_image.sum() == 0:
                    continue

                # 获得当前有效query
                valid_query = query[on_the_image, batch_idx:batch_idx+1, :]     # (N_valid, 1, C)
                # 获得当前的key 和 value
                cur_key = key[view_idx*n_key:(view_idx+1)*n_key, batch_idx:batch_idx+1, :]      # (N_key=H*W, 1, C)
                cur_value = value[view_idx*n_key:(view_idx+1)*n_key, batch_idx:batch_idx+1, :]      # (N_value=H*W, 1, C)

                # 获得当前mask
                cur_key_padding_mask = key_padding_mask[batch_idx:batch_idx+1, view_idx*n_key:(view_idx+1)*n_key]       # (1, N_key)

                # (N_valid, 1, C=embed_dims)
                cur_out = self.attn(
                    query=valid_query,    # (N_valid, 1, C=embed_dims)
                    key=cur_key,        # (L=N_view*H*W, 1, C)
                    value=cur_value,    # (L=N_view*H*W, 1, C)
                    attn_mask=attn_mask,    # None
                    key_padding_mask=cur_key_padding_mask)[0]
                # print(view_idx, cur_out.shape)

                if self.aggregation_method == 'mean':
                    out[on_the_image, batch_idx:batch_idx + 1, :] += cur_out
                    hit_num[on_the_image, batch_idx:batch_idx + 1] += 1
                else:
                    raise NotImplementedError

            if self.aggregation_method == 'mean':
                vaild_mask = hit_num[:, batch_idx] > 0      # (Nq, )
                out[vaild_mask, batch_idx, :] /= hit_num[vaild_mask, batch_idx].unsqueeze(-1)

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))
