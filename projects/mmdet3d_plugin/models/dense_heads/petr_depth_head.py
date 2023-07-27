# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import cv2
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmdet3d.models import HEADS, build_loss
from .petr_head import PETRHead
from mmdet.models.utils.transformer import inverse_sigmoid
import math
from mmcv.runner import force_fp32
from mmdet.core import multi_apply
from mmcv.cnn import Conv2d, Linear
from mmdet.models.utils import NormedLinear
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes

from ..utils.depthnet import build_depthnet
from mmdet3d.models import builder


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


class SELayer(nn.Module):
    def __init__(self, depth_channel, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(depth_channel, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        """
        Args:
            x: coords_position_embeding  (B*N, embed_dims, H, W)
            x_se: depth distribution     (B*N, depth_num, H, W)
        Returns:
            3D PE:  (B*N, embed_dims, H, W)
        """
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)   # (B*N, embed_dims, H, W)
        return x * self.gate(x_se)


@HEADS.register_module()
class PETRDepthHead(PETRHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2
    def __init__(self,
                 with_depth_supervision=True,
                 depthnet=dict(
                     type='VanillaDepthNet',
                     in_channels=2048,
                     context_channels=256,
                     depth_channels=64,
                     mid_channels=512,
                 ),
                 with_filter=False,     # 取topK个3D坐标来生成3D PE
                 num_keep=5,
                 with_dpe=False,    # Distribution-guide position encoder
                 use_sigmoid=True,
                 use_detach=False,
                 loss_depth=dict(type='BinaryCrossEntropyLoss', with_logits=False,
                                 reduction='mean', loss_weight=3.0),
                 **kwargs):
        self.with_dpe = with_dpe
        self.with_depth_supervision = with_depth_supervision
        self.with_filter = with_filter
        self.use_sigmoid = use_sigmoid
        self.use_detach = use_detach

        if self.with_depth_supervision:
            kwargs['in_channels'] = depthnet['context_channels']
            if with_filter:
                kwargs['depth_num'] = num_keep
                self.num_keep = num_keep
        super(PETRDepthHead, self).__init__(**kwargs)

        if self.with_depth_supervision:
            self.depth_net = build_depthnet(depthnet)
            self.depth_num = self.depth_net.depth_channels
            self.loss_depth = builder.build_loss(loss_depth)
        else:
            self.depth_net = None

        if self.with_dpe:
            self.dpe = SELayer(self.depth_num, self.embed_dims)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # 因为没有box_refine, 共享权重.
        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_position:
            # self.position_dim = 3 * self.depth_num      # D*3 3:(x, y, z)
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        # 在3D空间初始化一组0-1之间均匀分布的learnable anchor points.
        self.reference_points = nn.Embedding(self.num_query, 3)
        # anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def position_embeding(self, img_feats, img_metas, masks=None, depth_score=None):
        """
        Args:
            img_feats: List[(B, N_view, C, H, W), ]
            img_metas:
            masks: (B, N_view, H, W)
            depth_score: (B, N_view, D, H, W)
        Returns:
            coords_position_embeding: (B, N_view, embed_dims, H, W)
            coords_mask: (B, N_view, H, W)
        """
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats[self.position_level].shape
        # 映射到原图尺度上，得到对应的像素坐标.
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H      # (H, )
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W      # (W, )

        if self.LID:
            # (D, )
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        # (3, W, H, D)  --> (W, H, D, 3)    3: (u, v, d)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0).contiguous()
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)      # (W, H, D, 4)    4: (u, v, d, 1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)      # (W, H, D, 4)    4: (du, dv, d, 1)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)      # (B, N_view, 4, 4)

        # (1, 1, W, H, D, 4, 1) --> (B, N_view, W, H, D, 4, 1)
        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)

        # (B, N_view, K, H, W),  (B, N_view, K, H, W)
        if depth_score is not None and self.with_filter:
            _, keep_indices = torch.topk(depth_score, k=self.num_keep, dim=2)
            keep_indices = keep_indices.permute(0, 1, 4, 3, 2).contiguous()     # (B, N_view, W, H, K)
            keep_indices = keep_indices[..., None, None].repeat(1, 1, 1, 1, 1, 4, 1)
            coords = torch.gather(coords, dim=4, index=keep_indices)
            D = self.num_keep

        # (B, N_view, 1, 1, 1, 4, 4) --> (B, N_view, W, H, D, 4, 4)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)

        # 图像中每个像素对应的frustum points，借助img2lidars投影到lidar系中.
        # (B, N_view, W, H, D, 4, 4) @ (B, N_view, W, H, D, 4, 1) --> (B, N_view, W, H, D, 4, 1)
        # --> (B, N_view, W, H, D, 3)   3: (x, y, z)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        # 借助position_range，对3D坐标进行归一化.
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)     # (B, N_view, W, H, D, 3), 超出range的points mask
        # (B, N_view, W, H), 若该像素对应的frustum points 有过多的点超出range， 则对应的coords_mask=1
        # 在后续attention过程中， 会消除这些像素的影响.
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)   # (B, N_view, H, W)
        # (B, N_view, W, H, D, 3) --> (B, N_view, D, 3, H, W) --> (B*N_view, D*3, H, W)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)    # (B*N_view, D*3, H, W)
        # 3D position embedding(PE)
        coords_position_embeding = self.position_encoder(coords3d)      # (B*N_view, embed_dims, H, W)
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N_view, C, H, W).    # List[(B, N_view, C'=256, H'/16, W'/16), (B, N_view, C'=256, H'/32, W'/32), ]
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        x = mlvl_feats[0]   # (B, N_view, C, H, W)  只选择一个level的图像特征.
        batch_size, num_cams, fH, fW = x.size(0), x.size(1), x.size(3), x.size(4)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        # 建立masks，图像中pad的部分为1, 用于在attention过程中消除pad部分的影响.
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))    # (B, N_view, img_H, img_W)

        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0

        x = x.flatten(0, 1)    # (B*N_view, C, H, W)
        if self.with_depth_supervision:
            # 获得相机内外参
            intrinsics_list = []
            extrinsics_list = []
            for batch_id in range(len(img_metas)):
                cur_intrinsics = img_metas[batch_id]['intrinsics']    # List[(4, 4), (4, 4), ...]
                cur_extrinsics = img_metas[batch_id]['extrinsics']    # List[(4, 4), (4, 4), ...]
                cur_intrinsics = x.new_tensor(cur_intrinsics)    # (N_view, 4, 4)
                cur_extrinsics = x.new_tensor(cur_extrinsics)    # (N_view, 4, 4)
                intrinsics_list.append(cur_intrinsics)
                extrinsics_list.append(cur_extrinsics)
            intrinsics = torch.stack(intrinsics_list, dim=0)[..., :3, :3].contiguous()     # (B, N_view, 3, 3)
            extrinsics = torch.stack(extrinsics_list, dim=0).contiguous()        # (B, N_view, 4, 4)

            # (B*N_view, D, H, W), (B*N_view, C, H, W)
            depth, x = self.depth_net(x, intrinsics, extrinsics)
            if self.use_sigmoid:
                depth_score = depth.sigmoid()
            else:
                depth_score = depth.softmax(dim=1)

            # for vis
            # for j in range(depth.shape[0]):
            #     cur_depth_score = depth_score[j]  # (D, fH, fW)
            #     max_depth = torch.argmax(cur_depth_score, dim=0)  # (fH, fW)
            #     max_depth = max_depth.detach().cpu().numpy()
            #     max_depth = max_depth * 255 / 63
            #     max_depth = max_depth.astype(np.uint8)
            #     depth_score_map = cv2.applyColorMap(max_depth, cv2.COLORMAP_RAINBOW)
            #     cv2.imshow("score_map", depth_score_map)
            #     cv2.waitKey(0)

            self.depth_score = depth_score
            depth_score = depth_score.view(batch_size, num_cams, -1, fH, fW)
        else:
            depth_score = None

        # (B*N_view, C, H, W) --> (B*N_view, C'=embed_dim, H, W)
        x = self.input_proj(x)
        x = x.view(batch_size, num_cams, *x.shape[-3:])     # (B, N_view, C'=embed_dim, H, W)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)    # (B, N_view, H, W)

        if self.with_position:
            # 额, 但是这里没有使用coords_mask.
            # 3D PE: (B, N_view, embed_dims, H, W)
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks, depth_score)
            if self.with_dpe:
                if self.use_detach:
                    coords_position_embeding = self.dpe(
                        coords_position_embeding.flatten(0, 1), depth_score.detach().flatten(0, 1)).view(
                        x.size())   # (B, N, embed_dims, H, W)
                else:
                    coords_position_embeding = self.dpe(
                        coords_position_embeding.flatten(0, 1), depth_score.flatten(0, 1)).view(x.size())  # (B, N, embed_dims, H, W)

            pos_embed = coords_position_embeding
            if self.with_multiview:
                # 加入 2D PE 和 multi-view prior
                sin_embed = self.positional_encoding(masks)    # (B, N_view, num_feats*3=embed_dims*3/2, H, W)
                # (B, N_view, num_feats*3=embed_dims*3/2, H, W) --> (B*N_view, num_feats*3=embed_dims*3/2, H, W)
                # --> (B*N_view, embed_dims, H, W) --> (B, N_view, embed_dims, H, W)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed   # (B, N_view, embed_dims, H, W)
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        reference_points = self.reference_points.weight
        # 3D anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
        # (N_query, 3) --> (N_query, num_feats*3=embed_dims*3/2) --> (N_query, embed_dims)
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))

        # (1, N_query, 3) --> (B, N_query, 3)
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)

        # key为image feature, key_pos为生成的3D PE(+ 2D PE 、multi-view prior)
        # key+key_pos 即对应3D position-aware的特征.
        # import time
        # torch.cuda.synchronize()
        # time1 = time.time()
        outs_dec, _ = self.transformer(x,   # (B, N_view, embed_dim, H, W)
                                       masks,   # (B, N_view, H, W)
                                       query_embeds,    # (N_query, embed_dims)
                                       pos_embed,       # (B, N_view, embed_dims, H, W)
                                       self.reg_branches    # 没有进行box_refine, 因此没有用到reg_branches.
                                       )
        outs_dec = torch.nan_to_num(outs_dec)       # (num_layers, B, N_query, C=embed_dims)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("time = %f ms" % ((time2 - time1) * 1000))

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())   # (B, N_query, 3)
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])   # (B, N_query, n_cls)
            # (B, N_query, code_size)     code_size: (tx, ty, log(dx), log(dy), tz, log(dz), sin(rot), cos(rot), vx, vy)
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()     # (normalized_cx, normalized_cy)
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()     # normalized_cz

            # (B, N_query, code_size)  code_size: (normalized_cx, normalized_cy, log(dx), log(dy), normalized_cz, log(dz), sin(rot), cos(rot), vx, vy)
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)   # (num_layers, B, N_query, n_cls)
        all_bbox_preds = torch.stack(outputs_coords)    # (num_layers, B, N_query, code_size)

        # (B, N_query, code_size)  code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
        }
        if self.depth_net is not None:
            outs['depth'] = depth_score.view(batch_size*num_cams, -1, fH, fW)
        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             depth_map,
             depth_map_mask,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            depth_map:  # (B*N_views, fH, fW)
            depth_map_mask:  # (B*N_views, fH, fW)
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']      # (num_layers, B, N_query, n_cls)
        all_bbox_preds = preds_dicts['all_bbox_preds']      # (num_layers, B, N_query, code_size)   code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # 分别计算每层decoder layer的loss
        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        if self.with_depth_supervision:
            depth = preds_dicts['depth']    # (B*N_view, D, H, W)
            depth_loss = self.depth_loss(depth, depth_map, depth_map_mask)
            loss_dict['depth_loss'] = depth_loss

        return loss_dict

    def mask_points_by_dist(self, depth_map, depth_map_mask, min_dist, max_dist):
        mask = depth_map.new_ones(depth_map.shape, dtype=torch.bool)
        mask = torch.logical_and(mask, depth_map >= min_dist)
        mask = torch.logical_and(mask, depth_map < max_dist)
        depth_map_mask[~mask] = 0
        depth_map[~mask] = 0
        return depth_map, depth_map_mask

    def depth_loss(self, depth, depth_map, depth_map_mask):
        """
        Args:
            depth: (B*N_view, D, H, W)
            depth_map: (B*N_view, H, W)
            depth_map_mask: (B*N_view, H, W)
        Returns:

        """
        depth = depth.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_num)      # (B*N_view*H*W, D)
        depth_map = depth_map.view(-1)                          # (B*N_view*H*W, )
        depth_map_mask = depth_map_mask.view(-1).float()                # (B*N_view*H*W, )

        min_dist = self.depth_start
        if self.LID:
            bin_size = 2 * (self.position_range[3] - min_dist) / (self.depth_num * (1 + self.depth_num))
            depth_label = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - min_dist) / bin_size)   # (H, W)
        else:
            bin_size = (self.position_range[3] - min_dist) / self.depth_num
            depth_label = (depth_map - min_dist) / bin_size

        depth_label = depth_label.long()
        depth_label, depth_map_mask = self.mask_points_by_dist(depth_label, depth_map_mask, 0, self.depth_num)

        # for vis
        # depth_gt = depth_label.float().detach().cpu().numpy().reshape((6, 16, 44))
        # depth_mask = depth_map_mask.float().detach().cpu().numpy().reshape((6, 16, 44))
        # depth_gt = depth_gt / self.depth_num * 255
        # depth_gt = depth_gt.astype(np.uint8)
        # depth_mask *= 255
        # depth_mask = depth_mask.astype(np.uint8)
        # for i in range(6):
        #     depth_gt_vis = cv2.applyColorMap(depth_gt[i], colormap=cv2.COLORMAP_RAINBOW)
        #     cv2.imshow(f'depth {i}', depth_gt_vis)
        #     cv2.imshow(f'depth_mask{i}', depth_mask[i])
        #     cv2.waitKey(0)

        loss_depth = self.loss_depth(depth, depth_label, depth_map_mask,
                                     avg_factor=max(depth_map_mask.sum().float(), 1.0))
        return loss_depth


# --------------------------利用预测的depth map将pixel映射到3D point， 然后生成3D PE ----------------------------------------
class Balancer(nn.Module):
    def __init__(self, fg_weight, bg_weight, downsample_factor):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def compute_fg_mask(self, gt_boxes3d, shape, lidar2imgs, device):
        """
        Args:
            gt_boxes3d: List[(N0, 9), (N1, 9), ...]
            shape: (B, N_view, H, W)
            lidar2imgs: (B, N_view, 4, 4)
            img_metas: List[img_meta, ...]
        Returns:
            fg_mask: (B, N_view, H, W)
        """
        batch_size, n_view, fH, fW = shape
        fg_mask = torch.zeros((batch_size, n_view, fH, fW), dtype=torch.bool, device=device)  # (B, N_view, H, W)
        for batch_idx in range(batch_size):
            cur_gt_boxes3d = gt_boxes3d[batch_idx]  # (N, 9)
            cur_gt_boxes3d = LiDARInstance3DBoxes(cur_gt_boxes3d, box_dim=9, origin=(0.5, 0.5, 0.5))
            cur_corners3d = cur_gt_boxes3d.corners      # (N, 8, 3)
            cur_corners3d = torch.cat([cur_corners3d, torch.ones_like(cur_corners3d[..., 0:1])],
                                      dim=-1)  # (N, 8, 4)
            cur_corners3d = cur_corners3d.view(-1, 4)  # (N*8, 4)

            cur_lidar2imgs = lidar2imgs[batch_idx]      # (N_view, 4, 4)

            for view_id in range(n_view):
                cur_lidar2img = cur_lidar2imgs[view_id]  # (4, 4)
                cur_corners3d_proj = cur_corners3d @ cur_lidar2img.T  # (N*8, 4)
                cur_corners2ds = cur_corners3d_proj[:, :2] / cur_corners3d_proj[:, 2:3]  # (N*8, 2)
                cur_corners_depths = cur_corners3d_proj[:, 2].view(-1, 8)    # (N, 8)
                cur_corners2ds = cur_corners2ds.view(-1, 8, 2)  # (N, 8, 2)

                for box_id in range(len(cur_corners2ds)):
                    cur_corners_depth = cur_corners_depths[box_id]  # (8, )
                    cur_corners2d = cur_corners2ds[box_id]      # (8, 2)
                    in_front = cur_corners_depth > 1e-5     # (8, )
                    if in_front.sum() == 0:
                        continue

                    cur_valid_corners2d = cur_corners2d[in_front]   # (valid, 2)
                    cur_valid_corners2d /= self.downsample_factor

                    min_uv, _ = torch.min(cur_valid_corners2d, dim=0)  # (2, )
                    max_uv, _ = torch.max(cur_valid_corners2d, dim=0)  # (2, )
                    cur_boxes2d = torch.cat([min_uv, max_uv], dim=0)  # (4, ) 4: (u1, v1, u2, v2)

                    cur_boxes2d[:2] = torch.floor(cur_boxes2d[:2])
                    cur_boxes2d[2:] = torch.ceil(cur_boxes2d[2:])

                    cur_boxes2d[0] = torch.clamp(cur_boxes2d[0], min=0, max=fW-1)
                    cur_boxes2d[1] = torch.clamp(cur_boxes2d[1], min=0, max=fH-1)
                    cur_boxes2d[2] = torch.clamp(cur_boxes2d[2], min=0, max=fW-1)
                    cur_boxes2d[3] = torch.clamp(cur_boxes2d[3], min=0, max=fH-1)

                    cur_boxes2d = cur_boxes2d.long()
                    u1, v1, u2, v2 = cur_boxes2d
                    fg_mask[batch_idx, view_id, v1:v2, u1:u2] = True

        # for vis
        # fg_mask = fg_mask[0]    # (N_view, H, W)
        # for view_id in range(n_view):
        #     cur_fg_mask = fg_mask[view_id]  # (H, W)
        #     cur_fg_mask = cur_fg_mask.float().cpu().detach().numpy()
        #     cur_fg_mask *= 255
        #     cur_fg_mask = cur_fg_mask.astype(np.uint8)
        #     cv2.imshow("img", cur_fg_mask)
        #     cv2.waitKey(0)

        return fg_mask

    def forward(self, loss, shape, gt_boxes3d, depth_map_mask, img_metas):
        """
        Forward pass
        Args:
            loss: (B*N_view*H*W, ), Pixel-wise loss
            shape: (B, N_view, H, W)
            gt_boxes2d: List[(N0, 9), (N1, 9), ...]
            depth_map_mask: (B*N_view*H*W, )
            img_metas: List[img_meta0, img_meta1, ...]

        Returns:
            loss: (1), Total loss after foreground/background balancing
            tb_dict: dict[float], All losses to log in tensorboard
        """
        lidar2imgs = []
        for img_meta in img_metas:
            lidar2img = []
            for i in range(len(img_meta['lidar2img'])):
                lidar2img.append(img_meta['lidar2img'][i])
            lidar2imgs.append(np.asarray(lidar2img))
        lidar2img = np.asarray(lidar2imgs)
        lidar2img = loss.new_tensor(lidar2img)      # (B, N_view, 4, 4)

        # Compute masks
        fg_mask = self.compute_fg_mask(gt_boxes3d=gt_boxes3d,
                                       shape=shape,
                                       lidar2imgs=lidar2img,
                                       device=loss.device)    # (B, N_view, H, W)
        fg_mask = fg_mask.view(-1)
        bg_mask = ~fg_mask

        depth_map_mask = depth_map_mask.bool()
        fg_mask = fg_mask[depth_map_mask]
        bg_mask = bg_mask[depth_map_mask]

        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        # num_pixels = fg_mask.sum() + bg_mask.sum()
        num_pixels = weights.sum()

        # Compute losses
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels

        return fg_loss, bg_loss


@HEADS.register_module()
class PETRDepthHeadV2(PETRDepthHead):
    def __init__(self,
                 use_dfl=False,
                 use_detach=False,
                 share_pe_encoder=True,
                 with_2dpe_only=False,
                 use_prob_depth=True,
                 use_balancer=False,
                 loss_dfl=dict(type='DistributionFocalLoss', reduction='mean', loss_weight=1.0),
                 balancer_cfg=dict(fg_weight=5.0, bg_weight=1.0, downsample_factor=16),
                 with_pos_info=False,
                 **kwargs):
        self.share_pe_encoder = share_pe_encoder
        self.with_2dpe_only = with_2dpe_only
        self.with_pos_info = with_pos_info
        super(PETRDepthHeadV2, self).__init__(**kwargs)
        self.use_dfl = use_dfl
        self.use_detach = use_detach
        self.use_balancer = use_balancer

        if self.with_depth_supervision:
            self.with_pgd = getattr(self.depth_net, 'with_pgd', False)
            if self.use_dfl:
                self.loss_dfl = build_loss(loss_dfl)

            self.use_prob_depth = use_prob_depth
            if self.use_prob_depth:
                index = torch.arange(start=0, end=self.depth_num, step=1).float()  # (D, )
                bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num - 1)
                depth_bin = self.depth_start + bin_size * index  # (D, )
                self.register_buffer('project', depth_bin)  # (D, )
            if not self.use_prob_depth:
                assert self.depth_num == 1, 'depth_num setting is wrong'
                assert self.with_pgd is False, 'direct depth prediction cannot be combined with pgd'
                assert self.use_dfl is False, 'direct depth prediction cannot be combined with dfl'

            if self.use_balancer:
                assert self.loss_depth.reduction == 'none', 'reduction must be none when use_balancer is True'
                self.balancer = Balancer(**balancer_cfg)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # 因为没有box_refine, 共享权重.
        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        if self.with_2dpe_only:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        # 在3D空间初始化一组0-1之间均匀分布的learnable anchor points.
        self.reference_points = nn.Embedding(self.num_query, 3)

        if self.share_pe_encoder:
            position_encoder = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
            if self.with_position:
                self.position_encoder = position_encoder

            # anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
            self.query_embedding = position_encoder
        else:
            if self.with_position:
                # self.position_dim = 3 * self.depth_num      # D*3 3:(x, y, z)
                self.position_encoder = nn.Sequential(
                    nn.Linear(self.embed_dims*3//2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                )
            self.query_embedding = nn.Sequential(
                    nn.Linear(self.embed_dims*3//2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                )

        if self.with_pos_info:
            self.extra_position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )

    def integral(self, depth_pred):
        """
        Args:
            depth_pred: (N, D)
        Returns:
            depth_val: (N, )
        """
        depth_score = F.softmax(depth_pred, dim=-1)     # (N, D)
        depth_val = F.linear(depth_score, self.project.type_as(depth_score))  # (N, D) * (D, )  --> (N, )
        return depth_val

    def position_embeding(self, img_feats, img_metas, masks=None, depth_map=None):
        """
        Args:
            img_feats: List[(B, N_view, C, H, W), ]
            img_metas:
            masks: (B, N_view, H, W)
            depth_map: (B, N_view, H, W)
            depth_map_mask: (B, N_view, H, W)
        Returns:
            coords_position_embeding: (B, N_view, embed_dims, H, W)
            coords_mask: (B, N_view, H, W)
        """
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats[self.position_level].shape
        # 映射到原图尺度上，得到对应的像素坐标.
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H  # (H, )
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W  # (W, )

        # (2, W, H)  --> (W, H, 2)    2: (u, v)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(1, 2, 0).contiguous()
        coords = coords.view(1, 1, W, H, 2).repeat(B, N, 1, 1, 1)       # (B, N_view, W, H, 2)

        depth_map = depth_map.permute(0, 1, 3, 2).contiguous()      # (B, N_view, W, H)

        # inplace
        # coords = torch.cat((coords, depth_map.unsqueeze(dim=-1)), dim=-1)       # (B, N_view, W, H, 3)   3:(u, v, d)
        # coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)    # (B, N_view, W, H, 4)    4: (u, v, d, 1)
        # coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(
        #     coords[..., 2:3]) * eps)  # (B, N_view, W, H, 4)    4: (du, dv, d, 1)

        depth_map = depth_map.unsqueeze(dim=-1)     # (B, N_view, W, H, 1)
        coords = coords * torch.maximum(depth_map, torch.ones_like(depth_map) * eps)  # (B, N_view, W, H, 2)    (du, dv)
        coords = torch.cat([coords, depth_map], dim=-1)     # (B, N_view, W, H, 3)   (du, dv, d)
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)  # (B, N_view, W, H, 4)   (du, dv, d, 1)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N_view, 4, 4)

        coords = coords.unsqueeze(dim=-1)       # (B, N_view, W, H, 4, 1)
        # (B, N_view, 1, 1, 4, 4) --> (B, N_view, W, H, 4, 4)
        img2lidars = img2lidars.view(B, N, 1, 1, 4, 4).repeat(1, 1, W, H, 1, 1)

        # 图像中每个像素对应的frustum points，借助img2lidars投影到lidar系中.
        # (B, N_view, W, H, 4, 4) @ (B, N_view, H, D, 4, 1) --> (B, N_view, W, H, 4, 1)
        # --> (B, N_view, W, H, 3)   3: (x, y, z)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        # 借助position_range，对3D坐标进行归一化.
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
                    self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
                    self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
                    self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)  # (B, N_view, W, H, 3), 超出range的points mask
        coords_mask = coords_mask.sum(dim=-1) > 0       # (B, N_view, W, H)
        # 在后续attention过程中， 会消除这些像素的影响.
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)  # (B, N_view, H, W)

        coords3d = coords3d.permute(0, 1, 3, 2, 4).contiguous().view(B*N, H, W, 3)      # (B*N_view, H, W, 3)
        coords3d = inverse_sigmoid(coords3d)    # (B*N_view, H, W, 3)
        # 3D position embedding(PE)
        coords_position_embeding = self.position_encoder(pos2posemb3d(coords3d))  # (B*N_view, H, W, embed_dims)
        coords_position_embeding = coords_position_embeding.permute(0, 3, 1, 2).contiguous()    # (B*N_view, embed_dims, H, W)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N_view, C, H, W).    # List[(B, N_view, C'=256, H'/16, W'/16), (B, N_view, C'=256, H'/32, W'/32), ]
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        x = mlvl_feats[0]  # (B, N_view, C, H, W)  只选择一个level的图像特征.
        batch_size, num_cams, fH, fW = x.size(0), x.size(1), x.size(3), x.size(4)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        # 建立masks，图像中pad的部分为1, 用于在attention过程中消除pad部分的影响.
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))  # (B, N_view, img_H, img_W)

        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0

        x = x.flatten(0, 1)  # (B*N_view, C, H, W)
        if self.with_depth_supervision:
            # 获得相机内外参
            intrinsics_list = []
            extrinsics_list = []
            for batch_id in range(len(img_metas)):
                cur_intrinsics = img_metas[batch_id]['intrinsics']  # List[(4, 4), (4, 4), ...]
                cur_extrinsics = img_metas[batch_id]['extrinsics']  # List[(4, 4), (4, 4), ...]
                cur_intrinsics = x.new_tensor(cur_intrinsics)  # (N_view, 4, 4)
                cur_extrinsics = x.new_tensor(cur_extrinsics)  # (N_view, 4, 4)
                intrinsics_list.append(cur_intrinsics)
                extrinsics_list.append(cur_extrinsics)
            intrinsics = torch.stack(intrinsics_list, dim=0)[..., :3, :3].contiguous()  # (B, N_view, 3, 3)
            extrinsics = torch.stack(extrinsics_list, dim=0).contiguous()  # (B, N_view, 4, 4)

            # (B*N_view, D, H, W), (B*N_view, C, H, W), (B*N_view, 1, H, W)
            if self.with_pgd:
                depth, x, depth_direct = self.depth_net(x, intrinsics, extrinsics)
            else:
                # (B * N_view, D/1, H, W),  (B*N_view, C, H, W)
                depth, x = self.depth_net(x, intrinsics, extrinsics)
            # for vis
            # for j in range(depth.shape[0]):
            #     cur_depth_score = depth_score[j]  # (D, fH, fW)
            #     max_depth = torch.argmax(cur_depth_score, dim=0)  # (fH, fW)
            #     max_depth = max_depth.detach().cpu().numpy()
            #     max_depth = max_depth * 255 / 63
            #     max_depth = max_depth.astype(np.uint8)
            #     depth_score_map = cv2.applyColorMap(max_depth, cv2.COLORMAP_RAINBOW)
            #     cv2.imshow("score_map", depth_score_map)
            #     cv2.waitKey(0)

            if self.use_prob_depth:
                self.depth_score = depth   # 未经过softmax
                depth_prob = depth.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_num)   # (B*N_view, H, W, D) --> (B*N_view*H*W, D)
                depth_prob_val = self.integral(depth_prob)      # (B*N_view*H*W, )
                depth_map_pred = depth_prob_val

                if self.with_pgd:
                    sig_alpha = torch.sigmoid(self.depth_net.fuse_lambda)
                    depth_direct_val = depth_direct.view(-1)      # (B*N_view*H*W, )
                    depth_pgd_fuse = sig_alpha * depth_direct_val + (1 - sig_alpha) * depth_prob_val
                    depth_map_pred = depth_pgd_fuse
            else:
                # direct depth
                depth_map_pred = depth.exp().view(-1)     # (B*N_view*H*W, )
        else:
            depth_map_pred = None

        # (B*N_view, C, H, W) --> (B*N_view, C'=embed_dim, H, W)
        x = self.input_proj(x)
        x = x.view(batch_size, num_cams, *x.shape[-3:])  # (B, N_view, C'=embed_dim, H, W)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)  # (B, N_view, H, W)

        if self.with_position:
            # 额, 但是这里没有使用coords_mask.
            # 3D PE: (B, N_view, embed_dims, H, W)
            if self.use_detach:
                depth_map = depth_map_pred.detach()
            else:
                depth_map = depth_map_pred
            depth_map = depth_map.view(batch_size, num_cams, fH, fW)
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks, depth_map)
            pos_embed = coords_position_embeding
            if self.with_multiview:
                # 加入 2D PE 和 multi-view prior
                sin_embed = self.positional_encoding(masks)  # (B, N_view, num_feats*3=embed_dims*3/2, H, W)
                # (B, N_view, num_feats*3=embed_dims*3/2, H, W) --> (B*N_view, num_feats*3=embed_dims*3/2, H, W)
                # --> (B*N_view, embed_dims, H, W) --> (B, N_view, embed_dims, H, W)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed  # (B, N_view, embed_dims, H, W)
            elif self.with_2dpe_only:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embed = pos_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            elif self.with_2dpe_only:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)
            else:
                pos_embed = x.new_zeros(x.size())

        reference_points = self.reference_points.weight
        # 3D anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
        # (N_query, 3) --> (N_query, num_feats*3=embed_dims*3/2) --> (N_query, embed_dims)
        query_embeds = self.query_embedding(pos2posemb3d(inverse_sigmoid(reference_points)))

        # (1, N_query, 3) --> (B, N_query, 3)
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)

        # key为image feature, key_pos为生成的3D PE(+ 2D PE 、multi-view prior)
        # key+key_pos 即对应3D position-aware的特征.
        # import time
        # torch.cuda.synchronize()
        # time1 = time.time()
        outs_dec, _ = self.transformer(x,  # (B, N_view, embed_dim, H, W)
                                       masks,  # (B, N_view, H, W)
                                       query_embeds,  # (N_query, embed_dims)
                                       pos_embed,  # (B, N_view, embed_dims, H, W)
                                       self.reg_branches  # 没有进行box_refine, 因此没有用到reg_branches.
                                       )
        outs_dec = torch.nan_to_num(outs_dec)  # (num_layers, B, N_query, C=embed_dims)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("time = %f ms" % ((time2 - time1) * 1000))

        if self.with_pos_info:
            pos_feat = self.extra_position_encoder(inverse_sigmoid(reference_points))   # (B, N_query, C)
            outs_dec = outs_dec + pos_feat[None, ...]   # (num_layers, B, N_query, C=embed_dims)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())  # (B, N_query, 3)
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])  # (B, N_query, n_cls)
            # (B, N_query, code_size)     code_size: (tx, ty, log(dx), log(dy), tz, log(dz), sin(rot), cos(rot), vx, vy)
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()  # (normalized_cx, normalized_cy)
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()  # normalized_cz

            # (B, N_query, code_size)  code_size: (normalized_cx, normalized_cy, log(dx), log(dy), normalized_cz, log(dz), sin(rot), cos(rot), vx, vy)
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)  # (num_layers, B, N_query, n_cls)
        all_bbox_preds = torch.stack(outputs_coords)  # (num_layers, B, N_query, code_size)

        # (B, N_query, code_size)  code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        if self.depth_net is not None:
            outs['depth_map_pred'] = depth_map_pred    # (B*N_view*H*W, )
            if self.use_prob_depth:
                outs['depth_prob'] = depth_prob     # (B*N_view*H*W, D)
                if self.with_pgd:
                    outs['depth_prob_val'] = depth_prob_val  # (B*N_view*H*W, )
                    outs['depth_direct_val'] = depth_direct_val  # (B*N_view*H*W, )

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             depth_map,
             depth_map_mask,
             img_metas=None,
             gt_bboxes_ignore=None
             ):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            depth_map:  # (B*N_views, fH, fW)
            depth_map_mask:  # (B*N_views, fH, fW)
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            img_metas:
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']      # (num_layers, B, N_query, n_cls)
        all_bbox_preds = preds_dicts['all_bbox_preds']      # (num_layers, B, N_query, code_size)   code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        # 分别计算每层decoder layer的loss
        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        if self.with_depth_supervision:
            depth_map_pred = preds_dicts['depth_map_pred']    # (B*N_view*H*W, )
            depth_prob = preds_dicts.get('depth_prob', None)    # (B*N_view*H*W, D)

            depth_loss_dict = self.depth_loss(depth_map_pred, depth_map, depth_map_mask, depth_prob,
                                              gt_boxes3d=gt_bboxes_list, img_metas=img_metas)
            loss_dict.update(depth_loss_dict)

        self.loss_dict = loss_dict
        return loss_dict

    def depth_loss(self, depth_map_pred, depth_map_tgt, depth_map_mask, depth_prob=None, gt_boxes3d=None,
                   img_metas=None):
        """
        Args:
            depth_map_pred: (B*N_view*H*W, )
            gt_boxes3d: (B*N_view, H, W)
            depth_map_mask: (B*N_view, H, W)
            depth_prob: (B*N_view*H*W, D)
            gt_boxes3d: List[(N0, 9), (N1, 9), ...]
            img_metas: List[img_meta0, img_meta1, ...]
        Returns:

        """
        batch_size = len(gt_boxes3d)
        n_view = depth_map_tgt.shape[0] // batch_size
        fH, fW = depth_map_tgt.shape[1], depth_map_tgt.shape[2]

        depth_map_mask = depth_map_mask.view(-1).float()                # (B*N_view*H*W, )
        depth_map_tgt = depth_map_tgt.view(-1)                          # (B*N_view*H*W, )

        min_dist = self.depth_start
        depth_map_tgt, depth_map_mask = self.mask_points_by_dist(depth_map_tgt, depth_map_mask,
                                                                 min_dist=min_dist,
                                                                 max_dist=self.position_range[3])

        valid = depth_map_mask > 0
        valid_depth_pred = depth_map_pred[valid]      # (N_valid, )

        loss_dict = {}
        loss_depth = self.loss_depth(pred=valid_depth_pred, target=depth_map_tgt[valid],
                                     avg_factor=max(depth_map_mask.sum().float(), 1.0))

        if self.use_balancer:
            shape = (batch_size, n_view, fH, fW)
            fg_loss_depth, bg_loss_depth = self.balancer(loss_depth, shape, gt_boxes3d, depth_map_mask, img_metas)
            loss_dict['fg_loss_depth'] = fg_loss_depth
            loss_dict['bg_loss_depth'] = bg_loss_depth
        else:
            loss_dict['loss_depth'] = loss_depth

        if self.use_dfl and depth_prob is not None:
            bin_size = (self.position_range[3] - min_dist) / (self.depth_num - 1)
            depth_label_clip = (depth_map_tgt - min_dist) / bin_size
            depth_map_clip, depth_map_mask = self.mask_points_by_dist(depth_label_clip, depth_map_mask, 0,
                                                                      self.depth_num - 1)      # (B*N_view*H*W, )

            valid = depth_map_mask > 0      # (B*N_view*H*W, )
            valid_depth_prob = depth_prob[valid]    # (N_valid, )
            loss_dfl = self.loss_dfl(pred=valid_depth_prob, target=depth_map_clip[valid],
                                     avg_factor=max(depth_map_mask.sum().float(), 1.0))

            if self.use_balancer:
                shape = (batch_size, n_view, fH, fW)
                fg_loss_dfl, bg_loss_dfl = self.balancer(loss_dfl, shape, gt_boxes3d, depth_map_mask, img_metas)
                loss_dict['fg_loss_dfl'] = fg_loss_dfl
                loss_dict['bg_loss_dfl'] = bg_loss_dfl
            else:
                loss_dict['loss_dfl'] = loss_dfl

        return loss_dict


# -------------------------- 在PETRDepthHeadV2基础上支持box_refine ------------------------------------------------------
@HEADS.register_module()
class PETRDepthHeadV2_Refine(PETRDepthHeadV2):
    def __init__(self, with_box_refine=False, **kwargs):
        self.with_box_refine = with_box_refine
        super(PETRDepthHeadV2_Refine, self).__init__(**kwargs)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        if self.with_box_refine:
            # 如果with_box_refine， 那么fc_cls、reg_branch不共享权重.
            self.cls_branches = _get_clones(fc_cls, self.num_pred)
            self.reg_branches = _get_clones(reg_branch, self.num_pred)
        else:
            # 因为没有box_refine, 共享权重.
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        if self.with_2dpe_only:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        # 在3D空间初始化一组0-1之间均匀分布的learnable anchor points.
        self.reference_points = nn.Embedding(self.num_query, 3)

        if self.share_pe_encoder:
            position_encoder = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
            if self.with_position:
                self.position_encoder = position_encoder

            # anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
            self.query_embedding = position_encoder
        else:
            if self.with_position:
                # self.position_dim = 3 * self.depth_num      # D*3 3:(x, y, z)
                self.position_encoder = nn.Sequential(
                    nn.Linear(self.embed_dims*3//2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                )
            self.query_embedding = nn.Sequential(
                    nn.Linear(self.embed_dims*3//2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                )

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N_view, C, H, W).    # List[(B, N_view, C'=256, H'/16, W'/16), (B, N_view, C'=256, H'/32, W'/32), ]
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        x = mlvl_feats[0]  # (B, N_view, C, H, W)  只选择一个level的图像特征.
        batch_size, num_cams, fH, fW = x.size(0), x.size(1), x.size(3), x.size(4)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        # 建立masks，图像中pad的部分为1, 用于在attention过程中消除pad部分的影响.
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))  # (B, N_view, img_H, img_W)

        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0

        x = x.flatten(0, 1)  # (B*N_view, C, H, W)
        if self.with_depth_supervision:
            # 获得相机内外参
            intrinsics_list = []
            extrinsics_list = []
            for batch_id in range(len(img_metas)):
                cur_intrinsics = img_metas[batch_id]['intrinsics']  # List[(4, 4), (4, 4), ...]
                cur_extrinsics = img_metas[batch_id]['extrinsics']  # List[(4, 4), (4, 4), ...]
                cur_intrinsics = x.new_tensor(cur_intrinsics)  # (N_view, 4, 4)
                cur_extrinsics = x.new_tensor(cur_extrinsics)  # (N_view, 4, 4)
                intrinsics_list.append(cur_intrinsics)
                extrinsics_list.append(cur_extrinsics)
            intrinsics = torch.stack(intrinsics_list, dim=0)[..., :3, :3].contiguous()  # (B, N_view, 3, 3)
            extrinsics = torch.stack(extrinsics_list, dim=0).contiguous()  # (B, N_view, 4, 4)

            # (B*N_view, D, H, W), (B*N_view, C, H, W), (B*N_view, 1, H, W)
            if self.with_pgd:
                depth, x, depth_direct = self.depth_net(x, intrinsics, extrinsics)
            else:
                # (B * N_view, D/1, H, W),  (B*N_view, C, H, W)
                depth, x = self.depth_net(x, intrinsics, extrinsics)
            # for vis
            # for j in range(depth.shape[0]):
            #     cur_depth_score = depth_score[j]  # (D, fH, fW)
            #     max_depth = torch.argmax(cur_depth_score, dim=0)  # (fH, fW)
            #     max_depth = max_depth.detach().cpu().numpy()
            #     max_depth = max_depth * 255 / 63
            #     max_depth = max_depth.astype(np.uint8)
            #     depth_score_map = cv2.applyColorMap(max_depth, cv2.COLORMAP_RAINBOW)
            #     cv2.imshow("score_map", depth_score_map)
            #     cv2.waitKey(0)

            if self.use_prob_depth:
                self.depth_score = depth   # 未经过softmax
                depth_prob = depth.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_num)   # (B*N_view, H, W, D) --> (B*N_view*H*W, D)
                depth_prob_val = self.integral(depth_prob)      # (B*N_view*H*W, )
                depth_map_pred = depth_prob_val

                if self.with_pgd:
                    sig_alpha = torch.sigmoid(self.depth_net.fuse_lambda)
                    depth_direct_val = depth_direct.view(-1)      # (B*N_view*H*W, )
                    depth_pgd_fuse = sig_alpha * depth_direct_val + (1 - sig_alpha) * depth_prob_val
                    depth_map_pred = depth_pgd_fuse
            else:
                # direct depth
                depth_map_pred = depth.exp().view(-1)     # (B*N_view*H*W, )
        else:
            depth_map_pred = None

        # (B*N_view, C, H, W) --> (B*N_view, C'=embed_dim, H, W)
        x = self.input_proj(x)
        x = x.view(batch_size, num_cams, *x.shape[-3:])  # (B, N_view, C'=embed_dim, H, W)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)  # (B, N_view, H, W)

        if self.with_position:
            # 额, 但是这里没有使用coords_mask.
            # 3D PE: (B, N_view, embed_dims, H, W)
            if self.use_detach:
                depth_map = depth_map_pred.detach()
            else:
                depth_map = depth_map_pred
            depth_map = depth_map.view(batch_size, num_cams, fH, fW)
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks, depth_map)
            pos_embed = coords_position_embeding
            if self.with_multiview:
                # 加入 2D PE 和 multi-view prior
                sin_embed = self.positional_encoding(masks)  # (B, N_view, num_feats*3=embed_dims*3/2, H, W)
                # (B, N_view, num_feats*3=embed_dims*3/2, H, W) --> (B*N_view, num_feats*3=embed_dims*3/2, H, W)
                # --> (B*N_view, embed_dims, H, W) --> (B, N_view, embed_dims, H, W)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed  # (B, N_view, embed_dims, H, W)
            elif self.with_2dpe_only:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embed = pos_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            elif self.with_2dpe_only:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)
            else:
                pos_embed = x.new_zeros(x.size())

        reference_points = self.reference_points.weight
        # 3D anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
        # (N_query, 3) --> (N_query, num_feats*3=embed_dims*3/2) --> (N_query, embed_dims)
        query_embeds = self.query_embedding(pos2posemb3d(inverse_sigmoid(reference_points)))

        # (1, N_query, 3) --> (B, N_query, 3)
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)

        # key为image feature, key_pos为生成的3D PE(+ 2D PE 、multi-view prior)
        # key+key_pos 即对应3D position-aware的特征.
        hs, init_reference, inter_references = self.transformer(x,  # (B, N_view, embed_dim, H, W)
                                       masks,  # (B, N_view, H, W)
                                       query_embeds,  # (N_query, embed_dims)
                                       pos_embed,  # (B, N_view, embed_dims, H, W)
                                       reference_points,   # (B, N_query, 3)
                                       self.query_embedding,
                                       self.reg_branches if self.with_box_refine else None, # 没有进行box_refine, 因此没有用到reg_branches.
                                       )
        # hs: (N_decoders, N_query, B, C=embed_dims)
        # init_reference_out: (B, N_query, 3)
        # inter_references: (N_decoders, B, N_query, 3)

        hs = hs.permute(0, 2, 1, 3)     # (N_decoders, B, N_query, C=embed_dims)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)      # (B, N_query, 3)
            outputs_class = self.cls_branches[lvl](hs[lvl])     # (B, N_query, n_cls)
            tmp = self.reg_branches[lvl](hs[lvl])       # (B, N_q=300, code_size)   code_size: (tx, ty, log(dx), log(dy), tz, log(dz), sin(rot), cos(rot), vx, vy)

            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()     # (normalized_cx, normalized_cy)
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()     # normalized_cz
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])      # cx
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])      # cy
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])      # cz

            outputs_coord = tmp     # (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)      # (num_decoders, B, N_query, n_cls)
        outputs_coords = torch.stack(outputs_coords)        # (num_decoders, B, N_query, code_size)

        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        if self.depth_net is not None:
            outs['depth_map_pred'] = depth_map_pred    # (B*N_view*H*W, )
            if self.use_prob_depth:
                outs['depth_prob'] = depth_prob     # (B*N_view*H*W, D)
                if self.with_pgd:
                    outs['depth_prob_val'] = depth_prob_val  # (B*N_view*H*W, )
                    outs['depth_direct_val'] = depth_direct_val  # (B*N_view*H*W, )

        return outs


# -------------------------- 将reference points 转换到每个view的视锥空间中    ------ --------------------------------------
@HEADS.register_module()
class PETRDepthHeadV3(PETRDepthHeadV2):
    def __init__(self, coarsen_depth=False, **kwargs):
        self.coarsen_depth = coarsen_depth
        super(PETRDepthHeadV3, self).__init__(**kwargs)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # 因为没有box_refine, 共享权重.
        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        # 在3D空间初始化一组0-1之间均匀分布的learnable anchor points.
        self.reference_points = nn.Embedding(self.num_query, 3)

        self.pe_encoder_3d_1 = nn.Sequential(
            nn.Conv1d(3, self.embed_dims, kernel_size=1, bias=True),
            nn.BatchNorm1d(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1, bias=True)
        )

        self.pe_encoder_3d_2 = nn.Sequential(
            nn.Conv1d(3, self.embed_dims, kernel_size=1, bias=True),
            nn.BatchNorm1d(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1, bias=True)
        )

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def position_embeding(self, img_feats, img_metas, masks=None, depth_map=None):
        """
        Args:
            img_feats: List[(B, N_view, C, H, W), ]
            img_metas:
            masks: (B, N_view, H, W)
            depth_map: (B, N_view, H, W)
            depth_map_mask: (B, N_view, H, W)
        Returns:
            coords_position_embeding: (B, N_view, H, W, embed_dims)
        """
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats[self.position_level].shape
        # 映射到原图尺度上，得到对应的像素坐标.
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H  # (H, )
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W  # (W, )

        # (2, W, H)  --> (W, H, 2)    2: (u, v)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(1, 2, 0).contiguous()
        coords = coords.view(1, 1, W, H, 2).repeat(B, N, 1, 1, 1)       # (B, N_view, W, H, 2)

        depth_map = depth_map.permute(0, 1, 3, 2).contiguous()      # (B, N_view, W, H)
        depth_map = depth_map.unsqueeze(dim=-1)     # (B, N_view, W, H, 1)

        if self.coarsen_depth:
            coords = coords
            num_bins = 15
            bin_size = self.position_range[3] / num_bins
            depth_map = (depth_map / bin_size).int()
            coords3d = torch.cat([coords, depth_map], dim=-1)  # (B, N_view, W, H, 3)   (u, v, d)

            # 归一化
            coords3d[..., 0:1] = coords3d[..., 0:1] / pad_w
            coords3d[..., 1:2] = coords3d[..., 1:2] / pad_h
            coords3d[..., 2:3] = coords3d[..., 2:3] / num_bins
        else:
            coords3d = torch.cat([coords, depth_map], dim=-1)     # (B, N_view, W, H, 3)   (u, v, d)

            # 归一化
            coords3d[..., 0:1] = coords3d[..., 0:1] / pad_w
            coords3d[..., 1:2] = coords3d[..., 1:2] / pad_h
            coords3d[..., 2:3] = coords3d[..., 2:3] / self.position_range[3]

        # (B, N_view, W, H, 3) --> (B, N_view, 3, H, W) --> (B*N_view, 3, H*W)
        coords3d = coords3d.permute(0, 1, 4, 3, 2).contiguous().view(B*N, 3, H*W)
        # (B*N_view, 3, H*W) --> (B*N_view, C=embed_dims, H*W)
        coords_position_embeding = self.pe_encoder_3d_1(coords3d)
        # (B*N_view, C=embed_dims, H*W) --> (B*N_view, H*W, C) -->  (B, N_view, H, W, embed_dims)
        coords_position_embeding = coords_position_embeding.permute(0, 2, 1).contiguous().view(B, N, H, W, -1)

        return coords_position_embeding

    def cross_query_embedding(self, reference_points, img_metas):
        """
        Args:
            reference_points: (B, N_query, 3)
            img_metas:

        Returns:
            cross_query_pe: (B, N_view, N_q, C)
            on_the_image: (B*N_view, N_query)

        """
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]

        lidar2imgs = []
        for img_meta in img_metas:
            lidar2img = []
            for i in range(len(img_meta['lidar2img'])):
                lidar2img.append(img_meta['lidar2img'][i])
            lidar2imgs.append(np.asarray(lidar2img))
        lidar2img = np.asarray(lidar2imgs)
        lidar2img = reference_points.new_tensor(lidar2img)      # (B, N_view, 4, 4)

        bs, n_view = lidar2img.shape[0], lidar2img.shape[1]
        n_query = reference_points.shape[1]

        reference_points_realmetric = torch.zeros_like(reference_points)    # (B, N_query, 3)
        reference_points_realmetric[:, :, 0] = reference_points[:, :, 0] * (self.pc_range[3] - self.pc_range[0]) + \
                                               self.pc_range[0]
        reference_points_realmetric[:, :, 1] = reference_points[:, :, 1] * (self.pc_range[4] - self.pc_range[1]) + \
                                               self.pc_range[1]
        reference_points_realmetric[:, :, 2] = reference_points[:, :, 2] * (self.pc_range[5] - self.pc_range[2]) + \
                                               self.pc_range[2]

        points_3d = torch.cat([reference_points_realmetric, torch.ones_like(reference_points_realmetric[..., :1])], dim=-1)   # (B, N_query, 4)  4: (x, y, z, 1)
        points_3d = points_3d.unsqueeze(dim=1).repeat([1, n_view, 1, 1]).view(-1, n_query, 4)    # (B*N_view, N_query, 4)
        lidar2img = lidar2img.view(bs*n_view, 4, 4)     # (B*N_view, 4, 4)

        # (B*N_view, 4, 4) --> (B*N_view, N_query, 4, 4)
        lidar2img = lidar2img.unsqueeze(dim=1).repeat([1, n_query, 1, 1])
        # (B*N_view, N_query, 4, 4) @ # (B*N_view, N_query, 4, 1)--> (B*N_view, N_query, 4)   4: (du, dv, d, 1)
        pts_proj = torch.matmul(lidar2img, points_3d.unsqueeze(dim=-1)).squeeze(dim=-1)

        eps = 1e-5
        pts_depth = pts_proj[..., 2]       # (B*N_view, N_query)
        on_the_image = pts_depth > eps     # (B*N_view, N_query)
        pts_2d = pts_proj[..., :2] / torch.maximum(pts_proj[..., 2:3],
                                                   torch.ones_like(pts_proj[..., 2:3]) * eps)  # (B*N_view, N_query, 2)

        # 判断该参考点的投影点是否落在该图像上
        on_the_image = on_the_image & (pts_2d[..., 0] >= 0) & (pts_2d[..., 0] < pad_w) & (pts_2d[..., 1] >= 0) & (
                    pts_2d[..., 1] < pad_h)   # (B*N_view, N_query)

        if self.coarsen_depth:
            pts_2d = pts_2d
            num_bins = 15
            bin_size = self.position_range[3] / num_bins
            pts_depth = (pts_depth / bin_size).int()
            pts_3d = torch.cat([pts_2d, pts_depth.unsqueeze(dim=-1)],
                               dim=-1)  # (B*N_view, N_query, 3)       # (u, v, d)

            pts_3d[..., 0:1] = pts_3d[..., 0:1] / pad_w
            pts_3d[..., 1:2] = pts_3d[..., 1:2] / pad_h
            pts_3d[..., 2:3] = pts_3d[..., 2:3] / num_bins
        else:
            pts_3d = torch.cat([pts_2d, pts_depth.unsqueeze(dim=-1)], dim=-1)   # (B*N_view, N_query, 3)   (u, v, d)
            # 归一化
            pts_3d[..., 0:1] = pts_3d[..., 0:1] / pad_w
            pts_3d[..., 1:2] = pts_3d[..., 1:2] / pad_h
            pts_3d[..., 2:3] = pts_3d[..., 2:3] / self.position_range[3]

        # 这里可以任意设置, 因为后续利用on_the_image来设置attention_mask, 直接消除无效View的影响.
        pts_3d[~on_the_image] = -1      # (B*N_view, N_query, 3)

        # (B*N_view, N_query, 3) --> (B*N_view, 3, N_query)
        pts_3d = pts_3d.permute(0, 2, 1).contiguous()
        # (B*N_view, 3, N_query) --> (B*N_view, C, N_query)
        pts_3d_pe = self.pe_encoder_3d_2(pts_3d)
        # (B*N_view, C, N_query) --> (B*N_view, N_q, C) --> (B, N_view, N_q, C)
        cross_query_pe = pts_3d_pe.transpose(1, 2).view(bs, n_view, n_query, self.embed_dims)

        return cross_query_pe, on_the_image

    def integral(self, depth_pred):
        """
        Args:
            depth_pred: (N, D)
        Returns:
            depth_val: (N, )
        """
        depth_score = F.softmax(depth_pred, dim=-1)     # (N, D)
        depth_val = F.linear(depth_score, self.project.type_as(depth_score))  # (N, D) * (D, )  --> (N, )
        return depth_val

    def forward(self, mlvl_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N_view, C, H, W).    # List[(B, N_view, C'=256, H'/16, W'/16), (B, N_view, C'=256, H'/32, W'/32), ]
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        x = mlvl_feats[0]  # (B, N_view, C, H, W)  只选择一个level的图像特征.
        batch_size, num_cams, fH, fW = x.size(0), x.size(1), x.size(3), x.size(4)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        # 建立masks，图像中pad的部分为1, 用于在attention过程中消除pad部分的影响.
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))  # (B, N_view, img_H, img_W)

        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0

        x = x.flatten(0, 1)  # (B*N_view, C, H, W)
        if self.with_depth_supervision:
            # 获得相机内外参
            intrinsics_list = []
            extrinsics_list = []
            for batch_id in range(len(img_metas)):
                cur_intrinsics = img_metas[batch_id]['intrinsics']  # List[(4, 4), (4, 4), ...]
                cur_extrinsics = img_metas[batch_id]['extrinsics']  # List[(4, 4), (4, 4), ...]
                cur_intrinsics = x.new_tensor(cur_intrinsics)  # (N_view, 4, 4)
                cur_extrinsics = x.new_tensor(cur_extrinsics)  # (N_view, 4, 4)
                intrinsics_list.append(cur_intrinsics)
                extrinsics_list.append(cur_extrinsics)
            intrinsics = torch.stack(intrinsics_list, dim=0)[..., :3, :3].contiguous()  # (B, N_view, 3, 3)
            extrinsics = torch.stack(extrinsics_list, dim=0).contiguous()  # (B, N_view, 4, 4)

            # (B*N_view, D, H, W), (B*N_view, C, H, W), (B*N_view, 1, H, W)
            if self.with_pgd:
                depth, x, depth_direct = self.depth_net(x, intrinsics, extrinsics)
            else:
                # (B * N_view, D/1, H, W),  (B*N_view, C, H, W)
                depth, x = self.depth_net(x, intrinsics, extrinsics)
            # for vis
            # for j in range(depth.shape[0]):
            #     cur_depth_score = depth_score[j]  # (D, fH, fW)
            #     max_depth = torch.argmax(cur_depth_score, dim=0)  # (fH, fW)
            #     max_depth = max_depth.detach().cpu().numpy()
            #     max_depth = max_depth * 255 / 63
            #     max_depth = max_depth.astype(np.uint8)
            #     depth_score_map = cv2.applyColorMap(max_depth, cv2.COLORMAP_RAINBOW)
            #     cv2.imshow("score_map", depth_score_map)
            #     cv2.waitKey(0)

            if self.use_prob_depth:
                self.depth_score = depth  # 未经过softmax
                depth_prob = depth.permute(0, 2, 3, 1).contiguous().view(-1,
                                                                         self.depth_num)  # (B*N_view, H, W, D) --> (B*N_view*H*W, D)
                depth_prob_val = self.integral(depth_prob)  # (B*N_view*H*W, )
                depth_map_pred = depth_prob_val

                if self.with_pgd:
                    sig_alpha = torch.sigmoid(self.depth_net.fuse_lambda)
                    depth_direct_val = depth_direct.exp().view(-1)  # (B*N_view*H*W, )
                    depth_pgd_fuse = sig_alpha * depth_direct_val + (1 - sig_alpha) * depth_prob_val
                    depth_map_pred = depth_pgd_fuse
            else:
                # direct depth
                depth_map_pred = depth.exp().view(-1)  # (B*N_view*H*W, )
        else:
            depth_map_pred = None

        # (B*N_view, C, H, W) --> (B*N_view, C'=embed_dim, H, W)
        x = self.input_proj(x)
        x = x.view(batch_size, num_cams, *x.shape[-3:])  # (B, N_view, C'=embed_dim, H, W)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)  # (B, N_view, H, W)

        if self.use_detach:
            depth_map = depth_map_pred.detach()
        else:
            depth_map = depth_map_pred

        depth_map = depth_map.view(batch_size, num_cams, fH, fW)
        # 额, 但是这里没有使用coords_mask.
        # 3D PE: (B, N_view, H, W, C=embed_dims)
        coords_position_embeding = self.position_embeding(mlvl_feats, img_metas, masks, depth_map)
        pos_embed = coords_position_embeding

        reference_points = self.reference_points.weight
        # 3D anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
        # (N_query, 3) --> (N_query, num_feats*3=embed_dims*3/2) --> (N_query, embed_dims)
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))

        # (1, N_query, 3) --> (B, N_query, 3)
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)

        # (B, N_view, N_q, C),  (B * N_view, N_query)
        cross_query_pe, on_the_image = self.cross_query_embedding(reference_points, img_metas, )

        # key为image feature, key_pos为生成的3D PE(+ 2D PE 、multi-view prior)
        # key+key_pos 即对应3D position-aware的特征.
        # import time
        # torch.cuda.synchronize()
        # time1 = time.time()
        outs_dec, _ = self.transformer(x,  # (B, N_view, embed_dim, H, W)
                                       masks,  # (B, N_view, H, W)
                                       query_embeds,  # (N_query, embed_dims)
                                       pos_embed,  # (B, N_view, H, W, C=embed_dims)
                                       cross_query_pe,  # (B, N_view, N_q, C)
                                       on_the_image,    # (B*N_view, N_query)
                                       )
        outs_dec = torch.nan_to_num(outs_dec)  # (num_layers, B, N_query, C=embed_dims)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("time = %f ms" % ((time2 - time1) * 1000))

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())  # (B, N_query, 3)
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])  # (B, N_query, n_cls)
            # (B, N_query, code_size)     code_size: (tx, ty, log(dx), log(dy), tz, log(dz), sin(rot), cos(rot), vx, vy)
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()  # (normalized_cx, normalized_cy)
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()  # normalized_cz

            # (B, N_query, code_size)  code_size: (normalized_cx, normalized_cy, log(dx), log(dy), normalized_cz, log(dz), sin(rot), cos(rot), vx, vy)
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)  # (num_layers, B, N_query, n_cls)
        all_bbox_preds = torch.stack(outputs_coords)  # (num_layers, B, N_query, code_size)

        # (B, N_query, code_size)  code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        if self.depth_net is not None:
            outs['depth_map_pred'] = depth_map_pred    # (B*N_view*H*W, )
            if self.use_prob_depth:
                outs['depth_prob'] = depth_prob     # (B*N_view*H*W, D)
                if self.with_pgd:
                    outs['depth_prob_val'] = depth_prob_val  # (B*N_view*H*W, )
                    outs['depth_direct_val'] = depth_direct_val  # (B*N_view*H*W, )
        return outs


# -------------------------- 利用 GT depth map 将pixel映射到3D point， 然后生成3D PE --------------------------------------
@HEADS.register_module()
class PETRDepthGTHead(PETRHead):
    def __init__(self, add_noise, std=1.0, **kwargs):
        self.add_noise = add_noise
        self.std = std
        super(PETRDepthGTHead, self).__init__(**kwargs)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # 因为没有box_refine, 共享权重.
        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        position_encoder = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        if self.with_position:
            # self.position_dim = 3 * self.depth_num      # D*3 3:(x, y, z)
            self.position_encoder = position_encoder

        # 在3D空间初始化一组0-1之间均匀分布的learnable anchor points.
        self.reference_points = nn.Embedding(self.num_query, 3)
        # anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
        self.query_embedding = position_encoder

    def position_embeding(self, img_feats, img_metas, masks=None, depth_map=None, depth_map_mask=None):
        """
        Args:
            img_feats: List[(B, N_view, C, H, W), ]
            img_metas:
            masks: (B, N_view, H, W)
            depth_map: (B, N_view, H, W)
            depth_map_mask: (B, N_view, H, W)
        Returns:
            coords_position_embeding: (B, N_view, embed_dims, H, W)
            coords_mask: (B, N_view, H, W)
        """
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, C, H, W = img_feats[self.position_level].shape
        # 映射到原图尺度上，得到对应的像素坐标.
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H  # (H, )
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W  # (W, )

        # (2, W, H)  --> (W, H, 2)    2: (u, v)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(1, 2, 0).contiguous()
        coords = coords.view(1, 1, W, H, 2).repeat(B, N, 1, 1, 1)       # (B, N_view, W, H, 2)

        if self.add_noise:
            noise = torch.normal(mean=0.0, std=self.std, size=depth_map.size(), device=depth_map.device)
            depth_map += noise

        depth_map[~depth_map_mask] = self.position_range[3]
        depth_map = depth_map.permute(0, 1, 3, 2).contiguous()
        depth_map[depth_map > self.position_range[3]] = self.position_range[3]

        coords = torch.cat((coords, depth_map.unsqueeze(dim=-1)), dim=-1)       # (B, N_view, W, H, 3)   3:(u, v, d)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)    # (B, N_view, W, H, 4)    4: (u, v, d, 1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(
            coords[..., 2:3]) * eps)  # (B, N_view, W, H, 4)    4: (du, dv, d, 1)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N_view, 4, 4)

        coords = coords.unsqueeze(dim=-1)       # (B, N_view, W, H, 4, 1)
        # (B, N_view, 1, 1, 4, 4) --> (B, N_view, W, H, 4, 4)
        img2lidars = img2lidars.view(B, N, 1, 1, 4, 4).repeat(1, 1, W, H, 1, 1)

        # 图像中每个像素对应的frustum points，借助img2lidars投影到lidar系中.
        # (B, N_view, W, H, 4, 4) @ (B, N_view, H, D, 4, 1) --> (B, N_view, W, H, 4, 1)
        # --> (B, N_view, W, H, 3)   3: (x, y, z)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        # 借助position_range，对3D坐标进行归一化.
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
                    self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
                    self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
                    self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)  # (B, N_view, W, H, 3), 超出range的points mask
        coords_mask = coords_mask.sum(dim=-1) > 0       # (B, N_view, W, H)
        # 在后续attention过程中， 会消除这些像素的影响.
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)  # (B, N_view, H, W)

        coords3d = coords3d.permute(0, 1, 3, 2, 4).contiguous().view(B*N, H, W, 3)      # (B*N_view, H, W, 3)
        coords3d = inverse_sigmoid(coords3d)    # (B*N_view, H, W, 3)
        # 3D position embedding(PE)
        coords_position_embeding = self.position_encoder(pos2posemb3d(coords3d))  # (B*N_view, H, W, embed_dims)
        coords_position_embeding = coords_position_embeding.permute(0, 3, 1, 2).contiguous()    # (B*N_view, embed_dims, H, W)

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def forward(self, mlvl_feats, img_metas, depth_map, depth_map_mask):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N_view, C, H, W).    # List[(B, N_view, C'=256, H'/16, W'/16), (B, N_view, C'=256, H'/32, W'/32), ]
            depth_map: (B*N_view, fH, fW)
            depth_map_mask: (B*N_view, fH, fW)
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        x = mlvl_feats[0]  # (B, N_view, C, H, W)  只选择一个level的图像特征.
        batch_size, num_cams, fH, fW = x.size(0), x.size(1), x.size(3), x.size(4)
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        # 建立masks，图像中pad的部分为1, 用于在attention过程中消除pad部分的影响.
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))  # (B, N_view, img_H, img_W)

        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0

        x = x.flatten(0, 1)  # (B*N_view, C, H, W)

        # (B*N_view, C, H, W) --> (B*N_view, C'=embed_dim, H, W)
        x = self.input_proj(x)
        x = x.view(batch_size, num_cams, *x.shape[-3:])  # (B, N_view, C'=embed_dim, H, W)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)  # (B, N_view, H, W)

        depth_map = depth_map.view(batch_size, num_cams, fH, fW)     # (B, N_view, fH, fW)
        depth_map_mask = depth_map_mask.view(batch_size, num_cams, fH, fW)      # (B, N_view, fH, fW)
        if self.with_position:
            # 额, 但是这里没有使用coords_mask.
            # 3D PE: (B, N_view, embed_dims, H, W)
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks, depth_map, depth_map_mask)

            pos_embed = coords_position_embeding
            if self.with_multiview:
                # 加入 2D PE 和 multi-view prior
                sin_embed = self.positional_encoding(masks)  # (B, N_view, num_feats*3=embed_dims*3/2, H, W)
                # (B, N_view, num_feats*3=embed_dims*3/2, H, W) --> (B*N_view, num_feats*3=embed_dims*3/2, H, W)
                # --> (B*N_view, embed_dims, H, W) --> (B, N_view, embed_dims, H, W)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed  # (B, N_view, embed_dims, H, W)
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        reference_points = self.reference_points.weight
        # 3D anchor points先生成位置编码，然后利用query_embedding生成初始的object queries.
        # (N_query, 3) --> (N_query, num_feats*3=embed_dims*3/2) --> (N_query, embed_dims)
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))

        # (1, N_query, 3) --> (B, N_query, 3)
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)

        # key为image feature, key_pos为生成的3D PE(+ 2D PE 、multi-view prior)
        # key+key_pos 即对应3D position-aware的特征.
        # import time
        # torch.cuda.synchronize()
        # time1 = time.time()
        outs_dec, _ = self.transformer(x,  # (B, N_view, embed_dim, H, W)
                                       masks,  # (B, N_view, H, W)
                                       query_embeds,  # (N_query, embed_dims)
                                       pos_embed,  # (B, N_view, embed_dims, H, W)
                                       self.reg_branches  # 没有进行box_refine, 因此没有用到reg_branches.
                                       )
        outs_dec = torch.nan_to_num(outs_dec)  # (num_layers, B, N_query, C=embed_dims)
        # torch.cuda.synchronize()
        # time2 = time.time()
        # print("time = %f ms" % ((time2 - time1) * 1000))

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())  # (B, N_query, 3)
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])  # (B, N_query, n_cls)
            # (B, N_query, code_size)     code_size: (tx, ty, log(dx), log(dy), tz, log(dz), sin(rot), cos(rot), vx, vy)
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()  # (normalized_cx, normalized_cy)
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()  # normalized_cz

            # (B, N_query, code_size)  code_size: (normalized_cx, normalized_cy, log(dx), log(dy), normalized_cz, log(dz), sin(rot), cos(rot), vx, vy)
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)  # (num_layers, B, N_query, n_cls)
        all_bbox_preds = torch.stack(outputs_coords)  # (num_layers, B, N_query, code_size)

        # (B, N_query, code_size)  code_size: (cx, cy, log(dx), log(dy), cz, log(dz), sin(rot), cos(rot), vx, vy)
        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs


