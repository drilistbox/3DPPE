# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from .dgcnn3d_head import DGCNN3DHead
from .detr3d_head import Detr3DHead
from .petr_head import PETRHead
from .petrv2_head import PETRv2Head
from .petr_head_seg import PETRHeadseg
from .petr_hit_head import PETRHitHead
from .petr_depth_head import PETRDepthHead, PETRDepthGTHead, PETRDepthHeadV2, PETRDepthHeadV2_Refine
from .petrv2_depth_head import PETRV2DepthHead, PETRV2DepthHeadV2

__all__ = ['DGCNN3DHead', 'Detr3DHead','PETRHead','PETRv2Head','PETRHeadseg', 'PETRHitHead', 'PETRDepthHead',
           'PETRDepthGTHead', 'PETRDepthHeadV2', 'PETRDepthHeadV2_Refine', 'PETRV2DepthHead', 'PETRV2DepthHeadV2']