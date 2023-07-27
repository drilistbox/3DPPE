from __future__ import division

import argparse
import copy
import mmcv
import os
import time

import numpy as np
import torch
from torch.utils import data
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp
import random

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmdet.datasets.builder import build_dataloader
from mmseg import __version__ as mmseg_version
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

import cv2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True #for accelerating the running

setup_seed(667)

cfg_file = "../projects/configs/petr_depth/petr_depth_topk_UD_r50dcn_wogridmask_p4.py"
cfg = Config.fromfile(cfg_file)
# import modules from plguin/xx, registry will be updated
if hasattr(cfg, 'plugin'):
    if cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)
        else:
            # import dir is the dirpath for the config file
            _module_dir = os.path.dirname(cfg_file)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            plg_lib = importlib.import_module(_module_path)


# datasets = build_dataset(cfg.data.train)
# dataloader = build_dataloader(dataset=datasets, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1, dist=False, seed=1024)


# cfg.data.val.ann_file = cfg.data.train.ann_file       # 使用train set的数据
datasets = build_dataset(cfg.data.val)
dataloader = build_dataloader(dataset=datasets, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1, dist=False, seed=1024,
                              shuffle=False)


model = build_model(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
model.init_weights()

cfg.load_from ="/home/zichen/Desktop/ckpts/epoch_2.pth"
load_checkpoint(model, cfg.load_from)
model = MMDataParallel(
            model.cuda(0), device_ids=[0])


import matplotlib.pyplot as plt
import numpy as np


def onMouse(event, x, y, flags, param):
    depth_score, imgs_for_show = param
    if event == cv2.EVENT_LBUTTONDOWN:
        px, py = x // 16, y // 16
        depth_score = depth_score[:, py, px].detach().cpu().numpy()
        x = np.arange(0, 64)

        fig, ax = plt.subplots()
        ax.bar(x, depth_score, edgecolor="white", linewidth=0.7, width=1)
        ax.set(xlim=(0, 64), xticks=np.arange(0, 64),
               ylim=(0, 1), yticks=np.linspace(0, 1, 9))

        plt.show()


model.eval()
for i, batch_data in enumerate(dataloader):
    # if i % 100 == 0:
    img_norm_cfg = batch_data['img_metas'][0].data[0][0]['img_norm_cfg']
    mean = img_norm_cfg['mean']
    std = img_norm_cfg['std']
    imgs = batch_data['img'][0].data[0][0].clone()

    imgs = imgs.numpy().transpose(0, 2, 3, 1)
    imgs[..., 0] = imgs[..., 0] * std[0] + mean[0]
    imgs[..., 1] = imgs[..., 1] * std[1] + mean[1]
    imgs[..., 2] = imgs[..., 2] * std[2] + mean[2]
    imgs_for_show = imgs.astype(np.uint8)
    print(imgs_for_show.shape)

    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **batch_data)
    depth_score = model.module.pts_bbox_head.depth_score

    for j in range(imgs_for_show.shape[0]):
        windowName = f'img{j}'
        cv2.namedWindow(windowName, 0)
        cur_img = imgs_for_show[j]
        cv2.setMouseCallback(windowName, onMouse, (depth_score[j], cur_img))

        cur_depth_score = depth_score[j]  # (D, fH, fW)
        max_depth = torch.argmax(cur_depth_score, dim=0)  # (fH, fW)
        max_depth = max_depth.detach().cpu().numpy()
        max_depth = max_depth * 255 / 63
        max_depth = max_depth.astype(np.uint8)
        depth_score_map = cv2.applyColorMap(max_depth, cv2.COLORMAP_RAINBOW)
        cv2.imshow("score_map", depth_score_map)

        depth_score_map = cv2.resize(src=depth_score_map, dsize=(cur_img.shape[1], cur_img.shape[0]))

        dst = 0.5 * cur_img + depth_score_map * 0.5
        dst = dst.astype(np.uint8)
        cv2.imshow(windowName, dst)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()




