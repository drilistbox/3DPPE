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


cfg_file = "../projects/configs/petr/petr_r50dcn_gridmask_c5.py"
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

datasets = build_dataset(cfg.data.val)
dataloader = build_dataloader(dataset=datasets, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1, dist=False, seed=1024,
                              shuffle=False)


model = build_model(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
model.init_weights()
if cfg.load_from:
    load_checkpoint(model, cfg.load_from)
model = MMDataParallel(
            model.cuda(0), device_ids=[0])


results = []
for i, batch_data in enumerate(dataloader):
    # print(batch_data['img_metas'][0]._data[0][0]['filename'])
    # out = model.train_step(batch_data, None)
    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **batch_data)
    print(out)
    # print(out)
    results.extend(out)
    break

# datasets.evaluate(results, metric='mAp')
# pipeline = datasets.pipeline.transforms
# datasets.show(results, ".", pipeline=pipeline)