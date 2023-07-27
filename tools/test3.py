from __future__ import division

import argparse
import copy

import cv2
import mmcv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
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
import matplotlib.gridspec as mg

from visualizer import get_local
get_local.activate()


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

cfg_file = "../projects/configs/petr/petr_r50dcn_gridmask_p4_704x256.py"
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
# model.init_weights()

cfg.load_from ="/home/zichen/Desktop/ckpts/PETR/PETR/petr_r50dcn_gridmask_p4_704x256/epoch_24.pth"
load_checkpoint(model, cfg.load_from)

model = MMDataParallel(
            model.cuda(0), device_ids=[0])

import matplotlib.pyplot as plt
import numpy as np


# # ------------------------- 计算图像中每个pixel 3D PE的相似度，并显示 ----------------------------------
# def onMouse(event, x, y, flags, param):
#     view_id, pe, imgs_for_show = param     # pe: (N_view=6, C, H, W);  imgs_for_show: (N_view, H, W, 3)
#     n_view, embed_dims, fH, fW = pe.shape
#     img_H, img_W = imgs_for_show.shape[1], imgs_for_show.shape[2]
#     down_sample = 16
#     if event == cv2.EVENT_LBUTTONDOWN:
#         px, py = x // down_sample, y // down_sample
#         chosen_pe = pe[view_id:view_id+1, :, py, px]      # (1, C)
#         total_pe = pe.permute(0, 2, 3, 1).contiguous().view(-1, embed_dims)     # (N_view*fH*fW, C)
#         similarity = F.cosine_similarity(chosen_pe, total_pe, dim=-1)       # (N_view*fH*fW, )
#         similarity = similarity.view(n_view, fH, fW)    # (N_view, fH, fW)
#
#         min_v = similarity.min()
#         max_v = similarity.max()
#         similarity = (similarity - min_v) / (max_v - min_v)
#         similarity = similarity ** 10
#         # min_v = similarity.min()
#         # max_v = similarity.max()
#         # similarity = (similarity - min_v) / (max_v - min_v)
#
#         # way1
#         # fig = plt.figure(figsize=(100, 300))
#         # for view_idx in range(n_view):
#         #     # cur_similarity = similarity[view_idx]   # (fH, fW)
#         #     # ax = fig.add_subplot(2, 3, view_idx+1)
#         #     # ax.imshow(cur_similarity, cmap='summer')
#         #
#         #     cur_similarity = similarity[view_idx]   # (fH, fW)
#         #     ax = fig.add_subplot(2, 3, view_idx+1)
#         #     cur_similarity = cur_similarity.cpu().numpy()
#         #     cur_similarity = cur_similarity * 255
#         #     cur_similarity = cur_similarity.astype(np.uint8)
#         #     cur_similarity = cv2.applyColorMap(cur_similarity, colormap=cv2.COLORMAP_SUMMER)
#         #     cur_similarity = cur_similarity[..., [2, 1, 0]]
#         #     ax.imshow(cur_similarity)
#         #
#         # plt.show()
#
#         # way2
#         fig = plt.figure(figsize=(100, 300))
#         for view_idx in range(n_view):
#             cur_similarity = similarity[view_idx]  # (fH, fW)
#             cur_img = imgs_for_show[view_idx]   # (H, W, 3)  3:(b, g, r)
#
#             cur_similarity = cur_similarity.cpu().numpy()
#             cur_similarity = cur_similarity * 255
#             cur_similarity = cur_similarity.astype(np.uint8)
#             cur_similarity = cv2.applyColorMap(cur_similarity, colormap=cv2.COLORMAP_SUMMER)
#             cur_similarity = cv2.resize(cur_similarity, dsize=(img_W, img_H))
#             cur_img_similarity = cur_img * 0.5 + cur_similarity * 0.5
#             cur_img_similarity = cur_img_similarity.astype(np.uint8)
#             cur_img_similarity = cur_img_similarity[..., [2, 1, 0]]
#             ax = fig.add_subplot(2, 3, view_idx + 1)
#             ax.imshow(cur_img_similarity)
#         plt.show()
#
#
# model.eval()
# for i, batch_data in enumerate(dataloader):
#     # if i % 100 == 0:
#     img_norm_cfg = batch_data['img_metas'][0].data[0][0]['img_norm_cfg']
#     mean = img_norm_cfg['mean']
#     std = img_norm_cfg['std']
#     imgs = batch_data['img'][0].data[0][0].clone()
#
#     imgs = imgs.numpy().transpose(0, 2, 3, 1)
#     imgs[..., 0] = imgs[..., 0] * std[0] + mean[0]
#     imgs[..., 1] = imgs[..., 1] * std[1] + mean[1]
#     imgs[..., 2] = imgs[..., 2] * std[2] + mean[2]
#     imgs_for_show = imgs.astype(np.uint8)       # (N_view, H, W, 3)
#     img_H, img_W = imgs_for_show.shape[1:3]
#
#     with torch.no_grad():
#         out = model(return_loss=False, rescale=True, **batch_data)
#
#     cache = get_local.cache
#     pe = cache['PETRHead.position_embeding'][0]  # (B*N_view=6, embed_dims, H, W)
#     pe = torch.from_numpy(pe)
#     print(pe.sum())
#
#     n_view = 6
#
#     for view_id in range(n_view):
#         windowName = f'img{view_id}'
#         cv2.namedWindow(windowName, 0)
#         cur_img = imgs_for_show[view_id]    # (H, W, 3)
#         cv2.setMouseCallback(windowName, onMouse, (view_id, pe, imgs_for_show))
#
#         cv2.imshow(windowName, cur_img)
#
#     k = cv2.waitKey(0)
#     if k == 27:
#         cv2.destroyAllWindows()


# ----------------------------------- decoder 中 attention weight 可视化 ------------------------------------------------
# def check_point_in_img(points, height, width):
#     """
#     Args:
#         points: (N, 2)
#         height:
#         width:
#     Returns:
#         valid: (N, )
#     """
#     valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
#     valid = np.logical_and(valid,
#                            np.logical_and(points[:, 0] < width,
#                                           points[:, 1] < height))
#     return valid
#
#
# def depth2color(depth):
#     gray = max(0, min((depth + 2.5) / 3.0, 1.0))
#     max_lumi = 200
#     colors = np.array([[max_lumi, 0, max_lumi], [max_lumi, 0, 0],
#                        [max_lumi, max_lumi, 0], [0, max_lumi, 0],
#                        [0, max_lumi, max_lumi], [0, 0, max_lumi]],
#                       dtype=np.float32)
#     if gray == 1:
#         return tuple(colors[-1].tolist())
#     num_rank = len(colors)-1
#     rank = np.floor(gray*num_rank).astype(np.int32)
#     diff = (gray-rank/num_rank)*num_rank
#     return tuple((colors[rank]+(colors[rank+1]-colors[rank])*diff).tolist())
#
#
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
# label2name = {}
# for i, cls_name in enumerate(class_names):
#     label2name[i] = cls_name
#
# model.eval()
# canva_size = 1000
# show_range = 51.2
# draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
# for i, batch_data in enumerate(dataloader):
#     # if i % 100 == 0:
#     sample_token = batch_data['img_metas'][0].data[0][0]['sample_idx']
#     img_norm_cfg = batch_data['img_metas'][0].data[0][0]['img_norm_cfg']
#     mean = img_norm_cfg['mean']
#     std = img_norm_cfg['std']
#     imgs = batch_data['img'][0].data[0][0].clone()
#
#     lidar_path = batch_data['img_metas'][0].data[0][0]['pts_filename']
#     lidar_points = np.fromfile(lidar_path, dtype=np.float32)
#     lidar_points = lidar_points.reshape(-1, 5)[:, :3]  # (npoints, 3)  3: (x, y, z)
#
#     imgs = imgs.numpy().transpose(0, 2, 3, 1)
#     imgs[..., 0] = imgs[..., 0] * std[0] + mean[0]
#     imgs[..., 1] = imgs[..., 1] * std[1] + mean[1]
#     imgs[..., 2] = imgs[..., 2] * std[2] + mean[2]
#     imgs_for_show = imgs.astype(np.uint8)       # (N_view, H, W, 3)
#     img_H, img_W = imgs_for_show.shape[1:3]
#
#     with torch.no_grad():
#         out = model(return_loss=False, rescale=True, **batch_data)
#
#     pts_bbox = out[0]['pts_bbox']
#     boxes_3d = pts_bbox['boxes_3d']     # (Max_num, 9)
#     scores_3d = pts_bbox['scores_3d']   # (Max_num, )
#     labels_3d = pts_bbox['labels_3d']   # (Max_num, )
#     cache = get_local.cache
#     bbox_index = cache['NMSFreeCoder.decode_single'][0]     # (Max_num, )
#
#     thres = 0.2
#     mask = scores_3d > thres
#     boxes_3d = boxes_3d[mask]       # (N_pred, 9)
#     scores_3d = scores_3d[mask]     # (N_pred, )
#     labels_3d = labels_3d[mask]     # (N_pred, )
#     # print(boxes_3d.tensor.shape)
#     # print(scores_3d.shape)
#     # print(labels_3d.shape)
#
#     reference_points = model.module.pts_bbox_head.reference_points.weight
#     valid_reference_points = reference_points[bbox_index][mask]     # (N_pred, 3)
#     query_idx = bbox_index[mask]
#
#     cache = get_local.cache
#     attn_maps = cache['PETRMultiheadAttention.forward']
#     attn_maps = attn_maps[-1][0]    # (Nq, N_k=N_view*H*W)
#
#     canvas = np.ones((int(canva_size), int(canva_size), 3), dtype=np.uint8) * 255  # (H, W, 3)
#     lidar_points = lidar_points.reshape(-1, 5)[:, :3]  # (npoints, 3)  3: (x, y, z)
#     lidar_points[:, 1] = -lidar_points[:, 1]
#     lidar_points[:, :2] = (lidar_points[:, :2] + show_range) / \
#                           show_range / 2.0 * canva_size
#     for p in lidar_points:
#         if check_point_in_img(p.reshape(1, 3),
#                               canvas.shape[1],
#                               canvas.shape[0])[0]:
#             color = depth2color(p[2])
#             cv2.circle(canvas, (int(p[0]), int(p[1])),
#                        radius=0, color=color, thickness=1)
#
#     corners_lidar = boxes_3d.corners.numpy().reshape(-1, 3)       # (N_pred*8, 3)
#     corners_lidar = corners_lidar.reshape(-1, 8, 3)     # (N_pred, 8, 3)
#     corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
#     bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
#     bottom_corners_bev = (bottom_corners_bev + show_range) / \
#                          show_range / 2.0 * canva_size  # (N_pred, 4, 2)
#     bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)  # (N_pred, 4, 2)
#
#     center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)  # (N_pred, 2)
#     head_bev = corners_lidar[:, [4, 7], :2].mean(axis=1)  # (N_pred+, 2)
#     canter_canvas = (center_bev + show_range) / show_range / 2.0 * \
#                     canva_size  # (N_pred, 2)
#     center_canvas = canter_canvas.astype(np.int32)
#     head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size  # (N_pred, 2)
#     head_canvas = head_canvas.astype(np.int32)
#
#     for box_id in range(len(boxes_3d)):
#         for index in draw_boxes_indexes_bev:
#             cv2.line(canvas,
#                      tuple(bottom_corners_bev[box_id, index[0]]),
#                      tuple(bottom_corners_bev[box_id, index[1]]),
#                      (0, 0, 0),
#                      thickness=2
#                      )
#         cv2.line(canvas, tuple(center_canvas[box_id]), tuple(head_canvas[box_id]),
#                  color=(255, 255, 0), thickness=2, lineType=8)
#
#     for i in range(len(query_idx)):
#         cur_label = labels_3d[i]
#         cls_name = label2name[cur_label.item()]
#         print(f"detecting a {cls_name}")
#
#         query_id = query_idx[i]
#         cur_attn_map = attn_maps[query_id, :]      # (N_view*H*W, )
#
#         min = cur_attn_map.min()
#         max = cur_attn_map.max()
#         cur_attn_map = (cur_attn_map - min) / (max - min)
#         cur_attn_map = cur_attn_map ** 0.4
#
#         cur_attn_map = cur_attn_map.reshape((6, 16, 44))
#         cur_attn_map = cur_attn_map
#
#         fig = plt.figure(figsize=(50, 200))
#         gs = mg.GridSpec(2, 5)
#         lidar_ax = fig.add_subplot(gs[:2, :2])
#         canvas = canvas[..., [2, 1, 0]]
#         lidar_ax.imshow(canvas)
#
#         for view_id in range(len(cur_attn_map)):
#             att_map_cur_view = cur_attn_map[view_id]    # (fH, fW)
#             att_map_cur_view = att_map_cur_view * 255
#             att_map_cur_view = att_map_cur_view.astype(np.uint8)
#             att_map_cur_view = cv2.applyColorMap(att_map_cur_view, cv2.COLORMAP_JET)
#             att_map_cur_view = cv2.resize(att_map_cur_view, (img_W, img_H))
#
#             cur_imgs_for_show = imgs_for_show[view_id].astype(np.float)
#             cur_imgs_for_show = cur_imgs_for_show * 0.5 + att_map_cur_view * 0.5
#             cur_imgs_for_show = cur_imgs_for_show.astype(np.uint8)
#
#             row_id = view_id // 3
#             col_id = view_id % 3 + 2
#             ax = fig.add_subplot(gs[row_id, col_id])
#             cur_imgs_for_show = cur_imgs_for_show[..., [2, 1, 0]]
#             # Plot the heatmap
#             ax.imshow(cur_imgs_for_show)
#
#         plt.show()
#
#     break

# -------------------------------------------- 射线编码和点编码可视化 ------------------------------------------------------
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
    imgs_for_show = imgs.astype(np.uint8)       # (N_view, H, W, 3)
    img_H, img_W = imgs_for_show.shape[1:3]

    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **batch_data)

    cv2.imshow("img", imgs_for_show[0])

    # 射线编码
    # coor3d = model.module.pts_bbox_head.coords3d        # (B, N_view, W, H, D, 3)   3: (x, y, z)
    # coor3d = coor3d.permute(0, 1, 3, 2, 4, 5).contiguous()     # (B, N_view, H, W, D, 3)   3: (x, y, z)
    # coor3d = coor3d[0][0].cpu().numpy()   # (H, W, D, 3)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # x1 = coor3d[::4, ::4, 0, 0]      # (H, W)
    # y1 = coor3d[::4, ::4, 0, 1]      # (H, W)
    # z1 = coor3d[::4, ::4, 0, 2]      # (H, W)
    # x2 = coor3d[::4, ::4, 10, 0]     # (H, W)
    # y2 = coor3d[::4, ::4, 10, 1]     # (H, W)
    # z2 = coor3d[::4, ::4, 10, 2]     # (H, W)
    # ax.scatter(x1, y1, z1, c='red', s=1)
    # ax.scatter(x2, y2, z2, c='blue', s=1)
    #
    # fH, fW = x1.shape[:2]
    # for h_id in range(fH):
    #     for w_id in range(fW):
    #         x = [x1[h_id, w_id], x2[h_id, w_id]]
    #         y = [y1[h_id, w_id], y2[h_id, w_id]]
    #         z = [z1[h_id, w_id], z2[h_id, w_id]]
    #         ax.plot(x, y, z, color='green',)
    #
    # plt.show()

    # 点编码
    # depth_map = model.module.pts_bbox_head.depth_map    # (B, N_view, fH, FW)
    # depth_map = depth_map[0][0].cpu().numpy()     # (fH, FW)
    # depth_map = cv2.resize(depth_map, (img_W, img_H))   # (img_H, img_W)
    # depth_map = np.expand_dims(depth_map, axis=-1)      # (img_H, img_W, 1)
    # u, v = np.meshgrid(np.arange(0, img_W, 1), np.arange(0, img_H, 1))     # (img_H, img_W)
    # pts2d = np.stack([u, v], axis=-1)   # (img_H, img_W, 2)  2: (u, v)
    # pts2d = pts2d * np.maximum(depth_map, np.ones(depth_map.shape) * 1e-5)      # 2: (du, dv)
    # pts3d_cam = np.concatenate([pts2d, depth_map], axis=-1)     # 3: (du, dv, d)
    # pts3d_cam = np.concatenate([pts3d_cam, np.ones(depth_map.shape)], axis=-1)  # (img_H, img_W, 4) 4: (du, dv, d, 1)
    #
    # img2lidars = model.module.pts_bbox_head.img2lidars  # (B, N_view, 4, 4)
    # img2lidars = img2lidars[0][0].cpu().numpy()   # (4, 4)
    # img2lidars = img2lidars[None, None, ...]    # (1, 1, 4, 4)
    # img2lidars = np.tile(img2lidars, (img_H, img_W, 1, 1))
    #
    # coord3d = np.matmul(img2lidars, np.expand_dims(pts3d_cam, axis=-1))
    # coord3d = np.squeeze(coord3d, axis=-1)[..., :3]     # (img_H, img_W, 3)
    # print(coord3d.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(coord3d[..., 0], coord3d[..., 1], coord3d[..., 2], c='red', s=1)
    # plt.show()

    # import open3d
    # vis = open3d.visualization.Visualizer()
    # vis.create_window("nus")
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(coord3d.reshape((-1, 3)))
    # print("!!!")
    # vis.add_geometry(pcd)
    # vis.run()
    # vis.destroy_window()
    # break


    # 点编码
    coord_2d = model.module.pts_bbox_head.coords2d    # (B, N_view, fW, fH, 2)
    coord_3d = model.module.pts_bbox_head.coords3d    # (B, N_view, fW, fH, 3)   3: (x, y, z)
    coord_2d = coord_2d[0][0].permute(1, 0, 2).contiguous()      # (fH, fW, 2)
    coord_3d = coord_3d[0][0].permute(1, 0, 2).contiguous()      # (fH, fW, 3)
    coord_2d = coord_2d.view(-1, 2).cpu().numpy()
    coord_3d = coord_3d.view(-1, 3).cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(coord_3d[..., 0], coord_3d[..., 1], coord_3d[..., 2], c='red', s=1)
    plt.show()
    break