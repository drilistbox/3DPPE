_base_ = [
    './petr_depth_base_r50dcn_gridmask_c5.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)


model = dict(
    type='Petr3D_GTDepth',
    pts_bbox_head=dict(
        type='PETRDepthGTHead',
        add_noise=False,
    )
)



dataset_type = 'CustomNuScenesDataset'
data_root = '/datasets_220/nuScenes/v1.0-trainval/'
file_client_args = dict(backend='disk')

ida_aug_conf = {
        "resize_lim": (0.8, 1.0),
        "final_dim": (512, 1408),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='LoadDepthByMapplingPoints2Images',
        src_size=(900, 1600),
        input_size=(512, 1408),
        downsample=32
    ),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            training=True
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'depth_map', 'depth_map_mask'],
                           meta_keys=['filename', 'ori_shape', 'img_shape', 'lidar2img',
                                      'depth2img', 'cam2img', 'pad_shape',
                                      'scale_factor', 'flip', 'pcd_horizontal_flip',
                                      'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                                      'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                      'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                                      'transformation_3d_flow', 'img_info', 'intrinsics',
                                      'extrinsics'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(
        type='LoadDepthByMapplingPoints2Images',
        src_size=(900, 1600),
        input_size=(512, 1408),
        downsample=32
    ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img', 'depth_map', 'depth_map_mask'],
                 meta_keys=['filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'img_info', 'intrinsics',
                            'extrinsics'])
        ])
]

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, classes=class_names, modality=input_modality))


# cmd: bash tools/dist_train.sh projects/configs/petr_depth/petr_depth_gtpe_r50dcn_gridmask_c5.py 4
# Results writes to /tmp/tmpfb3i4wgk/results/pts_bbox/results_nusc.json                                                                                                                                                                         
# Evaluating bboxes of pts_bbox                                                                                                                                                                                                                 
# mAP: 0.4003                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                              
# mATE: 0.6896                                                                                                                                                                                                                                  
# mASE: 0.2766                                                                                                                                                                                                                                  
# mAOE: 0.6068                                                                                                                                                                                                                                  
# mAVE: 0.9809                                                                                                                                                                                                                                  
# mAAE: 0.2509                                                                                                                                                                                                                                  
# NDS: 0.4197                                                                                                                                                                                                                                   
# Eval time: 313.7s                                                                                                                                                                                                                             
                
# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.587   0.466   0.158   0.120   1.012   0.240
# truck   0.349   0.724   0.237   0.179   0.956   0.256
# bus     0.434   0.742   0.224   0.139   2.275   0.412
# trailer 0.168   1.154   0.244   0.674   0.501   0.117
# construction_vehicle    0.134   0.856   0.459   1.136   0.139   0.401
# pedestrian      0.540   0.535   0.303   1.120   0.838   0.335
# motorcycle      0.447   0.575   0.252   0.792   1.592   0.205
# bicycle 0.383   0.521   0.282   1.144   0.534   0.042
# traffic_cone    0.546   0.526   0.322   nan     nan     nan
# barrier 0.415   0.796   0.285   0.157   nan     nan
# 2022-08-12 12:53:42,186 - mmdet - INFO - Exp name: petr_depth_gtpe_r50dcn_gridmask_c5.py2022-08-12 12:53:42,187 - mmdet - INFO - Epoch(val) [24][1505]  pts_bbox_NuScenes/car_AP_dist_0.5: 0.2542, pts_bbox_NuScenes/car_AP_dist_1.0: 0.5172, pts_bbox_NuScenes/car_AP_dist_2.0: 0.7442, pts_bbox_NuScenes/car_AP_dist_4.0: 0.8311, pts_bbox_NuScenes/car_trans_err: 0.4664, pts_bbox_NuScenes/car_scale_err: 0.1577, pts_bbox_NuScenes/car_orient_err: 0.1200, pts_bbox_NuScenes/car_vel_err: 1.0117, pts_bbox_NuScenes/car_attr_err: 0.2398, pts_bbox_NuScenes/mATE: 0.6896, pts_bbox_NuScenes/mASE: 0.2766, pts_bbox_NuScenes/mAOE: 0.6068, pts_bbox_NuScenes/mAVE: 0.9809, pts_bbox_NuScenes/mAAE: 0.2509, pts_bbox_NuScenes/truck_AP_dist_0.5: 0.0390, pts_bbox_NuScenes/truck_AP_dist_1.0: 0.2405, pts_bbox_NuScenes/truck_AP_dist_2.0: 0.4977, pts_bbox_NuScenes/truck_AP_dist_4.0: 0.6192, pts_bbox_NuScenes/truck_trans_err: 0.7242, pts_bbox_NuScenes/truck_scale_err: 0.2374, pts_bbox_NuScenes/truck_orient_err: 0.1794, pts_bbox_NuScenes/truck_vel_err: 0.9564, pts_bbox_NuScenes/truck_attr_err: 0.2559, pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5: 0.0000, pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0: 0.0593, pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0: 0.1764, pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0: 0.3022, pts_bbox_NuScenes/construction_vehicle_trans_err: 0.8564, pts_bbox_NuScenes/construction_vehicle_scale_err: 0.4586, pts_bbox_NuScenes/construction_vehicle_orient_err: 1.1360, pts_bbox_NuScenes/construction_vehicle_vel_err: 0.1386, pts_bbox_NuScenes/construction_vehicle_attr_err: 0.4013, pts_bbox_NuScenes/bus_AP_dist_0.5: 0.0285, pts_bbox_NuScenes/bus_AP_dist_1.0: 0.2883, pts_bbox_NuScenes/bus_AP_dist_2.0: 0.6336, pts_bbox_NuScenes/bus_AP_dist_4.0: 0.7874, pts_bbox_NuScenes/bus_trans_err: 0.7416, pts_bbox_NuScenes/bus_scale_err: 0.2239, pts_bbox_NuScenes/bus_orient_err: 0.1389, pts_bbox_NuScenes/bus_vel_err: 2.2754, pts_bbox_NuScenes/bus_attr_err: 0.4117, pts_bbox_NuScenes/trailer_AP_dist_0.5: 0.0000, pts_bbox_NuScenes/trailer_AP_dist_1.0: 0.0091, pts_bbox_NuScenes/trailer_AP_dist_2.0: 0.2176, pts_bbox_NuScenes/trailer_AP_dist_4.0: 0.4442, pts_bbox_NuScenes/trailer_trans_err: 1.1545, pts_bbox_NuScenes/trailer_scale_err: 0.2440, pts_bbox_NuScenes/trailer_orient_err: 0.6736, pts_bbox_NuScenes/trailer_vel_err: 0.5012, pts_bbox_NuScenes/trailer_attr_err: 0.1169, pts_bbox_NuScenes/barrier_AP_dist_0.5: 0.0332, pts_bbox_NuScenes/barrier_AP_dist_1.0: 0.3034, pts_bbox_NuScenes/barrier_AP_dist_2.0: 0.6249, pts_bbox_NuScenes/barrier_AP_dist_4.0: 0.6995, pts_bbox_NuScenes/barrier_trans_err: 0.7962, pts_bbox_NuScenes/barrier_scale_err: 0.2854, pts_bbox_NuScenes/barrier_orient_err: 0.1571, pts_bbox_NuScenes/barrier_vel_err: nan, pts_bbox_NuScenes/barrier_attr_err: nan, pts_bbox_NuScenes/motorcycle_AP_dist_0.5: 0.1326, pts_bbox_NuScenes/motorcycle_AP_dist_1.0: 0.4239, pts_bbox_NuScenes/motorcycle_AP_dist_2.0: 0.5924, pts_bbox_NuScenes/motorcycle_AP_dist_4.0: 0.6398, pts_bbox_NuScenes/motorcycle_trans_err: 0.5747, pts_bbox_NuScenes/motorcycle_scale_err: 0.2518, pts_bbox_NuScenes/motorcycle_orient_err: 0.7925, pts_bbox_NuScenes/motorcycle_vel_err: 1.5918, pts_bbox_NuScenes/motorcycle_attr_err: 0.2047, pts_bbox_NuScenes/bicycle_AP_dist_0.5: 0.1406, pts_bbox_NuScenes/bicycle_AP_dist_1.0: 0.3788, pts_bbox_NuScenes/bicycle_AP_dist_2.0: 0.4874, pts_bbox_NuScenes/bicycle_AP_dist_4.0: 0.5233, pts_bbox_NuScenes/bicycle_trans_err: 0.5212, pts_bbox_NuScenes/bicycle_scale_err: 0.2818, pts_bbox_NuScenes/bicycle_orient_err: 1.1437, pts_bbox_NuScenes/bicycle_vel_err: 0.5344, pts_bbox_NuScenes/bicycle_attr_err: 0.0422, pts_bbox_NuScenes/pedestrian_AP_dist_0.5: 0.2003, pts_bbox_NuScenes/pedestrian_AP_dist_1.0: 0.5086, pts_bbox_NuScenes/pedestrian_AP_dist_2.0: 0.6890, pts_bbox_NuScenes/pedestrian_AP_dist_4.0: 0.7601, pts_bbox_NuScenes/pedestrian_trans_err: 0.5345, pts_bbox_NuScenes/pedestrian_scale_err: 0.3027, pts_bbox_NuScenes/pedestrian_orient_err: 1.1202, pts_bbox_NuScenes/pedestrian_vel_err: 0.8378, pts_bbox_NuScenes/pedestrian_attr_err: 0.3349, pts_bbox_NuScenes/traffic_cone_AP_dist_0.5: 0.1927, pts_bbox_NuScenes/traffic_cone_AP_dist_1.0: 0.5183, pts_bbox_NuScenes/traffic_cone_AP_dist_2.0: 0.6931, pts_bbox_NuScenes/traffic_cone_AP_dist_4.0: 0.7806, pts_bbox_NuScenes/traffic_cone_trans_err: 0.5263, pts_bbox_NuScenes/traffic_cone_scale_err: 0.3220, pts_bbox_NuScenes/traffic_cone_orient_err: nan, pts_bbox_NuScenes/traffic_cone_vel_err: nan, pts_bbox_NuScenes/traffic_cone_attr_err: nan, pts_bbox_NuScenes/NDS: 0.4197, pts_bbox_NuScenes/mAP: 0.4003
