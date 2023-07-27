_base_ = [
    './petr_depthv2_base_r50dcn_gridmask_p4_704x256.py'
]


model = dict(
    pts_bbox_head=dict(
        LID=False,
        depthnet=dict(
            type='CameraAwareDepthNet',
            in_channels=256,
            context_channels=256,
            depth_channels=10,
            mid_channels=256,
            with_depth_correction=True
        ),
        depth_num=10,
        use_detach=False,
        loss_depth=dict(_delete_=True, type='SmoothL1Loss', beta=1.0 / 9.0, reduction='mean', loss_weight=1.0),
    ))