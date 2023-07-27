_base_ = [
    './petr_depth_base_r50dcn_gridmask_c5.py'
]


model = dict(
    use_grid_mask=False,
    pts_bbox_head=dict(
        type='PETRDepthHeadV2',
        LID=False,
        depthnet=dict(
            type='CameraAwareDepthNet',
            in_channels=2048,
            context_channels=256,
            depth_channels=64,
            mid_channels=512,
            with_depth_correction=True
        ),
        use_detach=False,
        loss_depth=dict(_delete_=True, type='SmoothL1Loss', beta=1.0 / 9.0, reduction='mean', loss_weight=1.0),
    ))