_base_ = [
    './petr_depth_base_r50dcn_gridmask_p4_704x256.py'
]


model = dict(
    use_grid_mask=False,
    pts_bbox_head=dict(
        type='PETRDepthHeadV2',
        LID=False,
        depthnet=dict(
            type='CameraAwareDepthNet',
            in_channels=256,
            context_channels=256,
            depth_channels=64,
            mid_channels=256,
            with_depth_correction=True
        ),
        use_dfl=False,
        use_detach=False,
        use_prob_depth=True,
        use_balancer=True,
        balancer_cfg=dict(fg_weight=5.0, bg_weight=1.0, downsample_factor=16),
        loss_depth=dict(_delete_=True, type='SmoothL1Loss', beta=1.0 / 9.0, reduction='none', loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', reduction='none', loss_weight=0.25),
    )
)