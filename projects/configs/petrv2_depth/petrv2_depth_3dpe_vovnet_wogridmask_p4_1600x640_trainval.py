_base_ = [
    './petrv2_depth_base_vovnet_gridmask_p4_1600x640_trainval.py'
]


model = dict(
    use_grid_mask=False,
    pts_bbox_head=dict(
        type='PETRV2DepthHeadV2',
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
        loss_depth=dict(_delete_=True, type='SmoothL1Loss', beta=1.0 / 9.0, reduction='mean', loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', reduction='mean', loss_weight=0.25),
    )
)