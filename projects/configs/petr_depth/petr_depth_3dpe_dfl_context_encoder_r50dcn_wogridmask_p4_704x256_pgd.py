_base_ = [
    './petr_depth_3dpe_r50dcn_wogridmask_p4_704x256.py'
]


model = dict(
    pts_bbox_head=dict(
        depthnet=dict(
            with_context_encoder=True,
            with_pgd=True,
            depth_channels=16,
        ),
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        use_dfl=True,
        depth_num=16,
        loss_depth=dict(_delete_=True, type='SmoothL1Loss', beta=1.0 / 9.0, reduction='mean', loss_weight=0.1),
        loss_dfl=dict(type='DistributionFocalLoss', reduction='mean', loss_weight=0.25),
    )
)
