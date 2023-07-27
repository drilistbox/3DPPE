_base_ = [
    './petr_depth_base_r50dcn_gridmask_p4_704x256.py'
]

model = dict(
    use_grid_mask=False,
    pts_bbox_head=dict(
        LID=False,
        depthnet=dict(
            type='CameraAwareDepthNet',
            in_channels=256,
            context_channels=256,
            depth_channels=64,
            mid_channels=256,
            with_depth_correction=True,
        ),
        with_filter=True,
        num_keep=5,
    )
)
