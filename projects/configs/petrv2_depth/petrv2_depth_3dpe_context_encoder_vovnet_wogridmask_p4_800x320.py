_base_ = [
    './petrv2_depth_3dpe_vovnet_wogridmask_p4_800x320.py'
]


model = dict(
    pts_bbox_head=dict(
        depthnet=dict(
            with_context_encoder=True,
        ),
    )
)
