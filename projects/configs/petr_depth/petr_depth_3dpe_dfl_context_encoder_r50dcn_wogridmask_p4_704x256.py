_base_ = [
    './petr_depth_3dpe_r50dcn_wogridmask_p4_704x256.py'
]


model = dict(
    pts_bbox_head=dict(
        depthnet=dict(
            with_context_encoder=True,
        ),
        use_dfl=True,
    )
)
