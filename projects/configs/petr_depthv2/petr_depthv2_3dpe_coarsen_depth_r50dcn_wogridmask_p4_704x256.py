_base_ = [
    './petr_depthv2_3dpe_r50dcn_wogridmask_p4_704x256.py'
]


model = dict(
    pts_bbox_head=dict(
        coarse_depth=True,
    )
)