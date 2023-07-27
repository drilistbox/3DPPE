_base_ = [
    './petr_depth_gtpe_r50dcn_gridmask_c5.py'
]

model = dict(
    type='Petr3D_GTDepth',
    pts_bbox_head=dict(
        type='PETRDepthGTHead',
        add_noise=True,
        std=3.0,
    )
)
