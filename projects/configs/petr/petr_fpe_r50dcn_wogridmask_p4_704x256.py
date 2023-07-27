_base_ = [
    './petr_r50dcn_gridmask_p4_704x256.py'
]

model = dict(
    use_grid_mask=False,
    pts_bbox_head=dict(
        with_fpe=True,
    )
)