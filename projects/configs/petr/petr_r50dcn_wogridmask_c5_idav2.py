_base_ = [
    'petr_r50dcn_gridmask_c5_idav2.py'
]

model = dict(
    use_grid_mask=False
)