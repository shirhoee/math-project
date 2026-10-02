# Math Derivations

## Disparity to Depth
Z = 1 / (d/z_near + (1-d)/z_far)
Implemented in: `math_engine.py: disparity_to_depth`
Tested in: `tests/test_stage3.py: test_disparity_direction`

## Intrinsics from FOV
fx = (W/2) / tan(FOV/2)
Implemented in: `math_engine.py: __init__`

## Pivot Transform and 4x4 form
P_new = (P - c) * R^T + c + t
Implemented in: `math_engine.py: apply_transform`
Tested in: `tests/test_stage1.py: test_4x4_equivalence`

## Perspective Projection
u = fx * X/Z + cx, v = fy * Y/Z + cy
Implemented in: `math_engine.py: project_to_2d`
Tested in: `tests/test_stage1.py`

## Orthographic Projection and Idempotency
u = fx * X * scale + cx
Implemented in: `math_engine.py: project_orthographic`
Tested in: `tests/test_stage2.py`

## Orthogonality and Gimbal Lock
Rotation matrices are orthogonal: R^T = R^-1.

## Auto-Crop
Inscribed rectangle of projected corners, scaling fx, fy, cx, cy.
Implemented in: `app.py: auto_crop logic`
Tested in: `tools/run_autocrop_check.py`
