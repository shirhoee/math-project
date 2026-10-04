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

## Inverse 3x3 Matrix (Adjugate Formula)
The inverse of a 3x3 matrix $M$ can be computed without external libraries using the adjugate matrix:
$M^{-1} = \frac{1}{\det(M)} \text{adj}(M)$
where $\text{adj}(M)$ is the transpose of the cofactor matrix.
Implemented in: `mpi_renderer.py: inv3x3`
Tested in: `tests/test_mpi.py`

## Plane-Induced Homography
A 3D plane in the camera frame with normal $n$ at distance $d$ (equation $n^T X = d$) induces a 2D homography $H$ mapping pixels from the first view to the second view:
$H = K (R - \frac{t n^T}{d}) K^{-1}$
For an MPI layer $k$ perpendicular to the optical axis, $n = [0, 0, 1]^T$ and $d = Z_k$.
Implemented in: `mpi_renderer.py: layer_homographies`
Tested in: `tests/test_mpi.py`
