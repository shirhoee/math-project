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
The prompt's convention for the camera transform is $P' = R P + b$.
If the camera orbits around a center $c$ with an extra translation $t_{extra}$, then $b = c - R c + t_{extra}$.
The homography induced by a plane with normal $e_z = [0, 0, 1]^T$ at depth $d = Z_k$ under this transform is:
$H_d = K ( R + \frac{b e_z^T}{d} ) K^{-1}$

The report quotes $H = K ( R - \frac{t n^T}{d} ) K^{-1}$.
In our code, we define $n = -e_z = [0, 0, -1]^T$ and $t = b = c - R c + t_{extra}$.
Substituting these into the report's formula gives:
$H = K ( R - \frac{b (-e_z^T)}{d} ) K^{-1} = K ( R + \frac{b e_z^T}{d} ) K^{-1} = H_d$
which exactly matches the prompt's convention in one line.

Implemented in: `mpi_renderer.py: layer_homographies`
Tested in: `tests/test_mpi_stage4.py` (and against the point-based engine).
