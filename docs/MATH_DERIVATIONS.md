# Mathematical Derivations

This document outlines the core linear algebra formulas used in the custom 3D graphics engine.

## 1. Disparity to Depth Conversion
Depth Anything V2 outputs a disparity map $d$ (normalized to $[0, 1]$ where 1 is nearest). We convert this to physical depth $Z$ using a linear mapping:
$$ Z = z_{near} + (1 - d) \cdot (z_{far} - z_{near}) $$
This ensures $Z \ge z_{near} > 0$, preventing points from collapsing to the origin and preventing division by zero during projection.

## 2. Intrinsic Camera Matrix ($K$)
The intrinsic matrix maps 3D camera coordinates to 2D image pixels. Given a Field of View (FOV), we calculate the focal length:
$$ f_x = f_y = \frac{W / 2}{\tan(\text{FOV} / 2)} $$
With the principal point at $c_x = (W-1)/2, c_y = (H-1)/2$:
$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$
The inverse $K^{-1}$ is used to unproject 2D rays into 3D.

## 3. Pivot-Based Transform vs Camera Center
Rotating about the camera center ($P_{new} = P \cdot R^T + t$) results in a homography (an affine warp of the 2D image), yielding no parallax because the rays do not shift relative to one another. To reveal 3D structure, we pivot around the scene's median depth $c = (0, 0, Z_{pivot})$:
$$ P_{new} = (P - c) \cdot R^T + c + t $$

### Homogeneous Form
This is equivalent to the $4 \times 4$ homogeneous matrix composition:
$$ T_{total} = T(c+t) \cdot R_4 \cdot T(-c) $$

## 4. Perspective Projection
We multiply the rotated 3D points by the intrinsic matrix $K$:
$$ \begin{bmatrix} u' \\ v' \\ w \end{bmatrix} = K \cdot \begin{bmatrix} X_{new} \\ Y_{new} \\ Z_{new} \end{bmatrix} $$
The final 2D pixel coordinates require the perspective divide: $u = u'/w$ and $v = v'/w$.

## 5. Orthographic Projection Matrix
Orthographic projection drops the $Z$-coordinate:
$$ P_{ortho} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix} $$
This matrix is idempotent ($P_{ortho} \cdot P_{ortho} = P_{ortho}$), confirming it is a true mathematical projection.

## 6. Orthogonality and Gimbal Lock
Our rotation matrix $R = R_z \cdot R_y \cdot R_x$ is orthogonal, meaning $R^T \cdot R = I$ and $\det(R) = 1$. The Euler angle sequence (Pitch, Yaw, Roll) is susceptible to Gimbal Lock at $\pm 90^\circ$ pitch, which aligns the Yaw and Roll axes. We restrict UI sliders to $\pm 25^\circ$ to avoid this and to maintain high image quality before heavy occlusions ruin the view.
