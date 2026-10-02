# Mathematical Derivations

This document outlines the core linear algebra formulas used in the custom 3D graphics engine.

## 1. Intrinsic Camera Matrix ($K$)
The intrinsic matrix maps 3D camera coordinates to 2D image pixels. We assume the principal point $(c_x, c_y)$ is at the center of the image, and focal lengths $(f_x, f_y)$ are scaled based on the image size.

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

## 2. Unprojection (2D to 3D)
To map a 2D pixel $(u, v)$ with depth $Z$ back into 3D space $(X, Y, Z)$, we use the inverse of the intrinsic matrix ($K^{-1}$).

$$
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = Z \cdot K^{-1} \cdot \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$
This operation is applied in a vectorized manner across the $(N, 3)$ matrix of all pixels, where $N = \text{width} \times \text{height}$.

## 3. Orthogonal Rotation Matrices
To rotate the point cloud in 3D space, we apply Orthogonal Rotation Matrices. 

**Pitch (Rotation around X-axis):**
$$
R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}
$$

**Yaw (Rotation around Y-axis):**
$$
R_y(\phi) = \begin{bmatrix} \cos\phi & 0 & \sin\phi \\ 0 & 1 & 0 \\ -\sin\phi & 0 & \cos\phi \end{bmatrix}
$$

**Roll (Rotation around Z-axis):**
$$
R_z(\gamma) = \begin{bmatrix} \cos\gamma & -\sin\gamma & 0 \\ \sin\gamma & \cos\gamma & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

**Combined Rotation:** $R = R_z \cdot R_y \cdot R_x$
**Transformation:** $P_{rotated} = P_{original} \cdot R^T$

## 4. Perspective Projection (3D to 2D)
To render the rotated 3D points back to the 2D screen, we multiply by the intrinsic matrix $K$ and divide by the new depth $Z_{new}$ to account for perspective scaling.

$$
\begin{bmatrix} u_{new} \\ v_{new} \\ w \end{bmatrix} = K \cdot \begin{bmatrix} X_{new} \\ Y_{new} \\ Z_{new} \end{bmatrix}
$$
The final 2D pixel coordinates are $x = u_{new}/w$ and $y = v_{new}/w$.
