# 03 Math and Geometry

**Intrinsics from FOV**
- Derivation: $f_x = (W/2) / \tan(FOV/2)$
- Shapes: Scalar inputs/outputs.
- Implemented: `src.geometry.camera`

**Relative-Depth to Z Conversion**
- Derivation: $Z = 1 / (d/z_{near} + (1-d)/z_{far})$
- Shapes: $d \in \mathbb{R}^{H \times W} \rightarrow Z \in \mathbb{R}^{H \times W}$
- Implemented: `src.geometry.projection`

**Unprojection**
- Derivation: $X = (u - c_x) \cdot Z / f_x$, $Y = (v - c_y) \cdot Z / f_y$
- Shapes: $u, v, Z \in \mathbb{R}^{H \times W} \rightarrow P \in \mathbb{R}^{H \times W \times 3}$
- Implemented: `src.geometry.projection`

**Rotations**
- Derivation: Standard Euler angle (yaw, pitch, roll) to $3 \times 3$ matrix.
- Shapes: $\mathbb{R}^3 \rightarrow \mathbb{R}^{3 \times 3}$
- Implemented: `src.geometry.transform`

**Pivot Transform**
- Derivation: $P_{new} = (P - c) R^T + c + t$
- Shapes: $P \in \mathbb{R}^{N \times 3}$, $R \in \mathbb{R}^{3 \times 3}$, $t \in \mathbb{R}^3 \rightarrow P_{new} \in \mathbb{R}^{N \times 3}$
- Implemented: `src.geometry.transform`

**Plane-Induced Homography**
- Derivation: $H_d = K ( R + \frac{b e_z^T}{d} ) K^{-1}$
- Shapes: $K, R \in \mathbb{R}^{3 \times 3}$, $b \in \mathbb{R}^3 \rightarrow H \in \mathbb{R}^{3 \times 3}$
- Implemented: `src.geometry.homography`

**Adjugate Inverse**
- Derivation: $M^{-1} = \frac{1}{\det(M)} \text{adj}(M)$
- Shapes: $\mathbb{R}^{3 \times 3} \rightarrow \mathbb{R}^{3 \times 3}$
- Implemented: `src.geometry.math_utils`

## Preview of Stage B Math
- **8-point algorithm:** $A f = 0 \rightarrow$ SVD to find Fundamental matrix.
- **Pose Decomposition:** $E = U \Sigma V^T \rightarrow R = U W V^T, t = U_3$.
- **DLT Triangulation:** $A X = 0$ for point correspondences.
- **Least-Squares Scale Fit:** $s = \arg \min \sum (Z_i - s \cdot Z_{tri, i})^2$.
