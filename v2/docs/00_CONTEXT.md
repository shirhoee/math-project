# 00 Context

**Why this project exists:**
This project is part of the "Mathematical Foundation for AI and Data Science" course. It demonstrates the application of linear algebra and projective geometry to construct interactive 3D visualizations from single 2D photographs.

**History (V1 Summary):**
V1 implemented a 3D Photo Viewer using Depth Anything V2 for depth estimation. Initially, it used NumPy point-splatting and later moved to Multi-Plane Image (MPI) layers. It served an interface via Streamlit, pre-rendering a grid of 123 frames to allow smooth interpolation.

*What worked:*
- The geometry engine (intrinsics, pivot transform, perspective/orthographic projections).
- Strong mathematical foundations and test suite.

*What failed:*
- Slow architecture due to Streamlit reruns.
- Heavy base64 payload.
- Smearing artifacts at extreme viewing angles.
- Cluttered repository root.
- Specification drift over time.

**Why a rebuild was chosen:**
A clean rebuild (V2) allows separating the backend (FastAPI) from a static frontend, ensuring a much smoother, reactive, and maintainable application without the Streamlit overhead. It allows us to consolidate math into robust, vectorised NumPy code and cleanly handle performance, structure, and upcoming Stage B requirements.

**Ported Math Derivations (from V1):**
- **Disparity to Depth:** $Z = 1 / (d/z_{near} + (1-d)/z_{far})$
- **Intrinsics from FOV:** $f_x = (W/2) / \tan(FOV/2)$
- **Pivot Transform and 4x4 form:** $P_{new} = (P - c) * R^T + c + t$
- **Perspective Projection:** $u = f_x * X/Z + c_x, v = f_y * Y/Z + c_y$
- **Inverse 3x3 Matrix (Adjugate):** $M^{-1} = \frac{1}{\det(M)} \text{adj}(M)$
- **Plane-Induced Homography:** $H_d = K ( R + \frac{b e_z^T}{d} ) K^{-1}$ (where $b = c - R c + t_{extra}$)
