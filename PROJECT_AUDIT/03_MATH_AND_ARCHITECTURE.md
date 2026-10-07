# 03 MATH AND ARCHITECTURE

### Mathematical Foundations

This project relies entirely on geometric computer vision and linear algebra, explicitly avoiding external rendering libraries like OpenGL for the core engine. All math is implemented via NumPy.

1. **Disparity to Depth**
   The neural network outputs a relative disparity map $d \in [0, 1]$.
   $Z = \frac{1}{\frac{d}{Z_{near}} + \frac{1-d}{Z_{far}}}$
   *(Implemented in `math_engine.py: disparity_to_depth`)* [VERIFIED]

2. **Camera Intrinsics ($K$)**
   Constructed from Field of View (FOV):
   $f_x = f_y = \frac{W / 2}{\tan(FOV / 2)}$
   $K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$
   *(Implemented in `math_engine.py: TransformEngine.__init__`)* [VERIFIED]

3. **3D Unprojection (Point Cloud Mode)**
   Mapping 2D pixels $(u, v)$ to 3D space:
   $\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = Z \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$
   *(Implemented in `math_engine.py: unproject_to_3d`)* [VERIFIED]

4. **Coordinate Transformation (Orbit/Dolly)**
   $P_{new} = (P - c) R^T + c + t$
   where $c$ is the pivot center, $R$ is the rotation matrix (Euler angles), and $t$ is camera translation.
   *(Implemented in `math_engine.py: apply_transform`)* [VERIFIED]

5. **Multi-Plane Image (MPI) Homographies (Layer Mode)**
   For a given layer $k$ at depth $Z_k$, the homography $H_k$ induced by the camera motion is:
   $H_k = K ( R + \frac{t e_z^T}{Z_k} ) K^{-1}$
   where $t$ is the effective camera translation and $e_z = [0, 0, 1]^T$.
   *(Implemented in `mpi_renderer.py: layer_homographies`)* [VERIFIED]

6. **Adjugate Matrix Inversion**
   $M^{-1} = \frac{1}{\det(M)} \text{adj}(M)$
   *(Implemented in `mpi_renderer.py: inv3x3`)* [VERIFIED]

### Architecture & Data Flow

```mermaid
flowchart TD
    A[Input Image] -->|Depth Anything V2| B[Disparity Map]
    B -->|Percentile Norm + Guided Filter| C[Refined Disparity]
    
    C --> D{Renderer Selection}
    A --> D
    
    D -->|Point Splatting (V1)| E[Unproject to 3D Point Cloud]
    E --> F[Transform & Splat points to 2D]
    
    D -->|MPI Layers (V2)| G[Slice into N Depth Layers]
    G --> H[Inpaint Hidden Regions]
    H --> I[Warp Layers via Homography]
    I --> J[Composite Layers Back-to-Front]
    
    F --> K[Generate 41x3 Atlas Grid]
    J --> K
    
    K --> L[Streamlit Backend]
    L --> M[JavaScript Canvas (Frontend)]
    M --> N[60fps Mouse Interpolation]
```

### End-to-End Pipeline
1. **Input**: A single JPEG/PNG image.
2. **Depth**: `DepthEstimator` infers relative depth using a HuggingFace pipeline.
3. **Engine Choice**: 
   - **V1 (Point Cloud)**: Points are projected, rotated, translated, and splatted using a custom software rasterizer with Z-buffering.
   - **V2 (MPI)**: Image is sliced into RGBA planes. Each plane undergoes a 2D perspective warp (Homography) and is alpha-composited.
4. **Pre-rendering**: 123 views (41 yaw $\times$ 3 pitch) are rendered in parallel.
5. **Interactive UI**: The views are base64-encoded, sent to the browser, and drawn on an HTML5 Canvas that interpolates between frames based on cursor position.
