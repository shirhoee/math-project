# Mathematical Foundations of 2D to 3D View Synthesis
## Utilizing Linear Algebra and Projective Geometry for Novel View Generation

**Project Status:** Active Working Prototype (Phase 6 Complete)  
**Authors & Team Members:**
- **Shreyansh Raj** — Disparity Mapping & Depth Field Analytics
- **Shaurya Singh** — Projective Geometry & Transformation Matrices
- **Sarthak Bhowmik** — MPI Homographies & Compositing Logic

---

## 1. Executive Summary & Core Identity

The primary objective of this project is to develop a mathematically rigorous software engine capable of extrapolating 3D spatial geometry from a single 2D RGB image matrix and synthesizing novel viewpoint perspectives with realistic parallax. 

**This is fundamentally an applied Linear Algebra and Projective Geometry system.**
While modern generative AI often treats 3D view synthesis as an opaque, hallucinating black box, our engine establishes a deterministic geometric pipeline grounded in:
1. **Projective Camera Parameterization ($K \in \mathbb{R}^{3 \times 3}$)**
2. **Analytical Frustum Unprojection via Inverse Intrinsics ($K^{-1}$)**
3. **Rigid-Body Kinematics in the Special Orthogonal Lie Group $SO(3) \times \mathbb{R}^3$**
4. **Planar Homography Derivations ($H_k \in \mathbb{R}^{3 \times 3}$)**
5. **Multi-Plane Image (MPI) Slicing across Disparity-Linear Scalar Fields**
6. **Vectorized Back-to-Front Algebraic Alpha Compositing (Porter-Duff "Over" Operator)**

The deep learning model employed (`Depth-Anything-V2-Small-hf` via PyTorch / HuggingFace) is strictly an **isolated upstream pre-processor** used solely to estimate a 2D scalar disparity field from an RGB image. 

**PyTorch is NOT used for our linear algebra engine.** The moment disparity values are estimated, they are converted into a pure `numpy.ndarray`. Every single mathematical operation in our engine—matrix parameterization ($K$), analytical inversion ($K^{-1}$), coordinate unprojection, Lie group rotation kinematics ($SO(3)$), focal-pivot translations ($t$), planar homography derivation ($H_k$), coordinate remapping, and alpha compositing—is **implemented 100% in NumPy** using fundamental linear algebra principles (`np.ndarray`, matrix multiplication `@`, and vector broadcasting).

---

## 2. Project Architecture & Current Version (v1.6.0)

The project has advanced from an early monolithic prototype to a fully decoupled, production-grade microservice architecture:

```
+-----------------------------------------------------------------------------------+
|                        FRONTEND WEB CLIENT (Vite + React)                         |
|  - HTML5 Canvas Parallax Viewport (60-120 FPS requestAnimationFrame Loop)         |
|  - Live Mathematical Matrix Inspector (KaTeX LaTeX equations)                    |
|  - 0ms Drag-and-Drop Image Preview (URL.createObjectURL)                          |
|  - Bilinear Pitch/Yaw View Interpolation & Spring-Damper (Lerp) Physics          |
+------------------------------------------+----------------------------------------+
                                           | HTTP REST API (/api/v1/*)
                                           v
+-----------------------------------------------------------------------------------+
|                         BACKEND SERVICE (FastAPI + ASGI)                          |
|  - Endpoints: POST /render, GET /jobs/{id}/status, GET /jobs/{id}/result          |
|  - Static Asset Mounting: Serves compiled React production bundle at /            |
|  - Singleton Model Caching (_DEPTH_MODEL_CACHE)                                   |
+------------------------------------------+----------------------------------------+
                                           | Asynchronous Task Delegation
                                           v
+-----------------------------------------------------------------------------------+
|                   DISTRIBUTED TASK QUEUE (Celery + Redis / Eager)                 |
|  - Background Worker Execution                                                    |
|  - Dynamic Granular Progress Reporting (task_progress_registry)                   |
+------------------------------------------+----------------------------------------+
                                           | Linear Algebra Compute Core
                                           v
+-----------------------------------------------------------------------------------+
|                     PROJECTIVE GEOMETRY & MPI COMPUTE CORE                        |
|  1. Disparity-to-Depth Inversion (Z_k = 1 / d_k)                                  |
|  2. Pinhole Matrix Calibration: K and K^-1                                        |
|  3. 3D Rotation Matrix Construction: R in SO(3), Det(R) = 1, R R^T = I             |
|  4. Focal Pivot Translation Vector: t = c - R c + t_extra                         |
|  5. Per-Plane Homographies: H_k = K (R - t n^T / d_k) K^-1                        |
|  6. Vectorized Remapping & Algebraic Back-to-Front Alpha Compositing              |
+-----------------------------------------------------------------------------------+
```

---

## 3. Mathematical Foundations: The Linear Algebra Engine

### 3.1 Pinhole Camera Model & Intrinsic Calibration ($K$)
A central concept in projective geometry is mapping 3D Euclidean space $\mathbb{R}^3$ to the 2D projective image plane $\mathbb{P}^2$. Using homogeneous coordinates, a 3D point $P = [X, Y, Z]^T$ projects to image coordinates $p = [u, v, 1]^T$ via:

$$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}$$

where $\lambda = Z$ represents the projective scale factor, and $K \in \mathbb{R}^{3 \times 3}$ is the upper-triangular camera intrinsic matrix:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

- **Focal Lengths ($f_x, f_y$):** Derived directly from the horizontal Field of View ($\text{FOV}$) and image dimensions ($W \times H$):
  $$f_x = f_y = \frac{W}{2 \tan\left(\frac{\text{FOV}}{2}\right)}$$
- **Principal Point ($c_x, c_y$):** The optical center, parameterized at the image centroid:
  $$c_x = \frac{W}{2}, \quad c_y = \frac{H}{2}$$

### 3.2 Unprojection: From 2D Pixels to 3D Rays
Because $K$ is non-singular ($\det(K) = f_x f_y \neq 0$), its inverse $K^{-1}$ exists analytically:

$$K^{-1} = \begin{bmatrix} \frac{1}{f_x} & 0 & -\frac{c_x}{f_x} \\ 0 & \frac{1}{f_y} & -\frac{c_y}{f_y} \\ 0 & 0 & 1 \end{bmatrix}$$

Given an estimated metric depth scalar $Z(u, v)$ for each pixel, we compute the unprojected 3D spatial coordinate vector $(X, Y, Z)^T$ via matrix multiplication:

$$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = Z(u, v) \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

**Implementation Detail:** This transformation is fully vectorized across the entire image tensor using NumPy broadcasting ($3 \times (H \cdot W)$ linear systems evaluated simultaneously), eliminating slow iterative loops and executing in milliseconds.

### 3.3 3D Camera Kinematics in the Lie Group $SO(3)$
When the user moves the virtual camera, we define its orientation using a 3D rotation matrix $R \in SO(3)$ (Special Orthogonal group). For horizontal yaw ($\theta_y$) and vertical pitch ($\theta_p$):

$$R_y(\theta_y) = \begin{bmatrix} \cos\theta_y & 0 & \sin\theta_y \\ 0 & 1 & 0 \\ -\sin\theta_y & 0 & \cos\theta_y \end{bmatrix}, \quad R_x(\theta_p) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta_p & -\sin\theta_p \\ 0 & \sin\theta_p & \cos\theta_p \end{bmatrix}$$

The composite rotation is the matrix product:

$$R = R_y(\theta_y) R_x(\theta_p)$$

**Orthogonality Verification:**
In every render and in the client-side matrix inspector, we verify that $R$ preserves metric length and angles:
1. $R R^T = R^T R = I_3$ (Orthogonality)
2. $\det(R) = +1$ (Proper rotation, preserving orientation without reflection)

### 3.4 Focal Center Anchoring & Translation Vector ($t$)
A naive rotation around the camera origin $(0, 0, 0)$ would cause the scene to swing wildly out of view. To achieve an authentic parallax orbit, the camera must rotate around a focal subject point located at depth $Z_{\text{pivot}}$ along the optical axis: $c = [0, 0, Z_{\text{pivot}}]^T$.

Under a rigid-body change of coordinates, a point $P$ transforms to $P' = R(P - c) + c + t_{\text{extra}}$. Expanding this yields the canonical rigid-body form $P' = R P + t$, where the translation vector $t \in \mathbb{R}^3$ is defined as:

$$t = c - R c + t_{\text{extra}} = (I_3 - R) c + t_{\text{extra}}$$

This mathematical formulation anchors the central subject stationary in the frame while foreground and background planes move in opposite directions, creating convincing 3D parallax.

### 3.5 Planar Homography Derivation ($H_k$)
A Multi-Plane Image (MPI) represents the scene as a collection of $L$ fronto-parallel planar slices at depths $d_k$ ($k = 0, 1, \dots, L-1$). 

For a plane with unit normal vector $n = [0, 0, 1]^T$ at Euclidean distance $d_k$, any 3D point on the plane satisfies $n^T P = d_k$, or $\frac{n^T P}{d_k} = 1$. Substituting this into the rigid body equation:

$$P' = R P + t = \left( R + \frac{t n^T}{d_k} \right) P$$

Projecting back to the novel camera plane using the intrinsic matrix $K$:

$$p' \sim K P' = K \left( R + \frac{t n^T}{d_k} \right) P = K \left( R + \frac{t n^T}{d_k} \right) K^{-1} p$$

This yields the fundamental **induced planar homography matrix** $H_k \in \mathbb{R}^{3 \times 3}$:

$$H_k = K \left( R - \frac{t n^T}{d_k} \right) K^{-1}$$

### 3.6 Vectorized Inverse Warping
To avoid cracking and holes inherent to forward point projection, our engine executes **inverse warping**. For every destination pixel $(u', v')$ on depth layer $k$, we multiply by the matrix inverse $H_k^{-1}$:

$$\begin{bmatrix} \tilde{u} \\ \tilde{v} \\ \tilde{w} \end{bmatrix} = H_k^{-1} \begin{bmatrix} u' \\ v' \\ 1 \end{bmatrix}, \quad u = \frac{\tilde{u}}{\tilde{w}}, \quad v = \frac{\tilde{v}}{\tilde{w}}$$

This coordinate remapping is vectorized with SIMD hardware acceleration (`cv2.remap` with bilinear interpolation).

### 3.7 Back-to-Front Algebraic Alpha Compositing
After warping each layer, the final novel image is composited back-to-front (from $k = 0$ at $Z_{\text{far}}$ to $k = L-1$ at $Z_{\text{near}}$) using the Porter-Duff "Over" operator:

$$C_{\text{accumulated}}^{(k)} = C_k + (1 - \alpha_k) C_{\text{accumulated}}^{(k-1)}$$
$$\alpha_{\text{accumulated}}^{(k)} = \alpha_k + (1 - \alpha_k) \alpha_{\text{accumulated}}^{(k-1)}$$

This linear blending algebraically resolves occlusions, semi-transparent hair/edges, and depth discontinuities without depth buffer z-fighting.

---

## 4. Team Roles & Detailed Work Distribution

The workload was systematically divided into three interconnected mathematical domains, matching the project delegation:

```
+------------------------------------------------------------------------------------+
|                                    SHREYANSH RAJ                                   |
|                      Disparity Mapping & Depth Field Analytics                     |
|  - Maps relative disparity fields to absolute metric/relative depth Z              |
|  - Guided filter edge alignment (box filtering with O(1) integral sums)            |
|  - Non-linear interpolation (bicubic sampling, morphological hole-filling)          |
|  - Background dilation & occlusion boundary inpainting                             |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
|                                    SHAURYA SINGH                                   |
|                 Projective Geometry & Camera Transformation Matrices               |
|  - Parameterization of intrinsic matrix K and analytical inverse K^-1              |
|  - Vectorized unprojection engine mapping (u, v, 1)^T to 3D point cloud rays       |
|  - 3D Rotation matrices R in SO(3) with orthogonality & determinant validation     |
|  - Kinematics of spherical camera orbits & focal-pivot translation vector t        |
+------------------------------------------------------------------------------------+
                                           |
                                           v
+------------------------------------------------------------------------------------+
|                                   SARTHAK BHOWMIK                                  |
|                        MPI Homographies & Compositing Logic                        |
|  - Disparity-spaced Multi-Plane Image (MPI) slicing across L planes (8 to 32)      |
|  - Mathematical derivation and calculation of planar homographies H_k             |
|  - Analytical homography inversion H_k^-1 and vectorized bilinear remapping        |
|  - Vectorized back-to-front Porter-Duff alpha compositing engine                   |
+------------------------------------------------------------------------------------+
```

### 4.1 Shreyansh Raj — Disparity Mapping & Depth Field Analytics
* **Mathematical Focus**: Continuous scalar fields, inverse depth distributions, and edge-preserving filtering.
* **Key Achievements & How It Was Done**:
  1. **Disparity-to-Depth Inversion**: 
     - Neural depth estimation models output a relative disparity field $d(x, y) \in [0, 1]$, which is proportional to optical inverse depth:
       $$\text{disparity} \propto \frac{1}{Z}$$
     - Implemented dynamic normalization and metric bounds mapping:
       $$Z(x, y) = \frac{1}{\frac{1}{Z_{\text{far}}} + d(x, y) \left(\frac{1}{Z_{\text{near}}} - \frac{1}{Z_{\text{far}}}\right)}$$
       allowing fine-tuning of near and far clipping planes $[Z_{\text{near}}, Z_{\text{far}}]$.
  2. **Edge-Preserving Guided Filter**:
     - Raw depth maps exhibit blurred object boundaries that bleed background colors onto foreground silhouettes.
     - Implemented a fast guided filter using the RGB image as guidance $I$ and disparity as input $p$. Utilized $O(1)$ box filtering (`cv2.boxFilter`) to compute local covariance and variance:
       $$a_k = \frac{\text{cov}_{I, p}}{\sigma_I^2 + \epsilon}, \quad b_k = \bar{p}_k - a_k \bar{I}_k$$
       producing tack-sharp depth edges aligned with physical object contours.
  3. **Occlusion Hole-Filling & Inpainting**:
     - Synthesizing novel camera perspectives inherently reveals regions occluded in the original 2D image (disocclusions).
     - Built morphological dilation and background layer extension algorithms: foreground layers are clipped to their respective alpha masks, while background layers are inpainted using boundary-extrapolated bicubic sampling, ensuring disoccluded holes do not produce black voids.

### 4.2 Shaurya Singh — Projective Geometry & Transformation Matrices
* **Mathematical Focus**: Projective spaces $\mathbb{P}^2 \to \mathbb{P}^3$, Lie Group $SO(3)$, and rigid-body transforms.
* **Key Achievements & How It Was Done**:
  1. **Camera Matrix Parameterization & Inversion**:
     - Formulated the exact pinhole intrinsic matrix $K$ and its algebraic inverse $K^{-1}$ from Field of View (FOV) and image resolution.
     - Designed the vectorized unprojection engine in `src/core/math_engine.py`:
       $$\mathbf{P}_{\text{ray}} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad \mathbf{P}_{3D} = Z \cdot \mathbf{P}_{\text{ray}}$$
       broadcasting coordinate grids without explicit loops.
  2. **3D Rotation Kinematics in $SO(3)$**:
     - Implemented 3D rotation transformations $R = R_y(\theta_y) R_x(\theta_p)$ for horizontal and vertical camera orbits.
     - Implemented strict verification logic:
       $$\| R R^T - I_3 \|_F < 10^{-6}, \quad |\det(R) - 1| < 10^{-6}$$
       preventing matrix shear, non-uniform scaling, or chirality flips.
  3. **Spherical Orbit Translation ($t$)**:
     - Derived the closed-form translation vector $t = (I_3 - R) c$ around the focal subject center $c = [0, 0, Z_{\text{pivot}}]^T$.
     - Engineered dampening and motion limits ($\theta_y \le 8^\circ, \theta_p \le 3^\circ$) to maintain homography planar validity and eliminate visual tearing.

### 4.3 Sarthak Bhowmik — MPI Homographies & Compositing Logic
* **Mathematical Focus**: Planar projective transformations, homography inversion, and compositing algebras.
* **Key Achievements & How It Was Done**:
  1. **Disparity-Linear MPI Slicing**:
     - Divided the 3D viewing volume into $L$ discrete depth planes ($L = 8$ to $32$).
     - Proved that spacing planes linearly in Euclidean depth $Z$ causes unequal pixel shifts, whereas spacing planes linearly in **disparity space** ($\frac{1}{Z}$) guarantees constant parallax step size between adjacent planes:
       $$\frac{1}{d_k} = \frac{1}{Z_{\text{far}}} + \frac{k + 0.5}{L} \left( \frac{1}{Z_{\text{near}}} - \frac{1}{Z_{\text{far}}} \right)$$
  2. **Planar Homography Derivation & Computation**:
     - Implemented the full $3 \times 3$ homography calculation for each plane $k$:
       $$H_k = K \left( R - \frac{t n^T}{d_k} \right) K^{-1}$$
     - Evaluated $H_k$ analytically for each layer given the instantaneous camera pose $(R, t)$.
  3. **Vectorized Remapping & Alpha Compositing**:
     - Computed the algebraic inverse $H_k^{-1}$ via cofactor expansion.
     - Transferred per-layer pixel values via backward mapping (`cv2.remap`) with bilinear interpolation.
     - Implemented the vectorized Porter-Duff back-to-front compositing loop:
       $$C_{\text{out}} = C_k + (1 - \alpha_k) C_{\text{acc}}$$
       blending colors and transparencies accurately to resolve complex multi-layer occlusions.

---

## 5. Technology Stack & Tools Used

| Category | Tool / Framework | Specific Role & Implementation in Project |
| :--- | :--- | :--- |
| **Linear Algebra & Numerics** | **NumPy (1.26+)** | Matrix multiplications (`@`), analytical inversions (`np.linalg.inv`), tensor broadcasting for coordinate unprojection. |
| **Computer Vision & SIMD** | **OpenCV (cv2)** | SIMD-accelerated backward warping (`cv2.remap`), $O(1)$ integral image guided box filtering (`cv2.boxFilter`). |
| **Depth Perception** | **PyTorch & HuggingFace** | `Depth-Anything-V2-Small-hf` pipeline to generate initial scalar disparity fields from 2D images. |
| **Backend API** | **FastAPI & Uvicorn** | High-performance asynchronous REST endpoints (`/api/v1/render`, `/api/v1/jobs/*`, `/api/v1/samples`). |
| **Task Queue & Async** | **Celery & Redis** | Distributed background processing for render jobs with transparent eager local fallback. |
| **Frontend Framework** | **React 19 & Vite** | Modern Single Page Application (SPA) architecture with fast HMR and optimized production bundling. |
| **Interactive Viewport** | **HTML5 Canvas 2D API** | Custom 60–120 FPS render loop using `requestAnimationFrame`, spring-damper (`lerp`) cursor physics, and `createImageBitmap`. |
| **Math Rendering** | **KaTeX** | Real-time browser rendering of LaTeX matrices, determinants, and linear algebra formulas. |
| **Styling & Icons** | **Tailwind CSS & Lucide** | Responsive dark-mode user interface and technical icons. |
| **Quality & Testing** | **Pytest & Vitest** | Automated backend integration test suite and frontend linear algebra unit test suite. |

---

## 6. What We Have Achieved Until Now

1. **Complete Decoupling of Frontend and Math Engine**:
   - Replaced fragile script-reloading Streamlit UI with an independent, production-grade React SPA.
   - Built a high-performance REST API with FastAPI serving both the JSON endpoints and compiled static web assets.

2. **60 FPS Real-Time Parallax Viewport**:
   - Custom HTML5 `<canvas>` viewport delivering 60–120 FPS interactive mouse-tracking parallax.
   - Smooth auto-wiggle Lissajous figure-8 orbit on mouse idle.
   - 0ms client-side image preview via `URL.createObjectURL`.

3. **Live Mathematical Matrix Inspector**:
   - Direct on-screen inspection of the live intrinsic matrix $K$, rotation matrix $R$, orthogonality check $R R^T = I$, and per-layer homographies $H_k$.
   - Real-time updates as the user interacts with the canvas.

4. **Multi-Plane Image Slicing with Vectorized Compositing**:
   - Discrete slicing of 2D images into 8 to 32 depth layers based on disparity scalar fields.
   - Occlusion hole filling and edge-preserving guided filtering.
   - Back-to-front alpha compositing delivering clean, artifact-free novel views.

5. **Rigorous Test Coverage**:
   - Frontend Vitest suite verifying camera matrix transformations and orthogonality.
   - Backend Pytest suite verifying API endpoints, Celery workers, and static serving.

---

## 7. Next Steps & Future Work

1. **GPU Shader Acceleration (WebGL / WebGPU)**:
   - Port the homography inverse warping and back-to-front compositing directly into fragment shaders on the client GPU for instant, zero-latency local re-rendering.
2. **Dynamic Mesh Generation (Delaunay Triangulation)**:
   - Convert the 3D unprojected point cloud into an adaptive 3D surface mesh with depth discontinuity edge filtering.
3. **Multi-View Stereo & Epipolar Geometry**:
   - Extend the engine from single-image synthesis to two-view stereo geometry, computing Fundamental and Essential matrices ($F, E$) for feature correspondence.
