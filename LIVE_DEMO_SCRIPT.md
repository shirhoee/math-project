# Live Presentation Script & Teacher Demo Guide
## Project: Mathematical Foundations of 2D to 3D View Synthesis

**Team Members:** Shreyansh Raj, Shaurya Singh, Sarthak Bhowmik  
**Presentation Audience:** Mathematics / Linear Algebra Course Instructor  
**Target Duration:** 5 to 7 Minutes  

---

## Part 1: Pre-Demo Setup & Presenter Checklist

### 1. Launching the Live Application
Open PowerShell in the project directory (`e:\Math project`) and start the unified server:
```powershell
uvicorn src.api.server:app --app-dir v1 --port 8000
```
- Open Google Chrome or Microsoft Edge and navigate to:
  **`http://localhost:8000`**
- Press **F11** for clean fullscreen presentation mode.
- Have the **"Math Inspector"** tab open or easily clickable in the side drawer.
- Pre-load a sample preset (e.g., **Sample 4: Espresso Cup** or **Sample 1: Living Room**) so you have immediate interactive 3D parallax ready before you even upload a new image.

### 2. Strategic Presentation Mindset
> [!IMPORTANT]
> **Linear Algebra in Pure NumPy (Not PyTorch)!**
> Instructors in linear algebra evaluate whether you understand **vector spaces, matrix transformations, coordinate frames, and projective geometry**.
> - **We used NumPy (`np.ndarray`, matrix multiplications `@`, analytical matrix inverses, vector projections) for 100% of our linear algebra engine.**
> - Clarify to your teacher that PyTorch was **not** used to do the math—PyTorch was isolated to a single 1-step upstream task (fetching initial raw disparity values from a pre-trained model). The moment disparity is obtained, it is converted to a NumPy array, and all 3D unprojections, rotation groups ($SO(3)$), camera calibration matrices ($K$), planar homographies ($H_k$), and novel view blending are **pure applied linear algebra written in NumPy by our team**.

---

## Part 2: Step-by-Step Presentation Script

---

### Phase 1: The Hook & Problem Definition (1 Minute)
* **Goal**: Establish the mathematical core immediately.
* **On Screen**: The main application interface with the interactive canvas centered.

#### [What to Say]
> *"Good morning/afternoon, Professor.*
>
> *Today, our team—Shreyansh Raj, Shaurya Singh, and Sarthak Bhowmik—is presenting our project: **The Mathematical Foundations of 2D to 3D View Synthesis**.*
>
> *In computer vision and graphics, generating 3D views usually requires either physical LiDAR scans, multi-camera rigs, or black-box neural networks that hallucinate pixels. Our goal was different: **we built a deterministic mathematical engine that formulates 3D spatial geometry strictly through Applied Linear Algebra and Projective Geometry**.*
>
> *Given a single 2D image matrix, our software models the inverse projection of the 2D projective plane $\mathbb{P}^2$ into a 3D Euclidean viewing frustum $\mathbb{R}^3$, computes rigid-body kinematics in the 3D rotation group $SO(3)$, derives planar homographies $H_k$, and re-projects the scene to synthesize real-time parallax views with no physical 3D scanning."*

---

### Phase 2: The Linear Algebra Pipeline (2 Minutes)
* **Goal**: Walk through the core mathematical formulas and show that each team member solved a specific linear algebra challenge.
* **On Screen**: Open the **"Math Matrix Inspector"** on the right side of the interface.

#### [What to Say]
> *"Our pipeline is broken into three distinct mathematical stages, divided across our team:*
>
> **1. Disparity Mapping & Depth Analytics (Led by Shreyansh):**
> *"An image is initially just an $H \times W \times 3$ color matrix. Shreyansh’s module extracts a continuous scalar disparity field $d(u, v)$ and maps it to Euclidean depth using an inverse relation: $Z(u, v) = \frac{1}{d(u, v)}$. Because depth boundaries must align with physical object edges, he implemented a guided filter using $O(1)$ box-filtering integral sums to solve a local linear ridge regression, keeping depth contours razor-sharp while inpainting occlusion boundaries."*
>
> **2. Projective Geometry & Camera Transformations (Led by Shaurya):**
> *(Point to the matrix $K$ and $R$ on the screen)*
> *"Next, Shaurya established the pinhole camera geometry. He parameterized the intrinsic calibration matrix $K$ based on the Field of View, and derived its analytical inverse $K^{-1}$. Using vectorized tensor unprojection, we solve the linear system:*
>
> $$\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = Z \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
>
> *across the entire pixel grid simultaneously without slow loops.*
> *When the camera moves, Shaurya models the rotation in the Special Orthogonal group $SO(3)$, multiplying yaw and pitch matrices $R = R_y(\theta_y) R_x(\theta_p)$. You can see live on screen that we explicitly verify orthogonality: $R R^T = I$ and $\det(R) = +1$. Furthermore, to prevent the scene from swinging off-screen, he derived a focal pivot translation vector $t = (I - R)c$ that anchors the virtual camera orbit around a central subject."*
>
> **3. Multi-Plane Image Homographies & Compositing (Led by Sarthak):**
> *(Point to the Homography matrix $H_k$ and the Layer Viewer)*
> *"Finally, Sarthak handled the novel view synthesis using Multi-Plane Images. Rather than projecting points arbitrarily, he slices the volume into $L$ discrete depth planes spaced uniformly in disparity space.*
> *For each plane $k$ at distance $d_k$, Sarthak derived the induced planar homography:*
>
> $$H_k = K \left( R - \frac{t n^T}{d_k} \right) K^{-1}$$
>
> *He then inverts $H_k$ to perform backward coordinate remapping, and blends the layers back-to-front using Porter-Duff algebraic alpha compositing:*
>
> $$C_{\text{out}} = C_k + (1 - \alpha_k) C_{\text{acc}}$$
>
> *This resolves depth occlusions smoothly without the cracking or holes seen in naive point splatting."*

---

### Phase 3: The Live Interactive Demo (2 Minutes)
* **Goal**: Demonstrate 60 FPS performance, mouse responsiveness, live math inspection, and custom upload.
* **On Screen**: Close drawer, interact with the canvas.

#### [Action 1: Mouse Movement & Parallax]
* **Action**: Move your mouse slowly across the image from left to right, then up and down.
* **Say**:
  > *"Notice the parallax effect: as I move my cursor, foreground elements shift with a larger baseline displacement than background elements. This isn't a video or a pre-rendered 3D file—our HTML5 canvas is executing continuous frame interpolation at 60 FPS using spring-damper physics based on the instantaneous camera angles $\theta_y$ and $\theta_p$."*

#### [Action 2: Idle Auto-Wiggle]
* **Action**: Stop moving the mouse completely.
* **Say**:
  > *"When the mouse is idle, the engine smoothly falls into an automatic Lissajous figure-8 orbit, continuously executing the rotation matrix equations to provide natural depth perception."*

#### [Action 3: Live Matrix Inspector]
* **Action**: Click the **Math Matrix Inspector** button or toggle.
* **Say**:
  > *"Here is the live proof of our linear algebra calculations. As the camera position changes, you can see the updated rotation matrix $R$, its verified determinant of $1.000$, the identity product $R R^T = I$, and the calculated $3 \times 3$ homography matrix $H_k$ for any selected depth slice."*

#### [Action 4: Layer Slices Inspection]
* **Action**: Click on **Layer Slices Viewer** in the drawer.
* **Say**:
  > *"Here you can see Sarthak and Shreyansh's MPI decomposition. The 2D image has been sliced into fronto-parallel planes from near to far. Foreground layers contain only the isolated object silhouettes, while background layers have been inpainted to fill occluded gaps."*

#### [Action 5: Custom Image Upload / Preset Selection]
* **Action**: Click another sample preset or drag-and-drop a new photo.
* **Say**:
  > *"Our decoupled architecture provides a 0ms instant preview via browser object URLs, while delegating the depth estimation and homography atlas generation to our asynchronous FastAPI and Celery backend. Once computed, the full 3D interactive view is cached and immediately interactive."*

---

### Phase 4: Conclusion & Hand-off (30 Seconds)

#### [What to Say]
> *"To summarize: by bridging classical projective geometry with modern computational pipelines, we transformed a single 2D image into an interactive, mathematically consistent 3D viewpoint engine.*
>
> *Every coordinate, ray, and pixel blend you see is governed by verifiable linear transformations: $K^{-1}$ unprojection, $SO(3)$ rotations, planar homographies $H_k$, and alpha algebra.*
>
> *Thank you, Professor. We are now eager to answer any questions or walk through specific mathematical derivations."*

---

## Part 3: Anticipating Teacher Q&A (Defense Guide)

Here are the four most likely questions your teacher will ask, along with the precise mathematical answers to give:

### Q1: "Why do you compute planar homographies ($H_k$) instead of forward-projecting 3D points directly?"
* **Answer**:
  > *"Forward-projecting 3D points (Point Splatting) maps discrete coordinates from source to destination. Because destination coordinates are real numbers that must be rounded to integer pixels, forward projection inevitably creates 'cracking'—empty unmapped holes between points—requiring expensive radius splatting and Z-buffer arbitration.*
  > *In contrast, a planar homography $H_k$ is a continuous projective bijection between planes. By computing the inverse homography $H_k^{-1}$, we perform **backward mapping**: for every integer pixel on the novel canvas, we sample the exact fractional source coordinate using bilinear interpolation. This is strictly hole-free, differentiable, and executes in parallel via GPU or SIMD instructions."*

### Q2: "Why do you space the MPI layers linearly in disparity ($\frac{1}{Z}$) rather than linearly in Euclidean depth ($Z$)?"
* **Answer**:
  > *"Under perspective projection, image parallax displacement $\Delta u$ is inversely proportional to depth: $\Delta u \propto \frac{1}{Z}$.*
  > *If we spaced planes linearly in depth $Z$, nearby planes would exhibit massive, discontinuous jumps in pixel shift, while distant planes would have redundant, imperceptible sub-pixel shifts.*
  > *By quantizing uniformly in **disparity space** ($\Delta d = \text{constant}$), the pixel disparity between any two adjacent planes is strictly constant across the entire scene volume, guaranteeing smooth, artifact-free blending across all depths."*

### Q3: "How do you ensure your rotation matrix doesn't cause shear, distortion, or scale change?"
* **Answer**:
  > *"We construct the rotation matrix directly in the Special Orthogonal group $SO(3)$. We compute $R$ as the product of elementary Givens-style planar rotation matrices: $R = R_y(\theta_y) R_x(\theta_p)$.*
  > *Because both $R_y$ and $R_x$ are orthonormal with determinant $+1$, their product is guaranteed to satisfy $R R^T = I_3$ and $\det(R) = +1$. This mathematically guarantees that Euclidean distances and inner products are preserved—no shearing, stretching, or chirality inversions can occur. We even have unit tests in Vitest and Pytest asserting $\| R R^T - I \|_F < 10^{-6}$."*

### Q4: "Where does the AI model stop and where does your linear algebra begin? Did you use PyTorch for the math?"
* **Answer**:
  > *"We used **NumPy**, not PyTorch, for all of our linear algebra and geometry calculations!*
  > *PyTorch was strictly confined to an isolated initial step: running a single forward pass through a pre-trained vision transformer (`Depth-Anything-V2`) to get an initial 2D disparity scalar field. That output is immediately cast into a standard `numpy.ndarray`.*
  > *From that point on, **zero PyTorch is used**. The entire mathematical engine—the pinhole calibration matrix $K$, analytical inversion $K^{-1}$, coordinate unprojection, Lie group rotation matrices $R \in SO(3)$, focal translation vector $t$, per-plane homographies $H_k = K(R - \frac{t n^T}{d_k})K^{-1}$, guided filtering covariance formulas, and Porter-Duff alpha compositing—was written completely in **NumPy** using matrix multiplications (`@`), broadcasting, and linear algebra formulas."*

---

## Part 4: Presenter Cheat Sheet (Quick Cues)

| Timestamp | Presenter / Speaker | Primary Screen Action | Core Mathematical Concept |
| :--- | :--- | :--- | :--- |
| **0:00 - 1:00** | Team / Lead | Fullscreen live UI | Projective Geometry $\mathbb{P}^2 \to \mathbb{R}^3$, Frustum Modeling |
| **1:00 - 1:40** | Shreyansh Raj | Point to Depth/Disparity | Scalar field $Z = 1/d$, Guided Filter local covariance |
| **1:40 - 2:20** | Shaurya Singh | Open Math Inspector | Intrinsic $K$, Inverse $K^{-1}$, $SO(3)$ Rotation, $t = (I-R)c$ |
| **2:20 - 3:00** | Sarthak Bhowmik | Open Layer Slices | MPI Disparity Spacing, Homography $H_k$, Porter-Duff Over |
| **3:00 - 4:30** | Team | Move mouse & test presets | 60 FPS Canvas Parallax, Bilinear Lerp, Live KaTeX Matrices |
| **4:30 - End** | All | Answer questions | Orthogonality, Inverse Homography Warping vs Splatting |
