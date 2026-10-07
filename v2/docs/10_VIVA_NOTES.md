# 10 Viva Notes

## Core Concepts & Math
- **Intrinsics (K):** Computed using $f_x = (W/2) / \tan(FOV/2)$. Maps 3D points $(X, Y, Z)$ to 2D image coordinates $(u, v)$ via perspective division.
- **Extrinsics (R, t):** `R` is a $3 \times 3$ rotation matrix built from Euler angles. `t` is the translation vector. They transform the world points into the camera's local coordinate system: $P_{new} = P R^T + t$.
- **Adjugate Inverse:** We implemented a custom $3 \times 3$ matrix inverse using the adjugate method: $M^{-1} = \text{adj}(M) / \det(M)$. This perfectly matches `np.linalg.inv` and removes the need for complex linear algebra solvers.

## Architecture Choices
- **FastAPI + Static Web over Streamlit:** V1 used Streamlit, which re-ran the entire script on every interaction, causing massive overhead and forcing us to pre-render 123 frames and send heavy Base64 strings. V2 serves a lightweight JSON payload of binary-encoded Float32Arrays to a static HTML/JS/Three.js frontend, resulting in zero-latency rendering at 60 FPS.
- **Pure NumPy Geometry:** All unprojection and triangle filtering is heavily vectorised in NumPy. We construct the grid, map depth to Z, unproject, and filter discontinuities entirely in matrix operations without loops.

## Performance Optimizations
- **Caching:** Depth estimation is cached to disk using a SHA-256 hash of the image and settings, dropping load times from ~5s to <0.1s.
- **Thread Limits:** Restricted CPU threading (`OMP_NUM_THREADS=1` and `torch.set_num_threads(1)`) prevents process explosion when running the PyTorch model on a CPU.
- **Binary Payloads:** Vertex, index, and UV data are Base64 encoded as raw bytes instead of JSON text arrays, massively reducing network payload size and parsing time.

## Challenges & Solutions
- **Stretching at Edges:** When unprojecting depth, sharp depth jumps (e.g., a foreground object over a distant background) create "rubber-band" triangles. We solved this by measuring the max depth difference across the three vertices of every triangle and dropping any triangle where $\Delta Z > 10\%$ of the max depth.
- **Smearing at Camera Limits:** We clamped the OrbitControls in Three.js (e.g., azimuth between -30° and 30°) to prevent the user from viewing the scene from extreme angles where the 2.5D approximation breaks down.
