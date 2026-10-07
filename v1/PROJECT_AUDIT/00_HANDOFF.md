# 00 HANDOFF: Interactive 3D View Synthesis

### Project Summary
**Goal**: To transform a single flat 2D photograph into an interactive 3D scene using geometric computer vision and linear algebra, avoiding external 3D engines like OpenGL.

This project uses a neural network to estimate image depth, and then relies entirely on custom NumPy mathematical engines to unproject pixels into 3D, apply geometric camera transformations, and re-project them back to 2D. The app pre-renders 123 camera angles (41 yaw $\times$ 3 pitch) and serves them to a Streamlit frontend where an HTML5 Canvas allows the user to look around the scene in real-time at 60fps via mouse movement.

### Context & Deliverables
This appears to be a mathematics/linear-algebra course project demonstrating applied transformations. The main deliverable is the interactive web application (`app.py`), supported by rigorous mathematical derivations mapping code to theory (`MATH_DERIVATIONS.md`). 

### Mathematical Foundation
The project leans heavily on standard computer vision geometry:
1. **Camera Intrinsics**: $K$ matrix built from Field of View.
2. **Depth Conversion**: $Z = 1 / (d/Z_{near} + (1-d)/Z_{far})$.
3. **Rigid Body Transformations**: Euler rotation matrices ($R$) and translation vectors ($t$), applied relative to a pivot point $c$: $P' = (P-c)R^T + c + t$.
4. **Multi-Plane Image (MPI) Homographies**: Warping flat depth layers using planar homographies: $H = K(R + \frac{t e_z^T}{Z})K^{-1}$.
5. **Adjugate Matrix Inversion**: $3\times3$ inverses computed manually via adjugates.

### Architecture & Pipeline
1. **Depth Estimation**: `depth_estimator.py` runs "Depth Anything V2" (via HuggingFace) to extract relative disparity, then applies a box-filter based guided filter to snap depth edges to image boundaries.
2. **Rendering Engine**: 
   - *Original (Point Splatting)*: `math_engine.py` unprojects to a 3D point cloud, rotates/translates, and splats back to 2D with Z-buffering.
   - *Current (MPI Layers)*: `mpi_renderer.py` slices the image into $N$ RGBA layers. Hidden disoccluded regions are inpainted using a background-biased pyramid fill. Layers are warped via planar homographies and composited back-to-front.
3. **Parallel Generation**: `concurrent.futures` renders the 123 views in parallel.
4. **Frontend**: `app.py` (Streamlit) embeds base64 images into a custom JS canvas script that interpolates between the 4 nearest frames based on mouse $(x, y)$ coordinates.

### Tech Stack & How to Run
- **Python**: `numpy`, `pillow`, `streamlit`, `torch`, `transformers`, `opencv-python`.
- **Run the app**:
  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```
- **Run the tests**:
  ```bash
  pytest tests/
  ```
- **Run the Quality Check**:
  ```bash
  python tools/quality_check.py
  ```

### Current State
| Component | Status | Evidence | Notes |
|---|---|---|---|
| Depth Estimation | COMPLETE | `depth_estimator.py` / `quality_check.py` | Accurate and edge-aligned. |
| MPI Layer Renderer | COMPLETE | `mpi_renderer.py` / `quality_check.py` | High quality, fixes occlusion holes. |
| Point Renderer | COMPLETE | `math_engine.py` / `quality_check.py` | Available as a fallback toggle in UI. |
| Streamlit UI | PARTIAL | `app.py` runs, but `tests/test_app_smoke.py` fails. | The UI works, but a test asserts an outdated UI element. |
| Mathematics | COMPLETE | `MATH_DERIVATIONS.md` | All formulas explicitly mapped to code. |

### Known Issues & Blockers
- **Failing Test**: `test_app_smoke.py::test_app_loads` fails because it waits for a markdown string (`Loaded: 01_sample_1.jpg`) that was removed from `app.py` during a UI refactor.
- **Root Clutter**: The root directory is flooded with temporary images (`temp_resized.jpg`, `dummy.jpg`, etc.) and old iteration reports (`UPDATE_REPORT_*.md`), making it hard to navigate.

### Remaining Work (Prioritised)
1. **Fix Tests**: Update `test_app_smoke.py` to assert a valid UI element to restore a 100% passing test suite.
2. **Cleanup Root Directory**: Move temp files generation into a dedicated folder or use `tempfile`, and archive old `UPDATE_REPORT`s (see `05_CLEANUP_RECOMMENDATIONS.md`).
3. **Clean Code**: Remove unused `get_sum` function in `depth_estimator.py` and print statements in `math_engine.py`.

### Directory Map
- **Core Files**: `app.py`, `math_engine.py`, `mpi_renderer.py`, `depth_estimator.py`, `view_atlas.py`.
- **Documentation**: `README.md`, `MATH_DERIVATIONS.md`, `docs/`.
- **To Ignore/Archive**: `UPDATE_REPORT*.md`, `PROGRESS.md`, `temp_*.jpg`, `debug_layer0.png`.

### Open Questions
- Is `test_app_smoke.py` the only test that needs updating? The rest (41 tests) pass, but verify if they cover the new MPI renderer adequately.
