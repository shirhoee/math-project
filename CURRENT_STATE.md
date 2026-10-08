# Current State of the Codebase (v1)

## Architecture & Tech Stack
The `v1` codebase is a prototype mathematical computer vision application that estimates depth from 2D images and synthesizes novel 3D views. It operates entirely as a server-side rendered application.

- **Frontend & App State**: Built entirely in Streamlit (`v1/app.py`), which handles the UI, file uploads, and state management. 
- **Depth Estimation**: PyTorch and HuggingFace pipelines (`v1/depth_estimator.py`) running the `Depth-Anything-V2-Small-hf` model.
- **Rendering Engines**: 
  - `v1/mpi_renderer.py`: A Multi-Plane Image (MPI) based renderer.
  - `v1/math_engine.py`: A legacy point-cloud splatting renderer (used as a fallback/alternative).
- **Precomputation**: `v1/view_atlas.py` precomputes multiple view angles to serve to the UI.
- **Testing**: `pytest` (`v1/tests/`) and a suite of standalone diagnostic scripts (`v1/tools/`).

### How it Works
1. The user uploads an image via the Streamlit interface.
2. `app.py` orchestrates the pipeline, calling `depth_estimator.py` to extract a relative disparity map.
3. The depth map is used by `mpi_renderer.py` (or `math_engine.py`) alongside camera intrinsics to apply homography warping.
4. `view_atlas.py` pre-renders a grid of views (yaw and pitch variations).
5. The Streamlit UI displays these generated views dynamically using custom HTML/JS embedded within Streamlit to track mouse movement.

## Directory Tree

```
E:\Math project\v1
|   app.py                     # Main Streamlit application
|   depth_estimator.py         # PyTorch depth pipeline
|   math_engine.py             # Legacy point rendering
|   mpi_renderer.py            # Active MPI rendering
|   view_atlas.py              # Pre-rendering logic
|   requirements.txt
|   pytest.ini
|   [... numerous temp_*.jpg, UPDATE_REPORT*.md, and loose .png files]
|
+---assets/samples/            # Uploaded images and cache
+---docs/                      # Markdown documentation and derivations
+---PROJECT_AUDIT/             # Prior audit reports and handoffs
+---tests/                     # Pytest suite and fixtures
\---tools/                     # Diagnostics, benchmarking, and quality scripts
```

## Areas of Technical Debt
- **Root Directory Clutter**: The root of `v1` is severely polluted with temporary generation artifacts (`temp_resized.jpg`, `dummy.jpg`, `debug_layer0.png`) and deprecated status reports (`UPDATE_REPORT*.md`). 
- **Monolithic God Objects**: `app.py` handles Streamlit UI drawing, cache management, file I/O, base64 encoding, and orchestrates the complex rendering loop. It is difficult to read and maintain.
- **Duplicated Render Paths**: Having both `math_engine.py` and `mpi_renderer.py` introduces cognitive load and testing complexity.
- **Tight Coupling to UI Framework**: The core 3D generation logic is heavily entangled with Streamlit's `@st.cache_data` decorators and procedural execution flow, making it nearly impossible to expose as a standalone API.
- **Missing Input Validation & Types**: Heavy reliance on loosely typed NumPy arrays and missing validation for image uploads.
