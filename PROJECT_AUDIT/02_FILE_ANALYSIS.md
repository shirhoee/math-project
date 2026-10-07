# 02 FILE ANALYSIS

This document provides a deep analysis of meaningful source files, notebooks, and documents, establishing what they do, their dependencies, and any issues/TODOs observed.

### Core Source Files

#### `app.py`
- **Purpose**: Main entry point for the Streamlit web application. It orchestrates user inputs, depth estimation, point unprojection, and rendering (either original Point Splatting or new MPI Layers).
- **Key Functions/Classes**:
  - `start_job`, `run_build_job`: Asynchronously processes images, handles caching, estimates depth, and renders views.
  - Generates HTML/JS snippet for an interactive 3D viewer on the frontend using pre-rendered frames.
- **Dependencies**: `streamlit`, `numpy`, `PIL`, `depth_estimator`, `math_engine`, `view_atlas`, `mpi_renderer`, standard library tools.
- **Referenced by**: Entry point (not imported by others).
- **Issues/Debt**:
  - Hardcoded paths to `assets/samples/cache`.
  - Stub code/workarounds: Uses a threading hack (`add_script_run_ctx`) to run background generation in Streamlit.
  - Large JS blob inside Python code string.

#### `math_engine.py`
- **Purpose**: The core mathematical graphics engine for 3D View Synthesis. Strictly uses NumPy linear algebra operations.
- **Key Functions/Classes**:
  - `TransformEngine`: Main class.
  - `disparity_to_depth`: Converts disparity to depth $Z$.
  - `unproject_to_3d`: Maps 2D pixels and Z-depth to 3D point cloud using inverse intrinsics ($K^{-1}$).
  - `get_rotation_matrix`: Generates 3x3 rotation matrices from Euler angles.
  - `apply_transform`: Applies rotation and translation.
  - `project_to_2d`, `project_orthographic`: Projects 3D points back to 2D canvas with Z-buffering and splatting.
  - `_render_points`: Vectorized sub-routine for Z-buffering, replacing lexsort with stable single-key argsort.
  - `fill_holes_iterative`, `fill_holes_pyramid`: Inpainting for holes caused by disocclusions.
- **Dependencies**: `numpy`.
- **Referenced by**: `app.py`, `mpi_renderer.py`, `tools/quality_check.py`, tests.
- **Issues/Debt**:
  - `fill_holes_iterative` prints directly to stdout ("Hole filling cap reached").

#### `mpi_renderer.py`
- **Purpose**: Implements Multi-Plane Image (MPI) representation and rendering as a V2 engine replacement for point clouds.
- **Key Functions/Classes**:
  - `build_layers`: Slices image into $N$ layers based on depth. Fills hidden regions using pyramid fill.
  - `inv3x3`: Computes inverse of 3x3 matrices using adjugate matrix formula.
  - `layer_homographies`: Computes planar homographies for each layer.
  - `bicubic_sample`, `bilinear_sample`: Custom image sampling.
  - `render_mpi`: Renders final image from homographies.
  - `render_atlas_mpi`: Renders a grid of views in parallel using `concurrent.futures`.
  - `calibrate_motion`: Solves for the camera baseline needed to achieve a target parallax ratio.
- **Dependencies**: `numpy`, `PIL`, `depth_estimator`, `math_engine`, `concurrent.futures`.
- **Referenced by**: `app.py`, `tools/quality_check.py`.
- **Issues/Debt**:
  - Some logic (like the orbit computation in `render_atlas_mpi`) has inline comments discussing the evolution of the prompt/equations.

#### `depth_estimator.py`
- **Purpose**: Wrapper around "Depth Anything V2" model to estimate disparity from 2D images. Also includes guided filter logic.
- **Key Functions/Classes**:
  - `DepthEstimator`: Initializes the HuggingFace pipeline and processes images.
  - `estimate_depth`: Resizes, runs model, normalizes via percentiles, and refines with guided filter.
  - `box_filter`, `guided_filter`: Mathematical filters for edge-aligned depth refinement.
- **Dependencies**: `torch`, `transformers`, `PIL`, `numpy`, `cv2` (via other scripts?).
- **Referenced by**: `app.py`, `mpi_renderer.py`, tools.
- **Issues/Debt**:
  - "Stub code/commented-out blocks": `box_filter` has commented-out thought processes about 1-indexed cumulative sums and a `pass` block inside an unused inner function `get_sum`.

#### `README.md`
- **Purpose**: General project description, setup instructions, and high-level architectural overview.
- **Dependencies**: None.
- **Issues/Debt**: Mentions "V2 engine replaces point clouds with a layered Multi-Plane Image (MPI) renderer," indicating a pivot in architecture from V1.

#### `MATH_DERIVATIONS.md`
- **Purpose**: Records mathematical foundations, derivations, and maps them to code implementations.
- **Dependencies**: None.
- **Issues/Debt**: Explicitly mentions resolving a discrepancy between the project's codebase convention ($P' = R P + b$) and a report's formula.

### Observations [INFERRED]
- **Goal & Problem Statement**: The project aims to synthesize novel 3D views from a single 2D image by estimating depth, unprojecting to 3D (or slicing into MPI layers), applying transformations, and rendering them into an interactive atlas.
- **Architecture**:
  1. Depth Estimation (HF Transformers)
  2. Mathematical unprojection (Point cloud or MPI layers) (NumPy)
  3. Novel view rendering (Parallel processing)
  4. Frontend display (Streamlit + custom JS for 60fps interpolation)
- **Pivots**: The project initially used "Point Splatting" (V1) but pivoted to "Multi-Plane Image (MPI) Layers" (V2) for better rendering of disocclusions and speed. Evidence is scattered in `README.md` and `app.py` (which supports both renderers).
