> [!WARNING]
> ONLY V1: We are not working on v2 from now. All development is strictly restricted to v1.

# Refactoring PRD: Stabilization (v1 Codebase)

## Overview
This PRD outlines the stabilization strategy exclusively for the `v1` codebase. The architecture is currently a monolithic Streamlit application with heavy root-directory clutter and tightly coupled rendering logic. The goal is to enforce strict engineering standards, decouple the UI from the math engine, and prepare the codebase for future feature additions.

## Phases of Work

### Phase 1: Linting & Type Safety
*Enforcing strict rules to prevent runtime errors and ensure code clarity.*
- **Static Analysis**: Introduce `mypy` to enforce strong typing across all core logic files (`depth_estimator.py`, `math_engine.py`, `mpi_renderer.py`).
- **Formatting**: Implement a strict formatter (e.g., `black` or `ruff`) to standardize the currently inconsistent procedural code styles.
- **NumPy Types**: Annotate NumPy array inputs/outputs with explicit shapes and types (e.g., using `nptyping` or detailed docstrings) to prevent dimensionality mismatch bugs during matrix multiplications.

### Phase 2: File Restructuring
*Moving files into a logical folder structure to reduce clutter.*
- **Artifact Isolation**: Modify the codebase to write all `temp_*.jpg` files, debug outputs (`debug_layer0.png`), and caches into a dedicated, `.gitignore`d `outputs/` or `temp/` directory.
- **Documentation Cleanup**: Archive all `UPDATE_REPORT*.md` files and older `PROGRESS.md` files into an `archive/` or `docs/history/` folder.
- **Core Package**: Move the math and rendering scripts (`math_engine.py`, `mpi_renderer.py`, `view_atlas.py`, `depth_estimator.py`) into a dedicated `src/` module to separate them from the root application runner.

### Phase 3: Component/Logic Extraction
*Breaking down massive files into smaller, single-responsibility modules.*
- **Decouple Streamlit**: Extract all non-UI logic from `app.py`. The Streamlit app should only contain `st.*` calls, importing the heavy lifting from a newly formed `controller` or `pipeline` module.
- **Standardize Rendering API**: Create a unified interface or base class for both `math_engine.py` and `mpi_renderer.py` to prevent logic duplication and simplify testing. 
- **Tooling Consolidation**: Review the `tools/` directory (which currently contains 20+ scripts) and consolidate redundant benchmarking and diagnostic scripts into a cohesive CLI (e.g., using `argparse` or `click`).

### Phase 4: Feature X
*The actual new feature to build once the foundation is stable.*
- **Web API Exposure (Headless Mode)**:
  - Once the Streamlit UI is decoupled, expose the `mpi_renderer` pipeline via a lightweight REST API (e.g., FastAPI).
  - This allows the `v1` engine to act as a standalone microservice that accepts an image and returns precomputed 3D views, enabling future frontends to consume the data without relying on Streamlit.


### Phase 5: Robust Async Task Queue (Celery/Redis)
*Productionizing the backend by replacing raw threading with a distributed task queue.*
- **Celery Integration**: Extract rendering pipeline into Celery tasks backed by Redis with graceful local in-memory fallback.
- **State Management**: Refactor status polling to use Celery AsyncResult instead of global memory dicts.

### Phase 6: Standalone Modern Web Frontend (Vite + React + Tailwind CSS)
*Replacing fragile Streamlit UI with a decoupled, high-performance web client.*
- **Decoupled Architecture**: Scaffolding a Vite + React + Tailwind CSS application in `v1/frontend/` that communicates with the backend exclusively via FastAPI REST endpoints.
- **60 FPS Parallax Canvas**: Implementing a native HTML5 `<canvas>` viewport with `requestAnimationFrame`, spring/lerp cursor smoothing, asynchronous batch frame decoding (`createImageBitmap`), and bilinear pitch/yaw cross-fading.
- **Robust Drag-and-Drop Upload**: Direct client-side preview via `URL.createObjectURL`, decoupling file selection from server job execution to eliminate upload glitches.
- **Interactive Math Inspector**: Live client-side camera matrix and homography derivations with formatted KaTeX LaTeX equations.
- **FastAPI Static Serving**: Mounting the compiled frontend build in `v1/src/api/server.py` so the API and UI are served together via a single command.


