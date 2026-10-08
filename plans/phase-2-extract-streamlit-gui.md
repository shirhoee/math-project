# Phase 2: Extracting the Streamlit GUI

## Feature Description & User Stories
**Description**: The `v1` codebase currently mixes Streamlit UI rendering (`st.*` calls) directly with heavy orchestration logic, background threading, and file I/O within a monolithic `app.py`. Additionally, the root directory is cluttered with core math scripts and temporary artifacts. This phase will decouple the Streamlit interface from the underlying execution pipeline, reorganize the core packages into a dedicated `src/` directory, and isolate all temporary artifacts into an `outputs/` folder.
**User Stories**:
- As a developer, I want all core math and rendering scripts encapsulated within a `src/` module so that the project structure is clean and navigable.
- As a developer, I want all temporary files (caches, test outputs) routed to an `outputs/` directory so they don't pollute version control or the working directory.
- As a developer, I want the Streamlit UI to be completely decoupled from the rendering pipeline so that the core pipeline can be tested headlessly and eventually exposed via a REST API.

## Problem & Solution Statement
**Problem**: `app.py` handles user inputs, file uploads, background thread management (`threading.Thread`, `add_script_run_ctx`), and caching simultaneously. Core logic files sit in the root `v1/` directory alongside UI code and scattered scripts.
**Solution**: 
1. **File Restructuring**: We will create `v1/src/core/` and move `math_engine.py`, `mpi_renderer.py`, `depth_estimator.py`, and `view_atlas.py` into it. We will update all import paths across the codebase and tests.
2. **Artifact Isolation**: We will create an `outputs/` directory for temp caches and artifacts, updating the relevant hardcoded paths.
3. **Logic Extraction**: We will create `v1/src/pipeline.py` to encapsulate the orchestration logic (`run_build_job`, thread management, global state). `app.py` will be stripped down to pure Streamlit layout logic, communicating with `pipeline.py` via a simple API (e.g., `start_job()`, `get_status()`).

## CONTEXT REFERENCES
- `PRD.md` (Phases 2 and 3 regarding restructuring and decoupling)
- `v1/app.py` (Current monolithic orchestrator)
- `v1/tests/` (Test suite to ensure imports remain valid)

## New Files to Create
**Components**: None
**API**: None
**Utils**: 
- `v1/src/__init__.py`
- `v1/src/core/__init__.py`
- `v1/src/pipeline.py`
**Tests**: None

## STEP-BY-STEP TASKS

1. **Task 1: Setup Directory Structure & Artifact Isolation**
   - Create `v1/src/core/` and `v1/outputs/`.
   - Add `.gitignore` to `v1/outputs/` to ignore all contents (except `.gitkeep`).
   - Modify any existing logic in `v1/depth_estimator.py` or `v1/app.py` that writes to `temp_cache` or root `cache` to use `outputs/cache/`.
   - **VALIDATE**: `mkdir -p v1/src/core v1/outputs && echo "*" > v1/outputs/.gitignore && dir v1\src`

2. **Task 2: Move Core Packages**
   - Move `depth_estimator.py`, `math_engine.py`, `mpi_renderer.py`, and `view_atlas.py` into `v1/src/core/`.
   - Update all import statements inside `v1/app.py`, `v1/tests/*.py`, and the `tools/` scripts to reflect the new `src.core` module paths.
   - **VALIDATE**: `pytest v1/tests/`

3. **Task 3: Extract the Pipeline Controller**
   - Create `v1/src/pipeline.py`.
   - Migrate `run_build_job`, `global_job_status`, `global_job_results`, and the threading invocation logic (`threading.Thread`, `add_script_run_ctx`) out of `app.py` and into `pipeline.py`.
   - Expose a clean interface like `submit_render_job(...)` and `poll_job_status(jid)` in `pipeline.py`.
   - **VALIDATE**: `mypy v1/src/pipeline.py`

4. **Task 4: Refactor Streamlit GUI (`app.py`)**
   - Refactor `app.py` to remove all background threading and core module imports. 
   - Have `app.py` solely import from `src.pipeline` to submit jobs and retrieve results for rendering.
   - **VALIDATE**: `mypy v1/app.py`

5. **Task 5: Format & Lint New Structure**
   - Run `ruff` against the newly structured `v1/` codebase to ensure import sorting and formatting remain compliant.
   - **VALIDATE**: `ruff format --check v1/ && ruff check v1/`

6. **Task 6: Full Regression Check**
   - Run the complete E2E test suite to verify that the extracted pipeline and restructured imports function exactly as they did before.
   - **VALIDATE**: `pytest v1/tests/`

## TESTING STRATEGY
- **Unit Tests**: Existing tests in `v1/tests/` will inherently test the new imports and structure.
- **E2E Tests**: The Streamlit integration tests (`test_app_smoke.py`) will verify that `app.py` correctly delegates to `pipeline.py`.
- **Static Analysis**: `mypy` and `ruff` will validate the integrity of the abstracted module boundaries.

## ACCEPTANCE CRITERIA
- [x] `v1/src/core/` contains the engine files (`math_engine.py`, `mpi_renderer.py`, etc.).
- [x] `v1/outputs/` is used for all temporary artifacts and caches.
- [x] `v1/app.py` contains strictly Streamlit UI logic and zero `threading.Thread` or heavy orchestration code.
- [x] `v1/src/pipeline.py` acts as the headless orchestrator managing jobs.
- [x] Running `pytest v1/tests/` passes completely with zero import errors.
