---
trigger: model_decision
description: Apply these guidelines whenever working on the backend, API routes, database schemas, or server-side logic.
---


## Backend Architecture & Patterns

### 1. API Design & Typing
* **Strict Typing:** All Python functions MUST have type hints (`def func(a: int) -> str:`). Use `mypy` to enforce typing.
* **NumPy Arrays:** Document the shape and dtype of NumPy arrays and PyTorch tensors in docstrings (or use `nptyping`). E.g., `# Returns: np.ndarray shape (H, W, 3) dtype float32`.
* **Headless Design:** Ensure the core logic (MPI renderer, Math engine) can run completely headlessly without Streamlit.
* **FastAPI Microservice:** Expose decoupled rendering pipelines via FastAPI (`v1/src/api/server.py`) using strict Pydantic schemas (`v1/src/api/models.py`). Mount compiled frontend static assets via `StaticFiles(directory="frontend/dist", html=True)`.
* **CORS & Payload Optimization:** Always configure `CORSMiddleware` for frontend development. Avoid dumping raw float arrays into JSON; serialize only lightweight metadata, frame base64 strings, or compact data structures.

### 2. Math & Rendering
* **Modularization:** Break down massive scripts (`mpi_renderer.py`, `depth_estimator.py`) into single-responsibility functions/classes.
* **Avoid Duplication:** If logic exists in `math_engine.py` and `mpi_renderer.py`, abstract the shared logic (like homography computation or intrinsics) into a separate `geometry.py` module.
* **Vectorization:** Use NumPy/PyTorch vectorization instead of loops for any pixel-wise operations.

### 3. File System & Caching
* **Artifact Management:** NEVER write temporary files (`temp_*.jpg`) to the root directory. Use a dedicated `outputs/` or Python's `tempfile` module.
* **Pickle Caches:** When caching PyTorch inferences to disk using `pickle`, ensure paths are isolated to `assets/cache` and handle cache invalidation safely.

### 4. Safety & Concurrency
* **Graceful Task Queue Fallback:** Distributed task runners (Celery/Redis) must NEVER be hard, unhandled dependencies in core orchestration or client entrypoints. Always implement a graceful in-process fallback (e.g., `ThreadPoolExecutor` or direct execution) if Celery or Redis is unreachable or uninstalled.
* **Environment Independence:** Core modules must remain testable and runnable locally on standard environments without requiring external background daemons to boot.
* **PyTorch & Concurrency Testing:** When writing integration tests for PyTorch models, background inference threads must be cleanly managed or mocked out to prevent concurrency segmentation faults on Windows.
* **Variable Scoping:** Always ensure context managers (like `with open()`) use uniquely scoped variable names (e.g., `with open(...) as out_f:`) to prevent accidental variable shadowing of iterators or broader scoped variables.