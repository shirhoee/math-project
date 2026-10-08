---
trigger: model_decision
description: Apply these guidelines whenever working on the backend, API routes, database schemas, or server-side logic.
---


## Backend Architecture & Patterns

### 1. API Design & Typing
* **Strict Typing:** All Python functions MUST have type hints (`def func(a: int) -> str:`). Use `mypy` to enforce typing.
* **NumPy Arrays:** Document the shape and dtype of NumPy arrays and PyTorch tensors in docstrings (or use `nptyping`). E.g., `# Returns: np.ndarray shape (H, W, 3) dtype float32`.
* **Headless Design:** Ensure the core logic (MPI renderer, Math engine) can run completely headlessly without Streamlit.

### 2. Math & Rendering
* **Modularization:** Break down massive scripts (`mpi_renderer.py`, `depth_estimator.py`) into single-responsibility functions/classes.
* **Avoid Duplication:** If logic exists in `math_engine.py` and `mpi_renderer.py`, abstract the shared logic (like homography computation or intrinsics) into a separate `geometry.py` module.
* **Vectorization:** Use NumPy/PyTorch vectorization instead of loops for any pixel-wise operations.

### 3. File System & Caching
* **Artifact Management:** NEVER write temporary files (`temp_*.jpg`) to the root directory. Use a dedicated `outputs/` or Python's `tempfile` module.
* **Pickle Caches:** When caching PyTorch inferences to disk using `pickle`, ensure paths are isolated to `assets/cache` and handle cache invalidation safely.