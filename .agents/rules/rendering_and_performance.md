# Rendering & Performance Rules

## 1. Planar MPI Homographies & Distortion Boundaries
* **Angle Limits:** Single-image homographies must clamp novel view angles to `max_yaw <= 8.0°` and `max_pitch <= 3.0°`. Large angles (>15°) cause severe slice shearing and ghosting on foreground objects.
* **Motion Baseline:** Keep `motion_preset <= 0.05` for clean, solid 3D depth without parallax tearing.

## 2. Performance & Latency Constraints
* **Vectorized Warping:** Never iterate over pixel coordinates in pure Python for novel view warping. Use `cv2.remap` or `cv2.warpPerspective` with `INTER_LINEAR` which executes in C++/SIMD in milliseconds.
* **Filter Operations:** Prefer `cv2.boxFilter` over pure NumPy cumulative sum / meshgrid for edge alignment in guided depth filters.
* **Model Singleton Caching:** Never re-instantiate PyTorch / HuggingFace `pipeline(...)` or reload model checkpoints inside request handlers or task functions. Always retrieve from a module-level cache (`_DEPTH_MODEL_CACHE`).

## 3. Asynchronous Job & Progress Reporting
* **Dynamic Progress Reporting:** Background workers must write granular progress percentages to a shared job status registry (`task_progress_registry`). Polling endpoints must never return a hardcoded static progress value (e.g. 50%) during thread execution.
