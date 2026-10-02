# Project Progress Tracker

## Phase 1: Environment & AI Sensor Setup
- [x] Create virtual environment and install dependencies (`numpy`, `torch`, `transformers`, `streamlit`, `pillow`).
- [x] Implement `depth_estimator.py` to load an image and return a raw depth array using **Depth Anything V2** (superior to MiDaS for sharp edges).
- [x] Test with a sample image and visualize the raw depth map.
- [x] *Git Commit & Push*

## Phase 2: The Core Math Engine (Unprojection)
- [x] Implement Camera Intrinsic Matrix setup.
- [x] Write `unproject_to_3d()` using vectorized NumPy operations to map 2D + Depth to a 3D Point Cloud.
- [x] *Git Commit & Push*

## Phase 3: The Rotation Engine (Orthogonal Matrices)
- [ ] Write functions to generate $3 \times 3$ rotation matrices for X (Pitch), Y (Yaw), and Z (Roll).
- [ ] Implement `apply_transform()` to multiply the point cloud by the rotation matrices.
- [ ] *Git Commit & Push*

## Phase 4: The Projection Engine (3D to 2D)
- [ ] Implement `project_to_2d()` to map rotated 3D coordinates back to 2D pixel coordinates.
- [ ] Handle Z-buffering (ensuring points closer to the camera are drawn on top of points further away).
- [ ] Map original RGB colors to the new 2D canvas.
- [ ] *Git Commit & Push*

## Phase 5: Web UI (Streamlit)
- [ ] Build basic Streamlit layout (File Upload, Image Display).
- [ ] Add sliders for Pitch, Yaw, and Roll.
- [ ] Connect sliders to the Math Engine and optimize for real-time rendering.
- [ ] *Git Commit & Push*

## Phase 6: Polish & Documentation
- [ ] Refine handling of "holes" (occlusions) where possible.
- [ ] Finalize code comments with mathematical formulas.
- [ ] Complete README with usage instructions.
