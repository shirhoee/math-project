# Project Progress Tracker

## Phase 1: Environment & AI Sensor Setup
- [x] Create virtual environment and install dependencies.
- [x] Implement `depth_estimator.py` using **Depth Anything V2**.
- [x] Test and extract normalized disparity.

## Bug-fix pass (Stage A)
- [x] Fix disparity to depth conversion to avoid Z=0 inversion.
- [x] FOV-based Intrinsic Matrix.
- [x] Pivot-based transformations for true parallax.

## Phase 2: The Core Math Engine (Unprojection)
- [x] Implement Camera Intrinsic Matrix setup.
- [x] Write `unproject_to_3d()` mapping 2D + Z to 3D.

## Phase 3: The Rotation Engine (Orthogonal Matrices)
- [x] Generate $3 \times 3$ rotation matrices for Pitch, Yaw, Roll.
- [x] Apply $P_{new} = (P - c) R^T + c + t$.

## Phase 4: The Projection Engine (3D to 2D)
- [x] Implement `project_to_2d()` with perspective divide.
- [x] Robust Z-buffering (quantized single-key sort).
- [x] Adaptive Fill-Only Splatting.
- [x] Edge-masking for smooth surfaces.
- [x] Orthographic projection toggle.

## Phase 5: Performance & Verification
- [x] Vectorization Audit confirms 0 forbidden loops.
- [x] 100% Pytest pass rate (20/20 tests).

## Phase 6: Final Integration & UI
- [x] Streamlit app with sliders, cache limits, and auto-cropping.
- [x] Speed target met: Stride-2 Preview renders at 142.9 ms. HQ Stride-1 renders at 217.2 ms.

## Documentation
- [x] Update MATH_DERIVATIONS.md.
- [x] Finalize README.md.
