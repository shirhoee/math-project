# Progress

## Phase 1: Vectorization — Complete
- Disparity-to-depth, unprojection, rotation, perspective projection all vectorized with NumPy.
- Tests: `tests/test_stage1.py`

## Phase 2: Splatting — Complete
- Adaptive fill-only splatting with single-key argsort (quantized Z-buffer).
- Tests: `tests/test_stage2.py`, `tests/test_stage3.py::test_splatting`

## Phase 3: Z-buffer & Edges — Complete
- Edge masking with gradient threshold (`edge_tau`), demote mode (default).
- Tests: `tests/test_edge_mode.py`, `tests/test_stage3.py`

## Phase 4: UI & Tuning — Complete
- Streamlit app with sliders, auto-crop, orthographic projection.
- `st.cache_data` for depth estimation and unprojection.
- Tests: `tests/test_app_headless.py`

## Phase 5: Polish & Generalization (Task 5) — Complete
- Pyramid filler in preview and HQ (0.00% remaining holes).
- Render pass order: exact → demoted → splats → pyramid fill.
- Identity difference: 0.000% with `demote` defaults.
- Preview speed: 53.5 ms mean (stride 2 + fill).
- HQ speed: 209.7 ms mean (stride 1 + fill).
- Multi-image smoke tests on 3 synthetic images.
- Tests: `tests/test_multi_image_smoke.py`
- Report: `UPDATE_REPORT_5.md`
