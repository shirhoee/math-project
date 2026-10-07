# 04 STATE AND ISSUES

### Current State Classification

| Component | Status | Evidence | Notes |
|---|---|---|---|
| Depth Estimation | COMPLETE & WORKING | Verified in `app.py` and `depth_estimator.py`. Pipeline runs successfully using `Depth Anything V2`. | Fast and refines boundaries properly with guided filter. |
| Point Splatting (V1) | COMPLETE & WORKING | Verified by running `quality_check.py` and reading `app.py`. | Retained as a fallback renderer. |
| MPI Layers (V2) | COMPLETE & WORKING | Verified by running `quality_check.py`. Renders 123 views rapidly. | Solves disocclusion holes via pyramid fill. |
| Streamlit App (UI) | PARTIAL / BUGGY | `app.py` works but `test_app_smoke.py` fails on `test_app_loads`. | The test expects a `Loaded: <filename>` markdown string which no longer exists in `app.py`. |
| Interactive Viewer | COMPLETE & WORKING | Embedded JS handles 60fps interpolation inside `app.py`. | Mouse events are mapped to 41x3 grid index blending. |

### Execution Log & Evidence
- **Tests Execution**: `pytest tests/` ran 42 tests. 41 passed, 1 failed (`test_app_smoke.py::test_app_loads`).
  - *Traceback*: The test timed out after 30 seconds waiting for `Loaded: 01_sample_1.jpg` in the app's markdown. A code search confirmed `app.py` does not output this string.
- **Quality Check**: `python tools/quality_check.py` executed successfully.
  - *Output Snippet*: Evaluates synthetic sharpness and real image sharpness across renderers.
  - *Result*: "PASS: All quality gates met the target." showing both Point Splatting and Layers renderers work mathematically as intended.

### Known Bugs and Technical Debt
1. **Broken Test (`test_app_smoke.py`)**: As mentioned above, it asserts a UI element that no longer exists.
2. **Hardcoded Paths**:
   - `app.py` defaults to loading samples from `assets/samples/01_*.jpg`.
   - `quality_check.py` relies on `assets/samples/01_*.jpg`.
3. **Print Statements inside Core Logic**:
   - `math_engine.py` prints "Hole filling cap reached" directly to stdout, which pollutes the terminal when run as an API/Library.
4. **Unused / Commented Code**:
   - `depth_estimator.py` has a `box_filter` function containing inline comments and an unused `get_sum` function with a `pass` block, reflecting an aborted attempt at a 0-indexed integral image.

### Design Decisions & Pivots
- **Decision: Moving from Points to MPI Layers**:
  - *Evidence*: `UPDATE_REPORT_*.md` and `PROGRESS.md` detail the struggle with "black holes" in the 3D views. `math_engine.py` implements a complex `fill_holes_iterative`. `mpi_renderer.py` was introduced to fix this more robustly using `fill_holes_pyramid` and alpha blending over depth layers. `app.py` now supports both but defaults to Layers.

### What is Left to Do (Submission Ready)
- Fix the failing smoke test by updating the assertion to match `app.py`'s current UI state.
- Clean up dead code in `depth_estimator.py` and print statements in `math_engine.py`.
- Clear out the redundant `UPDATE_REPORT` files, keeping only the final one for the portfolio.
