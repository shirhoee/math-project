# 05 CLEANUP RECOMMENDATIONS

This document provides suggestions for cleaning up the project directory to make it submission/demo-ready. **No files have been modified or deleted.**

### 1. Consolidate Documentation (Archive/Merge)
There are multiple iteration reports that clutter the root directory.
- **Merge/Archive**:
  - `UPDATE_REPORT.md`
  - `UPDATE_REPORT_2.md`
  - `UPDATE_REPORT_3.md`
  - `UPDATE_REPORT_4.md`
  - `UPDATE_REPORT_5.md`
  - `PROGRESS.md`
  - `AI_RULES.md`
  *Recommendation*: Move all of these to a `docs/archive/` folder, or synthesize their conclusions into a single `docs/DEVELOPMENT_JOURNAL.md`. They are no longer needed for the final submission.

### 2. Remove Temporary and Obsolete Files
- **Delete root temp images**:
  - `temp_demo.jpg`
  - `temp_quality.jpg`
  - `temp_resized.jpg`
  - `temp_resized_app.jpg`
  - `temp_resized_precompute.jpg`
  - `temp_test.jpg`
  - `dummy.jpg`
  - `debug_layer0.png`
  - `Dramatic_center_near.png`, `Normal_center_near.png`, etc.
  *Recommendation*: Update scripts (like `app.py` and `quality_check.py`) to use the Python `tempfile` module or a dedicated `temp/` directory that is `.gitignore`d, rather than writing to the project root.

### 3. Move Loose Scripts
- `diag_fill.py`, `tune_settings.py`
  *Recommendation*: Move these into the `tools/` directory, as they appear to be utility/diagnostic scripts rather than core application files.

### 4. Code Cleanup
- **`depth_estimator.py`**: Remove the dead code and inline thought process in `box_filter` (the unused `get_sum` function).
- **`math_engine.py`**: Remove `print(f"Hole filling cap reached...")` or replace it with a proper `logging.warning`.
- **`tests/test_app_smoke.py`**: Fix the failing test by replacing the search for `"Loaded: {sample_name}"` with a check for a UI element that actually exists in `app.py`.

### Proposed Clean Structure
```text
Math project/
├── app.py
├── depth_estimator.py
├── math_engine.py
├── mpi_renderer.py
├── view_atlas.py
├── README.md
├── requirements.txt
├── pytest.ini
├── assets/
│   └── samples/
├── docs/
│   ├── MATH_DERIVATIONS.md
│   ├── DEMO_SCRIPT.md
│   └── results/
├── tests/
└── tools/
    ├── benchmark.py
    ├── quality_check.py
    ├── tune_settings.py
    └── ... (other utilities)
```
