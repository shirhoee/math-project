Requirement | Status (done / partial / missing) | Evidence (file, function, test name)
--- | --- | ---
Stage 1: Pyramid in preview | done | app.py (fill_holes_pyramid), benchmark.py (stride 2 benchmark)
Stage 2: Fix hole metrics | done | tools/hole_metrics.py (get_max_horizontal_run, get_max_thickness, ablation)
Stage 3: Demote edge masked | done | app.py, math_engine.py, test_edge_mode.py
Stage 4: Auto-crop off-centre | done | app.py, run_autocrop_check.py
Stage 5: Evidence | done | test_app_headless.py
