# 06 Testing and Quality

**Test Strategy:**
- Unit tests for all mathematical functions (`tests/test_geometry.py`).
- Integration tests for depth pipeline and mesh generation.
- FastAPI endpoint tests using `TestClient`.

**Quality Gates:**
1. **Performance Gate:** Uncached model load + inference < 15s. Cached < 3s.
2. **Math Gate:** `adjugate_inverse` must match `np.linalg.inv` within `1e-9`.
3. **Mesh Gate:** Output must not contain triangles spanning a depth difference $> 10\%$ of max depth (prevents stretching).
4. **Viewer Gate:** At identity camera pose, the rendered view must perfectly match the original image (pixel difference < 5%).

**Running Tests:**
`pytest tests/`

**Running Quality Checks:**
Custom scripts in `tests/` will measure timings and mesh validity.
