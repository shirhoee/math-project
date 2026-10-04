# UI Changes
1. **Interactive Viewer:** A responsive JS-based 3D viewer embedding base64 frames (no server roundtrips on mouse move). Code in `app.py`.
2. **Atlas Generation:** `view_atlas.py` precomputes 45 views safely (with hole checks).
3. **Redesigned Tabs:** 'How it works' (formulas/pipeline), 'Point Cloud' (external views), and 'Math' (live matrices). Code in `app.py`.
4. **GIF Export:** Server-side orbit GIF assembly via Pillow in `app.py`.
5. **No Debug Text:** Main page is clean; diagnostics are grouped into an expander.
6. **Per-Image Safety:** Automatic angle shrinking implemented in `app.py` if hole fraction exceeds 8%.
