# 07 Decisions

*Logs architectural and mathematical decisions made during the rebuild.*

- **D001:** Rebuild with FastAPI and static frontend instead of Streamlit to eliminate rerun overhead and base64 latency.
- **D002:** Stick to pure NumPy for geometry to fulfill constraints and ensure no hidden C-level library issues.
- **D003:** Use a single mesh with disconnected triangles at depth edges instead of Multi-Plane Images (MPI) to reduce overdraw and simplify the frontend renderer.
\n- **D004:** The v1 disparity_to_depth used a linear mapping: Z = z_near + (1-d)*(z_far-z_near). The docs spec and MATH_DERIVATIONS.md use the perspective-correct metric formula: Z = 1 / (d/z_near + (1-d)/z_far). We implemented the perspective-correct one.