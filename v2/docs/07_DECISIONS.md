# 07 Decisions

*Logs architectural and mathematical decisions made during the rebuild.*

- **D001:** Rebuild with FastAPI and static frontend instead of Streamlit to eliminate rerun overhead and base64 latency.
- **D002:** Stick to pure NumPy for geometry to fulfill constraints and ensure no hidden C-level library issues.
- **D003:** Use a single mesh with disconnected triangles at depth edges instead of Multi-Plane Images (MPI) to reduce overdraw and simplify the frontend renderer.
