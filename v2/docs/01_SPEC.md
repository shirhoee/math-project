# 01 Specification

**Goal:**
Build a robust, performant 3D viewer from a single image using depth estimation and vectorised NumPy geometry.

**60-Second Demo Script:**
1. Start the backend (`uvicorn src.server.main:app`).
2. Open the static web viewer (`index.html`).
3. Click "Load Sample 1". The model is cached; 3D view appears in under 3 seconds.
4. Drag to orbit. The camera pans smoothly in real-time.
5. Upload a new image. The depth model runs (under 15s on CPU). View updates.

**Functional Requirements:**
- Accept single image uploads.
- Estimate depth using a small, fast model (e.g. Depth Anything V2 Small).
- Generate a 3D representation (mesh).
- Serve the mesh and textures via FastAPI.
- A static frontend web viewer with orbit controls.

**Non-Functional Requirements & Targets:**
- Cached sample load to first view: under 3 seconds.
- Uncached on a CPU laptop: under 15 seconds (report actual).
- Camera drag: smooth interactive rate in the viewer.
- UI state: No blank screen at any time (loading indicators).

**Constraints:**
- No Streamlit.
- Geometry strictly vectorised NumPy (no open3d/scipy).
- No GPU required (must run acceptably on CPU).

**Non-Goals:**
- Perfectly accurate metric depth (relative depth is sufficient).
- Photorealistic view synthesis with complex inpainting.
- Multi-user concurrent load handling (local single-user focus).

**Success Criteria:**
- All quality gates and tests pass.
- Demo script runs flawlessly.
- Viewer shows smooth rotation without severe tearing or crashes.
