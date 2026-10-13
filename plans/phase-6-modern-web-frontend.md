# Phase 6: Standalone Modern Web Frontend (Vite + React + Tailwind CSS)

## Feature Description & User Stories
**Description**:
Replace the unstable, script-reloading Streamlit UI with a high-performance, decoupled Single Page Application (SPA) built using Vite, React, TypeScript, and Tailwind CSS. The new web client communicates with the backend exclusively via FastAPI REST endpoints and is served directly by FastAPI as compiled static assets. It features a native HTML5 Canvas parallax viewport operating at 60–120 FPS, drag-and-drop custom image uploading with instant 0ms preview, non-blocking asynchronous progress polling, and interactive mathematical matrix inspectors with KaTeX formatting.

**User Stories**:
- As a user, I want to drag-and-drop any custom image and see an immediate preview without page reloads or UI resets.
- As a user, I want to move my mouse over synthesized 3D views with buttery smooth, zero-jitter parallax tracking and automatic auto-wiggle.
- As a user, I want real-time progress updates during depth estimation and layer generation without UI lockups or crashes.
- As an educational user, I want to inspect camera rotation matrices, homographies, and MPI depth slices interactively using live LaTeX math.
- As a developer, I want a single command (`uvicorn src.api.server:app`) to launch both the backend API and the interactive frontend on port 8000.

## Problem & Solution Statement
**Problem**:
Streamlit's execution model re-runs the entire script upon every user interaction, causing custom file uploads to fail or drop state. Furthermore, embedding HTML5 canvases inside sandboxed Streamlit iframes creates severe input lag, while direct imports of backend worker packages (such as Celery) in the UI code cause fatal `ModuleNotFoundError` crashes when dependencies or workers are missing.
**Solution**:
1. Decouple the frontend entirely into a modern client located in `v1/frontend/` communicating over standard HTTP REST (`/api/v1/render`, `/api/v1/jobs/{id}/status`, `/api/v1/jobs/{id}/result`).
2. Implement a high-performance HTML5 `<canvas>` viewport using `requestAnimationFrame`, asynchronous batch decoding with `createImageBitmap`, spring-damper (`lerp`) cursor smoothing, and bilinear pitch-yaw frame blending.
3. Optimize FastAPI endpoints in `v1/src/api/server.py` to support CORS during local development and mount the production frontend build (`v1/frontend/dist`) via `StaticFiles`.

## CONTEXT REFERENCES
- `PRD.md` (Phase 6 Roadmap definition)
- `v1/src/api/server.py` (FastAPI REST service)
- `v1/src/api/models.py` (Pydantic schema definitions)
- `v1/src/core/renderers.py` (Strategy pattern for MPI and Point Splatting)
- `antigravity_global_boilerplate/.agents/rules/frontend.md` (Frontend architecture rules)
- `antigravity_global_boilerplate/.agents/rules/backend.md` (Backend architecture rules)

## New Files to Create
**Components**:
- `v1/frontend/src/components/ViewerCanvas.tsx` (60 FPS interactive HTML5 canvas viewport)
- `v1/frontend/src/components/UploadZone.tsx` (Drag-and-drop custom image uploader with 0ms preview)
- `v1/frontend/src/components/PresetGallery.tsx` (One-click sample preset selector)
- `v1/frontend/src/components/ControlsDrawer.tsx` (Motion presets, renderer options, depth sliders)
- `v1/frontend/src/components/MathMatrixInspector.tsx` (Live linear algebra derivations and KaTeX matrices)
- `v1/frontend/src/components/LayerSlicesViewer.tsx` (MPI depth slice inspector)
- `v1/frontend/src/components/ProgressBar.tsx` (Non-blocking polling status indicator)
- `v1/frontend/src/components/Header.tsx` (Application navigation bar and status indicators)

**API**:
- `v1/frontend/src/api/client.ts` (Typed HTTP fetch client for FastAPI endpoints)
- `v1/frontend/src/api/types.ts` (TypeScript interfaces matching Pydantic schemas)

**Utils**:
- `v1/frontend/src/utils/math.ts` (Client-side camera rotation $R$, intrinsics $K$, and homography $H$ calculations)
- `v1/frontend/src/utils/imageDecoder.ts` (Batch asynchronous `createImageBitmap` worker helper)

**Tests**:
- `v1/frontend/src/utils/__tests__/math.test.ts` (Unit tests for client matrix operations)
- `v1/tests/test_api_static.py` (Backend integration test verifying static file mounting and samples endpoint)

## STEP-BY-STEP TASKS

1. **Task 1: Backend API Preparation & CORS**
   - Add `CORSMiddleware` to `v1/src/api/server.py` allowing requests from `http://localhost:5173`.
   - Add `GET /api/v1/samples` to return filenames and relative paths of bundled sample images.
   - Optimize `clean_result` in `v1/src/api/server.py` to prevent serializing large raw numpy layer arrays into JSON.
   - **VALIDATE**: `pytest v1/tests/test_api.py`

2. **Task 2: Frontend Scaffolding with Vite & Tailwind CSS**
   - Initialize a Vite React TypeScript project in `v1/frontend`.
   - Install `tailwindcss`, `@tailwindcss/vite` (or `postcss` + `autoprefixer`), `lucide-react`, and `katex`.
   - Configure `vite.config.ts` with API proxy forwarding `/api` requests to `http://127.0.0.1:8000`.
   - **VALIDATE**: `npm --prefix v1/frontend run build`

3. **Task 3: API Client and TypeScript Schemas**
   - Create `v1/frontend/src/api/types.ts` defining `JobResponse`, `StatusResponse`, and `ResultResponse`.
   - Create `v1/frontend/src/api/client.ts` with `submitRenderJob`, `fetchJobStatus`, `fetchJobResult`, and `fetchSamples`.
   - Create unit tests for client models.
   - **VALIDATE**: `npm --prefix v1/frontend run build`

4. **Task 4: Interactive HTML5 Canvas Viewport**
   - Create `v1/frontend/src/components/ViewerCanvas.tsx`.
   - Implement `requestAnimationFrame` render loop with spring-damper (`lerp`) cursor smoothing.
   - Implement asynchronous parallel decoding using `createImageBitmap` to eliminate UI freezing.
   - Implement bilinear pitch-yaw frame blending across the discrete 2D atlas grid.
   - Implement auto-wiggle Lissajous figure-8 orbit on mouse idle.
   - **VALIDATE**: `npm --prefix v1/frontend run build`

5. **Task 5: Drag-and-Drop Upload Zone & Preset Gallery**
   - Create `v1/frontend/src/components/UploadZone.tsx` with drag-and-drop, MIME validation, and instant `URL.createObjectURL` preview.
   - Create `v1/frontend/src/components/PresetGallery.tsx` enabling one-click selection of bundled samples.
   - Create `v1/frontend/src/components/ProgressBar.tsx` for real-time polling progress feedback.
   - **VALIDATE**: `npm --prefix v1/frontend run build`

6. **Task 6: Live Mathematical Matrix Inspector**
   - Create `v1/frontend/src/utils/math.ts` implementing $3 \times 3$ Euler rotation matrices, orthogonality verification ($R R^T = I$, $\det(R) = 1$), camera intrinsics $K$, and plane homography $H_k = K (R - \frac{t n^T}{d_k}) K^{-1}$.
   - Create `v1/frontend/src/components/MathMatrixInspector.tsx` with formatted KaTeX LaTeX rendering.
   - Write unit tests in `v1/frontend/src/utils/__tests__/math.test.ts`.
   - **VALIDATE**: `npm --prefix v1/frontend test`

7. **Task 7: Parameter Controls Drawer & Layer Inspector**
   - Create `v1/frontend/src/components/ControlsDrawer.tsx` for tuning motion preset, layer counts (8–32), depth bounds ($Z_{near}, Z_{far}$), and unsharp masking.
   - Create `v1/frontend/src/components/LayerSlicesViewer.tsx` displaying individual MPI depth planes.
   - Assemble all components into `v1/frontend/src/App.tsx`.
   - **VALIDATE**: `npm --prefix v1/frontend run build`

8. **Task 8: Production Static Mounting in FastAPI**
   - Configure `v1/src/api/server.py` to mount `v1/frontend/dist` via `StaticFiles(directory="frontend/dist", html=True)`.
   - Write backend integration test in `v1/tests/test_api_static.py` asserting that `GET /` serves `index.html`.
   - Verify entire system compiles cleanly and runs headlessly.
   - **VALIDATE**: `pytest v1/tests/test_api_static.py`

## TESTING STRATEGY
- **Frontend Unit Tests**: Use Vitest for math transformation matrix computations, orthogonality verification, and API response parsing.
- **Backend API Integration Tests**: Use `pytest` with `fastapi.testclient.TestClient` to verify CORS headers, sample listings, and static file delivery.
- **Manual Verification**: Run `uvicorn src.api.server:app --app-dir v1 --port 8000` and open `http://localhost:8000` in the browser, verifying smooth parallax hover, instant image upload, and live math derivations.

## ACCEPTANCE CRITERIA
- [ ] Frontend operates as an independent Single Page Application with zero direct imports of backend Python modules.
- [ ] Custom image uploads display an instant 0ms preview and submit without page reloads or state drops.
- [ ] Viewport canvas operates with continuous 60+ FPS mouse-tracking parallax, lerp smoothing, and smooth auto-wiggle.
- [ ] Real-time progress updates poll cleanly from `/api/v1/jobs/{id}/status`.
- [ ] Mathematical matrix inspector dynamically validates and renders camera rotation and homography equations with KaTeX.
- [ ] Single backend command (`uvicorn src.api.server:app`) serves both API endpoints and the production web frontend.
- [ ] All frontend and backend tests pass cleanly.
