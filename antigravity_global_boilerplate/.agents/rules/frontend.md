---
trigger: model_decision
description: Apply these guidelines whenever working on the frontend, UI components, styling, or client-side React code.
---


## Frontend Architecture & Patterns

### 1. Component Design & Framework
* **Vite + React SPA:** Build the frontend as a standalone Single Page Application in `v1/frontend/` using Vite, React, and Tailwind CSS.
* **Strict API Boundary (Zero Backend Imports):** The UI layer must NEVER directly import PyTorch, rendering pipelines, or task queues. All data exchange occurs strictly via HTTP REST endpoints (`/api/v1/render`, `/api/v1/jobs/{id}/status`, `/api/v1/jobs/{id}/result`).
* **Component Modularity:** Decompose the UI into dedicated, testable components:
  - `<ViewerCanvas />`: 60 FPS interactive HTML5 canvas for parallax novel view display.
  - `<UploadZone />`: Drag-and-drop file upload with instant 0ms client preview.
  - `<PresetGallery />`: One-click loading of bundled sample images.
  - `<MathMatrixInspector />`: Live client-side linear algebra and KaTeX formula inspector.
  - `<LayerSlicesViewer />`: MPI depth layer slice browser.
  - `<ExportBar />`: Download buttons for rendered GIFs, WebPs, and depth maps.

### 2. Canvas & Parallax Rendering
* **HTML5 Canvas & RAF Loop:** Direct access to native `<canvas>` via `useRef`. Always use `requestAnimationFrame` for rendering instead of React state re-renders.
* **Asynchronous Batch Decoding:** Decode base64 frame arrays concurrently using `createImageBitmap()` to prevent UI thread micro-stutters.
* **Spring / Lerp Smoothing:** Apply linear interpolation (`current += (target - current) * 0.12`) to mouse cursor tracking to eliminate hand jitter.
* **Bilinear View Interpolation:** Blend across the 2D pitch-yaw atlas grid using corner weighting to ensure seamless transitions between discrete viewpoints.

### 3. State Management & Upload Lifecycle
* **Upload Isolation:** Generate immediate client-side previews using `URL.createObjectURL(file)` without blocking on network uploads. Send images as `FormData` to the REST API.
* **Non-Blocking Polling:** Poll `/api/v1/jobs/{id}/status` on an interval (e.g. 500ms) with clean cancellation on unmount.
* **Clean Error Handling:** If the backend API or rendering worker fails, display clear toast notifications or badges without tearing down existing canvas state.