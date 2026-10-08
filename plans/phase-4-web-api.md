# Phase 4: Web API Exposure (Headless Mode)

## Feature Description & User Stories
**Description**: With the `pipeline.py` now entirely decoupled from Streamlit, we need to wrap it in a lightweight, high-performance HTTP REST API using FastAPI. This transforms the `v1` repository from an interactive web demo into a scalable microservice capable of powering external frontends (e.g., React, Next.js, or mobile apps) asynchronously.
**User Stories**:
- As a frontend client, I want to `POST` a raw image to an endpoint and receive a unique `job_id` so that I can process images without blocking my UI.
- As a frontend client, I want to `GET` the progress and status of a given `job_id` so that I can show a loading bar.
- As a frontend client, I want to `GET` the final resulting payload (containing base64 frames and depth parameters) once the job is complete.

## Problem & Solution Statement
**Problem**: The application is currently only accessible via a Streamlit web GUI, restricting its usage in real-world microservice architectures where frontends and backends are disconnected over HTTP.
**Solution**:
1. **API Server Setup**: Create `v1/src/api/server.py` using `FastAPI` and `uvicorn`.
2. **REST Endpoints**:
   - `POST /api/v1/render`: Accepts a `UploadFile` (the image) and configuration form parameters (e.g., `renderer="Layers (MPI)"`). It saves the image to `outputs/` and invokes `submit_render_job(...)`, returning `{"job_id": 123}`.
   - `GET /api/v1/jobs/{job_id}/status`: Calls `poll_job_status(job_id)` and returns JSON `{"status": "...", "progress": 50}`.
   - `GET /api/v1/jobs/{job_id}/result`: Calls `get_job_result(job_id)` and returns the massive payload with `frames_b64`.
3. **Pydantic Validation**: Define strict request/response schema models in `v1/src/api/models.py`.

## CONTEXT REFERENCES
- `PRD.md` (Phase 4 requirements)
- `v1/src/pipeline.py` (`submit_render_job`, `poll_job_status`, `get_job_result` contracts)

## New Files to Create
**Components**:
- `v1/src/api/__init__.py`
- `v1/src/api/server.py` (FastAPI app and route definitions)
- `v1/src/api/models.py` (Pydantic schema definitions)
**API**: None
**Utils**: None
**Tests**:
- `v1/tests/test_api.py` (FastAPI `TestClient` E2E testing)

## STEP-BY-STEP TASKS

1. **Task 1: Setup API Dependencies**
   - We must install FastAPI and Uvicorn. However, since we are only simulating this locally and keeping it dependency-light, we will add them to a requirements check or just use them assuming they are in the environment (we will just test it). *Note: The environment already has `fastapi` and `uvicorn` and `python-multipart`.*
   - **VALIDATE**: `python -c "import fastapi, uvicorn, multipart"`

2. **Task 2: Define Pydantic Models**
   - Create `v1/src/api/models.py`.
   - Define `JobResponse`, `StatusResponse`, and `ResultResponse` Pydantic models.
   - **VALIDATE**: `mypy v1/src/api/models.py`

3. **Task 3: Build the FastAPI Endpoints**
   - Create `v1/src/api/server.py`.
   - Setup a standard FastAPI application.
   - Implement `POST /api/v1/render` taking an `UploadFile` and writing it to `outputs/uploads/` via `shutil.copyfileobj`. Generate a UUID-based or incremented `job_id`, and submit it to the pipeline.
   - Implement `GET /api/v1/jobs/{job_id}/status`.
   - Implement `GET /api/v1/jobs/{job_id}/result`.
   - **VALIDATE**: `mypy v1/src/api/server.py`

4. **Task 4: Create API Integration Tests**
   - Create `v1/tests/test_api.py` using `fastapi.testclient.TestClient`.
   - Write a test `test_render_workflow` that `POST`s a dummy image to the render endpoint, loops on the status endpoint until complete, and then asserts the result endpoint returns frames.
   - **VALIDATE**: `pytest v1/tests/test_api.py`

5. **Task 5: Format & Lint**
   - Run `ruff` formatting against the new `v1/src/api` module.
   - **VALIDATE**: `ruff format v1/src/api/ && ruff check v1/src/api/`

## TESTING STRATEGY
- **Integration Tests**: `pytest` and `TestClient` will simulate an external frontend firing off a multi-step job via HTTP. We will bypass real inference using `FAST_TEST=1` in the tests.

## ACCEPTANCE CRITERIA
- [x] FastAPI server is fully defined in `v1/src/api/server.py`.
- [x] Endpoints exist for submitting jobs, checking status, and getting results.
- [x] Pydantic models strictly validate the inputs/outputs.
- [x] `pytest v1/tests/test_api.py` passes smoothly.
