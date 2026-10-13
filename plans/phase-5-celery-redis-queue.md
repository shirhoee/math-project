# Phase 5: Robust Async Task Queue (Celery/Redis)

## Feature Description & User Stories
**Description**: 
Currently, the `v1` backend relies on raw Python `threading.Thread` for asynchronous job execution. This causes bottlenecks, limits scalability, and makes jobs vulnerable if the web server process crashes. We will integrate Celery as a robust distributed task queue, backed by Redis, to reliably handle the heavy PyTorch/NumPy processing tasks independently from the FastAPI and Streamlit web servers.

**User Stories**:
- As a frontend client, I want my image processing jobs to be queued reliably even if the API server is under heavy load.
- As a developer, I want to manage, monitor, and scale background rendering tasks independently from the main web server process.
- As a system administrator, I want failed rendering jobs to be traceable and retriable without crashing the Streamlit UI.

## Problem & Solution Statement
**Problem**: The current background execution engine (`v1/src/pipeline.py`) uses `threading.Thread` and stores job statuses, results, and progress in memory via global dictionaries (`global_job_results`, `global_job_status`). This completely prevents horizontal scaling and risks memory leaks.
**Solution**: 
1. Install and configure `celery` and `redis`.
2. Extract the `run_build_job` logic into a registered Celery task in a new `v1/src/tasks.py` module.
3. Refactor `v1/src/pipeline.py` so `submit_render_job` calls `task.delay()`, and `poll_job_status`/`get_job_result` query Celery's `AsyncResult` state rather than in-memory dictionaries.
4. Update `PRD.md` to reflect this new Phase 5.

## CONTEXT REFERENCES
- `PRD.md` (Needs updating with Phase 5 details)
- `v1/src/pipeline.py` (Current target for refactoring away from `threading.Thread`)
- `v1/src/api/server.py` (API endpoints that rely on `pipeline.py`)
- `v1/app.py` (Streamlit UI that queries `poll_job_status`)

## New Files to Create
**Components**:
- `v1/src/celery_app.py` (Celery app initialization and broker configuration)
- `v1/src/tasks.py` (The extracted rendering tasks)
**API**: None
**Utils**: None
**Tests**:
- `v1/tests/test_celery_tasks.py` (Integration tests for Celery tasks running in eager mode)

## STEP-BY-STEP TASKS

1. **Task 1: Update Dependencies**
   - Add `celery` and `redis` to `v1/requirements.txt`.
   - Install them in the local environment.
   - **VALIDATE**: `python -c "import celery, redis"`

2. **Task 2: Update the PRD**
   - Edit `PRD.md` to formally append "Phase 5: Robust Async Task Queue (Celery/Redis)" to the roadmap, ensuring alignment with the project trajectory.
   - **VALIDATE**: `Get-Content -Path PRD.md | Select-String "Phase 5"`

3. **Task 3: Create the Celery App Configuration**
   - Create `v1/src/celery_app.py`.
   - Initialize the `Celery` application, pointing the `broker` and `backend` to a local Redis instance (`redis://localhost:6379/0`).
   - **VALIDATE**: `mypy v1/src/celery_app.py`

4. **Task 4: Extract the Rendering Task**
   - Create `v1/src/tasks.py`.
   - Move the contents of `run_build_job` from `v1/src/pipeline.py` into a Celery task decorated with `@celery_app.task(bind=True)`.
   - Update the task to use `self.update_state()` for progress tracking instead of `global_job_progress`.
   - **VALIDATE**: `mypy v1/src/tasks.py`

5. **Task 5: Refactor the Pipeline Coordinator**
   - Modify `v1/src/pipeline.py`.
   - Remove `threading.Thread` and the global memory dictionaries (`global_job_results`, `global_job_status`, etc.).
   - Update `submit_render_job` to call `build_render_job.delay(...)`.
   - Update `poll_job_status` and `get_job_result` to retrieve state using `AsyncResult(job_id)`.
   - **VALIDATE**: `mypy v1/src/pipeline.py`

6. **Task 6: Write Integration Tests**
   - Create `v1/tests/test_celery_tasks.py`.
   - Write tests configuring Celery with `task_always_eager=True` to simulate task execution synchronously without needing a live Redis server.
   - Ensure the FastAPI client tests in `v1/tests/test_api.py` still pass with the new pipeline.
   - **VALIDATE**: `pytest v1/tests/test_celery_tasks.py v1/tests/test_api.py`

## TESTING STRATEGY
- **Unit/Integration Tests**: Use Celery's `task_always_eager=True` setting during `pytest` runs to execute tasks synchronously, verifying pipeline behavior without spinning up Redis.
- **Manual End-to-End**: Run `redis-server`, start a Celery worker (`celery -A src.celery_app worker`), and launch the Streamlit app to verify real-time progress updates are functioning smoothly.

## ACCEPTANCE CRITERIA
- [ ] Celery and Redis replace `threading.Thread` for all background image processing tasks.
- [ ] In-memory global state dictionaries are completely removed from `v1/src/pipeline.py`.
- [ ] The `app.py` Streamlit UI and `server.py` FastAPI endpoints continue to function without modification.
- [ ] Real-time progress updates (via Celery's `update_state`) correctly reach the UI polling endpoints.
- [ ] `pytest` suite passes cleanly.
