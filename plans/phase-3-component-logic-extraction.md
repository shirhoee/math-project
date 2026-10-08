# Phase 3: Component/Logic Extraction

## Feature Description & User Stories
**Description**: The codebase is stable, but the rendering orchestration in `pipeline.py` is deeply branched based on the chosen rendering engine (Point Splatting vs. MPI). The `tools/` directory is also a chaotic mess of 20+ one-off scripts. This phase standardizes the rendering API using an object-oriented Strategy pattern, and cleans up the documentation and tools folders.
**User Stories**:
- As a developer, I want a unified `BaseRenderer` interface so that I can add new rendering engines (e.g., Gaussian Splatting) without polluting `pipeline.py` with more `if/else` statements.
- As a developer, I want all obsolete progress reports archived into `docs/history/` to keep the root directory strictly for active code.
- As a developer, I want a single `cli.py` entrypoint for running benchmarks, diagnostics, and demos instead of remembering 20 different script names.

## Problem & Solution Statement
**Problem**: `pipeline.py` contains 70+ lines of branching logic for MPI vs Point Splatting, handling data preparation and image encoding redundantly. The `tools/` folder contains redundant scripts like `bench_before.py`, `bench_after.py`, and `benchmark.py`.
**Solution**:
1. **Rendering API Extraction**: Create `v1/src/core/renderers.py` with an abstract `BaseRenderer` and concrete `MPIRenderer` / `PointSplatRenderer` classes. Each subclass will expose a `.render(...)` method returning the output payload.
2. **Pipeline Simplification**: Refactor `v1/src/pipeline.py` to instantiate the appropriate renderer dynamically and yield progress callbacks to the UI.
3. **Documentation Archival**: Move all `UPDATE_REPORT*.md` and `PROGRESS.md` into `v1/docs/history/`.
4. **Tooling CLI**: Create `v1/tools/cli.py` using `argparse`. Register `benchmark`, `make_demo`, and `quality_check` as subcommands.

## CONTEXT REFERENCES
- `PRD.md` (Phase 3 requirements)
- `v1/src/pipeline.py` (Current branched renderer logic)
- `v1/tools/` (Scripts to audit and consolidate)

## New Files to Create
**Components**:
- `v1/src/core/renderers.py` (Abstract and concrete renderer classes)
- `v1/tools/cli.py` (Unified CLI interface)
**API**: None
**Utils**: None
**Tests**: None

## STEP-BY-STEP TASKS

1. **Task 1: Documentation Archival**
   - Create `v1/docs/history/` directory.
   - Move `v1/UPDATE_REPORT*.md` and `v1/PROGRESS.md` into `v1/docs/history/`.
   - **VALIDATE**: `mkdir -p v1/docs/history && dir v1\docs\history`

2. **Task 2: Define the BaseRenderer API**
   - Create `v1/src/core/renderers.py`.
   - Define a `BaseRenderer` class with an abstract `render(self, img_array, disparity, engine, **kwargs)` method.
   - Migrate the MPI logic from `pipeline.py` into an `MPIRenderer` class.
   - Migrate the Point Splatting logic into a `PointSplatRenderer` class.
   - **VALIDATE**: `mypy v1/src/core/renderers.py`

3. **Task 3: Refactor the Pipeline**
   - Update `v1/src/pipeline.py` to import the renderers.
   - Replace the `if renderer == "Layers (MPI)": ... else: ...` block with a factory instantiation: `engine_impl = MPIRenderer() if renderer == "Layers (MPI)" else PointSplatRenderer()`.
   - Call `engine_impl.render(...)` and pass progress update callbacks so `pipeline.py` can still update `global_job_progress`.
   - **VALIDATE**: `mypy v1/src/pipeline.py`

4. **Task 4: Build the Unified CLI**
   - Create `v1/tools/cli.py` using standard `argparse` subparsers.
   - Hook up `make_demo.py` to a `demo` command, `benchmark.py` to a `benchmark` command, and `quality_check.py` to a `quality` command.
   - **VALIDATE**: `python v1/tools/cli.py --help`

5. **Task 5: Format & Lint**
   - Run `ruff` formatting against the `v1/src/` folder to clean up any loose ends.
   - **VALIDATE**: `ruff format v1/src/ && ruff check v1/src/`

6. **Task 6: E2E Regression Testing**
   - Ensure the decoupled object-oriented renderer pipeline still successfully integrates with Streamlit by passing the smoke tests.
   - **VALIDATE**: `pytest v1/tests/test_app_smoke.py`

## TESTING STRATEGY
- **Static Analysis**: `mypy` ensures the abstract base class and concrete implementations match signatures.
- **E2E Tests**: `test_app_smoke.py` triggers an end-to-end execution of both rendering engines via the Streamlit interface, ensuring `st.session_state` and pipeline state correctly synchronize with the new objects.

## ACCEPTANCE CRITERIA
- [x] `v1/UPDATE_REPORT*.md` and `PROGRESS.md` are archived in `v1/docs/history/`.
- [x] `v1/src/core/renderers.py` exports `BaseRenderer`, `MPIRenderer`, and `PointSplatRenderer`.
- [x] `v1/src/pipeline.py` has zero duplicated loop/encoding logic and invokes the abstract renderer API.
- [x] `v1/tools/cli.py` is executable and provides multiple subcommands.
- [x] `pytest v1/tests/test_app_smoke.py` passes successfully.
