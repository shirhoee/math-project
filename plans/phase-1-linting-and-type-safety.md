# Phase 1: Linting & Type Safety

## Feature Description & User Stories
**Description**: The `v1` codebase currently lacks static type checking and consistent formatting, leading to hidden bugs (especially with NumPy array dimensionality) and cognitive overhead when navigating the monolithic files. This phase aims to introduce `mypy`, strict NumPy annotations, and `ruff` (for formatting and linting) to create a robust, error-free foundation.
**User Stories**:
- As a developer, I want `mypy` to statically check my Python code so that I catch type mismatch errors before runtime.
- As a developer, I want NumPy arrays to be clearly annotated with their expected shapes and types so that matrix math is predictable.
- As a developer, I want the codebase to be automatically formatted by `ruff` so that code style is uniform and easy to read.

## Problem & Solution Statement
**Problem**: The core files (`depth_estimator.py`, `math_engine.py`, `mpi_renderer.py`, `app.py`) in `v1` are procedurally written without type hints. Matrix operations assume shapes that are hard to trace.
**Solution**: We will integrate `ruff` to enforce PEP 8 standards and auto-format the codebase. We will introduce `mypy` and configure it. We will then go through the core logic files and add comprehensive Python type hints and NumPy docstrings (or `nptyping` if installed, but docstring annotations are safer for immediate backward compatibility) to all function signatures and critical variables.

## CONTEXT REFERENCES
- `PRD.md` (Phase 1 specification)
- `.agents/rules/backend.md` (Strict Typing and NumPy Array documentation rules)
- Files to modify:
  - `v1/depth_estimator.py`
  - `v1/math_engine.py`
  - `v1/mpi_renderer.py`
  - `v1/app.py`
  - `v1/view_atlas.py`

## New Files to Create
**Components**: None
**API**: None
**Utils**: 
- `pyproject.toml` (in `v1/` to hold configuration for `ruff` and `mypy` if not already present)
**Tests**: None

## STEP-BY-STEP TASKS

1. **Task 1: Setup Linting & Formatting Configuration**
   - Create or update `v1/pyproject.toml` to configure `ruff` and `mypy`. Enable strict mode for `mypy` where feasible (or at least enforce typing for the core files).
   - Update `v1/requirements.txt` to include `ruff` and `mypy`.
   - **VALIDATE**: `pip install -r v1/requirements.txt && ruff check v1/ --dry-run`

2. **Task 2: Auto-Format the Codebase**
   - Run `ruff format` and `ruff check --fix` over the entire `v1/` directory to standardize the procedural codebase style without changing functionality.
   - **VALIDATE**: `ruff format --check v1/`

3. **Task 3: Type Annotate `depth_estimator.py`**
   - Add explicit Python type hints to all function signatures.
   - Document NumPy array shapes (e.g., `# Returns: np.ndarray shape (H, W) dtype float32`) in docstrings.
   - **VALIDATE**: `mypy v1/depth_estimator.py`

4. **Task 4: Type Annotate `math_engine.py` and `mpi_renderer.py`** (COMPLETED)
   - Add explicit type hints. Pay special attention to the transformation matrices and homography parameters.
   - Document the shapes of all input/output tensors and matrices.
   - **VALIDATE**: `mypy v1/math_engine.py v1/mpi_renderer.py`

5. **Task 5: Type Annotate `view_atlas.py` and `app.py`** (COMPLETED)
   - Add type hints to the pre-rendering logic in `view_atlas.py`.
   - Add type hints to the Streamlit orchestration logic in `app.py`.
   - **VALIDATE**: `mypy v1/view_atlas.py v1/app.py`

6. **Task 6: Full Regression Check** (COMPLETED)
   - Ensure that the type annotations and formatting did not break existing tests or functionality.
   - **VALIDATE**: `pytest v1/tests/`

## TESTING STRATEGY
- **Unit Tests**: No new unit tests are required for this phase, as we are not adding new logic. 
- **E2E Tests**: Existing tests in `v1/tests/` will serve as our regression suite to ensure formatting and typing did not introduce syntax errors or behavioral changes.
- **Static Analysis**: We will rely on `mypy --strict` (or similar strictness) and `ruff` checks to serve as static tests.

## ACCEPTANCE CRITERIA
- [x] `v1/requirements.txt` includes `ruff` and `mypy`.
- [x] `v1/pyproject.toml` contains configuration for both tools.
- [x] Running `ruff check v1/` passes with zero critical errors.
- [x] Running `ruff format --check v1/` passes indicating the code is fully formatted.
- [x] Running `mypy v1/` passes on all core logic files without type mismatch errors.
- [x] All functions in `v1/depth_estimator.py`, `v1/math_engine.py`, and `v1/mpi_renderer.py` have explicit type hints and documented NumPy shapes.
- [x] Running `pytest v1/tests/` passes completely, ensuring zero regressions.
