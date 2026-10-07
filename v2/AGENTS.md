# AGENTS.md

**Project Identity:** V2 Rewrite for Mathematical Foundation for AI and Data Science.
**Goal:** Build a robust, performant 3D viewer from a single image using depth estimation and vectorised NumPy geometry, with a static web viewer and a FastAPI backend.
**Stack:** FastAPI, Uvicorn, NumPy, PyTorch, Transformers, HTML/JS/Three.js.
**Folder Map:**
- `src/`: Backend logic (`geometry`, `depth`, `mesh`, `server`).
- `web/`: Static frontend.
- `docs/`: Extensive project documentation.
- `tests/`: Unit and integration tests.
- `cache/`, `outputs/`: Ignored directories for temp data.

**Command Cheat-Sheet:**
- Install: `pip install -r requirements.txt`
- Run: `uvicorn src.server.main:app --reload`
- Test: `pytest tests/`
- Quality Check: Run designated scripts in `tests/` or `outputs/`.

**Golden Rules:**
See `.agent/rules/01-golden-rules.md`

**INSTRUCTION:**
Before starting any task, read `docs/04_PHASES_AND_STATUS.md`.
After finishing any task, update it.
On any error, consult `docs/05_ERROR_PLAYBOOK.md` before improvising.

**Pointer to Docs:**
All docs reside in `docs/`. See `docs/00_CONTEXT.md` for background.
