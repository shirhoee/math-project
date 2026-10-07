# 04 Phases and Status

**Overall Status:** DONE (Stage A)
**Current Phase:** COMPLETED
**Last Updated:** 2026-10-07T19:53:00+05:30
**Last Commit Hash:** TBD

## Phase Table

| ID | Phase | Goal | Deliverables | Acceptance Criteria | Status | Evidence |
|:---|:---|:---|:---|:---|:---|:---|
| P0 | Archive & Scaffold | Clean slate for V2 | `v1/` archive, `v2/` scaffold, documentation | All docs exist, rules set | DONE | Docs exist, git commit |
| P1 | Geometry Port | Port math from v1 | `src/geometry` | Unit tests pass, math accurate | DONE | tests pass, committed |
| P2 | Depth Module | Model loading & caching | `src/depth`, precomputed samples | Load time logged, disk cache works | DONE | timing report, tests pass |
| P3 | Mesh Builder | Unproject depth to mesh | `src/mesh` | Discontinuities cut, texture UVs map correctly | DONE | tests pass |
| P4 | Backend & Viewer | FastAPI & WebGL UI | `src/server`, `web/` | Uploads work, interactive viewer | DONE | tests pass |
| P5 | Quality Pass | Reduce stretching, polish | Updated mesh/UI | Gates met, smear reduced | DONE | tests pass |
| P6 | Finalization | Matrices UI, reports | Demo script, Viva notes, matrix panel | Report complete | DONE | docs updated |
| B | Stage B (Two-View) | Two-view geometry | *PLANNED (DO NOT START)* | | NOT STARTED | |

## Task Checklists

### P0
- [x] Archive v1
- [x] Scaffold v2
- [x] Write documentation

### P1
- [x] Port camera intrinsics
- [x] Port projection
- [x] Port transform
- [x] Port homography and adjugate inverse
- [x] Write and pass unit tests

### P2
- [x] Port depth estimator
- [x] Disk caching
- [x] Precompute samples
- [x] Timing report

### P3
- [x] Unproject to grid mesh
- [x] Cut triangles across depth discontinuities
- [x] Texture UV mapping
- [x] NumPy reference renderer for tests

### P4
- [x] FastAPI endpoints (POST /process, GET /samples)
- [x] Static web viewer with OrbitControls
- [x] Camera limits
- [x] Loading indicator and error display
- [x] Sample picker and upload

### P5
- [x] Edge cutting in mesh builder
- [x] Tune camera range in UI
- [x] Tests for math, mesh, and viewer quality gates

### P6
- [x] Matrix panel (live display of K, R, t)
- [x] Finalize demo script
- [x] Finalize viva notes
- [x] Final verification

## Current State
- Stage A complete. All phases DONE.

## Blockers and Workarounds
- None.
