# 3D View Synthesis Engine

## Overview

NumPy-only 3D view synthesis engine. Estimates depth from a single image using Depth Anything V2, unprojects to a 3D point cloud, applies user-controlled rotation/translation, and re-renders at the new viewpoint with adaptive splatting, edge demotion, and pyramid hole filling.

## Architecture

1. **Depth estimation** — Depth Anything V2 (Small) produces a disparity map.
2. **Unprojection** — Disparity → depth → 3D point cloud via camera intrinsics.
3. **3D transform** — Pivot-based rotation (Euler angles) and translation.
4. **Perspective projection** — Points → 2D via pinhole model with Z-buffer.
5. **Render passes** — (1) exact points, (2) demoted edge points (fill-only), (3) adaptive splats (fill-only).
6. **Pyramid hole fill** — Multi-scale background-biased fill, 0.00% residual holes.

## Setup

```bash
pip install streamlit numpy pillow transformers torch
streamlit run app.py
```

## Parameters

| Name | Default | Meaning | Effect |
|---|---|---|---|
| FOV (deg) | 60.0 | Horizontal field of view | Controls perspective distortion |
| Z Pivot | -1.0 (auto) | Depth of rotation centre | -1 uses median depth |
| Splatting | True | Adaptive fill-only footprint expansion | Closes sub-pixel holes from projection |
| Edge Masking | True | Depth-gradient threshold (tau=0.05) | Flags foreground/background boundary pixels |
| Edge Mode | demote | How flagged pixels are handled | `demote`: fill-only pass; `drop`: discard |
| Auto Crop | True | Off-centre principal-point shift + focal scaling | Removes border voids after rotation |
| Hole Filling | True | Pyramid fill | Fills remaining holes with background-biased content |
| Roll (deg) | ±5.0 | Roll rotation limit | Limited to avoid severe over-cropping |

## Occlusion Limitation

Single-image depth cannot reveal surfaces hidden in the original view. At yaw 25 deg, 23.8% of pixels are filled by the pyramid with no real image data. Filled regions show blocky artifacts (mean gradient ratio 0.49 vs. unfilled regions).

## Performance (800 x 534, sample.jpg)

| Mode | Mean (ms) | p95 (ms) |
|---|---|---|
| Preview (stride 2, fill) | 53.5 | 75.3 |
| HQ (stride 1, fill) | 209.7 | 233.6 |

Identity difference with default settings: 0.000%.

## Results Gallery

Side-by-side images (original | no-fill | filled) are in `docs/results/`.
