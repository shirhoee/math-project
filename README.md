# 3D View Synthesis Engine
![Orbit GIF](docs/results/demo_indoor.gif)

This project transforms a single flat photo into an interactive 3D scene you can look around in. It uses a deep learning model to estimate a depth map, then applies pure NumPy linear algebra to unproject the pixels into a 3D point cloud and render it from new viewpoints. The entire rendering pipeline—including rotation, perspective projection, splatting, and hole filling—is built from scratch without standard 3D libraries.

## Run It
```bash
pip install -r requirements.txt
streamlit run app.py
```
*Note: On first run, the Depth Anything V2 model weights will be downloaded automatically.*

## How It Works
```text
[Original Image] -> (Depth Model) -> [Disparity Map] -> (Intrinsic K^-1) -> [3D Point Cloud] -> (Rotation R) -> [Transformed 3D Points] -> (Intrinsic K + Z-buffer) -> [New View]
```

## Occlusion Limitation
Because the input is a single 2D image, the system has no information about what lies behind foreground objects. When rotating the camera, these occluded areas are revealed as "holes." We use an iterative multi-scale pyramid fill to patch these holes using background colors, but at extreme angles, this can look stretched or blocky. For the best experience, keep the rotation angles small (±12°).

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
