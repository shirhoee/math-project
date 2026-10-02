# Interactive 3D View Synthesis

A fully vectorized Python engine for 3D View Synthesis (single image $\rightarrow$ depth map $\rightarrow$ 3D point cloud $\rightarrow$ rotation $\rightarrow$ reprojection $\rightarrow$ Streamlit UI).

## Setup
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

## Parameter Table
| Parameter | Description |
|---|---|
| **FOV / z_near / z_far** | Configures intrinsic matrix boundaries and perspective scaling. |
| **Pitch / Yaw / Roll** | 3-Axis Euler camera rotation bounded $\pm45^\circ$. |
| **Preview Mode** | Fast interactive Stride-2 rendering (142.9 ms with fill). |
| **High Quality Mode** | Full-resolution Stride-1 rendering (217.2 ms with pyramid fill). |
| **Splatting** | Fill-only adaptive footprint expansion to prevent sub-pixel cracking. |
| **Edge Masking** | Rubber-sheet masking on depth discontinuities (threshold 0.05). |
| **Auto Crop** | Exact geometrical ray clipping of border voids. |
| **Hole Filling** | Push-Pull (Pyramid) Background-biased occlusion interpolation. |

## Limitation: Occlusion
View synthesis relies solely on a single RGB perspective. Angles beyond $\pm15^\circ$ generate significant occlusive shadows ("holes") where background geometry is unknown. Hole filling algorithms interpolate these regions via background-color extension, but extreme angles will still result in localized smearing and loss of detail.

## Engine Math
See `MATH_DERIVATIONS.md` for full algebraic proofs regarding:
1. FOV-based Intrinsic Camera Matrix ($K$)
2. Unprojection ($2D \rightarrow 3D$) and Projection ($3D \rightarrow 2D$) 
3. Depth normalization and pivot-based Euler rotations.
