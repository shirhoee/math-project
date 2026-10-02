# 3D View Synthesis Engine

## Overview
NumPy-based 3D View Synthesis engine mapping 2D images to 3D and rendering at novel viewpoints.

## Setup
1. Install dependencies
2. `streamlit run app.py`

## Parameters
| Name | Default | Meaning | Effect |
|---|---|---|---|
| FOV | 60.0 | Field of View | Changes perspective distortion |
| Z Pivot | -1.0 | Center of rotation | Affects translation during rotation |
| Splatting | True | Footprint expansion | Closes holes from dilation |
| Edge Mode | demote | Edge handling | Prevents foreground stretching |
| Auto Crop | True | Dynamic crop | Removes border voids |

## Architecture
Uses depth estimation -> unprojection -> 3D transform -> 2D projection -> hole filling.

## Occlusion Limitation
Due to single-image depth, areas occluded in the original view appear as holes when rotated.

## Performance
Preview speed 52 ms, HQ render 222 ms (with fill on 800x534 images).
