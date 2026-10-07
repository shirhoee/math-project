# 3D Photo Viewer

Transforms a single flat photo into an interactive 3D scene using depth estimation and a Multi-Plane Image (MPI) homography renderer. It precomputes 123 views in the background and blends them on the frontend for instant, 60fps parallax motion.

## Demos

| Indoor | Kyoto | Mountains | Cup | Park |
|:---:|:---:|:---:|:---:|:---:|
| ![Indoor](docs/results/demo_01_sample_1.webp) | ![Kyoto](docs/results/demo_02_sample_2.webp) | ![Mountains](docs/results/demo_03_sample_3.webp) | ![Cup](docs/results/demo_04_sample_4.webp) | ![Park](docs/results/demo_05_sample_5.webp) |

## Setup & Run (Local)

1. Stop any old servers running on ports 8501 or 8502 (e.g. `Stop-Process -Name "python" -Force` on Windows).
2. Install requirements and start the app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
3. Open the Local URL printed in the terminal. The app will open instantly and load samples from cache.

## How It Works

The V2 engine replaces point clouds with a layered **Multi-Plane Image (MPI)** renderer.
1. The image depth is estimated and sliced into 16 layers (Z-planes).
2. For any camera pose, each layer undergoes a pure 2D perspective warp (Homography).
3. The warped layers are composited back-to-front.
4. A high-density grid (41 columns x 3 rows) of these views is pre-rendered into a WebGL-friendly atlas.
5. The JavaScript frontend smoothly interpolates between these pre-rendered frames as the user moves their mouse.

## Photo Credits
Sample photos were sourced from Wikimedia Commons. See `assets/samples/ATTRIBUTION.md` for full attribution URLs and authors.
