# Interactive 3D View Synthesis using Linear Transformations

## Project Overview
This project takes a single 2D image, extracts a depth map using a lightweight AI model, and builds a custom 3D mathematical rendering engine to allow users to interactively rotate and view the image in 3D space. 

This project is built for the **Mathematical Foundation for AI & Data Science** course and specifically focuses on practically applying concepts like:
* **Orthogonal Matrices** (3D Rotation)
* **Matrix Projections** (3D to 2D screen mapping)
* **Linear Transformations & Vectors**

## Architecture
1. **Depth Estimation:** Uses a pre-trained model (Depth Anything V2) to generate a relative disparity map, which is then converted to a Z-axis depth map.
2. **Unprojection Engine:** Custom NumPy matrix operations to convert (X, Y, Z) into a 3D Point Cloud.
3. **Transformation Engine:** Applies custom 3x3 Orthogonal Rotation Matrices (Pitch, Yaw, Roll) pivoting around the scene's median depth, with optional camera translation.
4. **Projection Engine:** Reprojects the transformed 3D points back to a 2D image canvas.
5. **Frontend:** Streamlit web interface for interactive sliders and image upload.

## Mathematical Limitations (Occlusions)
Because the 3D space is generated from a single 2D projection, the engine only possesses surface data visible to the camera. When rotating the generated 3D point cloud via orthogonal matrices, regions behind foreground objects (which were occluded in the original 2D projection) will appear as void spaces or "holes". Mathematically, this is because a single projection cannot contain data for coordinate points that were masked by objects with a smaller $Z$ value along the same view vector.

## Setup & Installation
```bash
# 1. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit UI
streamlit run app.py
```
