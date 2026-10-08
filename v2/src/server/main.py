import os
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.depth.estimator import DepthEstimator
from src.mesh.builder import build_mesh
from src.geometry.camera import intrinsics_from_fov

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(BASE_DIR / "outputs", exist_ok=True)
os.makedirs(BASE_DIR / "assets/samples", exist_ok=True)

# Mount static files
app.mount("/web", StaticFiles(directory=str(BASE_DIR / "web")), name="web")
app.mount("/samples", StaticFiles(directory=str(BASE_DIR / "assets/samples")), name="samples")

depth_estimator = DepthEstimator.get_instance()

@app.get("/")
def read_root():
    return FileResponse(str(BASE_DIR / "web/index.html"))

@app.get("/samples_list")
def get_samples():
    samples = []
    for f in os.listdir(BASE_DIR / "assets/samples"):
        if f.endswith((".jpg", ".png", ".jpeg")):
            samples.append({"name": f, "url": f"/samples/{f}"})
    return samples

def encode_array(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("utf-8")

@app.post("/process")
async def process_image(file: UploadFile = File(None), sample_name: str = Form(None)):
    try:
        if file is not None:
            image_data = await file.read()
            image = Image.open(BytesIO(image_data)).convert("RGB")
        elif sample_name is not None:
            path = BASE_DIR / "assets/samples" / sample_name
            image = Image.open(str(path)).convert("RGB")
        else:
            return JSONResponse({"error": "No image provided"}, status_code=400)
            
        # Limit texture resolution to avoid huge payloads and slow rendering
        max_tex_res = 1024
        W, H = image.size
        if max(H, W) > max_tex_res:
            scale = max_tex_res / max(H, W)
            new_W = int(W * scale)
            new_H = int(H * scale)
            image = image.resize((new_W, new_H), Image.Resampling.LANCZOS)
            W, H = image.size

        # Downsample further for depth and mesh to keep vertex count reasonable (e.g. 256 max)
        mesh_res = 384
        scale_mesh = min(mesh_res / max(H, W), 1.0)
        mesh_W = int(W * scale_mesh)
        mesh_H = int(H * scale_mesh)
        
        # Estimate depth
        Z_map = depth_estimator.estimate_depth(image, max_resolution=518)
        
        # Resize depth map to mesh resolution
        import cv2
        Z_mesh = cv2.resize(Z_map, (mesh_W, mesh_H), interpolation=cv2.INTER_LINEAR)
        
        # Camera intrinsics for mesh generation
        fov_deg = 60.0
        K = intrinsics_from_fov(mesh_W, mesh_H, fov_deg)
        
        # Build mesh
        vertices, indices, uvs = build_mesh(Z_mesh, K, depth_threshold=0.1)
        
        # Convert texture to base64
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        texture_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return {
            "vertices": encode_array(vertices),
            "indices": encode_array(indices),
            "uvs": encode_array(uvs),
            "texture": texture_b64,
            "K": K.tolist(),
            "width": mesh_W,
            "height": mesh_H
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
