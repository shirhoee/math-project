import os
import shutil
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.models import JobResponse, ResultResponse, StatusResponse
from src.pipeline import get_job_result, poll_job_status, submit_render_job

app = FastAPI(title="3D View Synthesis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/render", response_model=JobResponse)
def render_image(
    file: UploadFile = File(...),  # noqa: B008
    renderer: str = Form("Layers (MPI)"),
    motion_preset_str: str = Form("Normal (0.07)"),
    z_near: float = Form(1.0),
    z_far: float = Form(4.0),
    n_layers: int = Form(16),
    sharpen_amount: float = Form(0.35),
) -> Any:
    os.makedirs("outputs/uploads", exist_ok=True)
    job_id = str(uuid.uuid4())
    img_path = f"outputs/uploads/{job_id}_{file.filename}"

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    submit_render_job(
        job_id=job_id,
        img_path=img_path,
        renderer=renderer,
        motion_preset_str=motion_preset_str,
        z_near=z_near,
        z_far=z_far,
        n_layers=n_layers,
        sharpen_amount=sharpen_amount,
    )

    return JobResponse(job_id=job_id)


def clean_result(result: Any) -> Any:
    if not result:
        return result
    cleaned = {}
    for k, v in result.items():
        if k in ("layers", "P", "atlas"):
            # Avoid serializing massive multi-megabyte numpy float arrays directly into JSON
            if hasattr(v, "shape"):
                cleaned[f"{k}_shape"] = list(v.shape)
            continue
        if hasattr(v, "tolist"):
            cleaned[k] = v.tolist()
        else:
            cleaned[k] = v
    return cleaned


@app.get("/api/v1/jobs/{job_id}/status", response_model=StatusResponse)
def get_status(job_id: str) -> Any:
    status_dict = poll_job_status(job_id)
    if status_dict.get("result"):
        status_dict["result"] = clean_result(status_dict["result"])
    return StatusResponse(**status_dict)


@app.get("/api/v1/jobs/{job_id}/result", response_model=ResultResponse)
def get_result(job_id: str) -> Any:
    result = get_job_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job result not found or not finished.")

    return clean_result(result)


@app.get("/api/v1/samples")
def get_samples() -> list[dict[str, str]]:
    samples_dir = "assets/samples"
    if not os.path.exists(samples_dir):
        # Fallback if working dir is not v1
        samples_dir = "v1/assets/samples"
    
    if not os.path.exists(samples_dir):
        return []

    valid_exts = {".jpg", ".jpeg", ".png"}
    samples = []
    for fname in sorted(os.listdir(samples_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts and not fname.startswith("upload_") and not fname.startswith("temp_"):
            samples.append({
                "id": os.path.splitext(fname)[0],
                "name": fname.replace("_", " ").replace(".jpg", "").title(),
                "filename": fname,
                "url": f"/api/v1/samples/{fname}",
            })
    return samples


@app.get("/api/v1/samples/{filename}")
def get_sample_file(filename: str) -> Any:
    # Security: prevent directory traversal
    base_name = os.path.basename(filename)
    for candidate in ["assets/samples", "v1/assets/samples"]:
        path = os.path.join(candidate, base_name)
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Sample not found")


# Mount static frontend build (must be after /api routes)
base_api_dir = os.path.dirname(os.path.abspath(__file__))
possible_dist = os.path.normpath(os.path.join(base_api_dir, "../../frontend/dist"))

for static_cand in [possible_dist, "frontend/dist", "v1/frontend/dist"]:
    if os.path.exists(static_cand) and os.path.exists(os.path.join(static_cand, "index.html")):
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=static_cand, html=True), name="frontend")
        break



