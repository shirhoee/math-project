import os
import shutil
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.api.models import JobResponse, ResultResponse, StatusResponse
from src.pipeline import get_job_result, poll_job_status, submit_render_job

app = FastAPI(title="3D View Synthesis API")


@app.post("/api/v1/render", response_model=JobResponse)
async def render_image(
    file: UploadFile = File(...),
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

    # We remove numpy arrays before returning JSON, or we can just let Pydantic try to serialize them.
    # FastAPI/Pydantic cannot natively serialize numpy arrays, so we must clean them.
    return clean_result(result)
