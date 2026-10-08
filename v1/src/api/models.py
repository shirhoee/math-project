from typing import Any

from pydantic import BaseModel


class JobResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    status: str | None
    progress: int
    result: dict[str, Any] | None = None


class ResultResponse(BaseModel):
    # This payload can be quite large, representing the output of pipeline.py
    # We will just mirror the dict structure but define a basic Pydantic model for it.
    renderer: str
    H: int
    W: int
    frames_b64: list[str]
    # PointSplatting has P, atlas
    # MPI has layers, z_k, n_yaw, n_pitch, max_yaw, max_pitch, z_pivot, baseline_x, baseline_y, cache_key, settings_str, img_path

    # We'll allow arbitrary extra fields for now using Config
    model_config = {"extra": "allow"}
