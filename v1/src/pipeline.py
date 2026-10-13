import glob
import hashlib
import os
import pickle
from typing import Any

from src.celery_app import celery_app
from src.tasks import build_render_job
from celery.result import AsyncResult  # type: ignore[import-untyped]

job_id_mapping: dict[str, Any] = {}
task_progress_registry: dict[str, dict[str, Any]] = {}

def get_hash(img_path: str, settings_str: str) -> str:
    s = f"{img_path}_{settings_str}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

def submit_render_job(
    job_id: Any,
    img_path: str | None,
    renderer: str,
    motion_preset_str: str,
    z_near: float,
    z_far: float,
    n_layers: int,
    sharpen_amount: float,
) -> dict[Any, Any] | None:
    is_fast_test = os.environ.get("FAST_TEST") == "1"
    task_progress_registry[str(job_id)] = {"status": "Starting pipeline...", "progress": 5}

    if img_path and "upload_" not in img_path and renderer == "Layers (MPI)" and not is_fast_test:
        from PIL import Image
        import numpy as np
        image = Image.open(img_path)
        max_dim = 512
        W, H = image.width, image.height
        if max(W, H) > max_dim:
            ratio = max_dim / float(max(W, H))
            W, H = int(W * ratio), int(H * ratio)

        preset_ratio = float(motion_preset_str.split("(")[1].split(")")[0])
        parallax_total = 2 * preset_ratio * W
        n_yaw = min(29, max(17, int(np.ceil(parallax_total / 2.0)) | 1))
        n_pitch = 3
        settings_str = f"CodeV3_W{W}_H{H}_Grid{n_yaw}x{n_pitch}_Preset{preset_ratio}_Z{z_near}-{z_far}_L{n_layers}"
        h = get_hash(img_path, settings_str)
        cache_path = os.path.join(os.environ.get("CACHE_DIR", "outputs/cache"), f"{h}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                data["renderer"] = renderer
                task_progress_registry[str(job_id)] = {"status": "Complete (loaded from cache)!", "progress": 100}
                from typing import cast
                return cast(dict[Any, Any], data)
            except Exception:
                pass

    if is_fast_test:
        task = build_render_job.delay(
            img_path,
            renderer,
            motion_preset_str,
            z_near,
            z_far,
            n_layers,
            sharpen_amount,
            is_fast_test,
            job_id,
        )
        job_id_mapping[job_id] = task
    elif getattr(celery_app.conf, "task_always_eager", False):
        import concurrent.futures

        if not hasattr(submit_render_job, "_executor"):
            setattr(submit_render_job, "_executor", concurrent.futures.ThreadPoolExecutor(max_workers=2))
        executor = getattr(submit_render_job, "_executor")

        def _run_eager() -> Any:
            return build_render_job.delay(
                img_path,
                renderer,
                motion_preset_str,
                z_near,
                z_far,
                n_layers,
                sharpen_amount,
                is_fast_test,
                job_id,
            )

        future = executor.submit(_run_eager)
        job_id_mapping[job_id] = future
    else:
        task = build_render_job.delay(
            img_path,
            renderer,
            motion_preset_str,
            z_near,
            z_far,
            n_layers,
            sharpen_amount,
            is_fast_test,
            job_id,
        )
        job_id_mapping[job_id] = task.id
        
    return None

def poll_job_status(job_id: Any) -> dict[str, Any]:
    task_id_or_obj = job_id_mapping.get(job_id)
    if not task_id_or_obj:
        return {"status": None, "progress": 0, "result": None}

    reg = task_progress_registry.get(str(job_id), {})
    current_status = reg.get("status", "Synthesizing 3D novel views...")
    current_progress = reg.get("progress", 25)

    import concurrent.futures
    if isinstance(task_id_or_obj, concurrent.futures.Future):
        if task_id_or_obj.done():
            try:
                res = task_id_or_obj.result()
                if res and res.successful():
                    return {"status": None, "progress": 100, "result": res.result}
                else:
                    return {"status": f"Error: {res.result if res else 'Failed'}", "progress": 0, "result": None}
            except Exception as e:
                return {"status": f"Error: {e}", "progress": 0, "result": None}
        else:
            return {"status": current_status, "progress": current_progress, "result": None}
        
    if isinstance(task_id_or_obj, str):
        res = AsyncResult(task_id_or_obj)
    else:
        res = task_id_or_obj  # EagerResult
        
    if res.ready():
        if res.successful():
            return {"status": None, "progress": 100, "result": res.result}
        else:
            return {"status": f"Error: {res.result}", "progress": 0, "result": None}
            
    meta = res.info or {}
    if isinstance(meta, dict) and meta.get("progress"):
        status = meta.get("status", current_status)
        progress = meta.get("progress", current_progress)
    else:
        status = current_status
        progress = current_progress
        
    return {
        "status": status,
        "progress": progress,
        "result": None,
    }

def get_job_result(job_id: Any) -> dict[Any, Any] | None:
    task_id_or_obj = job_id_mapping.get(job_id)
    if not task_id_or_obj:
        return None

    import concurrent.futures
    if isinstance(task_id_or_obj, concurrent.futures.Future):
        if task_id_or_obj.done():
            try:
                res = task_id_or_obj.result()
                if res and res.successful():
                    return res.result
            except Exception:
                return None
        return None
        
    if isinstance(task_id_or_obj, str):
        res = AsyncResult(task_id_or_obj)
    else:
        res = task_id_or_obj
        
    if res.ready() and res.successful():
        return res.result
    return None

def cleanup_cache(cache_dir: str = "outputs/cache") -> None:
    if os.path.exists(cache_dir):
        for f in glob.glob(f"{cache_dir}/*.pkl"):
            try:
                with open(f, "rb") as fp:
                    data = pickle.load(fp)
                img_path = data.get("img_path", "")
                settings_str = data.get("settings_str", "")
                if get_hash(img_path, settings_str) != os.path.basename(f).replace(".pkl", ""):
                    os.remove(f)
            except Exception:
                try:
                    os.remove(f)
                except:
                    pass
