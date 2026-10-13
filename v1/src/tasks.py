import os
import numpy as np
from PIL import Image
from typing import Any

from src.core.depth_estimator import get_depth_model
from src.core.math_engine import TransformEngine
from src.core.renderers import MPIRenderer, PointSplatRenderer
from src.celery_app import celery_app
import hashlib

def get_hash(img_path: str, settings_str: str) -> str:
    s = f"{img_path}_{settings_str}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]

@celery_app.task(bind=True)
def build_render_job(
    self,
    img_path: str | None,
    renderer: str,
    motion_preset_str: str,
    z_near: float,
    z_far: float,
    n_layers: int,
    sharpen_amount: float,
    is_fast_test: bool,
    job_id: Any = None,
) -> dict[Any, Any] | None:
    def safe_update_state(state_str: str, meta_dict: dict[str, Any]) -> None:
        try:
            self.update_state(state=state_str, meta=meta_dict)
        except Exception:
            pass
        if job_id:
            try:
                from src.pipeline import task_progress_registry
                task_progress_registry[str(job_id)] = meta_dict
            except Exception:
                pass

    try:
        safe_update_state("PROGRESS", {"status": "Loading & preprocessing image...", "progress": 10})
        if not img_path:
            safe_update_state("FAILURE", {"status": "No image selected.", "progress": 0})
            return None
            
        image = Image.open(img_path).convert("RGB")

        max_dim = 256 if is_fast_test else 512
        if max(image.width, image.height) > max_dim:
            ratio = max_dim / float(max(image.width, image.height))
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        os.makedirs("outputs", exist_ok=True)
        temp_path = os.path.join("outputs", "temp_resized_app.jpg")
        image.save(temp_path)
        img_array = np.array(image)
        H, W, _ = img_array.shape

        preset_ratio = float(motion_preset_str.split("(")[1].split(")")[0])
        parallax_total = 2 * preset_ratio * W
        n_yaw = min(29, max(17, int(np.ceil(parallax_total / 2.0)) | 1))
        n_pitch = 3
        if is_fast_test:
            n_yaw, n_pitch = 3, 1

        settings_str = f"CodeV3_W{W}_H{H}_Grid{n_yaw}x{n_pitch}_Preset{preset_ratio}_Z{z_near}-{z_far}_L{n_layers}"
        cache_dir = os.environ.get("CACHE_DIR", "outputs/cache")
        os.makedirs(cache_dir, exist_ok=True)
        h = get_hash(img_path, settings_str)
        cache_path = os.path.join(cache_dir, f"{h}.pkl")

        safe_update_state("PROGRESS", {"status": "Estimating relative depth...", "progress": 25})
        model = get_depth_model("Small")
        disparity = model.estimate_depth(temp_path, refine_depth=not is_fast_test)

        # Maximized Field of View (72 degrees) reduces edge parallax distortion
        fov_deg = 72.0
        engine = TransformEngine(W, H, fov_deg)

        # Anchor camera rotation directly to the focal subject depth
        inv_z_near = 1.0 / z_near
        inv_z_far = 1.0 / z_far
        Z_map = 1.0 / (inv_z_far + disparity * (inv_z_near - inv_z_far))
        central_depth = Z_map[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4]
        z_pivot = float(np.median(central_depth))

        # Restored wide motion boundaries
        max_yaw = 16.0
        max_pitch = 5.0
        if is_fast_test:
            n_yaw, n_pitch = 3, 1
            max_yaw, max_pitch = 4.0, 0.0

        def status_callback(msg: str, prog: int) -> None:
            safe_update_state("PROGRESS", {"status": msg, "progress": prog})

        safe_update_state("PROGRESS", {"status": "Decomposing into 3D Multi-Plane layers...", "progress": 55})
        renderer_impl = MPIRenderer() if renderer == "Layers (MPI)" else PointSplatRenderer()

        result = renderer_impl.render(
            img_array=img_array,
            disparity=disparity,
            engine=engine,
            z_near=z_near,
            z_far=z_far,
            z_pivot=z_pivot,
            n_layers=n_layers,
            preset_ratio=preset_ratio,
            max_yaw=max_yaw,
            max_pitch=max_pitch,
            n_yaw=n_yaw,
            n_pitch=n_pitch,
            sharpen_amount=sharpen_amount,
            status_callback=status_callback,
        )

        if "upload_" not in img_path and not is_fast_test:
            result["cache_key"] = h
            result["settings_str"] = settings_str
            result["img_path"] = img_path
            with open(cache_path, "wb") as f:
                import pickle
                pickle.dump(result, f)

        safe_update_state("SUCCESS", {"status": "Complete!", "progress": 100})
        return result
    except Exception as e:
        safe_update_state("FAILURE", {"status": f"Error: {e!s}", "progress": 0})
        raise e

