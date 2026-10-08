import glob
import hashlib
import os
import pickle
import threading
from typing import Any

import numpy as np
from PIL import Image

from src.core.depth_estimator import get_depth_model
from src.core.math_engine import TransformEngine
from src.core.renderers import MPIRenderer, PointSplatRenderer

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx as _add_script_run_ctx

    def add_script_run_ctx(t: threading.Thread) -> threading.Thread:
        return _add_script_run_ctx(t)
except ImportError:

    def add_script_run_ctx(t: threading.Thread) -> threading.Thread:
        return t


global_job_results: dict[Any, Any] = {}
global_job_status: dict[Any, Any] = {}
global_job_progress: dict[Any, Any] = {}


def get_hash(img_path: str, settings_str: str) -> str:
    s = f"{img_path}_{settings_str}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def run_build_job(
    job_id: Any,
    img_path: str | None,
    renderer: str,
    motion_preset_str: str,
    z_near: float,
    z_far: float,
    n_layers: int,
    sharpen_amount: float,
    is_fast_test: bool,
) -> None:
    try:
        global_job_status[job_id] = "Loading image..."
        global_job_progress[job_id] = 5
        if not img_path:
            global_job_status[job_id] = "No image selected."
            return
        image = Image.open(img_path).convert("RGB")

        max_dim = 256 if is_fast_test else 512
        if max(image.width, image.height) > max_dim:
            ratio = max_dim / float(max(image.width, image.height))
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        temp_path = os.path.join("outputs", "temp_resized_app.jpg")
        image.save(temp_path)
        img_array = np.array(image)
        H, W, _ = img_array.shape

        preset_ratio = float(motion_preset_str.split("(")[1].split(")")[0])
        parallax_total = 2 * preset_ratio * W
        n_yaw = min(41, int(np.ceil(parallax_total / 2.5)) | 1)
        n_pitch = 3
        if is_fast_test:
            n_yaw, n_pitch = 3, 1

        settings_str = f"CodeV2_W{W}_H{H}_Grid{n_yaw}x{n_pitch}_Preset{preset_ratio}_Z{z_near}-{z_far}_L{n_layers}"
        cache_dir = os.environ.get("CACHE_DIR", "outputs/cache")
        os.makedirs(cache_dir, exist_ok=True)
        h = get_hash(img_path, settings_str)
        cache_path = os.path.join(cache_dir, f"{h}.pkl")

        global_job_status[job_id] = "Estimating depth..."
        global_job_progress[job_id] = 20
        model = get_depth_model("Small")
        disparity = model.estimate_depth(temp_path, refine_depth=not is_fast_test)

        fov_deg = 60.0
        engine = TransformEngine(W, H, fov_deg)
        z_pivot = 0.8 * z_far + 0.2 * z_near

        parallax_total = 2 * preset_ratio * W
        n_yaw = min(41, int(np.ceil(parallax_total / 2.5)) | 1)
        n_pitch = 3
        max_yaw = 20.0
        max_pitch = 5.0
        if is_fast_test:
            n_yaw, n_pitch = 3, 1
            max_yaw, max_pitch = 5.0, 0.0

        def status_callback(msg: str, prog: int) -> None:
            global_job_status[job_id] = msg
            global_job_progress[job_id] = prog

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

        global_job_results[job_id] = result
        global_job_status[job_id] = None
    except Exception as e:
        global_job_status[job_id] = f"Error: {e!s}"
        global_job_progress[job_id] = 0


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

    if img_path and "upload_" not in img_path and renderer == "Layers (MPI)" and not is_fast_test:
        image = Image.open(img_path)
        max_dim = 512
        W, H = image.width, image.height
        if max(W, H) > max_dim:
            ratio = max_dim / float(max(W, H))
            W, H = int(W * ratio), int(H * ratio)

        preset_ratio = float(motion_preset_str.split("(")[1].split(")")[0])
        parallax_total = 2 * preset_ratio * W
        n_yaw = min(41, int(np.ceil(parallax_total / 2.5)) | 1)
        n_pitch = 3
        settings_str = f"CodeV2_W{W}_H{H}_Grid{n_yaw}x{n_pitch}_Preset{preset_ratio}_Z{z_near}-{z_far}_L{n_layers}"
        h = get_hash(img_path, settings_str)
        cache_path = os.path.join(os.environ.get("CACHE_DIR", "outputs/cache"), f"{h}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                data["renderer"] = renderer
                global_job_status[job_id] = None
                global_job_results[job_id] = data
                from typing import cast

                return cast(dict[Any, Any], data)
            except Exception:
                pass

    t = threading.Thread(
        target=run_build_job,
        args=(
            job_id,
            img_path,
            renderer,
            motion_preset_str,
            z_near,
            z_far,
            n_layers,
            sharpen_amount,
            is_fast_test,
        ),
    )
    t.daemon = True
    add_script_run_ctx(t)
    t.start()
    return None


def poll_job_status(job_id: Any) -> dict[str, Any]:
    return {
        "status": global_job_status.get(job_id),
        "progress": global_job_progress.get(job_id, 0),
        "result": global_job_results.get(job_id),
    }


def get_job_result(job_id: Any) -> dict[Any, Any] | None:
    return global_job_results.get(job_id)


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

