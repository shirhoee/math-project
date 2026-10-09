import glob
import hashlib
import os
import pickle
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

print("Importing depth_estimator...")
from src.core.depth_estimator import DepthEstimator

print("Importing math_engine...")
from src.core.math_engine import TransformEngine

print("Importing mpi_renderer...")
from src.core.mpi_renderer import build_layers, calibrate_motion, render_atlas_mpi


def get_hash(img_path, settings_str):
    with open(img_path, "rb") as f:
        data = f.read()
    h = hashlib.md5(data + settings_str.encode("utf-8")).hexdigest()
    return h[:12]


def precompute():
    sample_imgs = sorted(glob.glob("assets/samples/01_*.jpg"))
    cache_dir = "assets/samples/cache"
    os.makedirs(cache_dir, exist_ok=True)

    model = DepthEstimator(size="Small")

    for img_path in sample_imgs:
        print(f"Precomputing {img_path}...")

        image = Image.open(img_path).convert("RGB")
        if image.width > 1024:
            ratio = 1024.0 / image.width
            new_size = (1024, int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        temp_path = "temp_resized_precompute.jpg"
        image.save(temp_path)

        img_array = np.array(image)
        H, W, _ = img_array.shape

        # Settings
        z_near = 1.0
        z_far = 4.0
        n_layers = 16
        fov = 60.0
        preset_ratio = 0.07  # Normal

        settings_str = f"W{W}_H{H}_Layers{n_layers}_Preset{preset_ratio}"
        h = get_hash(img_path, settings_str)
        cache_path = os.path.join(cache_dir, f"{h}.pkl")

        if os.path.exists(cache_path):
            print(f" Already cached at {cache_path}")
            continue

        disparity = model.estimate_depth(temp_path, refine_depth=True)

        engine = TransformEngine(W, H, fov)
        Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
        P = engine.unproject_to_3d(Z_map)
        z_pivot = float(np.median(P[:, 2]))

        layers, z_k = build_layers(img_array, disparity, z_near, z_far, n_layers=n_layers)

        parallax_total = 2 * preset_ratio * W
        n_yaw = min(41, int(np.ceil(parallax_total / 2.5)) | 1)
        n_pitch = 3
        max_yaw = 15.0
        max_pitch = 5.0
        sharpen_amount = 0.35

        baseline_x = calibrate_motion(z_near, np.max(z_k), engine.K[0, 0], W, max_yaw, preset_ratio)
        baseline_y = calibrate_motion(
            z_near, np.max(z_k), engine.K[1, 1], H, max_pitch, preset_ratio
        )

        from src.core.view_atlas import orbit_angles

        angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)

        frames_b64 = render_atlas_mpi(
            layers,
            z_k,
            engine.K,
            angles,
            baseline_x,
            baseline_y,
            max_yaw,
            max_pitch,
            sharpen_amount,
            z_pivot,
        )

        data = {
            "disparity": disparity,
            "layers": layers,
            "z_k": z_k,
            "frames_b64": frames_b64,
            "n_yaw": n_yaw,
            "n_pitch": n_pitch,
            "max_yaw": max_yaw,
            "max_pitch": max_pitch,
            "z_pivot": z_pivot,
            "H": H,
            "W": W,
        }

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

        print(f" Saved to {cache_path}")


if __name__ == "__main__":
    precompute()
