import glob
import os
import sys

import imageio
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from depth_estimator import DepthEstimator
from math_engine import TransformEngine
from mpi_renderer import build_layers, layer_homographies, render_mpi, unsharp_mask


def main():
    model = DepthEstimator("Small")
    os.makedirs("docs/results", exist_ok=True)

    samples = sorted(glob.glob("assets/samples/01_*.jpg"))
    for f in samples:
        name = os.path.basename(f)[:10]
        print(f"Processing {name}...")

        img = Image.open(f).convert("RGB")
        # Resize to 512px height
        ratio = 512.0 / img.height
        W = int(img.width * ratio)
        H = 512
        img = img.resize((W, H), Image.Resampling.LANCZOS)
        img_array = np.array(img)

        # Save temp image for depth estimator
        img.save("temp_demo.jpg")
        disparity = model.estimate_depth("temp_demo.jpg", refine_depth=True)

        z_near, z_far = 1.0, 4.0
        z_k = np.linspace(z_near, z_far, 16).astype(np.float32)
        z_pivot = 0.8 * z_far + 0.2 * z_near
        layers, z_k_ret = build_layers(img_array, disparity, z_near, z_far, n_layers=16)

        engine = TransformEngine(W, H, 60.0)
        max_yaw = 20.0

        n_yaw = 41
        yaws = np.linspace(-max_yaw, max_yaw, n_yaw)

        from mpi_renderer import calibrate_motion

        target_ratio = 0.11
        baseline_x = calibrate_motion(
            z_near, z_far, engine.K[0, 0], W, max_yaw, target_ratio, z_pivot
        )

        frames = []
        for yaw in yaws:
            R = engine.get_rotation_matrix(0.0, np.radians(yaw), 0.0)
            c = np.array([0, 0, z_pivot], dtype=np.float32)
            t = c - R @ c + np.array([baseline_x * (yaw / max_yaw), 0.0, 0.0], dtype=np.float32)

            shift_x = np.abs(t[0] * engine.K[0, 0] / z_near + engine.K[0, 0] * R[0, 2])
            zoom = min(1.05, 1.0 + shift_x / (W / 2.0))

            K_render = engine.K.copy()
            K_render[0, 0] *= zoom
            K_render[1, 1] *= zoom

            H_k = layer_homographies(engine.K, R, t, z_k, K_render=K_render)
            frame = render_mpi(layers, H_k)
            frame = unsharp_mask(frame, 0.35)
            frames.append(frame)

        # Left-to-right-to-left
        sweep = frames + frames[-2:0:-1]

        out_path = f"docs/results/demo_{name}.webp"
        imageio.mimsave(out_path, sweep, fps=24, loop=0)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
