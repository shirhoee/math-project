import numpy as np

from math_engine import TransformEngine
from mpi_renderer import calibrate_motion
from view_atlas import orbit_angles


def build_synthetic_planes(W, H, z_near, z_far, n_layers):
    layers = np.zeros((n_layers, H, W, 5), dtype=np.float32)
    z_k = np.linspace(z_near, z_far, n_layers)

    # Draw a single white square in the center (small so it doesn't clip)
    checker = np.zeros((H, W), dtype=np.float32)
    checker[H // 2 - 16 : H // 2 + 16, W // 2 - 16 : W // 2 + 16] = 1.0

    # Near plane
    layers[0, ..., 0] = checker
    layers[0, ..., 1] = checker
    layers[0, ..., 2] = checker
    layers[0, ..., 3] = 1.0  # alpha
    layers[0, ..., 4] = 1.0  # weight

    # Far plane (last layer)
    layers[-1, ..., 0] = checker
    layers[-1, ..., 1] = checker
    layers[-1, ..., 2] = checker
    layers[-1, ..., 3] = 1.0
    layers[-1, ..., 4] = 1.0

    # Set middle layers transparent
    for i in range(1, n_layers - 1):
        layers[i, ..., 3] = 0.0

    return layers, z_k


def measure_shift(img1, img2):
    # Just compute the centroid of the image and return the difference in X
    img1_gray = np.mean(img1, axis=2)
    img2_gray = np.mean(img2, axis=2)

    y, x = np.meshgrid(np.arange(img1.shape[0]), np.arange(img1.shape[1]), indexing="ij")

    m1 = np.sum(img1_gray)
    m2 = np.sum(img2_gray)

    if m1 == 0 or m2 == 0:
        return 0

    cx1 = np.sum(img1_gray * x) / m1
    cx2 = np.sum(img2_gray * x) / m2

    return cx2 - cx1


def run_measurement():
    W, H = 512, 512
    engine = TransformEngine(H, W)
    z_near, z_far = 1.0, 4.0
    z_pivot = 0.8 * z_far + 0.2 * z_near
    n_layers = 16

    layers, z_k = build_synthetic_planes(W, H, z_near, z_far, n_layers)

    presets = {"Subtle": 0.04, "Normal": 0.07, "Dramatic": 0.11}
    for name, preset in presets.items():
        max_yaw = 20.0
        max_pitch = 8.0
        bx = calibrate_motion(z_near, z_far, engine.K[0, 0], W, max_yaw, preset, z_pivot=z_pivot)
        by = calibrate_motion(z_near, z_far, engine.K[1, 1], H, max_pitch, preset, z_pivot=z_pivot)

        # Grid density
        parallax_total = 2 * preset * W
        n_yaw = min(41, int(np.ceil(parallax_total / 2.5)) | 1)
        n_pitch = 3

        angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)

        # Center frame is (1, n_yaw//2). Extreme is (1, 0)
        # But wait, render_atlas_mpi returns base64. Let's just use layer_homographies directly
        # to render to numpy array.
        import cv2

        import mpi_renderer

        # Center
        R_center = engine.get_rotation_matrix(0, 0, 0)
        c = np.array([0.0, 0.0, z_pivot], dtype=np.float32)
        t_center = c - R_center @ c + np.array([0.0, 0.0, 0.0], dtype=np.float32)
        K_render = engine.K.copy()

        # We need a fixed zoom across all frames. Let's compute it for max_yaw!
        R_ext = engine.get_rotation_matrix(0, np.radians(max_yaw), 0)
        t_ext = c - R_ext @ c + np.array([bx, 0.0, 0.0], dtype=np.float32)

        shift_x_px_ext = np.abs(t_ext[0] * engine.K[0, 0] / z_near + engine.K[0, 0] * R_ext[0, 2])
        shift_y_px_ext = np.abs(t_ext[1] * engine.K[1, 1] / z_near + engine.K[1, 1] * R_ext[1, 2])
        margin_ext = max(shift_x_px_ext / (W / 2.0), shift_y_px_ext / (H / 2.0))
        zoom = 1.0 + margin_ext
        K_render[0, 0] *= zoom
        K_render[1, 1] *= zoom

        H_center = mpi_renderer.layer_homographies(
            engine.K, R_center, t_center, z_k, K_render=K_render
        )

        layers_near = layers.copy()
        layers_near[1:] = 0
        frame_center_near = mpi_renderer.render_mpi(layers_near, H_center)

        H_ext = mpi_renderer.layer_homographies(engine.K, R_ext, t_ext, z_k, K_render=K_render)
        frame_ext_near = mpi_renderer.render_mpi(layers_near, H_ext)

        shift_near = measure_shift(frame_center_near, frame_ext_near)

        cv2.imwrite(
            f"{name}_center_near.png", (frame_center_near[..., ::-1] * 255).astype(np.uint8)
        )
        cv2.imwrite(f"{name}_ext_near.png", (frame_ext_near[..., ::-1] * 255).astype(np.uint8))

        layers_far = layers.copy()
        layers_far[:-1] = 0
        frame_center_far = mpi_renderer.render_mpi(layers_far, H_center)
        frame_ext_far = mpi_renderer.render_mpi(layers_far, H_ext)

        shift_far = measure_shift(frame_center_far, frame_ext_far)

        parallax = np.abs(shift_near - shift_far)
        ratio = parallax / (preset * W)
        print(
            f"{name}: shift_near={shift_near}, shift_far={shift_far}, parallax={parallax}, target={preset * W}, ratio={ratio:.3f}"
        )


if __name__ == "__main__":
    run_measurement()
