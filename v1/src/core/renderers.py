import base64
import io
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from src.core.math_engine import TransformEngine
from src.core.mpi_renderer import (
    build_layers,
    calibrate_motion,
    render_atlas_mpi,
)
from src.core.view_atlas import orbit_angles, render_atlas


class BaseRenderer(ABC):
    @abstractmethod
    def render(
        self,
        img_array: np.ndarray[Any, Any],
        disparity: np.ndarray[Any, Any],
        engine: TransformEngine,
        z_near: float,
        z_far: float,
        z_pivot: float,
        n_layers: int,
        preset_ratio: float,
        max_yaw: float,
        max_pitch: float,
        n_yaw: int,
        n_pitch: int,
        sharpen_amount: float,
        status_callback: Callable[[str, int], None],
    ) -> dict[str, Any]:
        """Render the 3D scene into a series of frames."""


class MPIRenderer(BaseRenderer):
    def render(
        self,
        img_array: np.ndarray[Any, Any],
        disparity: np.ndarray[Any, Any],
        engine: TransformEngine,
        z_near: float,
        z_far: float,
        z_pivot: float,
        n_layers: int,
        preset_ratio: float,
        max_yaw: float,
        max_pitch: float,
        n_yaw: int,
        n_pitch: int,
        sharpen_amount: float,
        status_callback: Callable[[str, int], None],
    ) -> dict[str, Any]:
        H, W, _ = img_array.shape
        status_callback("Building layers...", 50)
        layers, z_k = build_layers(img_array, disparity, z_near, z_far, n_layers)

        status_callback("Rendering views...", 75)
        # Using np.max on a typed array returns Any, so we ignore type issues here
        # or cast to float.
        z_k_max = float(np.max(z_k))
        baseline_x = calibrate_motion(
            z_near, z_k_max, float(engine.K[0, 0]), W, max_yaw, preset_ratio
        )
        baseline_y = calibrate_motion(
            z_near, z_k_max, float(engine.K[1, 1]), H, max_pitch, preset_ratio
        )
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

        return {
            "img_array": img_array,
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
            "renderer": "Layers (MPI)",
            "baseline_x": baseline_x,
            "baseline_y": baseline_y,
        }


class PointSplatRenderer(BaseRenderer):
    def render(
        self,
        img_array: np.ndarray[Any, Any],
        disparity: np.ndarray[Any, Any],
        engine: TransformEngine,
        z_near: float,
        z_far: float,
        z_pivot: float,
        n_layers: int,
        preset_ratio: float,
        max_yaw: float,
        max_pitch: float,
        n_yaw: int,
        n_pitch: int,
        sharpen_amount: float,
        status_callback: Callable[[str, int], None],
    ) -> dict[str, Any]:
        H, W, _ = img_array.shape
        status_callback("Rendering views (Point Splatting)...", 50)
        Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
        P = engine.unproject_to_3d(Z_map)
        colors = img_array.reshape(-1, 3)
        angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)

        atlas = render_atlas(
            P,
            colors,
            engine.K,
            H,
            W,
            angles,
            Z_pivot=z_pivot,
            edge_tau=0.05,
            edge_mode="demote",
            s_max=2,
            z_near=z_near,
        )

        frames_b64 = []
        for p in range(n_pitch):
            for y in range(n_yaw):
                img = Image.fromarray(atlas[p, y])
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                frames_b64.append(f"data:image/jpeg;base64,{b64}")

        return {
            "img_array": img_array,
            "disparity": disparity,
            "P": P,
            "atlas": atlas,
            "frames_b64": frames_b64,
            "n_yaw": n_yaw,
            "n_pitch": n_pitch,
            "max_yaw": max_yaw,
            "max_pitch": max_pitch,
            "z_pivot": z_pivot,
            "H": H,
            "W": W,
            "renderer": "Point Splatting",
        }
