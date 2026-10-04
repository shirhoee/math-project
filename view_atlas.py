import numpy as np
from math_engine import TransformEngine

def orbit_angles(n_yaw: int = 9, n_pitch: int = 5, max_yaw: float = 12.0, max_pitch: float = 8.0) -> np.ndarray:
    """
    Computes Euler angles for an orbit.
    Formula: Grid of (pitch, yaw) spaced linearly from -max to +max.
    Outputs: (n_pitch, n_yaw, 2) array of angles in degrees.
    """
    yaw = np.linspace(-max_yaw, max_yaw, n_yaw)
    pitch = np.linspace(max_pitch, -max_pitch, n_pitch)
    yy, pp = np.meshgrid(yaw, pitch)
    angles = np.stack((yy, pp), axis=-1)
    return angles

def render_atlas(points: np.ndarray, colors: np.ndarray, K: np.ndarray, H: int, W: int, angles: np.ndarray, **render_kwargs) -> np.ndarray:
    """
    Renders multiple views of a point cloud.
    Inputs: points (N, 3), colors (N, 3), K (3, 3), angles (n_pitch, n_yaw, 2)
    Outputs: (n_pitch, n_yaw, H, W, 3) uint8 array.
    """
    n_pitch, n_yaw, _ = angles.shape
    engine = TransformEngine(W, H, fov_deg=60)
    # Ensure fx, fy, cx, cy are matched to K (though TransformEngine creates its own, we override if needed, but we can just use the provided engine's methods or set them)
    engine.fx = K[0, 0]
    engine.fy = K[1, 1]
    engine.cx = K[0, 2]
    engine.cy = K[1, 2]
    
    atlas = np.zeros((n_pitch, n_yaw, H, W, 3), dtype=np.uint8)
    # Loop allowed over small constant set of views
    for p in range(n_pitch):
        for y in range(n_yaw):
            yaw_deg = angles[p, y, 0]
            pitch_deg = angles[p, y, 1]
            
            yaw_rad = np.radians(yaw_deg)
            pitch_rad = np.radians(pitch_deg)
            
            R = engine.get_rotation_matrix(pitch_rad, yaw_rad, 0.0)
            P_new = engine.apply_transform(points, R, Z_pivot=render_kwargs.get('Z_pivot', None))
            
            canvas, depth = engine.project_to_2d(P_new, colors, **render_kwargs)
            canvas_filled, _ = engine.fill_holes_pyramid(canvas, depth)
            
            atlas[p, y] = canvas_filled
            
    return atlas

def render_external_view(points: np.ndarray, colors: np.ndarray, K: np.ndarray, H: int, W: int, yaw: float = 55.0, pitch: float = 20.0, pullback: float = 3.0) -> np.ndarray:
    """
    Renders the point cloud from an external side viewpoint.
    Inputs: points (N, 3), K (3, 3)
    Outputs: (H, W, 3) uint8 array.
    """
    engine = TransformEngine(W, H, fov_deg=60)
    engine.fx = K[0, 0]
    engine.fy = K[1, 1]
    engine.cx = K[0, 2]
    engine.cy = K[1, 2]
    
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    R = engine.get_rotation_matrix(pitch_rad, yaw_rad, 0.0)
    
    Z_pivot = float(np.median(points[:, 2]))
    # Apply rotation and pull back the camera
    t = np.array([0.0, 0.0, pullback], dtype=np.float32)
    P_new = engine.apply_transform(points, R, t=t, Z_pivot=Z_pivot)
    
    # Larger splat size, no fill, dark background
    canvas, depth = engine.project_to_2d(P_new, colors, splat_gain=2.0, s_max=3)
    # Dark background is handled by default zero canvas
    return canvas

def depth_colormap(d: np.ndarray) -> np.ndarray:
    """
    Maps normalized disparity to a color map using a piecewise-linear lookup.
    Inputs: d (H, W) array in [0, 1]
    Outputs: (H, W, 3) uint8 array.
    """
    d_clip = np.clip(d, 0, 1)
    
    # Hand-written piecewise linear colormap (Turbo-like or similar)
    # Points: 0: blue, 0.25: cyan, 0.5: green, 0.75: yellow, 1: red
    cmap_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cmap_r = np.array([0, 0, 0, 255, 255])
    cmap_g = np.array([0, 255, 255, 255, 0])
    cmap_b = np.array([255, 255, 0, 0, 0])
    
    r = np.interp(d_clip, cmap_x, cmap_r)
    g = np.interp(d_clip, cmap_x, cmap_g)
    b = np.interp(d_clip, cmap_x, cmap_b)
    
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return rgb
