import numpy as np

def reference_render(vertices: np.ndarray, colors: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    A simple point-based reference renderer for tests.
    vertices: (N, 3)
    colors: (N, 3)
    Returns: (H, W, 3) image
    """
    P = (vertices @ R.T) + t
    
    # Drop behind camera
    valid = P[:, 2] > 1e-3
    P = P[valid]
    C = colors[valid]
    
    uvw = P @ K.T
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    Z = uvw[:, 2]
    
    u_int = np.rint(u).astype(np.int32)
    v_int = np.rint(v).astype(np.int32)
    
    valid_uv = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    u_int = u_int[valid_uv]
    v_int = v_int[valid_uv]
    Z = Z[valid_uv]
    C = C[valid_uv]
    
    # Z-buffer
    idx = v_int * W + u_int
    canvas = np.zeros((H * W, 3), dtype=np.float32)
    z_buf = np.full(H * W, np.inf, dtype=np.float32)
    
    if len(Z) > 0:
        order = np.argsort(Z)[::-1]  # Draw back to front (painter's algo)
        idx_s = idx[order]
        C_s = C[order]
        canvas[idx_s] = C_s
        
    return canvas.reshape((H, W, 3))
