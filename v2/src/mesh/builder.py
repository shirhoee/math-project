import numpy as np

def build_mesh(Z: np.ndarray, K: np.ndarray, depth_threshold: float = 0.10):
    """
    Builds a 3D mesh from a depth map.
    Z: (H, W) float array of depth values.
    K: (3, 3) intrinsic matrix.
    depth_threshold: fraction of max depth to cut discontinuities.
    Returns: vertices (N, 3), indices (M, 3), uvs (N, 2)
    """
    H, W = Z.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    
    vertices = np.stack((X, Y, Z), axis=-1).reshape(-1, 3).astype(np.float32)
    
    # UVs
    u_norm = uu / (W - 1)
    v_norm = vv / (H - 1)
    # y-axis is inverted in WebGL usually, but here we keep image coordinates
    # For three.js v is inverted, so v_norm = 1.0 - v_norm. Let's do that.
    v_norm = 1.0 - v_norm
    uvs = np.stack((u_norm, v_norm), axis=-1).reshape(-1, 2).astype(np.float32)
    
    # Indices
    idx = np.arange(H * W).reshape(H, W)
    
    top_left = idx[:-1, :-1].flatten()
    top_right = idx[:-1, 1:].flatten()
    bottom_left = idx[1:, :-1].flatten()
    bottom_right = idx[1:, 1:].flatten()
    
    # Tri 1: TL, BL, TR
    tri1 = np.stack((top_left, bottom_left, top_right), axis=-1)
    # Tri 2: TR, BL, BR
    tri2 = np.stack((top_right, bottom_left, bottom_right), axis=-1)
    
    indices = np.vstack((tri1, tri2))
    
    # Discontinuity filtering
    Z_flat = Z.flatten()
    z_tri = Z_flat[indices]
    z_max_tri = np.max(z_tri, axis=1)
    z_min_tri = np.min(z_tri, axis=1)
    
    max_depth = np.max(Z)
    valid_mask = (z_max_tri - z_min_tri) < (depth_threshold * max_depth)
    indices = indices[valid_mask].astype(np.uint32)
    
    return vertices, indices, uvs
