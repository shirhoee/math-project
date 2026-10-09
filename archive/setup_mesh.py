import os
from pathlib import Path

os.makedirs('v2/src/mesh', exist_ok=True)
Path('v2/src/mesh/__init__.py').write_text('')

builder_code = '''import numpy as np

def build_mesh(Z: np.ndarray, K: np.ndarray, depth_threshold: float = 0.10):
    \"\"\"
    Builds a 3D mesh from a depth map.
    Z: (H, W) float array of depth values.
    K: (3, 3) intrinsic matrix.
    depth_threshold: fraction of max depth to cut discontinuities.
    Returns: vertices (N, 3), indices (M, 3), uvs (N, 2)
    \"\"\"
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
'''
Path('v2/src/mesh/builder.py').write_text(builder_code)

renderer_code = '''import numpy as np

def reference_render(vertices: np.ndarray, colors: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int) -> np.ndarray:
    \"\"\"
    A simple point-based reference renderer for tests.
    vertices: (N, 3)
    colors: (N, 3)
    Returns: (H, W, 3) image
    \"\"\"
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
'''
Path('v2/src/mesh/renderer.py').write_text(renderer_code)

test_mesh_code = '''import numpy as np
from src.mesh.builder import build_mesh
from src.mesh.renderer import reference_render

def test_build_mesh():
    Z = np.ones((10, 10), dtype=np.float32)
    Z[:, 5:] = 2.0  # create a huge step
    K = np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1]], dtype=np.float32)
    
    v, i, uv = build_mesh(Z, K, depth_threshold=0.1)
    
    assert v.shape == (100, 3)
    assert uv.shape == (100, 2)
    
    # original grid has 9x9x2 = 162 triangles
    # the cut is 9 triangles along the boundary, each composed of 2 triangles
    # Wait, 9 squares = 18 triangles cut. So 162 - 18 = 144 triangles.
    assert i.shape == (144, 3)

def test_reference_render():
    H, W = 10, 10
    Z = np.ones((H, W), dtype=np.float32)
    K = np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1]], dtype=np.float32)
    v, _, uv = build_mesh(Z, K)
    
    # Create fake colors based on UVs
    colors = np.zeros((H*W, 3), dtype=np.float32)
    colors[:, 0] = uv[:, 0]
    
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    
    img = reference_render(v, colors, K, R, t, H, W)
    
    # Check that identity pose reproduces the input exactly (or very closely)
    assert img.shape == (H, W, 3)
    # For pixel (0,0), uv[0,0] is 0
    # For pixel (0,W-1), uv[0,0] is 1.0
    assert np.allclose(img[0, -1, 0], 1.0)
'''
Path('v2/tests/test_mesh.py').write_text(test_mesh_code)
