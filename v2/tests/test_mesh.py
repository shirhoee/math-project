import numpy as np
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
