import numpy as np
import pytest
from src.geometry.math_utils import adjugate_inverse
from src.mesh.builder import build_mesh
from src.mesh.renderer import reference_render

def test_math_gate():
    np.random.seed(0)
    M = np.random.rand(3, 3).astype(np.float32)
    inv1 = adjugate_inverse(M)
    inv2 = np.linalg.inv(M)
    assert np.allclose(inv1, inv2, atol=1e-9), "Math Gate failed: adjugate differs from linalg.inv"

def test_mesh_gate():
    Z = np.ones((50, 50), dtype=np.float32)
    # create a 2.0 depth step, which is 100% of min depth
    Z[:, 25:] = 2.0
    K = np.array([[50, 0, 25], [0, 50, 25], [0, 0, 1]], dtype=np.float32)
    
    threshold = 0.1
    v, i, uv = build_mesh(Z, K, depth_threshold=threshold)
    
    # Check if any triangle spans more than threshold
    # Z of vertices
    Z_v = v[:, 2]
    # Z of each triangle vertex
    Z_tri = Z_v[i]
    diff = np.max(Z_tri, axis=1) - np.min(Z_tri, axis=1)
    
    assert np.all(diff < (threshold * 2.0)), "Mesh Gate failed: found stretching triangles"

def test_viewer_gate():
    H, W = 10, 10
    Z = np.ones((H, W), dtype=np.float32) * 2.0
    K = np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1]], dtype=np.float32)
    v, _, uv = build_mesh(Z, K)
    
    # Original image
    colors = np.random.rand(H*W, 3).astype(np.float32)
    
    # Render at identity
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    
    img = reference_render(v, colors, K, R, t, H, W)
    
    # Since we drop stretched triangles in real mesh, our reference renderer
    # is point-based and just projects vertices back to the pixel grid.
    # We should get exactly the original colors.
    # Note: reference_render draws points, so it expects N points and renders them.
    # The valid mask might exclude some points if they fall out, but here they should perfectly map back.
    # Due to floating point issues, they might land on exactly integer coordinates.
    # Since u_int = rint(u), u = 0,1,2... they map back to original indices.
    
    # To compare, flatten image and compare to original colors
    img_flat = img.reshape(-1, 3)
    # They should match perfectly
    assert np.allclose(img_flat, colors, atol=1e-5), "Viewer Gate failed: identity render does not match original"
