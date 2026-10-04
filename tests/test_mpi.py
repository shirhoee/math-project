import numpy as np
from mpi_renderer import build_layers, inv3x3, layer_homographies, render_mpi

def test_build_layers():
    H, W = 10, 10
    rgb = np.ones((H, W, 3), dtype=np.float32)
    disparity = np.linspace(0, 1, H*W).reshape(H, W).astype(np.float32)
    
    layers, z_k = build_layers(rgb, disparity, 1.0, 10.0, n_layers=4, bleed_px=1)
    
    # Check shape
    assert layers.shape == (4, H, W, 4)
    assert z_k.shape == (4,)
    
    # Check partition of unity (alpha sums to 1 on valid pixels)
    # The extended regions will make the sum of alpha > 1 in some places, 
    # but for original pixels, it should be >= 1.
    alpha_sum = np.sum(layers[..., 3], axis=0)
    assert np.all(alpha_sum >= 0.99)

def test_inv3x3():
    np.random.seed(42)
    M = np.random.randn(5, 3, 3).astype(np.float32)
    M_inv = inv3x3(M)
    
    for i in range(5):
        I = M[i] @ M_inv[i]
        assert np.allclose(I, np.eye(3), atol=1e-4)

def test_render_mpi_identity():
    H, W = 10, 10
    layers = np.zeros((2, H, W, 4), dtype=np.float32)
    layers[0, ..., :3] = 0.5  # bg
    layers[0, ..., 3] = 1.0
    layers[1, 2:8, 2:8, :3] = 1.0 # fg square
    layers[1, 2:8, 2:8, 3] = 1.0
    
    H_k = np.array([np.eye(3), np.eye(3)], dtype=np.float32)
    
    out = render_mpi(layers, H_k)
    
    assert out.shape == (H, W, 3)
    assert np.all(out[5, 5] == 255)
    assert np.all(out[0, 0] == 127) # 0.5 * 255
