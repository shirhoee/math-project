import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mpi_renderer import bicubic_sample, bilinear_sample

def test_bicubic():
    # 1. Constant image
    img = np.ones((10, 10, 3), dtype=np.float32) * 128
    u = np.array([4.5, 4.2])
    v = np.array([5.5, 5.2])
    res = bicubic_sample(img, u, v)
    assert np.allclose(res, 128), "Constant image test failed"
    
    # 2. Linear ramp
    img = np.zeros((10, 10, 1), dtype=np.float32)
    img[:, :, 0] = np.arange(10)[None, :] * 10
    u = np.array([4.5, 2.2])
    v = np.array([5.0, 5.0])
    res = bicubic_sample(img, u, v)
    assert np.allclose(res[:, 0], [45.0, 22.0]), f"Linear ramp test failed: {res[:, 0]}"
    
    # 3. 1-px checker pattern shifted by 0.5 px
    img = np.zeros((10, 10, 1), dtype=np.float32)
    img[::2, ::2] = 1.0
    img[1::2, 1::2] = 1.0
    u = np.array([4.49])
    v = np.array([4.49])
    
    res_bicubic = bicubic_sample(img, u, v)[0, 0]
    res_bilinear = bilinear_sample(img, u[..., None], v[..., None]).flatten()[0]
    print(f"Bicubic checker contrast (0.49 shift): {float(res_bicubic):.4f}, Bilinear: {float(res_bilinear):.4f}")
    
    # 4. Center frame equals source
    from mpi_renderer import layer_homographies, render_mpi
    layers = np.zeros((1, 10, 10, 4), dtype=np.float32)
    layers[0, ..., :3] = np.random.rand(10, 10, 3)
    layers[0, ..., 3] = 1.0
    
    H_k = np.array([np.eye(3, dtype=np.float32)])
    mpi_center = render_mpi(layers, H_k)
    max_diff = np.max(np.abs(mpi_center.astype(np.float32) - layers[0, ..., :3] * 255.0))
    print(f"Max diff for center frame: {max_diff:.4f}")
    assert max_diff <= 1.0, f"Max diff too high: {max_diff}"
    
    print("Tests passed!")

if __name__ == "__main__":
    test_bicubic()
