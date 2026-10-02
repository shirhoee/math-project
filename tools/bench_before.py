import numpy as np
import time
import os
import platform
from math_engine import TransformEngine
from depth_estimator import DepthEstimator
from PIL import Image

def benchmark():
    img = Image.open('sample.jpg').convert('RGB')
    img_array = np.array(img)
    H, W, _ = img_array.shape
    
    estimator = DepthEstimator()
    disparity = estimator.estimate_depth('sample.jpg')
    
    engine = TransformEngine(W, H, fov_deg=60.0)
    Z_map = engine.disparity_to_depth(disparity, z_near=1.0, z_far=4.0)
    P = engine.unproject_to_3d(Z_map)
    colors = img_array.reshape(-1, 3)
    
    R = engine.get_rotation_matrix(np.radians(20), np.radians(20), 0)
    
    def run_pass(stride):
        P_s = P.reshape(H, W, 3)[::stride, ::stride].reshape(-1, 3)
        C_s = colors.reshape(H, W, 3)[::stride, ::stride].reshape(-1, 3)
        Z_s = Z_map[::stride, ::stride]
        engine_s = TransformEngine(W//stride, H//stride, fov_deg=60.0)
        
        times = []
        for i in range(21):
            t0 = time.perf_counter()
            P_new = engine_s.apply_transform(P_s, R, np.zeros(3), Z_pivot=None)
            c, d = engine_s.project_to_2d(P_new, C_s, splat_gain=1.0, s_max=1, z_near=1.0, edge_tau=0.05, Z_map_original=Z_s)
            c, d = engine_s.fill_holes(c, d, max_iters=15)
            t1 = time.perf_counter()
            if i > 0:
                times.append((t1 - t0) * 1000)
        
        return np.mean(times), np.std(times)
        
    m1, s1 = run_pass(1)
    m2, s2 = run_pass(2)
    
    with open("perf_before.txt", "w") as f:
        f.write(f"Stride 1: {m1:.1f} +- {s1:.1f} ms\n")
        f.write(f"Stride 2: {m2:.1f} +- {s2:.1f} ms\n")
        f.write(f"CPU: {platform.processor()} ({os.cpu_count()} cores)\n")

if __name__ == "__main__":
    benchmark()
