"""
Benchmark script for the math engine.
Runs 1 warm-up and 21 timed iterations for stride 1 and 2, with and without fill.
"""
import numpy as np
import time
import os
import sys

# Ensure math_engine and depth_estimator can be imported from current directory
sys.path.insert(0, os.getcwd())

from math_engine import TransformEngine
from PIL import Image

def benchmark():
    fixture_dir = r"E:\Math project\tests\fixtures"
    disparity_path = os.path.join(fixture_dir, 'sample_disparity.npy')
    sample_path = os.path.join(fixture_dir, 'sample.jpg')
    
    if not os.path.exists(disparity_path) or not os.path.exists(sample_path):
        print("Fixtures not found.")
        return
        
    img = Image.open(sample_path).convert('RGB')
    img_array = np.array(img)
    H, W, _ = img_array.shape
    
    disparity = np.load(disparity_path)
    
    engine = TransformEngine(W, H, fov_deg=60.0)
    Z_map = engine.disparity_to_depth(disparity, z_near=1.0, z_far=4.0)
    P = engine.unproject_to_3d(Z_map)
    colors = img_array.reshape(-1, 3)
    
    # 20 degree rotation for benchmark
    R = engine.get_rotation_matrix(np.radians(20), np.radians(20), 0)
    Z_ref = float(np.median(Z_map))
    
    def run_pass(stride, use_fill):
        # We need to manually stride the source points for the benchmark if splatting
        # to match app.py behavior exactly.
        P_s = P.reshape(H, W, 3)[::stride, ::stride].reshape(-1, 3)
        C_s = colors.reshape(H, W, 3)[::stride, ::stride].reshape(-1, 3)
        Z_s = Z_map[::stride, ::stride]
        engine_s = TransformEngine(W//stride, H//stride, fov_deg=60.0)
        
        times = []
        for i in range(22):
            t0 = time.perf_counter()
            # Handle API changes between passes
            if 'Z_pivot' in engine_s.apply_transform.__code__.co_varnames:
                P_new = engine_s.apply_transform(P_s, R, np.zeros(3), Z_pivot=Z_ref)
            else:
                P_new = engine_s.apply_transform(P_s, R, np.zeros(3))
                
            if 'Z_map_original' in engine_s.project_to_2d.__code__.co_varnames:
                c, d = engine_s.project_to_2d(P_new, C_s, splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z_s)
            else:
                c, d = engine_s.project_to_2d(P_new, C_s, splat_gain=1.0, s_max=1, edge_tau=0.05)
            
            if use_fill:
                if hasattr(engine_s, 'fill_holes_pyramid'):
                    c, d = engine_s.fill_holes_pyramid(c, d)
                else:
                    c, d = engine_s.fill_holes(c, d, max_iters=15)
                    
            t1 = time.perf_counter()
            if i > 0:
                times.append((t1 - t0) * 1000)
                
        return np.mean(times), np.std(times), np.percentile(times, 95)
        
    m1_nofill, s1_nofill, p1_nofill = run_pass(1, False)
    m1_fill, s1_fill, p1_fill = run_pass(1, True)
    m2_nofill, s2_nofill, p2_nofill = run_pass(2, False)
    m2_fill, s2_fill, p2_fill = run_pass(2, True)
    
    print(f"Stride 1 (No fill): {m1_nofill:.1f} +- {s1_nofill:.1f} ms")
    print(f"Stride 1 (Fill): {m1_fill:.1f} +- {s1_fill:.1f} ms")
    print(f"Stride 2 (No fill): {m2_nofill:.1f} +- {s2_nofill:.1f} ms")
    print(f"Stride 2 (Fill): {m2_fill:.1f} +- {s2_fill:.1f} ms (p95: {p2_fill:.1f} ms)")

if __name__ == "__main__":
    benchmark()
