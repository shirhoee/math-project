import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
from PIL import Image
from math_engine import TransformEngine

img = Image.open('tests/fixtures/sample.jpg').convert('RGB')
img_array = np.array(img)
H, W, _ = img_array.shape

disparity = np.load('tests/fixtures/sample_disparity.npy')
engine = TransformEngine(W, H, fov_deg=60.0)
Z_map = engine.disparity_to_depth(disparity, z_near=1.0, z_far=4.0)
P = engine.unproject_to_3d(Z_map)
colors = img_array.reshape(-1, 3)
Z_ref = float(np.median(Z_map))

print("Angle | Remaining (Iter 8) | Remaining (Pyr) | Max Width")
for angle in [5, 10, 15, 20, 25]:
    R = engine.get_rotation_matrix(0, np.radians(angle), 0)
    P_new = engine.apply_transform(P, R, np.zeros(3), Z_pivot=Z_ref)
    c_base, d_base = engine.project_to_2d(P_new, colors, splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z_map)
    
    # 1. Preview (stride 2) iteratively with 8 cap
    engine_s2 = TransformEngine(W//2, H//2, fov_deg=60.0)
    c_s2, d_s2 = engine_s2.project_to_2d(P_new.reshape(H,W,3)[::2,::2].reshape(-1,3), colors.reshape(H,W,3)[::2,::2].reshape(-1,3), splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z_map[::2,::2])
    _, d_fill_8 = engine_s2.fill_holes_iterative(c_s2, d_s2, max_iters=8)
    rem_iter = np.sum(np.isinf(d_fill_8)) / d_fill_8.size * 100
    
    # 2. HQ Pyramid
    _, d_fill_pyr = engine.fill_holes_pyramid(c_base.copy(), d_base.copy())
    rem_pyr = np.sum(np.isinf(d_fill_pyr)) / d_fill_pyr.size * 100
    
    # 3. Max hole width
    holes = np.isinf(d_base)
    padded = np.pad(holes, ((0,0), (1,1)), 'constant', constant_values=False)
    edges = np.diff(padded.astype(int), axis=1)
    starts = np.where(edges == 1)
    ends = np.where(edges == -1)
    if len(starts[0]) == 0:
        w = 0
    else:
        w = np.max(ends[1] - starts[1])
        
    print(f"{angle:2}    | {rem_iter:5.1f}%              | {rem_pyr:5.1f}%           | {w}")
