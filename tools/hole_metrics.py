import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
from PIL import Image
from math_engine import TransformEngine

def get_max_horizontal_run(mask):
    # cumulative-sum / reset trick, no per-pixel loops
    # mask is boolean (H, W)
    padded = np.pad(mask, ((0,0), (1,1)), 'constant', constant_values=False)
    edges = np.diff(padded.astype(int), axis=1)
    starts = np.where(edges == 1)
    ends = np.where(edges == -1)
    if len(starts[0]) == 0:
        return 0
    return np.max(ends[1] - starts[1])
def get_max_thickness(mask):
    if not np.any(mask): return 0
    count = 0
    curr = mask
    while np.any(curr):
        out = np.zeros_like(curr)
        # 3x3 erosion via slicing (9 neighbors)
        out[1:-1, 1:-1] = (
            curr[:-2, :-2] & curr[:-2, 1:-1] & curr[:-2, 2:] &
            curr[1:-1, :-2] & curr[1:-1, 1:-1] & curr[1:-1, 2:] &
            curr[2:, :-2] & curr[2:, 1:-1] & curr[2:, 2:]
        )
        curr = out
        count += 1
    return count

img = Image.open('tests/fixtures/sample.jpg').convert('RGB')
img_array = np.array(img)
H, W, _ = img_array.shape

disparity = np.load('tests/fixtures/sample_disparity.npy')
engine = TransformEngine(W, H, fov_deg=60.0)
Z_map = engine.disparity_to_depth(disparity, z_near=1.0, z_far=4.0)
P = engine.unproject_to_3d(Z_map)
colors = img_array.reshape(-1, 3)
Z_ref = float(np.median(Z_map))

print("Angle | Max Run (px) | Max Thickness (px) | Edge Mask Ablation (Holes ON - Holes OFF %)")
for angle in [5, 10, 15, 20, 25]:
    R = engine.get_rotation_matrix(0, np.radians(angle), 0)
    P_new = engine.apply_transform(P, R, np.zeros(3), Z_pivot=Z_ref)
    
    # Render with Edge Mask ON (tau=0.05)
    _, d_on = engine.project_to_2d(P_new, colors, splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z_map)
    holes_on = np.isinf(d_on)
    
    # Render with Edge Mask OFF (tau=None)
    _, d_off = engine.project_to_2d(P_new, colors, splat_gain=1.0, s_max=1, edge_tau=None, Z_map_original=Z_map)
    holes_off = np.isinf(d_off)
    
    run = get_max_horizontal_run(holes_on)
    thick = get_max_thickness(holes_on)
    
    diff_frac = (np.sum(holes_on) - np.sum(holes_off)) / holes_on.size * 100
    print(f"{angle:2}    | {run:12} | {thick:18} | {diff_frac:5.2f}%")
