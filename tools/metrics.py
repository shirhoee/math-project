import numpy as np
from PIL import Image
from math_engine import TransformEngine
from depth_estimator import DepthEstimator

img = Image.open('sample.jpg').convert('RGB')
img_array = np.array(img)
H, W, _ = img_array.shape

estimator = DepthEstimator()
disparity = estimator.estimate_depth('sample.jpg')

engine = TransformEngine(W, H, fov_deg=60.0)
Z_map = engine.disparity_to_depth(disparity, 1.0, 4.0)
P = engine.unproject_to_3d(Z_map)
colors = img_array.reshape(-1, 3)

# Splat size stats
Z_ref = float(np.median(Z_map))
s_i = np.clip(np.round(1.0 * Z_ref / Z_map.flatten() - 0.5), 0, 1).astype(np.int32)
splat_fraction = np.sum(s_i >= 1) / len(s_i) * 100
print(f"Splat fraction: {splat_fraction:.2f}%")

# Hole filling stats
for angle in [10, 15, 20, 25]:
    R = engine.get_rotation_matrix(np.radians(angle), np.radians(angle), 0)
    P_new = engine.apply_transform(P, R, np.zeros(3), Z_pivot=Z_ref)
    
    # Just project without splat/edge to get pure holes
    c, d = engine.project_to_2d(P_new, colors, splat_gain=0, s_max=0, edge_tau=None)
    holes_before = np.sum(np.isinf(d)) / (H * W) * 100
    
    # Fill
    c_f, d_f = engine.fill_holes(c, d, max_iters=40)
    holes_after = np.sum(np.isinf(d_f)) / (H * W) * 100
    
    print(f"Angle {angle} deg: Holes before = {holes_before:.2f}%, Holes after = {holes_after:.2f}%")

# Edge mask stats
dZ_dx = np.zeros_like(Z_map)
dZ_dy = np.zeros_like(Z_map)
dZ_dx[:, :-1] = np.abs(Z_map[:, 1:] - Z_map[:, :-1])
dZ_dy[:-1, :] = np.abs(Z_map[1:, :] - Z_map[:-1, :])
g = np.maximum(dZ_dx, dZ_dy) / (Z_map + 1e-8)

for tau in [0.02, 0.05, 0.10]:
    dropped = np.sum(g > tau) / (H * W) * 100
    print(f"Edge tau={tau}: dropped {dropped:.2f}%")
