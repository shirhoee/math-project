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
Z = engine.disparity_to_depth(disparity, 1.0, 4.0)
P = engine.unproject_to_3d(Z)

c1, d1 = engine.project_to_2d(P, img_array.reshape(-1, 3), splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z, edge_mode="drop")
c2, d2 = engine.project_to_2d(P, img_array.reshape(-1, 3), splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z, edge_mode="demote")

diff_drop = np.mean(c1 != img_array) * 100
diff_demote = np.mean(c2 != img_array) * 100
print(f"diff_drop: {diff_drop}")
print(f"diff_demote: {diff_demote}")
