import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())
from math_engine import TransformEngine

def test_image(img_path, fov_deg=60.0):
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    H, W, _ = img_array.shape
    
    # Fake disparity for dummy images (flat wall)
    disparity = np.ones((H, W), dtype=np.float32)
    
    engine = TransformEngine(W, H, fov_deg=fov_deg)
    Z = engine.disparity_to_depth(disparity, 1.0, 4.0)
    P = engine.unproject_to_3d(Z)
    colors = img_array.reshape(-1, 3)
    
    # Render with Pass 4 uniform focal scaling (for max rotation, e.g. 15 deg)
    R = engine.get_rotation_matrix(0, np.radians(15.0), 0)
    P_new = engine.apply_transform(P, R, np.zeros(3), Z_pivot=2.0)
    
    t0 = time.time()
    c, d = engine.project_to_2d(P_new, colors, splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z, edge_mode="demote")
    t1 = time.time()
    
    holes_before = np.sum(np.isinf(d)) / d.size * 100
    
    c_fill, d_fill = engine.fill_holes_pyramid(c, d)
    t2 = time.time()
    
    holes_after = np.sum(np.isinf(d_fill)) / d_fill.size * 100
    
    print(f"{os.path.basename(img_path)}: Render {t1-t0:.3f}s | Fill {t2-t1:.3f}s | Holes Before: {holes_before:.2f}% | Holes After: {holes_after:.2f}%")

test_image('tests/fixtures/gen/indoor.jpg')
test_image('tests/fixtures/gen/landscape.jpg')
test_image('tests/fixtures/gen/closeup.jpg')
