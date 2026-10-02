import os
import numpy as np
from PIL import Image
from math_engine import TransformEngine

def test_image(img_path, fov_deg=60.0):
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    H, W, _ = img_array.shape
    
    # Fake disparity for dummy images
    disparity = np.ones((H, W), dtype=np.float32)
    
    engine = TransformEngine(W, H, fov_deg=fov_deg)
    Z = engine.disparity_to_depth(disparity, 1.0, 4.0)
    P = engine.unproject_to_3d(Z)
    
    # Test project_to_2d
    c, d = engine.project_to_2d(P, img_array.reshape(-1, 3), splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z, edge_mode="demote")
    
    holes = np.isinf(d).sum() / d.size * 100
    print(f"{os.path.basename(img_path)}: Holes={holes:.2f}%")

if __name__ == "__main__":
    test_image('tests/fixtures/gen/indoor.jpg')
    test_image('tests/fixtures/gen/landscape.jpg')
    test_image('tests/fixtures/gen/closeup.jpg')
