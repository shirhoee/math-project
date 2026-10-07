import os, sys, numpy as np; sys.path.insert(0, os.getcwd()); from PIL import Image; from math_engine import TransformEngine; 

def test_multi_image_smoke():
  for img_name in ['indoor.jpg', 'landscape.jpg', 'closeup.jpg']:
    img_path = f'tests/fixtures/gen/{img_name}'
    if not os.path.exists(img_path): continue
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    H, W, _ = img_array.shape
    eng = TransformEngine(W, H, 60.0)
    disparity = np.ones((H, W), dtype=np.float32)
    Z = eng.disparity_to_depth(disparity, 1.0, 4.0)
    P = eng.unproject_to_3d(Z)
    C = img_array.reshape(-1, 3)
    R = eng.get_rotation_matrix(0, np.radians(15.0), 0)
    P_new = eng.apply_transform(P, R, np.zeros(3), Z_pivot=2.0)
    c1, d1 = eng.project_to_2d(P_new, C, splat_gain=1.0, s_max=1, stride=2, edge_tau=0.05, Z_map_original=Z, edge_mode='demote')
    cf1, df1 = eng.fill_holes_pyramid(c1, d1)
    
    assert cf1.shape == (H, W, 3)
    assert cf1.dtype == np.uint8
    holes = np.mean(np.isinf(df1)) * 100
    assert holes < 1.0  # bounding the fraction
