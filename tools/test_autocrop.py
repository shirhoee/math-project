import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
from math_engine import TransformEngine

def run_crop(angle_deg):
    W, H = 800, 534
    engine = TransformEngine(W, H, fov_deg=60.0)
    R = engine.get_rotation_matrix(0, np.radians(angle_deg), 0)
    z_near, z_far = 1.0, 4.0
    
    corners_u = np.array([0, W-1, W-1, 0, 0, W-1, W-1, 0], dtype=np.float32)
    corners_v = np.array([0, 0, H-1, H-1, 0, 0, H-1, H-1], dtype=np.float32)
    corners_z = np.array([z_near]*4 + [z_far]*4, dtype=np.float32)
    
    c_uvw = np.stack([corners_u * corners_z, corners_v * corners_z, corners_z], axis=1)
    c_P = c_uvw @ engine.K_inv.T
    c_P_new = engine.apply_transform(c_P, R, np.zeros(3), Z_pivot=2.0)
    c_uvw_new = c_P_new @ engine.K.T
    c_u_new = c_uvw_new[:, 0] / c_P_new[:, 2]
    c_v_new = c_uvw_new[:, 1] / c_P_new[:, 2]
    
    L = np.max(c_u_new[[0, 3, 4, 7]])
    R_bound = np.min(c_u_new[[1, 2, 5, 6]])
    T = np.max(c_v_new[[0, 1, 4, 5]])
    B = np.min(c_v_new[[2, 3, 6, 7]])
    
    # a) Old 1 + tan(angle)
    scale_old = 1.0 + np.tan(np.radians(angle_deg))
    retained_old = 100.0 / (scale_old ** 2)
    
    # b) Pass-4 uniform scaling
    # In Pass 4, L and R_bound were used, but cx remained fixed. 
    # To keep cx fixed and keep the valid region inside [0, W],
    # the window [-L, W-R_bound] tells us the margins.
    # The uniform scale about center required: W_new = max(W - 2*L, W - 2*(W-R_bound)) ?
    # No, Pass 4 just did scale = max(W / (R_bound - L), H / (B - T)) and multiplied fx.
    # But scaling fx without shifting cx just expands about cx!
    # So if cx = W/2, distance to border is min(cx - L, R_bound - cx).
    # To keep valid, scale_pass4 = W / (2 * min(cx - L, R_bound - cx))
    cx = W / 2
    scale_pass4 = W / (2 * min(cx - L, R_bound - cx))
    retained_pass4 = 100.0 / (scale_pass4 ** 2)
    
    # c) New off-centre crop
    w_valid = R_bound - L
    h_valid = B - T
    scale_new = max(W / w_valid, H / h_valid)
    retained_new = 100.0 / (scale_new ** 2)
    
    print(f"{angle_deg:2} | Old: {retained_old:5.1f}% | Pass-4: {retained_pass4:5.1f}% | New: {retained_new:5.1f}%")

print("Angle | Retained FOV")
for a in [5, 10, 15, 20, 25]:
    run_crop(a)
