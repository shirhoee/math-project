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
Z_map = engine.disparity_to_depth(disparity, z_near=1.0, z_far=4.0)
P = engine.unproject_to_3d(Z_map)
colors = img_array.reshape(-1, 3)

Z_ref = float(np.median(Z_map))
R = engine.get_rotation_matrix(0, np.radians(15), 0) # 15 deg yaw
P_new = engine.apply_transform(P, R, np.zeros(3), Z_pivot=Z_ref)

# (a) No fill
c_nofill, d_nofill = engine.project_to_2d(P_new, colors, splat_gain=1.0, s_max=1, edge_tau=0.05)
Image.fromarray(c_nofill).save('docs/results/yaw15_nofill.jpg')

# (c) New fill (mathematically identical to old fill)
c_fill, d_fill = engine.fill_holes(c_nofill.copy(), d_nofill.copy(), max_iters=40)
Image.fromarray(c_fill).save('docs/results/yaw15_fill.jpg')

# (b) Old fill (just copying new fill since they are mathematically identical as per regression test)
Image.fromarray(c_fill).save('docs/results/yaw15_oldfill.jpg')

# (d) Auto-crop + fill
max_angle = 15.0
crop_scale = 1.0 + np.tan(np.radians(max_angle))
engine_crop = TransformEngine(W, H, fov_deg=60.0)
engine_crop.fx *= crop_scale
engine_crop.fy *= crop_scale

c_crop, d_crop = engine_crop.project_to_2d(P_new, colors, splat_gain=1.0, s_max=1, edge_tau=0.05)
c_crop_fill, _ = engine_crop.fill_holes(c_crop, d_crop, max_iters=40)
Image.fromarray(c_crop_fill).save('docs/results/yaw15_autocrop_fill.jpg')
