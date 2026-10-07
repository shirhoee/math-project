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

Z_ref = float(np.median(Z_map))

print("Angle | Total Holes | Border | Edge Mask | Cracks | Disocclusion")
for angle in [5, 10, 15, 20, 25]:
    R = engine.get_rotation_matrix(np.radians(angle), np.radians(angle), 0)
    P_new = engine.apply_transform(P, R, np.zeros(3), Z_pivot=Z_ref)
    
    # Base configuration: Splatting ON (which removes most cracks), Edge Mask ON.
    c_base, d_base = engine.project_to_2d(P_new, colors, splat_gain=1.0, s_max=1, edge_tau=0.05)
    holes_base_mask = np.isinf(d_base)
    total_holes = np.sum(holes_base_mask)
    
    # 1. Edge Masked: Turn off edge mask. Holes that vanish were caused by the mask.
    c_no_edge, d_no_edge = engine.project_to_2d(P_new, colors, splat_gain=1.0, s_max=1, edge_tau=None)
    holes_no_edge = np.isinf(d_no_edge)
    edge_masked = np.sum(holes_base_mask & ~holes_no_edge)
    
    # 2. Border (Out-of-frame): Use a larger canvas.
    engine_large = TransformEngine(W*2, H*2, fov_deg=60.0)
    # Translate points so they land in the middle of the 2x canvas
    P_large = P_new.copy()
    c_large, d_large = engine_large.project_to_2d(P_large, colors, splat_gain=1.0, s_max=1, edge_tau=None)
    # We need to map the large canvas back to the original to see if border pixels filled it.
    # Actually, a simpler way to find out-of-frame border holes:
    # Any hole pixel that is outside the bounding box of valid pixels, OR we can just check if any projected point landed outside.
    # Wait, the easiest way: project without bounds clipping. If they landed < 0 or >= W, they are border.
    uvw = P_new @ engine.K.T
    u = uvw[:, 0] / P_new[:, 2]
    v = uvw[:, 1] / P_new[:, 2]
    out_of_bounds = (u < 0) | (u >= W) | (v < 0) | (v >= H)
    # Wait, this tells us which SOURCE points went out of bounds. It doesn't tell us which TARGET pixels are empty because of it!
    # A target pixel is a border void if it lies outside the convex hull of valid pixels.
    # We can approximate border voids by finding the min/max u for each v, and everything outside is border.
    valid_mask = ~holes_no_edge
    border_mask = np.zeros_like(valid_mask)
    for y in range(H):
        valid_x = np.where(valid_mask[y])[0]
        if len(valid_x) > 0:
            border_mask[y, :valid_x[0]] = True
            border_mask[y, valid_x[-1]+1:] = True
        else:
            border_mask[y, :] = True
            
    border_holes = np.sum(border_mask & holes_base_mask)
    
    # 3. Cracks: Holes that vanish if we increase splat footprint.
    c_big_splat, d_big_splat = engine.project_to_2d(P_new, colors, splat_gain=2.0, s_max=2, edge_tau=0.05)
    holes_big_splat = np.isinf(d_big_splat)
    cracks = np.sum(holes_base_mask & ~holes_big_splat & ~border_mask & ~holes_no_edge)
    
    # 4. Disocclusion: The remainder.
    disocclusion = total_holes - edge_masked - border_holes - cracks
    
    N = H * W
    print(f"{angle:2}    | {total_holes/N*100:5.1f}% | {border_holes/N*100:5.1f}% | {edge_masked/N*100:5.1f}% | {cracks/N*100:5.1f}% | {disocclusion/N*100:5.1f}%")

