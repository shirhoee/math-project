import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from math_engine import TransformEngine
from mpi_renderer import build_layers, layer_homographies, render_mpi
from view_atlas import render_atlas

def psnr_mae(img1, img2, mask=None):
    if mask is None:
        mask = np.ones(img1.shape[:2], dtype=bool)
    
    if not np.any(mask):
        return 100.0, 0.0
        
    diff = (img1.astype(np.float32) - img2.astype(np.float32))[mask]
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    
    if mse == 0:
        return 100.0, 0.0
    return 20 * np.log10(255.0 / np.sqrt(mse)), mae

def build_analytic_scene(H, W):
    """
    Back wall: Z=4
    Floor: Y=ground
    Foreground box: Z=2
    """
    # Simple smooth texture: gradients
    Y, X = np.ogrid[:H, :W]
    
    img = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.float32)
    
    # Back wall (z=4)
    img[:, :] = (X / W * 255)[..., None]
    depth[:, :] = 4.0
    
    # Foreground box (z=2, center)
    box_h, box_w = H // 2, W // 2
    y0, x0 = H // 4, W // 4
    y1, x1 = y0 + box_h, x0 + box_w
    
    box_texture = (Y[y0:y1, :] / H * 255)[..., None]
    img[y0:y1, x0:x1, 1] = box_texture[..., 0, 0] # Greenish
    depth[y0:y1, x0:x1] = 2.0
    
    return img, depth

def main():
    H, W = 256, 256
    img_array, depth = build_analytic_scene(H, W)
    
    fov = 60.0
    z_near, z_far = 1.0, 4.0
    engine = TransformEngine(W, H, fov)
    
    disparity = (1.0 - (depth - z_near) / (z_far - z_near))
    Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
    P = engine.unproject_to_3d(Z_map)
    colors = img_array.reshape(-1, 3)
    
    z_pivot = float(np.median(P[:, 2]))
    
    layers, z_k = build_layers(img_array, disparity, z_near, z_far, n_layers=16, bleed_px=0)
    
    from mpi_renderer import inv3x3, bilinear_sample
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([x.flatten(), y.flatten(), np.ones_like(x).flatten()], axis=0).astype(np.float32)
    
    def render_true_plane(H_matrix, src_img, valid_mask):
        H_inv = inv3x3(H_matrix)
        src_coords = H_inv @ coords
        src_w = src_coords[2]
        src_u = (src_coords[0] / src_w).reshape(H, W)
        src_v = (src_coords[1] / src_w).reshape(H, W)
        sampled = bilinear_sample(src_img.astype(np.float32)/255.0, src_u, src_v)
        in_bounds = (src_u >= 0) & (src_u < W) & (src_v >= 0) & (src_v < H) & valid_mask(src_u, src_v)
        return sampled * 255.0, in_bounds

    def mask_bg(u, v): return np.ones_like(u, dtype=bool)
    def mask_fg(u, v):
        box_h, box_w = H // 2, W // 2
        y0, x0 = H // 4, W // 4
        y1, x1 = y0 + box_h, x0 + box_w
        return (u >= x0) & (u < x1) & (v >= y0) & (v < y1)

    for yaw in [5.0, 10.0, 15.0]:
        R = engine.get_rotation_matrix(0.0, np.radians(yaw), 0.0)
        pivot_pos = np.array([0, 0, z_pivot], dtype=np.float32)
        t = pivot_pos - R @ pivot_pos
        
        H_bg = engine.K @ (R + (t.reshape(3,1) @ np.array([[0,0,1]])) / 4.0) @ engine.K_inv
        H_fg = engine.K @ (R + (t.reshape(3,1) @ np.array([[0,0,1]])) / 2.0) @ engine.K_inv
        
        bg_img, bg_valid = render_true_plane(H_bg, img_array, mask_bg)
        fg_img, fg_valid = render_true_plane(H_fg, img_array, mask_fg)
        
        true_img = bg_img.copy()
        true_img[fg_valid] = fg_img[fg_valid]
        true_img = np.clip(true_img, 0, 255).astype(np.uint8)
        
        angles = np.array([[[yaw, 0.0]]], dtype=np.float32)
        atlas_old = render_atlas(P, colors, engine.K, H, W, angles, Z_pivot=z_pivot, edge_tau=None, s_max=0, z_near=z_near)
        pt_render = atlas_old[0, 0]
        
        H_k = layer_homographies(engine.K, R, t, z_k)
        mpi_render = render_mpi(layers, H_k)
        
        # Disocclusion mask (areas only visible in true_img that are behind the foreground object, but wait, the true_img has NO disocclusions handled, it just overlaps. That's fine for simple plane.)
        p_pt, m_pt = psnr_mae(pt_render, true_img)
        p_mpi, m_mpi = psnr_mae(mpi_render, true_img)
        
        print(f"Point vs True @ {yaw}deg | PSNR: {p_pt:.2f} dB | MAE: {m_pt:.2f}")
        print(f"MPI vs True   @ {yaw}deg | PSNR: {p_mpi:.2f} dB | MAE: {m_mpi:.2f}")

if __name__ == "__main__":
    main()
