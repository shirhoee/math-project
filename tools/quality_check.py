import glob
import cv2
import numpy as np
from PIL import Image
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from depth_estimator import DepthEstimator
from math_engine import TransformEngine
from mpi_renderer import build_layers, layer_homographies, render_mpi
from view_atlas import render_atlas

def laplacian_variance(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(img, cv2.CV_64F).var()

def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    model = DepthEstimator("Small")
    
    samples = sorted(glob.glob("assets/samples/01_*.jpg"))
    if not samples:
        print("No samples found.")
        sys.exit(1)
        
    print("Running Quality Gates...")
    
    contact_sheets = []
    
    all_mpi_sharpness = []
    all_mpi_psnr = []
    
    for i, path in enumerate(samples):
        print(f"Processing {path}...")
        img = Image.open(path).convert("RGB")
        img_array = np.array(img)
        H, W, _ = img_array.shape
        
        disparity = model.estimate_depth(path, refine_depth=True)
        
        fov = 60.0
        z_near, z_far = 1.0, 4.0
        
        engine = TransformEngine(W, H, fov)
        Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
        P = engine.unproject_to_3d(Z_map)
        colors = img_array.reshape(-1, 3)
        
        z_pivot = float(np.median(P[:, 2]))
        
        # Original Point Splatting Max Yaw (12 deg)
        angles_max = np.array([[[12.0, 0.0]]], dtype=np.float32)
        atlas_old = render_atlas(P, colors, engine.K, H, W, angles_max, Z_pivot=z_pivot, edge_tau=0.05, edge_mode='demote', s_max=2, z_near=z_near)
        old_max_yaw = atlas_old[0, 0]
        
        # MPI Engine
        t0 = time.time()
        layers, z_k = build_layers(img_array, disparity, z_near, z_far, n_layers=16)
        
        # Centre frame (0 deg)
        H_k_center = layer_homographies(engine.K, np.eye(3), np.zeros(3), z_k)
        mpi_center = render_mpi(layers, H_k_center)
        
        # Max yaw frame (12 deg)
        from view_atlas import orbit_angles
        # purely translation for MPI orbit equivalent to 12 deg
        # baseline for 12 deg. 12/max_yaw ... let's use a standard translation
        # baseline = calibrate_motion(z_near, engine.K[0,0], W, 0.05)
        # Instead, let's match rotation to Point Splatting for exact comparison
        R_12 = engine.get_rotation_matrix(0.0, np.radians(12.0), 0.0)
        # Pivot compensation t = z_pivot * [0,0,1] - R @ (z_pivot * [0,0,1])
        pivot_pos = np.array([0, 0, z_pivot], dtype=np.float32)
        t_12 = pivot_pos - R_12 @ pivot_pos
        
        H_k_max = layer_homographies(engine.K, R_12, t_12, z_k)
        mpi_max_yaw = render_mpi(layers, H_k_max)
        
        # Metrics
        sharp_orig = laplacian_variance(img_array)
        sharp_mpi = laplacian_variance(mpi_center)
        sharp_ratio = sharp_mpi / sharp_orig
        
        p = psnr(mpi_center, img_array)
        
        print(f"  Sharpness Ratio: {sharp_ratio:.3f}")
        print(f"  PSNR (center): {p:.2f} dB")
        
        all_mpi_sharpness.append(sharp_ratio)
        all_mpi_psnr.append(p)
        
        # Contact sheet row: Original | Disparity | Old Max Yaw | MPI Max Yaw
        # Resize to small for contact sheet to avoid massive memory
        scale = 320 / W
        s_w, s_h = 320, int(H * scale)
        
        d_vis = (disparity * 255).astype(np.uint8)
        d_vis = cv2.cvtColor(d_vis, cv2.COLOR_GRAY2RGB)
        
        imgs = [img_array, d_vis, old_max_yaw, mpi_max_yaw]
        imgs_resized = [cv2.resize(im, (s_w, s_h)) for im in imgs]
        row = np.concatenate(imgs_resized, axis=1)
        contact_sheets.append(row)
        
    final_sheet = np.concatenate(contact_sheets, axis=0)
    Image.fromarray(final_sheet).save("assets/samples/contact_sheet.jpg")
    print("Saved contact_sheet.jpg")
    
    mean_sharp = np.mean(all_mpi_sharpness)
    mean_psnr = np.mean(all_mpi_psnr)
    
    success = True
    if mean_sharp < 0.90:
        print(f"FAIL: Mean sharpness ratio {mean_sharp:.3f} < 0.90")
        success = False
    else:
        print(f"PASS: Mean sharpness ratio {mean_sharp:.3f} >= 0.90")
        
    if mean_psnr < 30.0:
        print(f"FAIL: Mean PSNR {mean_psnr:.2f} < 30.0 dB")
        success = False
    else:
        print(f"PASS: Mean PSNR {mean_psnr:.2f} >= 30.0 dB")
        
    if not success:
        sys.exit(1)
        
if __name__ == "__main__":
    main()
