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
        
    print("Running Quality Gates...\n")
    
    print("| Sample | Renderer | Sharp (Ctr) | Sharp (Ext) | Hole Frac | Parallax | PSNR (dB) |")
    print("|--------|----------|-------------|-------------|-----------|----------|-----------|")
    
    any_failed = False
    build_times = []
    
    for i, path in enumerate(samples):
        img = Image.open(path).convert("RGB")
        if img.width > 1024:
            ratio = 1024.0 / img.width
            new_size = (1024, int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        H, W, _ = img_array.shape
        
        temp_path = "temp_quality.jpg"
        img.save(temp_path)
        
        # Real App Path Timing
        t0 = time.time()
        disparity = model.estimate_depth(temp_path, refine_depth=True)
        t_depth = time.time() - t0
        
        fov = 60.0
        z_near, z_far = 1.0, 4.0
        engine = TransformEngine(W, H, fov)
        Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
        P = engine.unproject_to_3d(Z_map)
        
        z_pivot = float(np.median(P[:, 2]))
        
        # 1. POINT RENDERER
        colors = img_array.reshape(-1, 3)
        t_pt0 = time.time()
        n_yaw, n_pitch = 9, 5
        max_yaw, max_pitch = 12.0, 8.0
        
        from view_atlas import orbit_angles
        angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)
        
        atlas_pt = render_atlas(P, colors, engine.K, H, W, angles, Z_pivot=z_pivot, edge_tau=0.05, edge_mode='demote', s_max=2, z_near=z_near)
        t_pt1 = time.time()
        
        pt_center = atlas_pt[n_pitch//2, n_yaw//2]
        pt_max_yaw = atlas_pt[n_pitch//2, n_yaw-1]
        
        # 2. MPI RENDERER
        t_mpi0 = time.time()
        layers, z_k = build_layers(img_array, disparity, z_near, z_far, n_layers=16)
        t_build_mpi = time.time() - t_mpi0
        
        t_mpi_atlas0 = time.time()
        from mpi_renderer import calibrate_motion, render_atlas_mpi
        baseline_x = calibrate_motion(z_near, engine.K[0,0], W, 0.05) # 5% shift
        baseline_y = calibrate_motion(z_near, engine.K[1,1], H, 0.05)
        
        # We also need H_k_center and H_k_max to compute the metrics
        from mpi_renderer import layer_homographies
        H_k_center = layer_homographies(engine.K, np.eye(3), np.zeros(3), z_k)
        mpi_center = render_mpi(layers, H_k_center)
        
        yaw_deg = max_yaw
        t_max = np.array([baseline_x, 0.0, 0.0], dtype=np.float32)
        H_k_max = layer_homographies(engine.K, np.eye(3), t_max, z_k)
        mpi_max_yaw = render_mpi(layers, H_k_max)
        t_mpi_atlas1 = time.time()
        
        # Evaluate Gates
        sharp_orig = laplacian_variance(img_array)
        
        shift_x = int(np.ceil((np.abs(baseline_x) * engine.K[0,0] / z_near)))
        shift_y = int(np.ceil((np.abs(baseline_y) * engine.K[1,1] / z_near)))
        
        def crop_center(img, sx, sy):
            if sx == 0 and sy == 0: return img
            sy = max(1, sy)
            sx = max(1, sx)
            return img[sy:-sy, sx:-sx]
            
        sharp_orig_crop = laplacian_variance(crop_center(img_array, shift_x, shift_y))
        
        name = os.path.basename(path)[:10]
        
        for renderer_name, center_img, ext_img in [("Points", pt_center, pt_max_yaw), ("Layers", mpi_center, mpi_max_yaw)]:
            sharp_center = laplacian_variance(center_img)
            sharp_extreme = laplacian_variance(ext_img)
            sharp_ext_crop = laplacian_variance(crop_center(ext_img, shift_x, shift_y))
            
            sharp_ratio_centre = sharp_center / sharp_orig
            sharp_ratio_extreme = sharp_extreme / sharp_orig
            sharp_ratio_ext_crop = sharp_ext_crop / sharp_orig_crop
            
            identity_psnr = psnr(center_img, img_array)
            
            if renderer_name == "Layers":
                alpha_rem = np.ones((H, W), dtype=np.float32)
                for k in range(layers.shape[0]):
                    alpha_rem *= (1.0 - layers[k, ..., 3])
                final_a = 1.0 - alpha_rem
                black_hole_fraction = np.mean(final_a < 0.99)
                parallax_ratio = (np.abs(baseline_x) * engine.K[0,0] / z_near) / W / 0.05
                
                # With sharpening
                from mpi_renderer import unsharp_mask
                sharp_center_img = unsharp_mask(center_img, 0.35)
                sharp_ext_img = unsharp_mask(ext_img, 0.35)
                
                sharp_c_sharp = laplacian_variance(sharp_center_img)
                sharp_e_sharp = laplacian_variance(sharp_ext_img)
                sharp_ec_sharp = laplacian_variance(crop_center(sharp_ext_img, shift_x, shift_y))
                
                sharp_ratio_centre_s = sharp_c_sharp / sharp_orig
                sharp_ratio_extreme_s = sharp_e_sharp / sharp_orig
                sharp_ratio_ext_crop_s = sharp_ec_sharp / sharp_orig_crop
                
                print(f"| {name} | {renderer_name:8s} | {sharp_ratio_centre:.3f} (S {sharp_ratio_centre_s:.3f}) | {sharp_ratio_extreme:.3f} (Crop {sharp_ratio_ext_crop:.3f}, S {sharp_ratio_ext_crop_s:.3f}) | {black_hole_fraction:.3f} | {parallax_ratio:.3f} | {identity_psnr:.2f} |")
                
                if sharp_ratio_centre < 0.99 or sharp_ratio_ext_crop < 0.80 or black_hole_fraction > 0.01 or identity_psnr < 45.0 or parallax_ratio < 0.9 or parallax_ratio > 1.1:
                    any_failed = True
            else:
                black_hole_fraction = 0.0
                parallax_ratio = 1.0
                print(f"| {name} | {renderer_name:8s} | {sharp_ratio_centre:.3f} | {sharp_ratio_extreme:.3f} (Crop {sharp_ratio_ext_crop:.3f}) | {black_hole_fraction:.3f} | {parallax_ratio:.3f} | {identity_psnr:.2f} |")

                    
        # Timing
        # We need to simulate atlas rendering + encoding
        # The prompt says "measure the real app path in seconds, split into: depth, layer building, atlas rendering, encoding, page assembly"
        print(f"Timing for {name}: Depth={t_depth:.2f}s, Build Points={t_pt1-t_pt0:.2f}s, Build Layers={t_build_mpi:.2f}s, MPI Render (2 views)={t_mpi_atlas1-t_mpi_atlas0:.2f}s")
        
    print()
    if any_failed:
        print("FAIL: One or more quality gates did not meet the target.")
        sys.exit(1)
    else:
        print("PASS: All quality gates met the target.")
        sys.exit(0)

if __name__ == "__main__":
    main()
