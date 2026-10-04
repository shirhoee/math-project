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
    
    print("| Sample | Sharp (Ctr) | Sharp (Ext) | Hole Frac | Parallax | PSNR (dB) | Build (s) |")
    print("|--------|-------------|-------------|-----------|----------|-----------|-----------|")
    
    metrics = []
    
    for i, path in enumerate(samples):
        img = Image.open(path).convert("RGB")
        if img.width > 1024:
            ratio = 1024.0 / img.width
            new_size = (1024, int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(img)
        H, W, _ = img_array.shape
        
        # Save temporary for depth estimator
        temp_path = "temp_quality.jpg"
        img.save(temp_path)
        
        disparity = model.estimate_depth(temp_path, refine_depth=True)
        
        fov = 60.0
        z_near, z_far = 1.0, 4.0
        
        engine = TransformEngine(W, H, fov)
        Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
        P = engine.unproject_to_3d(Z_map)
        
        z_pivot = float(np.median(P[:, 2]))
        
        t0 = time.time()
        layers, z_k = build_layers(img_array, disparity, z_near, z_far, n_layers=16)
        t1 = time.time()
        build_time_s = t1 - t0
        
        H_k_center = layer_homographies(engine.K, np.eye(3), np.zeros(3), z_k)
        mpi_center = render_mpi(layers, H_k_center)
        
        R_12 = engine.get_rotation_matrix(0.0, np.radians(12.0), 0.0)
        pivot_pos = np.array([0, 0, z_pivot], dtype=np.float32)
        t_12 = pivot_pos - R_12 @ pivot_pos
        
        H_k_max = layer_homographies(engine.K, R_12, t_12, z_k)
        mpi_max_yaw = render_mpi(layers, H_k_max)
        
        # Calculate final alpha
        alpha_rem = np.ones((H, W), dtype=np.float32)
        for k in range(layers.shape[0]):
            alpha_rem *= (1.0 - layers[k, ..., 3])
        final_a = 1.0 - alpha_rem
        black_hole_fraction = np.mean(final_a < 0.99)
        
        sharp_orig = laplacian_variance(img_array)
        sharp_center = laplacian_variance(mpi_center)
        sharp_extreme = laplacian_variance(mpi_max_yaw)
        
        sharp_ratio_centre = sharp_center / sharp_orig
        sharp_ratio_extreme = sharp_extreme / sharp_orig
        
        identity_psnr = psnr(mpi_center, img_array)
        
        # Parallax ratio (approximation of max shift over W)
        parallax_ratio = (np.abs(t_12[0]) * engine.K[0,0] / z_near) / W
        
        name = os.path.basename(path)[:10]
        print(f"| {name} | {sharp_ratio_centre:.3f} | {sharp_ratio_extreme:.3f} | {black_hole_fraction:.3f} | {parallax_ratio:.3f} | {identity_psnr:.2f} | {build_time_s:.2f} |")
        
        metrics.append({
            "psnr": identity_psnr,
            "sharp_ctr": sharp_ratio_centre,
        })
        
    print()
    mean_psnr = np.mean([m["psnr"] for m in metrics])
    mean_sharp = np.mean([m["sharp_ctr"] for m in metrics])
    
    psnr_pass = mean_psnr >= 45.0
    sharp_pass = mean_sharp >= 0.95
    
    print(f"Identity PSNR (Target >= 45.0 dB): {mean_psnr:.2f} dB -> {'PASS' if psnr_pass else 'FAIL'}")
    print(f"Sharpness Ratio (Target >= 0.95): {mean_sharp:.3f} -> {'PASS' if sharp_pass else 'FAIL'}")

if __name__ == "__main__":
    main()
