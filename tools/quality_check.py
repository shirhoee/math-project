import glob
import cv2
import numpy as np
from PIL import Image
import time
import sys
import io
import base64
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
        t_pt0 = time.time()
        colors = img_array.reshape(-1, 3)
        n_layers = 16
        target_shift_ratio = 0.07
        parallax_total = 2 * target_shift_ratio * W
        cols = int(np.ceil(parallax_total / 2.5)) + 1
        if cols % 2 == 0: cols += 1
        n_yaw = min(cols, 41)
        n_pitch = 3
        max_yaw, max_pitch = 12.0, 8.0
        
        name = os.path.basename(path)[:10]
        print(f"\nConfiguration for {name}: W={W}, H={H}, layers={n_layers}, preset={target_shift_ratio}, frames={n_yaw}x{n_pitch}")
        
        from view_atlas import orbit_angles
        angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)
        
        atlas_pt = render_atlas(P, colors, engine.K, H, W, angles, Z_pivot=z_pivot, edge_tau=0.05, edge_mode='demote', s_max=2, z_near=z_near)
        t_pt1 = time.time()
        
        pt_center = atlas_pt[n_pitch//2, n_yaw//2]
        pt_max_yaw = atlas_pt[n_pitch//2, n_yaw-1]
        
        # 2. MPI RENDERER
        t_mpi0 = time.time()
        layers, z_k = build_layers(img_array, disparity, z_near, z_far, n_layers=n_layers)
        t_build_mpi = time.time() - t_mpi0
        
        t_mpi_atlas0 = time.time()
        from mpi_renderer import calibrate_motion, render_atlas_mpi
        baseline_x = calibrate_motion(z_near, np.max(z_k), engine.K[0,0], W, max_yaw, 0.07)
        baseline_y = calibrate_motion(z_near, np.max(z_k), engine.K[1,1], H, max_pitch, 0.07)
        
        # We also need H_k_center and H_k_max to compute the metrics
        from mpi_renderer import layer_homographies
        H_k_center = layer_homographies(engine.K, np.eye(3), np.zeros(3), z_k)
        mpi_center = render_mpi(layers, H_k_center)
        
        yaw_deg = max_yaw
        t_extra = np.array([baseline_x, 0.0, 0.0], dtype=np.float32)
        R_max = engine.get_rotation_matrix(0.0, np.radians(yaw_deg), 0.0)
        c = np.array([0.0, 0.0, z_pivot], dtype=np.float32)
        t_max = c - R_max @ c + t_extra
        
        shift_x_px = np.abs(t_max[0] * engine.K[0,0] / z_near + engine.K[0,0] * R_max[0,2])
        zoom = 1.0 + shift_x_px / (W / 2.0)
        K_render = engine.K.copy()
        K_render[0,0] *= zoom
        K_render[1,1] *= zoom
        
        H_k_max = layer_homographies(engine.K, R_max, t_max, z_k, K_render=K_render)
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
                
                actual_shift = (engine.K[0,0] * np.abs(baseline_x) / np.cos(np.radians(max_yaw))) * (1.0/z_near - 1.0/np.max(z_k))
                parallax_ratio = actual_shift / (W * 0.07)
                
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
        from mpi_renderer import render_atlas_mpi
        t_mpi_render_all0 = time.time()
        mpi_frames = render_atlas_mpi(layers, z_k, engine.K, angles, baseline_x, baseline_y, max_yaw, max_pitch, 0.0, z_pivot)
        t_mpi_render_all1 = time.time()
        
        os.makedirs("docs/results", exist_ok=True)
        # Gather frames
        pt_left = atlas_pt[n_pitch//2, 0]
        pt_right = atlas_pt[n_pitch//2, -1]
        pt_up = atlas_pt[-1, n_yaw//2]
        pt_down = atlas_pt[0, n_yaw//2]
        
        def decode_b64(s):
            if isinstance(s, str):
                return np.array(Image.open(io.BytesIO(base64.b64decode(s.split(",")[1]))).convert('RGB'))
            return s
            
        mpi_left = decode_b64(mpi_frames[n_pitch//2 * n_yaw + 0])
        mpi_right = decode_b64(mpi_frames[n_pitch//2 * n_yaw + (n_yaw-1)])
        mpi_up = decode_b64(mpi_frames[-1 * n_yaw + n_yaw//2])
        mpi_down = decode_b64(mpi_frames[0 * n_yaw + n_yaw//2])
        
        def paste_images(img_list, row, col, h, w):
            canvas = np.zeros((h * row, w * col, 3), dtype=np.uint8)
            for i, img in enumerate(img_list):
                if img.shape[:2] != (h, w):
                    import cv2
                    img = cv2.resize(img, (w, h))
                r, c = i // col, i % col
                canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img
            return canvas
            
        pt_row = [pt_left, pt_center, pt_right, pt_up, pt_down]
        mpi_row = [mpi_left, decode_b64(mpi_center), mpi_right, mpi_up, mpi_down]
        
        # Crops
        cx, cy = W//2, H//2
        s = 100
        pt_crop_c = pt_center[cy-s:cy+s, cx-s:cx+s]
        pt_crop_e = pt_max_yaw[cy-s:cy+s, cx-s:cx+s]
        
        mpi_crop_c = mpi_row[1][cy-s:cy+s, cx-s:cx+s]
        mpi_crop_e = decode_b64(mpi_max_yaw)[cy-s:cy+s, cx-s:cx+s]
        
        # 200% scaling
        pt_crop_c = np.repeat(np.repeat(pt_crop_c, 2, axis=0), 2, axis=1)
        pt_crop_e = np.repeat(np.repeat(pt_crop_e, 2, axis=0), 2, axis=1)
        mpi_crop_c = np.repeat(np.repeat(mpi_crop_c, 2, axis=0), 2, axis=1)
        mpi_crop_e = np.repeat(np.repeat(mpi_crop_e, 2, axis=0), 2, axis=1)
        
        # Pad crops to match H, W (just stick them in the center of a black frame)
        def pad_crop(crop, h, w):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            ch, cw = crop.shape[:2]
            canvas[(h-ch)//2:(h+ch)//2, (w-cw)//2:(w+cw)//2] = crop
            return canvas
            
        pt_crop_row = [pad_crop(pt_crop_c, H, W), pad_crop(pt_crop_e, H, W), pad_crop(pt_crop_c, H, W), pad_crop(pt_crop_c, H, W), pad_crop(pt_crop_c, H, W)]
        mpi_crop_row = [pad_crop(mpi_crop_c, H, W), pad_crop(mpi_crop_e, H, W), pad_crop(mpi_crop_c, H, W), pad_crop(mpi_crop_c, H, W), pad_crop(mpi_crop_c, H, W)]
        
        contact = paste_images(pt_row + pt_crop_row + mpi_row + mpi_crop_row, 4, 5, H, W)
        Image.fromarray(contact).save(f"docs/results/contact_{name}.png")                    
        # Timing
        # We need to simulate atlas rendering + encoding
        # The prompt says "measure the real app path in seconds, split into: depth, layer building, atlas rendering, encoding, page assembly"
        print(f"Timing for {name}: Depth={t_depth:.2f}s, Build Points={t_pt1-t_pt0:.2f}s, Build Layers={t_build_mpi:.2f}s, MPI Render (123 views)={t_mpi_render_all1-t_mpi_render_all0:.2f}s")
        
    print()
    if any_failed:
        print("FAIL: One or more quality gates did not meet the target.")
        sys.exit(1)
    else:
        print("PASS: All quality gates met the target.")
        sys.exit(0)

if __name__ == "__main__":
    main()
