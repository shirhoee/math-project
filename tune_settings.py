import numpy as np
import glob
from PIL import Image
import os
from depth_estimator import DepthEstimator
from math_engine import TransformEngine
from view_atlas import orbit_angles

def main():
    model = DepthEstimator()
    samples = sorted(glob.glob("assets/samples/*.jpg"))
    os.makedirs("docs/results", exist_ok=True)
    
    for img_path in samples:
        name = os.path.basename(img_path).split('.')[0]
        print(f"Processing {name}...")
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        H, W, _ = img_array.shape
        
        disparity = model.estimate_depth(img_path)
        
        # Tuning defaults
        fov = 60.0
        z_near = 1.0
        z_far = 4.0
        edge_tau = 0.05
        s_max = 2
        edge_mode = "demote"
        
        engine = TransformEngine(W, H, fov)
        Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
        P = engine.unproject_to_3d(Z_map)
        colors = img_array.reshape(-1, 3)
        
        z_pivot = float(np.median(P[:, 2]))
        
        max_yaw = 12.0
        max_pitch = 8.0
        
        # Check extremes for holes
        extreme_angles = np.array([
            [[max_yaw, max_pitch]],
            [[-max_yaw, -max_pitch]],
            [[max_yaw, -max_pitch]],
            [[-max_yaw, max_pitch]]
        ])
        
        hole_fraction = 0
        for pt in extreme_angles:
            yaw_rad = np.radians(pt[0, 0])
            pitch_rad = np.radians(pt[0, 1])
            R = engine.get_rotation_matrix(pitch_rad, yaw_rad, 0.0)
            P_new = engine.apply_transform(P, R, Z_pivot=z_pivot)
            _, depth = engine.project_to_2d(P_new, colors, edge_tau=edge_tau, edge_mode=edge_mode, s_max=s_max, z_near=z_near)
            holes = np.sum(np.isinf(depth)) / depth.size
            hole_fraction = max(hole_fraction, holes)
            
        print(f"Max hole fraction at extremes: {hole_fraction*100:.2f}%")
        shrink_factor = 1.0
        if hole_fraction > 0.08:
            # We want holes <= 8%. A rough linear approximation: shrink_factor = 0.08 / hole_fraction
            shrink_factor = 0.08 / hole_fraction
            print(f"Shrinking angle range by factor {shrink_factor:.2f}")
            max_yaw *= shrink_factor
            max_pitch *= shrink_factor
            
        n_yaw, n_pitch = 9, 5
        angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)
        
        scale = min(1.0, 480.0 / W)
        W_s, H_s = int(W * scale), int(H * scale)
        K_s = engine.K.copy()
        K_s[0, 0] *= scale; K_s[1, 1] *= scale
        K_s[0, 2] *= scale; K_s[1, 2] *= scale
        
        engine_render = TransformEngine(W_s, H_s, fov)
        engine_render.fx = K_s[0, 0]
        engine_render.fy = K_s[1, 1]
        engine_render.cx = K_s[0, 2]
        engine_render.cy = K_s[1, 2]
        
        atlas = np.zeros((n_pitch, n_yaw, H_s, W_s, 3), dtype=np.uint8)
        
        for p in range(n_pitch):
            for y in range(n_yaw):
                yaw_deg = angles[p, y, 0]
                pitch_deg = angles[p, y, 1]
                R = engine_render.get_rotation_matrix(np.radians(pitch_deg), np.radians(yaw_deg), 0.0)
                P_new = engine_render.apply_transform(P, R, Z_pivot=z_pivot)
                canvas, depth = engine_render.project_to_2d(P_new, colors, edge_tau=edge_tau, edge_mode=edge_mode, s_max=s_max, z_near=z_near)
                canvas_filled, _ = engine_render.fill_holes_pyramid(canvas, depth)
                atlas[p, y] = canvas_filled
                
        # Save GIF
        gif_frames = []
        for t in np.linspace(0, 2*np.pi, 20):
            p = int((n_pitch-1)/2 + np.sin(t)*(n_pitch-1)/2)
            y = int((n_yaw-1)/2 + np.cos(t)*(n_yaw-1)/2)
            gif_frames.append(Image.fromarray(atlas[p, y]))
        gif_frames[0].save(f"docs/results/demo_{name}.gif", save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
        print(f"Saved docs/results/demo_{name}.gif")
        
        # Save 5 key frames
        Image.fromarray(atlas[n_pitch//2, n_yaw//2]).save(f"docs/results/demo_{name}_center.jpg")
        Image.fromarray(atlas[0, 0]).save(f"docs/results/demo_{name}_topleft.jpg")
        Image.fromarray(atlas[n_pitch-1, n_yaw-1]).save(f"docs/results/demo_{name}_bottomright.jpg")
        Image.fromarray(atlas[n_pitch//2, 0]).save(f"docs/results/demo_{name}_left.jpg")
        Image.fromarray(atlas[n_pitch//2, n_yaw-1]).save(f"docs/results/demo_{name}_right.jpg")

if __name__ == "__main__":
    main()
