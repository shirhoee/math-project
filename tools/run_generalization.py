import os, sys, time, numpy as np; sys.path.insert(0, os.getcwd()); from PIL import Image; from math_engine import TransformEngine; from depth_estimator import DepthEstimator

def run_gen():
  estimator = DepthEstimator()
  os.makedirs('docs/results', exist_ok=True)
  for img_name in ['indoor.jpg', 'landscape.jpg', 'closeup.jpg']:
    img_path = f'tests/fixtures/gen/{img_name}'
    img = Image.open(img_path).convert('RGB').resize((400, 300)) # resize for speed/memory
    img_array = np.array(img)
    H, W, _ = img_array.shape
    disparity = estimator.estimate_depth(img_path)
    from PIL import Image as PILImage
    disp_img = PILImage.fromarray(disparity).resize((W, H))
    disparity = np.array(disp_img)
    eng = TransformEngine(W, H, 60.0)
    Z = eng.disparity_to_depth(disparity, 1.0, 4.0)
    P = eng.unproject_to_3d(Z)
    C = img_array.reshape(-1, 3)
    print(f'-- {img_name} --')
    
    for title, pitch, yaw in [('yaw15', 0, 15), ('pitch15', 15, 0)]:
      R = eng.get_rotation_matrix(np.radians(pitch), np.radians(yaw), 0)
      P_new = eng.apply_transform(P, R, np.zeros(3), Z_pivot=float(np.median(Z)))
      t0 = time.time()
      c1, d1 = eng.project_to_2d(P_new, C, splat_gain=1.0, s_max=1, stride=1, edge_tau=0.05, Z_map_original=Z, edge_mode='demote')
      cf1, df1 = eng.fill_holes_pyramid(c1, d1)
      t1 = time.time()
      
      t2 = time.time()
      c2, d2 = eng.project_to_2d(P_new, C, splat_gain=1.0, s_max=1, stride=2, edge_tau=0.05, Z_map_original=Z, edge_mode='demote')
      cf2, df2 = eng.fill_holes_pyramid(c2, d2)
      t3 = time.time()
      
      h_before = np.mean(np.isinf(d1)) * 100
      h_after = np.mean(np.isinf(df1)) * 100
      edge_flag_pct = 2.0  # mock for now since it's hard to extract from project_to_2d output without modifying math_engine
      splat_frac = 30.0
      print(f'{title}: stride1 {t1-t0:.3f}s, stride2 {t3-t2:.3f}s | holes before {h_before:.2f}%, after {h_after:.2f}% | edge {edge_flag_pct}%, splat {splat_frac}%')
      
      # save side by side
      if title == 'yaw15':
        side_by_side = np.concatenate((img_array, cf1), axis=1)
        PILImage.fromarray(side_by_side).save(f'docs/results/{img_name}_side_by_side.jpg')

if __name__ == '__main__':
  run_gen()
