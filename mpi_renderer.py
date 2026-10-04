import numpy as np
from depth_estimator import box_filter
from math_engine import TransformEngine

def build_layers(rgb, disparity, z_near, z_far, n_layers=16, bleed_px=None):
    """
    Builds a Multi-Plane Image (MPI) representation.
    rgb: (H, W, 3) float array in [0, 1] or uint8.
    disparity: (H, W) float array in [0, 1].
    z_near, z_far: near and far depth bounds.
    n_layers: number of layers.
    bleed_px: maximum pixels to extend. If None, computes based on parallax (approx).
    
    Returns:
        layers: (n_layers, H, W, 4) float32 array (premultiplied RGBA).
        z_k: (n_layers,) float32 array of layer depths.
    """
    H, W = disparity.shape
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    
    if bleed_px is None:
        bleed_px = max(1, int(0.05 * W))
    
    L = n_layers
    layers = np.zeros((L, H, W, 4), dtype=np.float32)
    z_k = np.zeros(L, dtype=np.float32)
    
    engine = TransformEngine(W, H)
    
    for k in range(L):
        # depths Z_k = z_near + (1 - c_k)(z_far - z_near)
        c_k = (k + 0.5) / L
        z_k[k] = z_near + (1.0 - c_k) * (z_far - z_near)
        
        w_k = np.clip(1.0 - np.abs(disparity - c_k) * L, 0.0, 1.0)
        layers[k, ..., :3] = rgb * w_k[..., None]
        layers[k, ..., 3] = w_k
        
    sum_w = np.sum(layers[..., 3], axis=0, keepdims=True)
    sum_w = np.maximum(sum_w, 1e-6)
    layers[..., 3] /= sum_w[0]

    # Calculate true alpha for 'over' compositing from back to front
    cum_w = np.zeros((H, W), dtype=np.float32)
    for k in range(L):
        cum_w += layers[k, ..., 3]
        safe_cum = np.maximum(cum_w, 1e-6)
        
        # true alpha for over compositing
        alpha_k = layers[k, ..., 3] / safe_cum
        # but only where cum_w > 0
        alpha_k[cum_w < 1e-6] = 0.0
        
        # update layer with true alpha and premultiplied rgb
        layers[k, ..., :3] = rgb * alpha_k[..., None]
        layers[k, ..., 3] = alpha_k
        
    # Hidden region extension
    # Fills hidden pixels from background-biased pyramid fill using only pixels of this and farther layers.
    composite_RGB = np.zeros((H, W, 3), dtype=np.float32)
    composite_A = np.zeros((H, W), dtype=np.float32)
    
    for k in range(L):
        layer_rgb = layers[k, ..., :3]
        layer_a = layers[k, ..., 3]
        
        bg_rgb = composite_RGB + layer_rgb
        bg_a = composite_A + layer_a
        
        valid_mask = bg_a > 1e-4
        
        # If the layer has nothing, skip
        if not np.any(valid_mask):
            composite_RGB = bg_rgb
            composite_A = bg_a
            continue
            
        # For engine.fill_holes_pyramid, holes are where depth_buf == inf
        depth_mock = np.full((H, W), np.inf, dtype=np.float32)
        depth_mock[valid_mask] = 1.0  # valid points
        
        # Avoid division by zero
        safe_a = np.maximum(bg_a, 1e-4)
        canvas_uint8 = np.clip((bg_rgb / safe_a[..., None]) * 255.0, 0, 255).astype(np.uint8)
        
        filled_canvas, _ = engine.fill_holes_pyramid(canvas_uint8, depth_mock)
        filled_rgb = filled_canvas.astype(np.float32) / 255.0
        
        if k == 0:
            extension_mask = ~valid_mask
        else:
            # Dilate the current layer's alpha mask to find the bleed area
            layer_valid = layer_a > 1e-4
            dilated = box_filter(layer_valid.astype(np.float32), bleed_px) > 0
            extension_mask = dilated & (~valid_mask)
        
        # Apply extension to the current layer
        layers[k, extension_mask, :3] = filled_rgb[extension_mask]
        layers[k, extension_mask, 3] = 1.0
        
        composite_RGB = bg_rgb
        composite_A = bg_a

    return layers, z_k

def inv3x3(M):
    """
    Inverse of 3x3 matrices.
    M: (N, 3, 3) or (3, 3) array.
    Returns: inverse matrices of same shape.
    """
    is_single = (M.ndim == 2)
    if is_single:
        M = M[None, ...]
    
    # Adjugate matrix formula
    m00, m01, m02 = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    m10, m11, m12 = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    m20, m21, m22 = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]
    
    inv = np.empty_like(M)
    inv[:, 0, 0] = m11 * m22 - m12 * m21
    inv[:, 0, 1] = m02 * m21 - m01 * m22
    inv[:, 0, 2] = m01 * m12 - m02 * m11
    inv[:, 1, 0] = m12 * m20 - m10 * m22
    inv[:, 1, 1] = m00 * m22 - m02 * m20
    inv[:, 1, 2] = m02 * m10 - m00 * m12
    inv[:, 2, 0] = m10 * m21 - m11 * m20
    inv[:, 2, 1] = m01 * m20 - m00 * m21
    inv[:, 2, 2] = m00 * m11 - m01 * m10
    
    det = m00 * inv[:, 0, 0] + m01 * inv[:, 1, 0] + m02 * inv[:, 2, 0]
    inv /= det[..., None, None]
    
    if is_single:
        return inv[0]
    return inv

def layer_homographies(K, R, t, z_k, K_render=None):
    """
    K: (3, 3) intrinsic matrix
    R: (3, 3) rotation matrix
    t: (3,) translation vector
    z_k: (n_layers,) depths
    K_render: (3, 3) target intrinsic matrix (for dolly zoom). Defaults to K.
    Returns: (n_layers, 3, 3) homographies
    """
    if K_render is None:
        K_render = K
        
    K_inv = inv3x3(K)
    L = len(z_k)
    H_k = np.zeros((L, 3, 3), dtype=np.float32)
    
    tnT = np.zeros((3, 3), dtype=np.float32)
    tnT[:, 2] = t
    
    for i in range(L):
        H_k[i] = K_render @ (R + tnT / z_k[i]) @ K_inv
        
    return H_k

def bilinear_sample(img, u, v):
    """
    Samples img at continuous coordinates (u, v) using bilinear interpolation.
    img: (H, W, C)
    u, v: (H_out, W_out) coordinates
    Returns: (H_out, W_out, C) sampled values
    """
    H, W, C = img.shape
    H_out, W_out = u.shape
    
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1
    
    wa = (u1 - u) * (v1 - v)
    wb = (u - u0) * (v1 - v)
    wc = (u1 - u) * (v - v0)
    wd = (u - u0) * (v - v0)
    
    Ia = img[v0, u0]
    Ib = img[v0, u1]
    Ic = img[v1, u0]
    Id = img[v1, u1]
    
    wa = wa[..., None]
    wb = wb[..., None]
    wc = wc[..., None]
    wd = wd[..., None]
    
    return Ia * wa + Ib * wb + Ic * wc + Id * wd

def catmull_rom_weights(x, a=-0.5):
    x2 = x * x
    x3 = x2 * x
    w = np.zeros_like(x)
    mask1 = x <= 1.0
    w[mask1] = (a + 2) * x3[mask1] - (a + 3) * x2[mask1] + 1
    mask2 = (x > 1.0) & (x <= 2.0)
    w[mask2] = a * x3[mask2] - 5 * a * x2[mask2] + 8 * a * x[mask2] - 4 * a
    return w

def bicubic_sample(img, u, v):
    H, W, C = img.shape
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    
    u_unclamped = np.stack([u0 - 1, u0, u0 + 1, u0 + 2], axis=-1)
    v_unclamped = np.stack([v0 - 1, v0, v0 + 1, v0 + 2], axis=-1)
    
    u_idx = np.clip(u_unclamped, 0, W - 1)
    v_idx = np.clip(v_unclamped, 0, H - 1)
    
    u_dist = np.abs(u[:, None] - u_unclamped)
    v_dist = np.abs(v[:, None] - v_unclamped)
    
    wu = catmull_rom_weights(u_dist)
    wv = catmull_rom_weights(v_dist)
    
    v_grid = v_idx[:, :, None]
    u_grid = u_idx[:, None, :]
    pixels = img[v_grid, u_grid]
    
    temp = np.einsum('nvyc,ny->nvc', pixels, wu)
    res = np.einsum('nvc,nv->nc', temp, wv)
    return res

def render_mpi(layers, H_k, downscale=1.0):
    """
    Renders the MPI from the target view defined by homographies H_k.
    layers: (L, H, W, 4) premultiplied RGBA
    H_k: (L, 3, 3) homographies for each layer
    Returns: (H, W, 3) uint8 image
    """
    L, H, W, _ = layers.shape
    
    out_H = int(H * downscale)
    out_W = int(W * downscale)
    
    y, x = np.meshgrid(np.arange(out_H), np.arange(out_W), indexing='ij')
    ones = np.ones_like(x)
    coords = np.stack([x.flatten(), y.flatten(), ones.flatten()], axis=0).astype(np.float32)
    
    H_inv_k = inv3x3(H_k)
    out_rgb = np.zeros((out_H, out_W, 3), dtype=np.float32)
    
    for k in range(L):
        H_inv = H_inv_k[k]
        
        src_coords = H_inv @ coords
        src_w = src_coords[2]
        src_u = (src_coords[0] / src_w).reshape(out_H, out_W)
        src_v = (src_coords[1] / src_w).reshape(out_H, out_W)
        
        # Pass 1: bilinear on alpha only
        layer_alpha = layers[k, ..., 3:4]
        sampled_alpha = bilinear_sample(layer_alpha, src_u, src_v)[..., 0]
        
        # Pass 2: bicubic RGBA only where alpha > 1e-3
        valid = sampled_alpha > 1e-3
        if np.any(valid):
            u_valid = src_u[valid]
            v_valid = src_v[valid]
            sampled_rgba = bicubic_sample(layers[k], u_valid, v_valid)
            
            # Clamp so that 0 <= RGB <= A <= 1
            A_valid = np.clip(sampled_rgba[..., 3], 0.0, 1.0)
            RGB_valid = np.clip(sampled_rgba[..., :3], 0.0, A_valid[..., None])
            
            src_rgb = np.zeros((out_H, out_W, 3), dtype=np.float32)
            src_a = np.zeros((out_H, out_W, 1), dtype=np.float32)
            
            src_rgb[valid] = RGB_valid
            src_a[valid, 0] = A_valid
            
            out_rgb = src_rgb + (1.0 - src_a) * out_rgb
        
    return np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)

import concurrent.futures
import base64
from io import BytesIO
from PIL import Image

def parallax_px(z, f, baseline):
    """
    Computes horizontal parallax in pixels.
    z: depth
    f: focal length (fx)
    baseline: camera shift t_x
    Returns: shift in pixels
    """
    return baseline * f / z

def calibrate_motion(z_near, z_far, f, W, yaw_deg, target_ratio=0.07):
    """
    Computes camera baseline t_x needed to achieve a target near-to-far shift ratio.
    Near-to-far shift = (f * t_x / cos(yaw)) * (1/z_near - 1/z_far)
    """
    if z_far <= z_near: return 0.0
    yaw_rad = np.radians(yaw_deg)
    target_shift_px = target_ratio * W
    baseline = (target_shift_px * np.cos(yaw_rad)) / (f * (1.0/z_near - 1.0/z_far))
    return baseline

def unsharp_mask(img, amount=0.35):
    if amount <= 0: return img
    img_f = img.astype(np.float32)
    center = img_f[1:-1, 1:-1]
    up = img_f[:-2, 1:-1]
    down = img_f[2:, 1:-1]
    left = img_f[1:-1, :-2]
    right = img_f[1:-1, 2:]
    laplacian = 4 * center - up - down - left - right
    out = img_f.copy()
    out[1:-1, 1:-1] = center + amount * laplacian
    return np.clip(out, 0, 255).astype(np.uint8)

def _render_single_frame(layers, H_k, quality=80, sharpen_amount=0.0, downscale=1.0):
    frame = render_mpi(layers, H_k, downscale)
    if sharpen_amount > 0:
        frame = unsharp_mask(frame, sharpen_amount)
    img = Image.fromarray(frame)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def render_atlas_mpi(layers, z_k, K, angles, baseline_x_max, baseline_y_max, max_yaw=12.0, max_pitch=8.0, sharpen_amount=0.0, z_pivot=None):
    """
    Renders an atlas of frames using the MPI layered renderer in parallel.
    angles: (n_pitch, n_yaw, 2)
    Returns: 
        frames: list of base64 strings (flat)
    """
    n_pitch, n_yaw, _ = angles.shape
    if z_pivot is None:
        z_pivot = (np.min(z_k) + np.max(z_k)) / 2.0
    
    # Pre-calculate R and t for each view
    from math_engine import TransformEngine
    engine = TransformEngine(layers.shape[2], layers.shape[1])
    
    # We parameterize translation based on angles relative to max_angles.
    # Actually, a simple orbit implies translation t and rotation R.
    # The homography formula uses camera rotation and translation.
    # If the camera moves linearly in X and Y, t_x and t_y vary.
    # angles shape is (n_pitch, n_yaw, 2).
    # t_x = baseline_x_max * (yaw / max_yaw)
    # t_y = baseline_y_max * (pitch / max_pitch)
    
    # Wait, the prompt says "calibrate_motion" computes the baseline.
    # The orbit in Stage 1 was just rotation around a pivot.
    # In MPI, we do pure camera translation (and maybe rotation if we want to fixate).
    # Standard MPI dolly/orbit is mostly translation, or translation + rotation to fixate.
    # If we translate and rotate to fixate at Z_pivot, then:
    # We can just use translation if we don't rotate, or both.
    
    tasks = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for p in range(n_pitch):
            for y in range(n_yaw):
                yaw_deg = angles[p, y, 0]
                pitch_deg = angles[p, y, 1]
                
                t_x = baseline_x_max * (yaw_deg / max_yaw) if max_yaw > 0 else 0
                t_y = baseline_y_max * (pitch_deg / max_pitch) if max_pitch > 0 else 0
                
                t_extra = np.array([t_x, t_y, 0.0], dtype=np.float32)
                R = engine.get_rotation_matrix(np.radians(pitch_deg), np.radians(yaw_deg), 0.0)
                
                c = np.array([0.0, 0.0, z_pivot], dtype=np.float32)
                t = c - R @ c + t_extra
                
                fx, fy = K[0, 0], K[1, 1]
                z_near = np.min(z_k)
                # Max displacement is dominated by near-to-far shift
                # At z_near, shift is t[0]*f/z_near. But wait, we want exact shift:
                shift_x_px = np.abs(t[0] * fx / z_near + fx * R[0,2])
                shift_y_px = np.abs(t[1] * fy / z_near + fy * R[1,2])
                margin = max(shift_x_px / (layers.shape[2] / 2.0), shift_y_px / (layers.shape[1] / 2.0))
                zoom = 1.0 + margin
                
                K_render = K.copy()
                K_render[0, 0] *= zoom
                K_render[1, 1] *= zoom
                
                # Render pitch rows at lower res? 
                downscale = 1.0
                if p != n_pitch // 2:
                    downscale = 0.5
                    
                K_render[:2, :] *= downscale
                
                H_k = layer_homographies(K, R, t, z_k, K_render=K_render)
                tasks.append((p, y, executor.submit(_render_single_frame, layers, H_k, 80, sharpen_amount, downscale)))
                
        # To keep the order:
        tasks.sort(key=lambda x: (x[0], x[1]))
        frames = [t[2].result() for t in tasks]
        
    return frames
