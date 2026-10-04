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
    layers[..., :3] /= sum_w[..., None]
    layers[..., 3:4] /= sum_w[..., None]
        
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
        H_k[i] = K_render @ (R - tnT / z_k[i]) @ K_inv
        
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

def render_mpi(layers, H_k):
    """
    Renders the MPI from the target view defined by homographies H_k.
    layers: (L, H, W, 4) premultiplied RGBA
    H_k: (L, 3, 3) homographies for each layer
    Returns: (H, W, 3) uint8 image
    """
    L, H, W, _ = layers.shape
    
    # Target grid
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    ones = np.ones_like(x)
    
    # coords: (3, H*W)
    coords = np.stack([x.flatten(), y.flatten(), ones.flatten()], axis=0).astype(np.float32)
    
    H_inv_k = inv3x3(H_k)
    
    out_rgb = np.zeros((H, W, 3), dtype=np.float32)
    
    # Back to front composite (k=0 is farthest)
    for k in range(L):
        H_inv = H_inv_k[k]
        
        src_coords = H_inv @ coords
        src_w = src_coords[2]
        src_u = (src_coords[0] / src_w).reshape(H, W)
        src_v = (src_coords[1] / src_w).reshape(H, W)
        
        sampled = bilinear_sample(layers[k], src_u, src_v)
        
        src_rgb = sampled[..., :3]
        src_a = sampled[..., 3:4]
        
        out_rgb = src_rgb + (1.0 - src_a) * out_rgb
        
    return np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
