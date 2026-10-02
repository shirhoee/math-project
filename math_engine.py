import numpy as np

class TransformEngine:
    """
    The core mathematical graphics engine for 3D View Synthesis.
    All transformations are strictly built using NumPy linear algebra operations.
    """
    def __init__(self, width: int, height: int, fov_deg: float = 60.0):
        self.width = width
        self.height = height
        
        self.cx = (width - 1) / 2.0
        self.cy = (height - 1) / 2.0
        
        fov_rad = np.radians(fov_deg)
        self.fx = (width / 2.0) / np.tan(fov_rad / 2.0)
        self.fy = self.fx
        
        # Coordinate Convention: +X right, +Y down, +Z forward
        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ]
        ], dtype=np.float32)
        
        self.K_inv = np.linalg.inv(self.K)

    def disparity_to_depth(self, d: np.ndarray, z_near: float = 1.0, z_far: float = 4.0) -> np.ndarray:
        """
        Converts normalized disparity (d in [0, 1]) to depth Z.
        Input: d (H, W) array
        Output: Z (H, W) array
        """
        return z_near + (1.0 - d) * (z_far - z_near)

    def unproject_to_3d(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Maps the 2D pixels and Z-depth into a 3D Point Cloud.
        Mathematical Formula: [X, Y, Z]^T = Z * K^-1 * [u, v, 1]^T
        Inputs: depth_map (H, W)
        Outputs: (N, 3) matrix
        """
        H, W = depth_map.shape
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        
        uu = uu.flatten().astype(np.float32)
        vv = vv.flatten().astype(np.float32)
        zz = depth_map.flatten().astype(np.float32)
        
        ones = np.ones_like(uu)
        uv1 = np.vstack((uu, vv, ones)).T  # (N, 3)
        
        normalized_rays = uv1 @ self.K_inv.T
        point_cloud = normalized_rays * zz[:, np.newaxis]
        return point_cloud.astype(np.float32)

    def get_rotation_matrix(self, pitch: float, yaw: float, roll: float) -> np.ndarray:
        """
        Creates a 3x3 Orthogonal Rotation Matrix based on Euler angles (in radians).
        Order: R_z * R_y * R_x
        """
        cx, sx = np.cos(pitch), np.sin(pitch)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        
        cy, sy = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        
        cz, sz = np.cos(roll), np.sin(roll)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        
        return Rz @ (Ry @ Rx)

    def apply_transform(self, point_cloud: np.ndarray, R: np.ndarray, t: np.ndarray = None, Z_pivot: float = None) -> np.ndarray:
        """
        Applies linear transformation (pivot-based rotation and translation) to the 3D point cloud.
        Mathematical Formula: P_new = (P - c) @ R^T + c + t
        """
        if t is None:
            t = np.zeros(3, dtype=np.float32)
        if Z_pivot is None:
            Z_pivot = float(np.median(point_cloud[:, 2]))
            
        c = np.array([0, 0, Z_pivot], dtype=np.float32)
        centered = point_cloud - c
        return (centered @ R.T).astype(np.float32) + c + t

    def build_homogeneous_transform(self, R: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
        T_minus_c = np.eye(4, dtype=np.float32)
        T_minus_c[0:3, 3] = -c
        R4 = np.eye(4, dtype=np.float32)
        R4[0:3, 0:3] = R
        T_plus_c_t = np.eye(4, dtype=np.float32)
        T_plus_c_t[0:3, 3] = c + t
        return T_plus_c_t @ (R4 @ T_minus_c)

    def project_to_2d(self, P: np.ndarray, colors: np.ndarray, z_clip: float = 1e-3, splat_gain: float = 1.0, s_max: int = 1, stride: int = 1, Z_ref: float = None, edge_tau: float = 0.05, Z_map_original: np.ndarray = None, z_near: float = 1.0) -> tuple:
        H, W = self.height, self.width
        
        if edge_tau is not None and Z_map_original is not None:
            dZ_dx = np.zeros_like(Z_map_original)
            dZ_dy = np.zeros_like(Z_map_original)
            dZ_dx[:, :-1] = np.abs(Z_map_original[:, 1:] - Z_map_original[:, :-1])
            dZ_dy[:-1, :] = np.abs(Z_map_original[1:, :] - Z_map_original[:-1, :])
            g = np.maximum(dZ_dx, dZ_dy) / (Z_map_original + 1e-8)
            
            valid_edges = (g < edge_tau).flatten()
            P = P[valid_edges]
            colors = colors[valid_edges]

        Z = P[:, 2]
        valid_z = Z > z_clip
        P = P[valid_z]
        colors = colors[valid_z]
        Z = Z[valid_z]
        
        uvw = P @ self.K.T
        u = uvw[:, 0] / Z
        v = uvw[:, 1] / Z
        
        return self._render_points(u, v, Z, colors, H, W, splat_gain, s_max, stride, Z_ref)
        
    def project_orthographic(self, P: np.ndarray, colors: np.ndarray, scale: float = 1.0, splat_gain: float = 1.0, s_max: int = 1, stride: int = 1, Z_ref: float = None, edge_tau: float = 0.05, Z_map_original: np.ndarray = None, z_near: float = 1.0) -> tuple:
        H, W = self.height, self.width
        
        if edge_tau is not None and Z_map_original is not None:
            dZ_dx = np.zeros_like(Z_map_original)
            dZ_dy = np.zeros_like(Z_map_original)
            dZ_dx[:, :-1] = np.abs(Z_map_original[:, 1:] - Z_map_original[:, :-1])
            dZ_dy[:-1, :] = np.abs(Z_map_original[1:, :] - Z_map_original[:-1, :])
            g = np.maximum(dZ_dx, dZ_dy) / (Z_map_original + 1e-8)
            
            valid_edges = (g < edge_tau).flatten()
            P = P[valid_edges]
            colors = colors[valid_edges]

        Pi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        P_ortho = P @ Pi.T
        
        u = P_ortho[:, 0] * self.fx * scale + self.cx
        v = P_ortho[:, 1] * self.fy * scale + self.cy
        Z = P[:, 2] 
        
        return self._render_points(u, v, Z, colors, H, W, splat_gain, s_max, stride, Z_ref)
        
    def _render_points(self, u: np.ndarray, v: np.ndarray, Z: np.ndarray, colors: np.ndarray, H: int, W: int, splat_gain: float, s_max: int, stride: int, Z_ref: float) -> tuple:
        """
        Vectorized sub-routine for Z-buffering and 2-Pass Fill-Only Adaptive Splatting.
        Quantization logic replaces lexsort with stable single-key argsort for massive perf gains.
        Quantization error bound: (z_max - z_min) / 65535 ~ 4.5e-5 for a depth range of 3.
        """
        if Z_ref is None and len(Z) > 0:
            Z_ref = float(np.median(Z))
            
        u_int = np.rint(u).astype(np.int64)
        v_int = np.rint(v).astype(np.int64)
        
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        u_int, v_int, Z, colors = u_int[valid], v_int[valid], Z[valid], colors[valid]
        
        idx = v_int * W + u_int
        
        canvas = np.zeros((H * W, 3), dtype=np.uint8)
        depth_buf = np.full(H * W, np.inf, dtype=np.float32)
        
        if len(Z) == 0:
            return canvas.reshape((H, W, 3)), depth_buf.reshape((H, W))
            
        def zbuffer_pass(curr_idx, curr_Z, curr_C, target_empty_only=False):
            if len(curr_Z) == 0: return
            if target_empty_only:
                empty_mask = np.isinf(depth_buf[curr_idx])
                curr_idx, curr_Z, curr_C = curr_idx[empty_mask], curr_Z[empty_mask], curr_C[empty_mask]
                if len(curr_Z) == 0: return
                
            z_min, z_max = curr_Z.min(), curr_Z.max()
            # Prevent div by zero
            z_range = max(float(z_max - z_min), 1e-8)
            z_quant = np.clip((curr_Z - z_min) / z_range * 65535, 0, 65535).astype(np.int64)
            key = curr_idx.astype(np.int64) * 65536 + z_quant
            order = np.argsort(key) # stable argsort using single integer key
            
            idx_s = curr_idx[order]
            first = np.ones(idx_s.shape, dtype=bool)
            first[1:] = idx_s[1:] != idx_s[:-1]
            winners = order[first]
            
            canvas[curr_idx[winners]] = curr_C[winners]
            depth_buf[curr_idx[winners]] = curr_Z[winners]

        # Pass 1: Exact un-splatted points
        zbuffer_pass(idx, Z, colors, target_empty_only=False)
        
        # Pass 2: Splatting (Adaptive, Fill-only)
        if s_max > 0:
            # Adaptive Splat Footprint Formula
            s_i = np.clip(np.floor(splat_gain * Z_ref / Z), 0, s_max).astype(np.int32)
            mask = s_i >= 1
            
            if np.any(mask):
                u_s = u_int[mask]
                v_s = v_int[mask]
                Z_s = Z[mask]
                C_s = colors[mask]
                si_s = s_i[mask]
                
                u_cands, v_cands, z_cands, c_cands = [], [], [], []
                curr_smax = int(si_s.max())
                
                for dx in range(-curr_smax, curr_smax + 1):
                    for dy in range(-curr_smax, curr_smax + 1):
                        if dx == 0 and dy == 0:
                            continue
                        m = si_s >= max(abs(dx), abs(dy))
                        if np.any(m):
                            u_cands.append(u_s[m] + dx)
                            v_cands.append(v_s[m] + dy)
                            z_cands.append(Z_s[m])
                            c_cands.append(C_s[m])
                            
                if u_cands:
                    u_c = np.concatenate(u_cands)
                    v_c = np.concatenate(v_cands)
                    Z_c = np.concatenate(z_cands)
                    C_c = np.concatenate(c_cands, axis=0)
                    
                    val_c = (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H)
                    idx_c = v_c[val_c] * W + u_c[val_c]
                    
                    zbuffer_pass(idx_c, Z_c[val_c], C_c[val_c], target_empty_only=True)
                    
        return canvas.reshape((H, W, 3)), depth_buf.reshape((H, W))

    def fill_holes_iterative(self, canvas: np.ndarray, depth_buf: np.ndarray, max_iters: int = 15) -> tuple:
        """
        Fills holes (pixels with infinite depth) iteratively using background-biased neighbor interpolation.
        Automatically stops when no holes remain.
        """
        depth_buf = depth_buf.astype(np.float32)
        canvas = canvas.astype(np.uint8)
        H, W = depth_buf.shape
        max_iters = min(max_iters, 40)
        
        holes = np.isinf(depth_buf)
        if not np.any(holes):
            return canvas, depth_buf
            
        rows, cols = np.nonzero(holes)
        r_min, r_max = max(0, rows.min() - 1), min(H, rows.max() + 2)
        c_min, c_max = max(0, cols.min() - 1), min(W, cols.max() + 2)
        
        box_H = r_max - r_min
        box_W = c_max - c_min
        d_box = depth_buf[r_min:r_max, c_min:c_max]
        c_box = canvas[r_min:r_max, c_min:c_max]
        
        d_pad = np.pad(d_box, ((1, 1), (1, 1)), mode='constant', constant_values=np.inf)
        c_pad = np.pad(c_box, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        
        shifts = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for i in range(max_iters):
            box_holes = np.isinf(d_box)
            if not np.any(box_holes):
                break
                
            best_d = np.full_like(d_box, -1.0)
            best_c = np.zeros_like(c_box)
            
            for dy, dx in shifts:
                cand_d = d_pad[1+dy:box_H+1+dy, 1+dx:box_W+1+dx]
                cand_c = c_pad[1+dy:box_H+1+dy, 1+dx:box_W+1+dx]
                
                valid = ~np.isinf(cand_d)
                m = box_holes & valid & (cand_d > best_d)
                
                best_d = np.where(m, cand_d, best_d)
                best_c = np.where(m[..., None], cand_c, best_c)
                
            filled = best_d > -0.5
            d_box[filled] = best_d[filled]
            c_box[filled] = best_c[filled]
            
            d_pad[1:-1, 1:-1] = d_box
            c_pad[1:-1, 1:-1] = c_box
            
            if i == max_iters - 1 and np.any(np.isinf(d_box)):
                print(f"Hole filling cap reached ({max_iters} iters).")
                
        depth_buf[r_min:r_max, c_min:c_max] = d_box
        canvas[r_min:r_max, c_min:c_max] = c_box
                
        return canvas, depth_buf

    def fill_holes_pyramid(self, canvas: np.ndarray, depth_buf: np.ndarray) -> tuple:
        depth_buf = depth_buf.astype(np.float32)
        canvas = canvas.astype(np.uint8)
        
        holes = np.isinf(depth_buf)
        if not np.any(holes):
            return canvas, depth_buf
            
        pyr_d = [depth_buf]
        pyr_c = [canvas]
        
        d_curr = depth_buf
        c_curr = canvas
        
        # Max 6 levels
        for _ in range(6):
            H, W = d_curr.shape
            H_next, W_next = H // 2, W // 2
            if H_next == 0 or W_next == 0:
                break
                
            d_view = d_curr[:H_next*2, :W_next*2].reshape(H_next, 2, W_next, 2)
            c_view = c_curr[:H_next*2, :W_next*2].reshape(H_next, 2, W_next, 2, 3)
            
            d_safe = np.where(np.isinf(d_view), -1.0, d_view)
            
            d_next = np.max(d_safe, axis=(1, 3))
            d_next = np.where(d_next == -1.0, np.inf, d_next)
            
            d_flat = d_safe.reshape(H_next, W_next, 4)
            idx = np.argmax(d_flat, axis=-1)
            
            c_flat = c_view.reshape(H_next, W_next, 4, 3)
            c_next = np.take_along_axis(c_flat, idx[..., None, None], axis=2).squeeze(2)
            
            pyr_d.append(d_next)
            pyr_c.append(c_next)
            
            d_curr = d_next
            c_curr = c_next
            
        d_fill = pyr_d[-1].copy()
        c_fill = pyr_c[-1].copy()
        
        for i in range(len(pyr_d) - 2, -1, -1):
            d_target = pyr_d[i].copy()
            c_target = pyr_c[i].copy()
            H, W = d_target.shape
            
            d_up = np.repeat(np.repeat(d_fill, 2, axis=0), 2, axis=1)[:H, :W]
            c_up = np.repeat(np.repeat(c_fill, 2, axis=0), 2, axis=1)[:H, :W]
            
            holes = np.isinf(d_target) & ~np.isinf(d_up)
            
            d_target[holes] = d_up[holes]
            c_target[holes] = c_up[holes]
            
            d_fill = d_target
            c_fill = c_target
            
        return c_fill, d_fill
