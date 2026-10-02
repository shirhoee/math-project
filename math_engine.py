import numpy as np

class TransformEngine:
    """
    The core mathematical graphics engine for 3D View Synthesis.
    All transformations are strictly built using NumPy linear algebra operations.
    """
    def __init__(self, width: int, height: int, fov_deg: float = 60.0):
        self.width = width
        self.height = height
        
        # Principal point (center of the image)
        self.cx = (width - 1) / 2.0
        self.cy = (height - 1) / 2.0
        
        # Focal length from FOV
        fov_rad = np.radians(fov_deg)
        self.fx = (width / 2.0) / np.tan(fov_rad / 2.0)
        self.fy = self.fx
        
        # 1. Intrinsic Camera Matrix (K)
        # Coordinate Convention: +X right, +Y down, +Z forward (into scene)
        # Shape: (3, 3)
        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ]
        ], dtype=np.float32)
        
        # Inverse Intrinsic Camera Matrix (K^-1)
        # Shape: (3, 3)
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
        Inputs: 
            depth_map: (H, W) array of Z values.
        Outputs: 
            point_cloud: (N, 3) matrix where N = H * W, representing (X, Y, Z) coordinates.
        """
        H, W = depth_map.shape
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        
        uu = uu.flatten()
        vv = vv.flatten()
        zz = depth_map.flatten()
        
        ones = np.ones_like(uu)
        uv1 = np.vstack((uu, vv, ones)).T  # Shape: (N, 3)
        
        # Vectorized Inverse Projection
        # (N, 3) @ (3, 3) -> (N, 3)
        normalized_rays = uv1 @ self.K_inv.T
        
        # Multiply each ray by its depth (Z)
        # (N, 3) * (N, 1) -> (N, 3)
        point_cloud = normalized_rays * zz[:, np.newaxis]
        
        return point_cloud

    def get_rotation_matrix(self, pitch: float, yaw: float, roll: float) -> np.ndarray:
        """
        Creates a 3x3 Orthogonal Rotation Matrix based on Euler angles (in radians).
        
        Mathematical Formula: R = R_z * R_y * R_x
        Convention: +X right, +Y down, +Z forward.
        Inputs: 
            pitch: Rotation around X-axis
            yaw: Rotation around Y-axis
            roll: Rotation around Z-axis
        Output: 
            R: (3, 3) combined orthogonal rotation matrix
        """
        cx, sx = np.cos(pitch), np.sin(pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ], dtype=np.float32)
        
        cy, sy = np.cos(yaw), np.sin(yaw)
        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ], dtype=np.float32)
        
        cz, sz = np.cos(roll), np.sin(roll)
        Rz = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return Rz @ (Ry @ Rx)

    def apply_transform(self, point_cloud: np.ndarray, R: np.ndarray, t: np.ndarray = None, Z_pivot: float = None) -> np.ndarray:
        """
        Applies linear transformation (pivot-based rotation and translation) to the 3D point cloud.
        
        Mathematical Formula: P_new = (P - c) @ R^T + c + t
        (Using R^T because our points are stored as row vectors)
        
        Inputs:
            point_cloud: (N, 3) matrix
            R: (3, 3) orthogonal rotation matrix
            t: (3,) translation vector (optional)
            Z_pivot: Depth value (scalar) around which to pivot the rotation. Defaults to median depth.
        Output:
            transformed_cloud: (N, 3) matrix
        """
        if t is None:
            t = np.zeros(3, dtype=np.float32)
            
        if Z_pivot is None:
            Z_pivot = float(np.median(point_cloud[:, 2]))
            
        c = np.array([0, 0, Z_pivot], dtype=np.float32)
        
        # (N, 3) - (3,) -> (N, 3)
        centered = point_cloud - c
        # (N, 3) @ (3, 3) + (3,) + (3,) -> (N, 3)
        transformed_cloud = (centered @ R.T) + c + t
        return transformed_cloud

    def build_homogeneous_transform(self, R: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Builds the 4x4 homogeneous transformation matrix for pivot rotation and translation.
        Formula: T = T(c+t) @ R4 @ T(-c)
        Inputs:
            R: (3, 3) matrix
            t: (3,) matrix
            c: (3,) matrix
        Output:
            T: (4, 4) matrix
        """
        T_minus_c = np.eye(4, dtype=np.float32)
        T_minus_c[0:3, 3] = -c
        
        R4 = np.eye(4, dtype=np.float32)
        R4[0:3, 0:3] = R
        
        T_plus_c_t = np.eye(4, dtype=np.float32)
        T_plus_c_t[0:3, 3] = c + t
        
        # (4,4) @ (4,4) @ (4,4) -> (4,4)
        return T_plus_c_t @ (R4 @ T_minus_c)

    def project_to_2d(self, P: np.ndarray, colors: np.ndarray, z_clip: float = 1e-3, splat_gain: float = 1.0, s_max: int = 1, z_near: float = 1.0, edge_tau: float = 0.05, Z_map_original: np.ndarray = None) -> tuple:
        """
        Projects 3D points back to 2D pixels with Z-buffering, splatting, and edge masking.
        
        Mathematical Formula: [u' v' w]^T = K @ [X Y Z]^T
        Perspective divide: u = u'/w, v = v'/w
        Inputs:
            P: (N, 3) rotated point cloud
            colors: (N, 3) flat color array aligned with P
        Output:
            canvas: (H, W, 3) uint8 image
            depth_buf: (H, W) float depth buffer
        """
        H, W = self.height, self.width
        
        # Stretched edge masking
        if edge_tau is not None and Z_map_original is not None:
            dZ_dx = np.zeros_like(Z_map_original)
            dZ_dy = np.zeros_like(Z_map_original)
            dZ_dx[:, :-1] = np.abs(Z_map_original[:, 1:] - Z_map_original[:, :-1])
            dZ_dy[:-1, :] = np.abs(Z_map_original[1:, :] - Z_map_original[:-1, :])
            g = np.maximum(dZ_dx, dZ_dy) / (Z_map_original + 1e-8)
            
            valid_edges = (g < edge_tau).flatten()
            P = P[valid_edges]
            colors = colors[valid_edges]

        # 1. Near-plane clip
        Z = P[:, 2]
        valid_z = Z > z_clip
        P = P[valid_z]
        colors = colors[valid_z]
        Z = Z[valid_z]
        
        # 2. Perspective Projection
        # (N, 3) @ (3, 3) -> (N, 3)
        uvw = P @ self.K.T
        u = uvw[:, 0] / Z
        v = uvw[:, 1] / Z
        
        return self._render_points(u, v, Z, colors, H, W, splat_gain, s_max, z_near)
        
    def project_orthographic(self, P: np.ndarray, colors: np.ndarray, scale: float = 1.0, splat_gain: float = 1.0, s_max: int = 1, z_near: float = 1.0, edge_tau: float = 0.05, Z_map_original: np.ndarray = None) -> tuple:
        """
        Projects 3D points using Orthographic Projection Matrix.
        Pi = diag(1, 1, 0)
        
        Mathematical Formula: [X' Y' 0]^T = Pi @ [X Y Z]^T
        Inputs:
            P: (N, 3) point cloud
            colors: (N, 3) color array
        Output: (N, 3) -> (H, W, 3) canvas
        """
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

        Pi = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        
        # (N, 3) @ (3, 3) -> (N, 3)
        P_ortho = P @ Pi.T
        
        # Shift to center and scale
        u = P_ortho[:, 0] * self.fx * scale + self.cx
        v = P_ortho[:, 1] * self.fy * scale + self.cy
        Z = P[:, 2] 
        
        return self._render_points(u, v, Z, colors, H, W, splat_gain, s_max, z_near)
        
    def _render_points(self, u: np.ndarray, v: np.ndarray, Z: np.ndarray, colors: np.ndarray, H: int, W: int, splat_gain: float, s_max: int, z_near: float) -> tuple:
        """
        Vectorized sub-routine for Z-buffering and Splatting.
        """
        if s_max > 0:
            s_i = np.clip(np.ceil(splat_gain * z_near / Z), 0, s_max).astype(np.int32)
            u_cands, v_cands, z_cands, c_cands = [], [], [], []
            
            # Loop over constant offset set [-s_max, s_max]^2 (whitelisted)
            for dx in range(-s_max, s_max + 1):
                for dy in range(-s_max, s_max + 1):
                    mask = s_i >= max(abs(dx), abs(dy))
                    if np.any(mask):
                        u_cands.append(u[mask] + dx)
                        v_cands.append(v[mask] + dy)
                        z_cands.append(Z[mask])
                        c_cands.append(colors[mask])
                        
            if u_cands:
                u = np.concatenate(u_cands)
                v = np.concatenate(v_cands)
                Z = np.concatenate(z_cands)
                colors = np.concatenate(c_cands, axis=0)

        # 3. Round to int
        u_int = np.rint(u).astype(np.int64)
        v_int = np.rint(v).astype(np.int64)
        
        # 4. Bounds mask
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        u_int = u_int[valid]
        v_int = v_int[valid]
        Z = Z[valid]
        colors = colors[valid]
        
        # Robust Z-buffer
        idx = v_int * W + u_int
        order = np.lexsort((Z, idx))
        idx_s = idx[order]
        
        first = np.ones(idx_s.shape, dtype=bool)
        first[1:] = idx_s[1:] != idx_s[:-1]
        winners = order[first]
        
        canvas = np.zeros((H * W, 3), dtype=np.uint8)
        depth_buf = np.full(H * W, np.inf, dtype=np.float32)
        
        canvas[idx[winners]] = colors[winners]
        depth_buf[idx[winners]] = Z[winners]
        
        return canvas.reshape((H, W, 3)), depth_buf.reshape((H, W))

    def fill_holes(self, canvas: np.ndarray, depth_buf: np.ndarray, max_iters: int = 15) -> tuple:
        """
        Fills holes (pixels with infinite depth) iteratively using background-biased neighbor interpolation.
        For each hole, it chooses the valid neighbor with the largest depth (furthest away), preventing
        foreground colors from bleeding into the background.
        
        Inputs:
            canvas: (H, W, 3) image
            depth_buf: (H, W) depth map
            max_iters: Number of dilation iterations
        Outputs:
            canvas, depth_buf (filled)
        """
        H, W = depth_buf.shape
        for _ in range(max_iters):
            holes = np.isinf(depth_buf)
            if not np.any(holes):
                break
                
            d_pad = np.pad(depth_buf, 1, constant_values=0)
            c_pad = np.pad(canvas, ((1, 1), (1, 1), (0, 0)), constant_values=0)
            
            d_pad_safe = np.where(np.isinf(d_pad), -1.0, d_pad)
            
            neighbors_d = []
            neighbors_c = []
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                neighbors_d.append(d_pad_safe[1+dy:H+1+dy, 1+dx:W+1+dx])
                neighbors_c.append(c_pad[1+dy:H+1+dy, 1+dx:W+1+dx])
                
            neighbors_d = np.stack(neighbors_d, axis=0)
            neighbors_c = np.stack(neighbors_c, axis=0)
            
            max_idx = np.argmax(neighbors_d, axis=0)
            
            max_d = np.take_along_axis(neighbors_d, max_idx[np.newaxis, ...], axis=0)[0]
            
            max_idx_c = np.broadcast_to(max_idx[np.newaxis, ..., np.newaxis], (1, H, W, 3))
            max_c = np.take_along_axis(neighbors_c, max_idx_c, axis=0)[0]
            
            valid_fill = holes & (max_d > 0)
            depth_buf[valid_fill] = max_d[valid_fill]
            canvas[valid_fill] = max_c[valid_fill]
            
        return canvas, depth_buf

if __name__ == "__main__":
    print("Testing Stage A fixes...")
    engine = TransformEngine(width=800, height=534)
    print(f"Intrinsic Matrix K:\n{engine.K}")
