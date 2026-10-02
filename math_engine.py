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

if __name__ == "__main__":
    print("Testing Stage A fixes...")
    engine = TransformEngine(width=800, height=534)
    print(f"Intrinsic Matrix K:\n{engine.K}")
