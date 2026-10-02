import numpy as np

class TransformEngine:
    """
    The core mathematical graphics engine for 3D View Synthesis.
    All transformations are strictly built using NumPy linear algebra operations.
    """
    def __init__(self, width: int, height: int, fov_scale: float = 1.0):
        self.width = width
        self.height = height
        
        # Principal point (center of the image)
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        # Focal length (estimated standard FOV)
        self.fx = max(width, height) * fov_scale
        self.fy = self.fx
        
        # 1. Intrinsic Camera Matrix (K)
        # Shape: (3, 3)
        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ]
        ], dtype=np.float32)
        
        # Inverse Intrinsic Camera Matrix (K^-1)
        # Shape: (3, 3)
        self.K_inv = np.linalg.inv(self.K)

    def unproject_to_3d(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Maps the 2D pixels and Z-depth into a 3D Point Cloud.
        
        Mathematical Formula:
        [X, Y, Z]^T = Z * K^-1 * [u, v, 1]^T
        
        Inputs:
            depth_map: (H, W) array of Z values.
            
        Outputs:
            point_cloud: (N, 3) matrix where N = H * W, representing (X, Y, Z) coordinates.
        """
        H, W = depth_map.shape
        
        # Create a grid of (u, v) pixel coordinates
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        
        # Flatten into 1D vectors for vectorization
        # Shape of uu, vv, zz: (N,)
        uu = uu.flatten()
        vv = vv.flatten()
        zz = depth_map.flatten()
        
        # Create homogeneous 2D coordinates: matrix of [u, v, 1]
        # Shape: (N, 3)
        ones = np.ones_like(uu)
        uv1 = np.vstack((uu, vv, ones)).T
        
        # Vectorized Inverse Projection
        # Multiply (N, 3) array by the transpose of the (3, 3) Inverse Intrinsic Matrix
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
        Inputs: 
            pitch: Rotation around X-axis
            yaw: Rotation around Y-axis
            roll: Rotation around Z-axis
        Output: 
            R: (3, 3) combined orthogonal rotation matrix
        """
        # Pitch (X-axis rotation)
        cx, sx = np.cos(pitch), np.sin(pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ], dtype=np.float32)
        
        # Yaw (Y-axis rotation)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ], dtype=np.float32)
        
        # Roll (Z-axis rotation)
        cz, sz = np.cos(roll), np.sin(roll)
        Rz = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combined rotation matrix
        return Rz @ Ry @ Rx

    def apply_transform(self, point_cloud: np.ndarray, R: np.ndarray, t: np.ndarray = None) -> np.ndarray:
        """
        Applies linear transformation (rotation and translation) to the 3D point cloud.
        
        Mathematical Formula: P_new = P @ R^T + t
        (Using R^T because our points are stored as row vectors)
        
        Inputs:
            point_cloud: (N, 3) matrix
            R: (3, 3) orthogonal rotation matrix
            t: (3,) translation vector (optional)
        Output:
            transformed_cloud: (N, 3) matrix
        """
        if t is None:
            t = np.zeros(3, dtype=np.float32)
            
        # (N, 3) @ (3, 3) -> (N, 3)
        transformed_cloud = (point_cloud @ R.T) + t
        return transformed_cloud

if __name__ == "__main__":
    # Quick math test
    print("Testing Math Engine...")
    engine = TransformEngine(width=800, height=534)
    
    # Create a dummy depth map (all depth = 5.0)
    dummy_depth = np.full((534, 800), 5.0)
    
    print("\n--- Phase 2: Unprojection ---")
    pc = engine.unproject_to_3d(dummy_depth)
    print(f"Generated Point Cloud Shape: {pc.shape}")
    print(f"Sample 3D point (center pixel): {pc[534*800//2 + 400]}")
    
    print("\n--- Phase 3: Transformations ---")
    # Rotate 90 degrees around Y axis (yaw)
    R = engine.get_rotation_matrix(pitch=0.0, yaw=np.pi/2, roll=0.0)
    print(f"Orthogonal Rotation Matrix (90 deg Yaw):\n{np.round(R, 3)}")
    
    rotated_pc = engine.apply_transform(pc, R)
    print(f"Sample 3D point after rotation: {np.round(rotated_pc[534*800//2 + 400], 3)}")
