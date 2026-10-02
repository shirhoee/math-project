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

if __name__ == "__main__":
    # Quick math test
    print("Testing Phase 2 Math Engine...")
    engine = TransformEngine(width=800, height=534)
    
    # Create a dummy depth map (all depth = 5.0)
    dummy_depth = np.full((534, 800), 5.0)
    
    pc = engine.unproject_to_3d(dummy_depth)
    print(f"Intrinsic Matrix K:\n{engine.K}")
    print(f"Generated Point Cloud Shape: {pc.shape}")
    print(f"Sample 3D point (pixel 0,0): {pc[0]}")
    print(f"Sample 3D point (center pixel): {pc[534*800//2 + 400]}")
