import torch
from transformers import pipeline
from PIL import Image
import numpy as np

class DepthEstimator:
    """
    A mathematical wrapper class that uses Depth Anything V2 as a "sensor" 
    to extract the Z-axis depth values from a 2D image.
    
    This class treats the AI model as a black-box function: 
    f(image) -> depth_map
    """
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        print(f"Loading depth estimation model: {model_name}...")
        
        # Check for GPU (CUDA) or Apple Silicon (MPS) for faster matrix operations
        if torch.cuda.is_available():
            self.device = 0
            print("Using GPU (CUDA) for depth estimation.")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Silicon (MPS) for depth estimation.")
        else:
            self.device = -1
            print("Using CPU for depth estimation. This might take a few seconds per image.")
            
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.device)
        print("Model loaded successfully.")

    def estimate_depth(self, image_path: str) -> np.ndarray:
        """
        Extracts the depth map from an image.
        
        Formula/Concept:
        Extracts a scalar Z value for each (u, v) pixel coordinate.
        
        Input:
            image_path (str): Path to the input 2D image.
            
        Output:
            depth_map (np.ndarray): A 2D numpy array of shape (H, W) where 
                                    each element represents the relative depth (Z) 
                                    of that pixel.
        """
        image = Image.open(image_path).convert("RGB")
        
        # Run inference
        result = self.pipe(image)
        
        # The pipeline returns a PIL Image for 'depth'. Convert to NumPy array.
        depth_image = result["depth"]
        depth_map = np.array(depth_image, dtype=np.float32)
        
        # Normalize the depth map to a standard scale (e.g., 0 to 1) 
        # to ensure the math remains stable during unprojection.
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)
        
        return depth_map

if __name__ == "__main__":
    # A simple test block to verify the sensor works
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        print("No image provided. Please provide an image path to test.")
        sys.exit(1)
        
    estimator = DepthEstimator()
    depth = estimator.estimate_depth(test_image)
    print(f"Successfully extracted depth map!")
    print(f"Depth map shape: {depth.shape}")
    print(f"Max depth value: {np.max(depth):.4f}, Min depth value: {np.min(depth):.4f}")
