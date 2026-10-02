import torch
from transformers import pipeline
from PIL import Image
import numpy as np

class DepthEstimator:
    """
    A mathematical wrapper class that uses Depth Anything V2 as a "sensor" 
    to extract the relative disparity from a 2D image.
    """
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        print(f"Loading depth estimation model: {model_name}...")
        if torch.cuda.is_available():
            self.device = 0
            print("Using GPU (CUDA) for depth estimation.")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Silicon (MPS) for depth estimation.")
        else:
            self.device = -1
            print("Using CPU for depth estimation.")
            
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.device)
        print("Model loaded successfully.")

    def estimate_depth(self, image_path: str) -> np.ndarray:
        """
        Extracts the disparity map from an image.
        Depth Anything outputs disparity (closer = larger value).
        
        Input:
            image_path (str): Path to the input 2D image.
        Output:
            disparity (np.ndarray): (H, W) array of disparity normalized to [0, 1].
        """
        image = Image.open(image_path).convert("RGB")
        result = self.pipe(image)
        
        depth_image = result["depth"]
        disparity = np.array(depth_image, dtype=np.float32)
        
        d_min, d_max = np.min(disparity), np.max(disparity)
        if d_max > d_min:
            disparity = (disparity - d_min) / (d_max - d_min)
        else:
            disparity = np.zeros_like(disparity)
            
        return disparity
