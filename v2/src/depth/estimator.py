import os
import hashlib
import pickle
import numpy as np
import torch
from transformers import pipeline
from PIL import Image

def box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """
    Box filter using cumulative sum (integral image) to compute local means.
    img: (H, W) float array
    r: radius
    Returns local means (H, W).
    """
    H, W = img.shape
    S_cum = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float32)
    img_pad = np.pad(img, r, mode='constant', constant_values=0)
    S_cum[1:, 1:] = img_pad.cumsum(axis=0).cumsum(axis=1)
    
    C_cum = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float32)
    ones_pad = np.pad(np.ones_like(img), r, mode='constant', constant_values=0)
    C_cum[1:, 1:] = ones_pad.cumsum(axis=0).cumsum(axis=1)
    
    y_start = np.arange(H)
    y_end = y_start + 2 * r + 1
    x_start = np.arange(W)
    x_end = x_start + 2 * r + 1
    
    Y_start, X_start = np.meshgrid(y_start, x_start, indexing='ij')
    Y_end, X_end = np.meshgrid(y_end, x_end, indexing='ij')
    
    sum_img = S_cum[Y_end, X_end] - S_cum[Y_start, X_end] - S_cum[Y_end, X_start] + S_cum[Y_start, X_start]
    count_img = C_cum[Y_end, X_end] - C_cum[Y_start, X_end] - C_cum[Y_end, X_start] + C_cum[Y_start, X_start]
    
    return sum_img / count_img

def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float = 1e-3) -> np.ndarray:
    """
    Guided filter for edge-aligned depth.
    I: Guide image (grayscale, normalized [0, 1]) (H, W)
    p: Filtering input (disparity map) (H, W)
    """
    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    corr_I = box_filter(I * I, r)
    corr_Ip = box_filter(I * p, r)
    
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    a = cov_Ip / (np.maximum(var_I, 0) + eps)
    b = mean_p - a * mean_I
    
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)
    
    q = mean_a * I + mean_b
    return q

class DepthEstimator:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = None
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_model(self):
        if self.pipe is None:
            # Limits torch threads to 1 if on CPU to prevent process explosion
            if self.device == -1:
                torch.set_num_threads(1)
            self.pipe = pipeline(task="depth-estimation", model=self.model_name, device=self.device)
            
    def compute_hash(self, image: Image.Image, settings: dict) -> str:
        h = hashlib.sha256()
        h.update(image.tobytes())
        h.update(str(settings).encode("utf-8"))
        return h.hexdigest()

    def estimate_depth(self, image: Image.Image, max_resolution=518, refine_depth=True, gamma=1.0) -> np.ndarray:
        settings = {"res": max_resolution, "refine": refine_depth, "gamma": gamma}
        img_hash = self.compute_hash(image, settings)
        cache_file = os.path.join(self.cache_dir, f"{img_hash}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
                
        self.load_model()
        
        W, H = image.size
        scale = min(max_resolution / max(H, W), 1.0)
        new_W = int(W * scale)
        new_H = int(H * scale)
        new_W = new_W - (new_W % 14)
        new_H = new_H - (new_H % 14)
        if new_W == 0 or new_H == 0:
            new_W, new_H = 14, 14
            
        image_resized = image.resize((new_W, new_H), Image.Resampling.LANCZOS)
        result = self.pipe(image_resized)
        depth_image = result["depth"]
        depth_image = depth_image.resize((W, H), Image.Resampling.LANCZOS)
        disparity = np.array(depth_image, dtype=np.float32)
        
        p_min = np.percentile(disparity, 2)
        p_max = np.percentile(disparity, 98)
        if p_max > p_min:
            disparity = np.clip((disparity - p_min) / (p_max - p_min), 0, 1)
        else:
            disparity = np.zeros_like(disparity)
            
        if gamma != 1.0:
            disparity = np.power(disparity, gamma)
            
        if refine_depth:
            I_gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
            r = max(1, int(0.01 * W))
            eps = 1e-3
            disparity = guided_filter(I_gray, disparity, r, eps)
            disparity = np.clip(disparity, 0, 1)
            
        with open(cache_file, "wb") as f:
            pickle.dump(disparity, f)
            
        return disparity
