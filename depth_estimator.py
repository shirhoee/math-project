import torch
from transformers import pipeline
from PIL import Image
import numpy as np
import time

def box_filter(img, r):
    """
    Box filter using cumulative sum (integral image) to compute local means in O(1) per pixel.
    img: (H, W) float array
    r: radius of the box filter (window is 2r+1 x 2r+1)
    Returns: (H, W) array of local sums, and (H, W) array of local counts.
    """
    H, W = img.shape
    # Pad to handle borders correctly without shrinking the window
    # Actually, we can pad with edge values, or just compute sums over valid pixels.
    # To compute sums over valid pixels easily:
    # pad img with zeros
    S = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float32)
    counts = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float32)
    
    img_pad = np.pad(img, r, mode='constant', constant_values=0)
    ones_pad = np.pad(np.ones_like(img), r, mode='constant', constant_values=0)
    
    S_cum = img_pad.cumsum(axis=0).cumsum(axis=1)
    counts_cum = ones_pad.cumsum(axis=0).cumsum(axis=1)
    
    # window sum for point (y, x) involves cumsums at bounds
    # S_cum has shape (H+2r, W+2r). We want sum of img_pad[y:y+2r+1, x:x+2r+1]
    y_min, y_max = 0, H
    x_min, x_max = 0, W
    
    # For a pixel (y, x) in original image, its window in padded image is
    # [y, y+2r] x [x, x+2r].
    # Sum = S_cum[y+2r, x+2r] - S_cum[y-1, x+2r] - S_cum[y+2r, x-1] + S_cum[y-1, x-1]
    
    # We can compute this for all y in [0, H-1] and x in [0, W-1]
    y1 = np.arange(H)
    y2 = y1 + 2 * r
    x1 = np.arange(W)
    x2 = x1 + 2 * r
    
    Y1, X1 = np.meshgrid(y1, x1, indexing='ij')
    Y2, X2 = np.meshgrid(y2, x2, indexing='ij')
    
    def get_sum(cum_array):
        sum_arr = cum_array[Y2, X2]
        # Since y1=0 maps to y-1 which is out of bounds, we handle the -1 index by using 0
        # Actually cumsum array is better built 1-indexed to avoid -1
        # Let's rebuild 1-indexed
        pass
    
    # Let's do 1-indexed cumsum
    S_cum = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float32)
    S_cum[1:, 1:] = img_pad.cumsum(axis=0).cumsum(axis=1)
    
    C_cum = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float32)
    C_cum[1:, 1:] = ones_pad.cumsum(axis=0).cumsum(axis=1)
    
    # y1, x1 are the start indices (inclusive) in the padded array
    # y2, x2 are the end indices (inclusive) in the padded array
    # padded array indices for window around (y, x) are y to y+2r, x to x+2r
    # in 1-indexed cumsum, sum is S[y2+1, x2+1] - S[y1, x2+1] - S[y2+1, x1] + S[y1, x1]
    y_start = np.arange(H)
    y_end = y_start + 2 * r + 1
    x_start = np.arange(W)
    x_end = x_start + 2 * r + 1
    
    Y_start, X_start = np.meshgrid(y_start, x_start, indexing='ij')
    Y_end, X_end = np.meshgrid(y_end, x_end, indexing='ij')
    
    sum_img = S_cum[Y_end, X_end] - S_cum[Y_start, X_end] - S_cum[Y_end, X_start] + S_cum[Y_start, X_start]
    count_img = C_cum[Y_end, X_end] - C_cum[Y_start, X_end] - C_cum[Y_end, X_start] + C_cum[Y_start, X_start]
    
    return sum_img / count_img

def guided_filter(I, p, r, eps=1e-3):
    """
    Guided filter for edge-aligned depth.
    I: Guide image (grayscale, normalized [0, 1]) (H, W)
    p: Filtering input (disparity map) (H, W)
    r: Radius of the filter
    eps: Regularization term
    Formulas:
        mean_I = f_mean(I, r)
        mean_p = f_mean(p, r)
        corr_I = f_mean(I * I, r)
        corr_Ip = f_mean(I * p, r)
        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = f_mean(a, r)
        mean_b = f_mean(b, r)
        q = mean_a * I + mean_b
    """
    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    corr_I = box_filter(I * I, r)
    corr_Ip = box_filter(I * p, r)
    
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)
    
    q = mean_a * I + mean_b
    return q

class DepthEstimator:
    """
    A mathematical wrapper class that uses Depth Anything V2 as a "sensor" 
    to extract the relative disparity from a 2D image.
    """
    MODELS = {
        "Small": "depth-anything/Depth-Anything-V2-Small-hf",
        "Base": "depth-anything/Depth-Anything-V2-Base-hf",
        "Large": "depth-anything/Depth-Anything-V2-Large-hf"
    }
    
    def __init__(self, size="Small"):
        model_name = self.MODELS.get(size, self.MODELS["Small"])
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

    def estimate_depth(self, image_path: str, max_resolution=1022, refine_depth=True, gamma=1.0) -> np.ndarray:
        """
        Extracts the disparity map from an image, applies percentile normalization,
        and optionally refines it with a guided filter.
        """
        image = Image.open(image_path).convert("RGB")
        
        # Calculate multiple of 14 resolution
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
        
        # Resize depth back to original image size
        depth_image = depth_image.resize((W, H), Image.Resampling.LANCZOS)
        disparity = np.array(depth_image, dtype=np.float32)
        
        # Percentile Normalization to prevent outliers from compressing the useful depth range
        p_min = np.percentile(disparity, 2)
        p_max = np.percentile(disparity, 98)
        if p_max > p_min:
            disparity = np.clip((disparity - p_min) / (p_max - p_min), 0, 1)
        else:
            disparity = np.zeros_like(disparity)
            
        if gamma != 1.0:
            disparity = np.power(disparity, gamma)
            
        # Guided Filter for Edge Alignment
        if refine_depth:
            I_gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
            r = max(1, int(0.01 * W))
            eps = 1e-3
            disparity = guided_filter(I_gray, disparity, r, eps)
            disparity = np.clip(disparity, 0, 1)
            
        return disparity

def get_depth_model(size="Small"):
    return DepthEstimator(size=size)
