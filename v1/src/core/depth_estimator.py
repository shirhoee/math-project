from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


import cv2


def box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """
    Box filter to compute local means in O(1) per pixel.
    Uses cv2.boxFilter for maximum CPU performance with fallback to cumsum.
    img: (H, W) float array
    r: radius of the box filter (window is 2r+1 x 2r+1)
    Returns: np.ndarray shape (H, W) dtype float32 of local means.
    """
    try:
        ksize = 2 * r + 1
        return cv2.boxFilter(img.astype(np.float32), ddepth=-1, ksize=(ksize, ksize), borderType=cv2.BORDER_REPLICATE)
    except Exception:
        H, W = img.shape
        S = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float32)
        counts = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float32)

        img_pad = np.pad(img, r, mode="constant", constant_values=0)
        ones_pad = np.pad(np.ones_like(img), r, mode="constant", constant_values=0)

        S_cum = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float32)
        S_cum[1:, 1:] = img_pad.cumsum(axis=0).cumsum(axis=1)

        C_cum = np.zeros((H + 2 * r + 1, W + 2 * r + 1), dtype=np.float32)
        C_cum[1:, 1:] = ones_pad.cumsum(axis=0).cumsum(axis=1)

        y_start = np.arange(H)
        y_end = y_start + 2 * r + 1
        x_start = np.arange(W)
        x_end = x_start + 2 * r + 1

        Y_start, X_start = np.meshgrid(y_start, x_start, indexing="ij")
        Y_end, X_end = np.meshgrid(y_end, x_end, indexing="ij")

        sum_img = (
            S_cum[Y_end, X_end]
            - S_cum[Y_start, X_end]
            - S_cum[Y_end, X_start]
            + S_cum[Y_start, X_start]
        )
        count_img = (
            C_cum[Y_end, X_end]
            - C_cum[Y_start, X_end]
            - C_cum[Y_end, X_start]
            + C_cum[Y_start, X_start]
        )

        return sum_img / count_img


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float = 1e-3) -> np.ndarray:
    """
    Guided filter for edge-aligned depth.
    I: Guide image (grayscale, normalized [0, 1]) (H, W)
    p: Filtering input (disparity map) (H, W)
    r: Radius of the filter
    eps: Regularization term

    Returns: np.ndarray shape (H, W) dtype float32
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

    MODELS: dict[str, str] = {
        "Small": "depth-anything/Depth-Anything-V2-Small-hf",
        "Base": "depth-anything/Depth-Anything-V2-Base-hf",
        "Large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    device: Any
    pipe: Any

    def __init__(self, size: str = "Small") -> None:
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

    def estimate_depth(
        self,
        image_path: str,
        max_resolution: int = 1022,
        refine_depth: bool = True,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """
        Extracts the disparity map from an image, applies percentile normalization,
        and optionally refines it with a guided filter.

        Returns: np.ndarray shape (H, W) dtype float32
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
        p_min = float(np.percentile(disparity, 2))
        p_max = float(np.percentile(disparity, 98))
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


_DEPTH_MODEL_CACHE: dict[str, DepthEstimator] = {}


def get_depth_model(size: str = "Small") -> DepthEstimator:
    """
    Returns a cached DepthEstimator singleton to avoid reloading PyTorch weights on every render.
    """
    if size not in _DEPTH_MODEL_CACHE:
        _DEPTH_MODEL_CACHE[size] = DepthEstimator(size=size)
    return _DEPTH_MODEL_CACHE[size]

