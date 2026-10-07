import os
import numpy as np
from PIL import Image
from src.depth.estimator import DepthEstimator, box_filter, guided_filter

def test_box_filter():
    img = np.ones((10, 10), dtype=np.float32)
    res = box_filter(img, 1)
    assert np.allclose(res, 1.0)

def test_guided_filter():
    I = np.ones((10, 10), dtype=np.float32) * 0.5
    p = np.ones((10, 10), dtype=np.float32) * 0.2
    res = guided_filter(I, p, 1)
    assert np.allclose(res, 0.2)
