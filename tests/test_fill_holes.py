import numpy as np
import pytest
from math_engine import TransformEngine

def fill_holes_old(canvas: np.ndarray, depth_buf: np.ndarray, max_iters: int = 15) -> tuple:
    H, W = depth_buf.shape
    max_iters = min(max_iters, 40)
    canvas = canvas.copy()
    depth_buf = depth_buf.copy()
    
    for i in range(max_iters):
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

def test_fill_holes_regression():
    engine = TransformEngine(20, 20)
    
    np.random.seed(42)
    for _ in range(3):
        canvas = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        depth = np.random.uniform(1.0, 5.0, (20, 20)).astype(np.float32)
        
        # Add random holes
        holes = np.random.rand(20, 20) > 0.5
        depth[holes] = np.inf
        
        c_old, d_old = fill_holes_old(canvas.copy(), depth.copy(), max_iters=3)
        c_new, d_new = engine.fill_holes(canvas.copy(), depth.copy(), max_iters=3)
        
        np.testing.assert_array_equal(d_old, d_new)
        np.testing.assert_array_equal(c_old, c_new)
