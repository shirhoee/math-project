import numpy as np
import pytest
from view_atlas import orbit_angles, render_atlas, render_external_view, depth_colormap
from math_engine import TransformEngine

def test_orbit_angles():
    angles = orbit_angles(n_yaw=9, n_pitch=5, max_yaw=12.0, max_pitch=8.0)
    assert angles.shape == (5, 9, 2)
    # Check bounds
    assert np.isclose(angles[..., 0].max(), 12.0)
    assert np.isclose(angles[..., 0].min(), -12.0)
    assert np.isclose(angles[..., 1].max(), 8.0)
    assert np.isclose(angles[..., 1].min(), -8.0)

def test_render_atlas():
    N = 100
    points = np.random.rand(N, 3).astype(np.float32)
    points[:, 2] += 2.0  # push in front of camera
    colors = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
    
    W, H = 320, 240
    engine = TransformEngine(W, H)
    K = engine.K
    
    angles = orbit_angles(n_yaw=3, n_pitch=3, max_yaw=10, max_pitch=10)
    atlas = render_atlas(points, colors, K, H, W, angles, s_max=1)
    
    assert atlas.shape == (3, 3, H, W, 3)
    assert atlas.dtype == np.uint8
    
    # Centre frame should be similar to original rendering
    # We can just check that frames at +/- max yaw differ from center
    diff = np.abs(atlas[1, 0].astype(int) - atlas[1, 1].astype(int))
    assert np.mean(diff) > 0
    
    # Two runs identical
    atlas2 = render_atlas(points, colors, K, H, W, angles, s_max=1)
    assert np.array_equal(atlas, atlas2)

def test_render_external_view():
    N = 100
    points = np.random.rand(N, 3).astype(np.float32)
    points[:, 2] += 2.0
    colors = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
    
    W, H = 320, 240
    engine = TransformEngine(W, H)
    K = engine.K
    
    external = render_external_view(points, colors, K, H, W, yaw=55, pitch=20, pullback=3.0)
    front = render_external_view(points, colors, K, H, W, yaw=0, pitch=0, pullback=0.0)
    
    assert external.shape == (H, W, 3)
    assert external.dtype == np.uint8
    assert np.mean(np.abs(external.astype(int) - front.astype(int))) > 0

def test_depth_colormap():
    d = np.linspace(0, 1, 100).reshape(10, 10)
    cmap = depth_colormap(d)
    assert cmap.shape == (10, 10, 3)
    assert cmap.dtype == np.uint8
    # Max range
    assert cmap.max() <= 255
    assert cmap.min() >= 0
