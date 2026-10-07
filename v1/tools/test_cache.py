import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np

# Mock streamlit cache so it actually works in standard python run
def cache_data(func):
    cache = {}
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

import app
app.st.cache_data = cache_data

from app import get_point_cloud

disparity = np.random.rand(10, 10).astype(np.float32)
print("Initial UI Mount...")
P, Z = get_point_cloud(disparity, 1.0, 4.0, 60.0, 10, 10)

print("Moving Yaw slider 10 times...")
for i in range(1, 11):
    yaw = i * 2.0
    # Simulate UI render loop which calls get_point_cloud on each frame
    P, Z = get_point_cloud(disparity, 1.0, 4.0, 60.0, 10, 10)
    
print("Done. Cache miss count should be exactly 1.")
