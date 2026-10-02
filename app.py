import streamlit as st
import numpy as np
from PIL import Image
from depth_estimator import DepthEstimator
from math_engine import TransformEngine
import time

st.set_page_config(page_title="3D View Synthesis", layout="wide")
st.title("Interactive 3D View Synthesis using Linear Transformations")

@st.cache_resource
def get_depth_model():
    return DepthEstimator()

@st.cache_data
def process_image(image_bytes, fov_deg):
    print(f"CACHE MISS: Running depth estimation for FOV {fov_deg}...")
    image = Image.open(image_bytes).convert("RGB")
    img_array = np.array(image)
    H, W, _ = img_array.shape
    
    with open("temp.jpg", "wb") as f:
        f.write(image_bytes.read())
        
    model = get_depth_model()
    disparity = model.estimate_depth("temp.jpg")
    
    return disparity, img_array, H, W

uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

st.sidebar.header("Transformations")
pitch = st.sidebar.slider("Pitch (deg)", -25.0, 25.0, 0.0)
yaw = st.sidebar.slider("Yaw (deg)", -25.0, 25.0, 0.0)
roll = st.sidebar.slider("Roll (deg)", -25.0, 25.0, 0.0)
tx = st.sidebar.slider("Translate X", -1.0, 1.0, 0.0)
ty = st.sidebar.slider("Translate Y", -1.0, 1.0, 0.0)
tz = st.sidebar.slider("Translate Z", -1.0, 1.0, 0.0)

st.sidebar.header("Parameters")
fov = st.sidebar.slider("FOV (deg)", 30.0, 120.0, 60.0)
z_near = st.sidebar.slider("Z Near", 0.1, 5.0, 1.0)
z_far = st.sidebar.slider("Z Far", 1.0, 20.0, 4.0)
z_pivot = st.sidebar.number_input("Z Pivot", value=-1.0, help="-1 uses median depth")

st.sidebar.header("Rendering Options")
auto_crop = st.sidebar.checkbox("Auto Crop Border Voids", value=True)
ortho = st.sidebar.checkbox("Orthographic Projection")
ortho_scale = st.sidebar.slider("Ortho Scale", 0.5, 2.0, 1.0)
splatting = st.sidebar.checkbox("Splatting", value=True)
edge_mask = st.sidebar.checkbox("Edge Masking", value=True)
fill_holes = st.sidebar.checkbox("Hole Filling", value=True)
hq_render = st.sidebar.button("High Quality Render")

@st.cache_data
def get_point_cloud(disparity, z_near, z_far, fov_deg, W, H):
    print("CACHE MISS: Running unprojection...")
    engine = TransformEngine(W, H, fov_deg)
    Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
    P = engine.unproject_to_3d(Z_map)
    return P, Z_map

if uploaded_file is not None:
    disparity, img_array, H, W = process_image(uploaded_file, fov)
    
    t0 = time.time()
    engine = TransformEngine(W, H, fov)
    P, Z_map = get_point_cloud(disparity, z_near, z_far, fov, W, H)
    colors = img_array.reshape(-1, 3)
    t1 = time.time()
    
    R = engine.get_rotation_matrix(np.radians(pitch), np.radians(yaw), np.radians(roll))
    t = np.array([tx, ty, tz], dtype=np.float32)
    pivot = None if z_pivot == -1.0 else z_pivot
    
    stride = 1 if hq_render else 2
    
    if stride > 1:
        H_s, W_s = engine.height // stride, engine.width // stride
        P_render = P.reshape(H, W, 3)[::stride, ::stride].reshape(-1, 3)
        colors_render = colors.reshape(H, W, 3)[::stride, ::stride].reshape(-1, 3)
        engine_render = TransformEngine(W_s, H_s, fov)
        Z_map_render = Z_map[::stride, ::stride]
    else:
        P_render = P
        colors_render = colors
        engine_render = engine
        Z_map_render = Z_map
        
    crop_scale = 1.0
    if auto_crop:
        # Compute roughly how much the image shifted due to rotation
        max_angle = max(abs(pitch), abs(yaw))
        crop_scale = 1.0 + np.tan(np.radians(max_angle))
        
    engine_render.fx *= crop_scale
    engine_render.fy *= crop_scale
    
    t2 = time.time()
    P_new = engine_render.apply_transform(P_render, R, t, Z_pivot=pivot)
    
    splat_gain = 1.0
    s_max = 1 if splatting else 0
    edge_tau = 0.05 if edge_mask else None
    
    if ortho:
        canvas, depth_buf = engine_render.project_orthographic(P_new, colors_render, scale=ortho_scale * crop_scale, splat_gain=splat_gain, s_max=s_max, edge_tau=edge_tau, Z_map_original=Z_map_render, z_near=z_near)
    else:
        canvas, depth_buf = engine_render.project_to_2d(P_new, colors_render, splat_gain=splat_gain, s_max=s_max, edge_tau=edge_tau, Z_map_original=Z_map_render, z_near=z_near)
    t3 = time.time()
    
    holes_before = np.sum(np.isinf(depth_buf))
    total_pixels = depth_buf.size
    
    if fill_holes:
        iters = 40 if stride == 1 else 8
        canvas, depth_buf = engine_render.fill_holes(canvas, depth_buf, max_iters=iters)
    t4 = time.time()
    
    holes_after = np.sum(np.isinf(depth_buf))
    
    st.image(canvas, caption=f"Rendered View (Stride: {stride})", use_container_width=True)
    st.image(img_array, caption="Original Image", width=300)
    
    total_render_time = (t4 - t0) * 1000
    st.markdown(f"### Render Time: **{total_render_time:.1f} ms**")
    st.markdown(f"### Hole Percentage: **{holes_after / total_pixels * 100:.2f}%** (Before fill: {holes_before / total_pixels * 100:.2f}%)")
    
    st.write(f"**Detailed Breakdown:**")
    st.write(f"- Point Cloud Unprojection: {(t1-t0)*1000:.1f} ms")
    st.write(f"- Subsampling & Transforms: {(t2-t1)*1000:.1f} ms")
    st.write(f"- Projection Engine: {(t3-t2)*1000:.1f} ms")
    if fill_holes:
        st.write(f"- Hole Filling: {(t4-t3)*1000:.1f} ms")
