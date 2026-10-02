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
def process_image(image_bytes, fov_deg, z_near, z_far):
    # Load image and model
    image = Image.open(image_bytes).convert("RGB")
    img_array = np.array(image)
    H, W, _ = img_array.shape
    
    # Save original to disk to give path to pipeline (or use a temp file)
    with open("temp.jpg", "wb") as f:
        f.write(image_bytes.read())
        
    model = get_depth_model()
    disparity = model.estimate_depth("temp.jpg")
    
    engine = TransformEngine(W, H, fov_deg)
    Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
    P = engine.unproject_to_3d(Z_map)
    colors = img_array.reshape(-1, 3)
    
    return engine, P, colors, Z_map, img_array

uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# Sliders
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
ortho = st.sidebar.checkbox("Orthographic Projection")
ortho_scale = st.sidebar.slider("Ortho Scale", 0.5, 2.0, 1.0)
splatting = st.sidebar.checkbox("Splatting", value=True)
edge_mask = st.sidebar.checkbox("Edge Masking", value=True)
fill_holes = st.sidebar.checkbox("Hole Filling", value=True)
hq_render = st.sidebar.button("High Quality Render")

if uploaded_file is not None:
    t0 = time.time()
    engine, P, colors, Z_map, img_array = process_image(uploaded_file, fov, z_near, z_far)
    t1 = time.time()
    
    R = engine.get_rotation_matrix(np.radians(pitch), np.radians(yaw), np.radians(roll))
    t = np.array([tx, ty, tz], dtype=np.float32)
    pivot = None if z_pivot == -1.0 else z_pivot
    
    stride = 1 if hq_render else 2
    
    if stride > 1:
        H, W = engine.height, engine.width
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        mask = (uu % stride == 0) & (vv % stride == 0)
        mask_flat = mask.flatten()
        P_render = P[mask_flat]
        colors_render = colors[mask_flat]
    else:
        P_render = P
        colors_render = colors
        
    t2 = time.time()
    P_new = engine.apply_transform(P_render, R, t, Z_pivot=pivot)
    
    splat_gain = 1.0
    s_max = 1 if splatting else 0
    edge_tau = 0.05 if edge_mask else None
    
    if ortho:
        canvas, depth_buf = engine.project_orthographic(P_new, colors_render, scale=ortho_scale, splat_gain=splat_gain, s_max=s_max, z_near=z_near, edge_tau=edge_tau, Z_map_original=Z_map)
    else:
        canvas, depth_buf = engine.project_to_2d(P_new, colors_render, splat_gain=splat_gain, s_max=s_max, z_near=z_near, edge_tau=edge_tau, Z_map_original=Z_map)
    t3 = time.time()
    
    if fill_holes:
        canvas, depth_buf = engine.fill_holes(canvas, depth_buf, max_iters=15)
    t4 = time.time()
    
    st.image(canvas, caption=f"Rendered View (Stride: {stride})", use_container_width=True)
    st.image(img_array, caption="Original Image", width=300)
    
    st.write(f"**Performance Metrics:**")
    st.write(f"- Depth Estimation & Caching: {(t1-t0)*1000:.1f} ms")
    st.write(f"- Point Cloud Subsampling: {(t2-t1)*1000:.1f} ms")
    st.write(f"- Transform & Projection: {(t3-t2)*1000:.1f} ms")
    if fill_holes:
        st.write(f"- Hole Filling: {(t4-t3)*1000:.1f} ms")
else:
    st.write("Please upload an image to begin.")
