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

if 'depth_runs' not in globals():
    depth_runs = 0

@st.cache_data
def process_image(image_bytes, fov_deg):
    global depth_runs
    depth_runs += 1
    image = Image.open(image_bytes).convert("RGB")
    img_array = np.array(image)
    H, W, _ = img_array.shape
    
    image_bytes.seek(0)
    with open("temp.jpg", "wb") as f:
        f.write(image_bytes.read())
        
    model = get_depth_model()
    disparity = model.estimate_depth("temp.jpg")
    
    return disparity, img_array, H, W

uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is None:
    # Fallback to sample for testing
    import io
    with open('tests/fixtures/sample.jpg', 'rb') as f:
        uploaded_file = io.BytesIO(f.read())

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

if 'unprojection_runs' not in globals():
    unprojection_runs = 0

@st.cache_data
def get_point_cloud(disparity, z_near, z_far, fov_deg, W, H):
    global unprojection_runs
    unprojection_runs += 1
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
        corners_u = np.array([0, W-1, W-1, 0, 0, W-1, W-1, 0], dtype=np.float32)
        corners_v = np.array([0, 0, H-1, H-1, 0, 0, H-1, H-1], dtype=np.float32)
        corners_z = np.array([z_near]*4 + [z_far]*4, dtype=np.float32)
        
        c_uvw = np.stack([corners_u * corners_z, corners_v * corners_z, corners_z], axis=1)
        c_P = c_uvw @ engine_render.K_inv.T
        c_P_new = engine_render.apply_transform(c_P, R, t, Z_pivot=pivot)
        c_uvw_new = c_P_new @ engine_render.K.T
        c_u_new = c_uvw_new[:, 0] / c_P_new[:, 2]
        c_v_new = c_uvw_new[:, 1] / c_P_new[:, 2]
        
        L = np.max(c_u_new[[0, 3, 4, 7]])
        R_bound = np.min(c_u_new[[1, 2, 5, 6]])
        T = np.max(c_v_new[[0, 1, 4, 5]])
        B = np.min(c_v_new[[2, 3, 6, 7]])
        
        if L < R_bound and T < B:
            w_valid = R_bound - L
            h_valid = B - T
            scale = max(W / w_valid, H / h_valid)
            
            center_u = (L + R_bound) / 2
            center_v = (T + B) / 2
            
            engine_render.cx = (engine_render.cx - center_u) * scale + W / 2.0
            engine_render.cy = (engine_render.cy - center_v) * scale + H / 2.0
            engine_render.fx *= scale
            engine_render.fy *= scale
    
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
        canvas, depth_buf = engine_render.fill_holes_pyramid(canvas, depth_buf)
    t4 = time.time()
    
    holes_after = np.sum(np.isinf(depth_buf))
    
    st.image(canvas, caption=f"Rendered View (Stride: {stride})", use_container_width=True)
    
    st.text(f"Depth Runs: {depth_runs}")
    st.text(f"Unprojection Runs: {unprojection_runs}")
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
