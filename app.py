import streamlit as st
import numpy as np
from PIL import Image
import base64
import time
import io
import os
import glob

from depth_estimator import DepthEstimator
from math_engine import TransformEngine
from view_atlas import orbit_angles, render_atlas, render_external_view, depth_colormap

st.set_page_config(page_title="3D View Synthesis", layout="wide")

# CSS for dark simple styling and hiding defaults
st.markdown("""
<style>
.viewer-container {
    position: relative;
    width: 100%;
    max-width: 600px;
    margin: 0 auto;
}
.viewer-container img {
    width: 100%;
    height: auto;
    display: block;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_depth_model():
    return DepthEstimator()

@st.cache_data(show_spinner="Estimating depth...")
def process_image(img_path):
    image = Image.open(img_path).convert("RGB")
    img_array = np.array(image)
    H, W, _ = img_array.shape
    
    # Save a temporary copy for the model
    temp_path = "temp.jpg"
    image.save(temp_path)
    
    model = get_depth_model()
    disparity = model.estimate_depth(temp_path)
    
    return disparity, img_array, H, W

@st.cache_data(show_spinner="Building 3D points...")
def get_point_cloud(disparity, z_near, z_far, fov_deg, W, H):
    engine = TransformEngine(W, H, fov_deg)
    Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
    P = engine.unproject_to_3d(Z_map)
    return P, Z_map

@st.cache_data(show_spinner="Rendering 45 views...")
def build_atlas(P, colors, K, H, W, n_yaw, n_pitch, max_yaw, max_pitch, z_pivot, edge_tau, edge_mode, s_max, z_near):
    angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)
    
    # Downscale for preview atlas
    scale = min(1.0, 480.0 / W)
    W_s, H_s = int(W * scale), int(H * scale)
    
    # K matrix scaling
    K_s = K.copy()
    K_s[0, 0] *= scale; K_s[1, 1] *= scale
    K_s[0, 2] *= scale; K_s[1, 2] *= scale
    
    atlas = render_atlas(
        P, colors, K_s, H_s, W_s, angles,
        Z_pivot=z_pivot, edge_tau=edge_tau, edge_mode=edge_mode, s_max=s_max, z_near=z_near
    )
    return atlas, angles, W_s, H_s

# Main UI
st.title("Interactive 3D View Synthesis")
st.markdown("A single photo becomes a 3D scene you can look around in: depth estimation plus linear algebra.")

# Samples and uploader
samples = sorted(glob.glob("assets/samples/*.jpg")) + sorted(glob.glob("tests/fixtures/gen/*.jpg"))
samples = list(dict.fromkeys(samples))[:3] # unique first 3

col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
clicked_sample = None
with col1:
    if len(samples) > 0 and st.button("Sample 1"): clicked_sample = samples[0]
with col2:
    if len(samples) > 1 and st.button("Sample 2"): clicked_sample = samples[1]
with col3:
    if len(samples) > 2 and st.button("Sample 3"): clicked_sample = samples[2]
with col4:
    uploaded_file = st.file_uploader("Or upload your own", type=['jpg', 'jpeg', 'png'])

# Initialize session state for selected image
if 'selected_image_path' not in st.session_state:
    st.session_state.selected_image_path = samples[0] if samples else None

if uploaded_file is not None:
    temp_path = "temp_uploaded.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.selected_image_path = temp_path
elif clicked_sample is not None:
    st.session_state.selected_image_path = clicked_sample

if st.session_state.selected_image_path is None:
    st.stop()

# --- SETTINGS (Collapsed) ---
with st.expander("Settings & Diagnostics"):
    scol1, scol2 = st.columns(2)
    with scol1:
        fov = st.slider("FOV (deg)", 30.0, 120.0, 60.0)
        z_near = st.slider("Z Near", 0.1, 5.0, 1.0)
        z_far = st.slider("Z Far", 1.0, 20.0, 4.0)
    with scol2:
        pivot_mode = st.selectbox("Pivot", ["Median Depth", "Center"])
        edge_mode = st.selectbox("Edge Mode", ["demote", "drop"])
        splat_size = st.slider("Splat Size", 0, 5, 2)
        fill_holes = st.checkbox("Hole Filling", value=True)
    
    st.markdown("### Diagnostics")
    diag_placeholder = st.empty()


# --- PROCESS IMAGE ---
t0 = time.time()
disparity, img_array, H, W = process_image(st.session_state.selected_image_path)
t1 = time.time()

engine = TransformEngine(W, H, fov)
P, Z_map = get_point_cloud(disparity, z_near, z_far, fov, W, H)
colors = img_array.reshape(-1, 3)
t2 = time.time()

z_pivot = float(np.median(P[:, 2])) if pivot_mode == "Median Depth" else (z_near + z_far)/2.0
edge_tau = 0.05

n_yaw, n_pitch = 9, 5
max_yaw, max_pitch = 12.0, 8.0

atlas, angles, W_s, H_s = build_atlas(
    P, colors, engine.K, H, W, n_yaw, n_pitch, max_yaw, max_pitch,
    z_pivot, edge_tau, edge_mode, splat_size, z_near
)
t3 = time.time()

# Diagnostics update
with diag_placeholder.container():
    st.write(f"- Depth Estimation: {(t1-t0)*1000:.1f} ms")
    st.write(f"- Unprojection: {(t2-t1)*1000:.1f} ms")
    st.write(f"- Atlas Rendering (45 views): {(t3-t2)*1000:.1f} ms")

# Generate Base64 frames for JS viewer
frames_b64 = []
for p in range(n_pitch):
    for y in range(n_yaw):
        img = Image.fromarray(atlas[p, y])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        frames_b64.append(f"data:image/jpeg;base64,{b64}")

js_frames_array = "[" + ",".join([f"'{f}'" for f in frames_b64]) + "]"

# --- TABS ---
tab_viewer, tab_how, tab_pc, tab_math = st.tabs(["3D Viewer", "How it works", "Point Cloud", "Math (live)"])

with tab_viewer:
    # 3D Viewer Interactive HTML
    html_code = f"""
    <div class="viewer-container" id="viewer" style="cursor: crosshair;">
        <img id="view-img" src="{frames_b64[(n_pitch//2)*n_yaw + (n_yaw//2)]}" draggable="false" />
    </div>
    <script>
    const frames = {js_frames_array};
    const nYaw = {n_yaw};
    const nPitch = {n_pitch};
    const img = document.getElementById('view-img');
    const container = document.getElementById('viewer');
    
    let isIdle = false;
    let idleTimer = null;
    let wiggleT = 0;
    
    function setFrame(p, y) {{
        p = Math.max(0, Math.min(nPitch - 1, p));
        y = Math.max(0, Math.min(nYaw - 1, y));
        img.src = frames[p * nYaw + y];
    }}
    
    function resetIdle() {{
        isIdle = false;
        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {{ isIdle = true; }}, 2000);
    }}
    
    container.addEventListener('mousemove', (e) => {{
        resetIdle();
        const rect = container.getBoundingClientRect();
        const nx = (e.clientX - rect.left) / rect.width;
        const ny = (e.clientY - rect.top) / rect.height;
        const yIdx = Math.floor(nx * nYaw);
        const pIdx = Math.floor(ny * nPitch);
        setFrame(pIdx, yIdx);
    }});
    
    container.addEventListener('touchmove', (e) => {{
        resetIdle();
        e.preventDefault();
        const rect = container.getBoundingClientRect();
        const touch = e.touches[0];
        const nx = (touch.clientX - rect.left) / rect.width;
        const ny = (touch.clientY - rect.top) / rect.height;
        const yIdx = Math.floor(nx * nYaw);
        const pIdx = Math.floor(ny * nPitch);
        setFrame(pIdx, yIdx);
    }}, {{passive: false}});
    
    // Auto wiggle
    setInterval(() => {{
        if (isIdle) {{
            wiggleT += 0.05;
            const yIdx = Math.floor((nYaw-1)/2 + Math.sin(wiggleT) * (nYaw-1)/2);
            const pIdx = Math.floor((nPitch-1)/2 + Math.cos(wiggleT*0.7) * (nPitch-1)/2);
            setFrame(pIdx, yIdx);
        }}
    }}, 50);
    
    resetIdle();
    </script>
    """
    st.components.v1.html(html_code, height=600)
    
    # Download GIF
    def create_gif():
        gif_frames = []
        # Circular path through atlas
        for t in np.linspace(0, 2*np.pi, 20):
            p = int((n_pitch-1)/2 + Math.sin(t)*(n_pitch-1)/2) if 'Math' in globals() else int((n_pitch-1)/2 + np.sin(t)*(n_pitch-1)/2)
            y = int((n_yaw-1)/2 + Math.cos(t)*(n_yaw-1)/2) if 'Math' in globals() else int((n_yaw-1)/2 + np.cos(t)*(n_yaw-1)/2)
            gif_frames.append(Image.fromarray(atlas[p, y]))
        buf = io.BytesIO()
        gif_frames[0].save(buf, format='GIF', save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
        return buf.getvalue()
        
    st.download_button("Download orbit GIF", data=create_gif(), file_name="orbit.gif", mime="image/gif")
    
    st.markdown("---")
    st.markdown("### Fallback Controls (High Quality Render)")
    @st.fragment
    def fallback_controls():
        fyaw = st.slider("Yaw", -max_yaw, max_yaw, 0.0, key="fyaw")
        fpitch = st.slider("Pitch", -max_pitch, max_pitch, 0.0, key="fpitch")
        if st.button("Render exact view"):
            R = engine.get_rotation_matrix(np.radians(fpitch), np.radians(fyaw), 0.0)
            P_new = engine.apply_transform(P, R, Z_pivot=z_pivot)
            canvas, d = engine.project_to_2d(P_new, colors, edge_tau=edge_tau, edge_mode=edge_mode, s_max=splat_size, z_near=z_near)
            if fill_holes:
                canvas, _ = engine.fill_holes_pyramid(canvas, d)
            st.image(canvas, use_container_width=True)
    fallback_controls()


with tab_how:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(img_array, caption="1. Original Image")
        st.latex(r"I(u, v)")
    with c2:
        d_map = engine.disparity_to_depth(disparity, 0, 1) # normalized for colormap
        st.image(depth_colormap(d_map), caption="2. Depth Map")
        st.latex(r"X = Z \cdot K^{-1} [u, v, 1]^T")
    with c3:
        ext_view = render_external_view(P, colors, engine.K, H, W)
        st.image(ext_view, caption="3. 3D Point Cloud")
        st.latex(r"P' = (P - c) R^T + c + t")
    with c4:
        st.image(atlas[n_pitch//2, 0], caption="4. New View")
        st.latex(r"[u', v', w]^T = K P'^T \\ u = u'/w")


with tab_pc:
    @st.fragment
    def pc_controls():
        ext_yaw = st.slider("External Yaw", -90.0, 90.0, 55.0)
        ext_pitch = st.slider("External Pitch", -90.0, 90.0, 20.0)
        ext_view = render_external_view(P, colors, engine.K, H, W, yaw=ext_yaw, pitch=ext_pitch)
        st.image(ext_view, use_container_width=True)
    pc_controls()


with tab_math:
    @st.fragment
    def math_controls():
        st.markdown("Move the sliders to see the matrices update live. A rotation about the camera gives no parallax. A rotation about a pivot inside the scene gives the 3D effect.")
        myaw = st.slider("Yaw", -45.0, 45.0, 15.0, key="myaw")
        mpitch = st.slider("Pitch", -45.0, 45.0, 5.0, key="mpitch")
        mroll = st.slider("Roll", -45.0, 45.0, 0.0, key="mroll")
        
        R = engine.get_rotation_matrix(np.radians(mpitch), np.radians(myaw), np.radians(mroll))
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Camera Intrinsic Matrix K**")
            st.dataframe(engine.K)
        with c2:
            st.markdown("**Composite Rotation Matrix R**")
            st.dataframe(R)
            
        I = np.eye(3)
        R_RT = R @ R.T
        max_diff = np.max(np.abs(R_RT - I))
        det_R = np.linalg.det(R)
        
        st.markdown(f"**Live Checks:** `max|R Rᵀ − I|` = {max_diff:.2e}, `det R` = {det_R:.4f}")
    math_controls()
