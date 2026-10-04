import streamlit as st
import numpy as np
from PIL import Image
import base64
import time
import io
import os
import glob
import math

from depth_estimator import DepthEstimator
from math_engine import TransformEngine
from view_atlas import orbit_angles, render_atlas, render_external_view, depth_colormap
from mpi_renderer import build_layers, render_atlas_mpi, calibrate_motion, render_mpi, layer_homographies

st.set_page_config(page_title="3D View Synthesis", layout="wide")

# CSS
st.markdown("""
<style>
.viewer-container {
    position: relative;
    width: 100%;
    max-width: 800px;
    margin: 0 auto;
}
.viewer-container canvas {
    width: 100%;
    height: auto;
    display: block;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_depth_model(size="Small"):
    return DepthEstimator(size=size)

@st.cache_data(show_spinner="Estimating depth...")
def process_image(img_path, model_size="Small", refine=True):
    image = Image.open(img_path).convert("RGB")
    
    # Resize to max 1024 px wide to stay under 15 MB payload
    if image.width > 1024:
        ratio = 1024.0 / image.width
        new_size = (1024, int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
    img_array = np.array(image)
    H, W, _ = img_array.shape
    
    # Save temporary resized image for depth model
    temp_path = "temp_resized.jpg"
    image.save(temp_path)
    
    model = get_depth_model(model_size)
    disparity = model.estimate_depth(temp_path, refine_depth=refine)
    
    return disparity, img_array, H, W

@st.cache_data(show_spinner="Building 3D points...")
def get_point_cloud(disparity, z_near, z_far, fov_deg, W, H):
    engine = TransformEngine(W, H, fov_deg)
    Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
    P = engine.unproject_to_3d(Z_map)
    return P, Z_map

@st.cache_data(show_spinner="Building MPI Layers...")
def build_mpi(img_array, disparity, z_near, z_far, n_layers=16):
    layers, z_k = build_layers(img_array, disparity, z_near, z_far, n_layers)
    return layers, z_k

@st.cache_data(show_spinner="Rendering views (Point Splatting)...")
def build_atlas_points(P, colors, K, H, W, n_yaw, n_pitch, max_yaw, max_pitch, z_pivot, edge_tau, edge_mode, s_max, z_near):
    scale = 1.0
    W_s, H_s = W, H
    K_s = K.copy()
    
    engine_render = TransformEngine(W_s, H_s, 60)
    engine_render.fx = K_s[0, 0]
    engine_render.fy = K_s[1, 1]
    engine_render.cx = K_s[0, 2]
    engine_render.cy = K_s[1, 2]
    
    angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)
    
    atlas = render_atlas(
        P, colors, K_s, H_s, W_s, angles,
        Z_pivot=z_pivot, edge_tau=edge_tau, edge_mode=edge_mode, s_max=s_max, z_near=z_near
    )
    
    # Generate Base64
    frames_b64 = []
    for p in range(n_pitch):
        for y in range(n_yaw):
            img = Image.fromarray(atlas[p, y])
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            frames_b64.append(f"data:image/jpeg;base64,{b64}")
            
    return frames_b64, atlas, angles

@st.cache_data(show_spinner="Rendering views (MPI)...")
def build_atlas_layered(layers, z_k, K, W, H, n_yaw, n_pitch, max_yaw, max_pitch, z_near, sharpen_amount=0.0):
    angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)
    
    baseline_x = calibrate_motion(z_near, K[0, 0], W, 0.05)
    baseline_y = calibrate_motion(z_near, K[1, 1], H, 0.05)
    
    frames_b64 = render_atlas_mpi(layers, z_k, K, angles, baseline_x, baseline_y, max_yaw, max_pitch, sharpen_amount)
    return frames_b64, angles, baseline_x, baseline_y

# --- STATE ---
if "selected_image_path" not in st.session_state:
    sample_imgs = sorted(glob.glob("assets/samples/01_*.jpg"))
    st.session_state.selected_image_path = sample_imgs[0] if sample_imgs else None

# --- UI ---
st.title("Interactive 3D Photo Viewer")
st.markdown("A single photo becomes a 3D scene you can look around in: depth estimation plus linear algebra.")

c1, c2, c3 = st.columns([1, 1, 2])
sample_imgs = sorted(glob.glob("assets/samples/01_*.jpg"))
for i, col in enumerate([c1, c2, c3]):
    if i < len(sample_imgs):
        img_path = sample_imgs[i]
        with col:
            st.image(img_path, use_container_width=True)
            if st.button(f"Load Sample {i+1}", key=f"load_{i}"):
                st.session_state.selected_image_path = img_path
                st.rerun()

uploaded = st.file_uploader("Upload your own photo (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    path = f"assets/samples/upload_{uploaded.name}"
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.session_state.selected_image_path = path
    st.rerun()

if not st.session_state.selected_image_path:
    st.stop()

# --- SETTINGS ---
with st.expander("Settings", expanded=False):
    scol1, scol2 = st.columns(2)
    with scol1:
        renderer = st.radio("Renderer", ["Layers (MPI)", "Point Splatting (Original)"])
        fov = st.slider("FOV (deg)", 30.0, 120.0, 60.0)
        z_near = st.slider("Z Near", 0.1, 5.0, 1.0)
        z_far = st.slider("Z Far", 1.0, 20.0, 4.0)
    with scol2:
        if renderer == "Point Splatting (Original)":
            pivot_mode = st.selectbox("Pivot", ["Median Depth", "Center"])
            edge_mode = st.selectbox("Edge Mode", ["demote", "drop"])
            splat_size = st.slider("Splat Size", 0, 5, 2)
            fill_holes = st.checkbox("Hole Filling", value=True)
        else:
            n_layers = st.slider("MPI Layers", 4, 32, 16)
            sharpen_amount = st.slider("Sharpen Amount", 0.0, 1.0, 0.35)
    
    st.markdown("### Diagnostics")
    diag_placeholder = st.empty()

# --- PROCESS IMAGE ---
t0 = time.time()
disparity, img_array, H, W = process_image(st.session_state.selected_image_path)
t1 = time.time()

engine = TransformEngine(W, H, fov)

n_yaw, n_pitch = 9, 5
max_yaw, max_pitch = 12.0, 8.0

if renderer == "Point Splatting (Original)":
    P, Z_map = get_point_cloud(disparity, z_near, z_far, fov, W, H)
    colors = img_array.reshape(-1, 3)
    t2 = time.time()
    
    z_pivot = float(np.median(P[:, 2])) if pivot_mode == "Median Depth" else (z_near + z_far)/2.0
    edge_tau = 0.05
    
    frames_b64, atlas, angles = build_atlas_points(
        P, colors, engine.K, H, W, n_yaw, n_pitch, max_yaw, max_pitch,
        z_pivot, edge_tau, edge_mode, splat_size, z_near
    )
    t3 = time.time()
else:
    layers, z_k = build_mpi(img_array, disparity, z_near, z_far, n_layers)
    t2 = time.time()
    
    frames_b64, angles, base_x, base_y = build_atlas_layered(
        layers, z_k, engine.K, W, H, n_yaw, n_pitch, max_yaw, max_pitch, z_near, sharpen_amount
    )
    t3 = time.time()

# Diagnostics update
with diag_placeholder.container():
    st.write(f"- Depth Estimation: {(t1-t0)*1000:.1f} ms")
    if renderer == "Point Splatting (Original)":
        st.write(f"- Unprojection: {(t2-t1)*1000:.1f} ms")
        st.write(f"- Atlas Rendering (45 views): {(t3-t2)*1000:.1f} ms")
    else:
        st.write(f"- Layer Building: {(t2-t1)*1000:.1f} ms")
        st.write(f"- MPI Atlas Rendering (45 views): {(t3-t2)*1000:.1f} ms")

js_frames_array = "[" + ",".join([f"'{f}'" for f in frames_b64]) + "]"

# --- TABS ---
tab_viewer, tab_how, tab_pc, tab_layers, tab_math = st.tabs(["3D Viewer", "How it works", "Point Cloud", "Layers", "Math (live)"])

with tab_viewer:
    # 3D Viewer Interactive HTML (Cross-blending with canvas)
    html_code = f"""
    <div class="viewer-container" id="viewer" style="cursor: crosshair;">
        <canvas id="view-canvas"></canvas>
    </div>
    <script>
    const framesData = {js_frames_array};
    const nYaw = {n_yaw};
    const nPitch = {n_pitch};
    const canvas = document.getElementById('view-canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('viewer');
    
    let isIdle = false;
    let idleTimer = null;
    let wiggleT = 0;
    
    // Preload images
    let images = [];
    let loadedCount = 0;
    
    for (let i = 0; i < framesData.length; i++) {{
        let img = new Image();
        img.onload = function() {{
            loadedCount++;
            if(loadedCount === 1) {{
                // Initialize canvas size on first load
                canvas.width = img.width;
                canvas.height = img.height;
                setFrameFloat((nPitch-1)/2, (nYaw-1)/2);
            }}
        }};
        img.src = framesData[i];
        images.push(img);
    }}
    
    // Draw blended frame
    function setFrameFloat(p, y) {{
        p = Math.max(0, Math.min(nPitch - 1.001, p));
        y = Math.max(0, Math.min(nYaw - 1.001, y));
        
        let p0 = Math.floor(p), p1 = p0 + 1;
        let y0 = Math.floor(y), y1 = y0 + 1;
        
        let wp1 = p - p0, wp0 = 1 - wp1;
        let wy1 = y - y0, wy0 = 1 - wy1;
        
        let w00 = wp0 * wy0;
        let w01 = wp0 * wy1;
        let w10 = wp1 * wy0;
        let w11 = wp1 * wy1;
        
        if (loadedCount < images.length) return;
        
        ctx.globalCompositeOperation = 'source-over';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.globalCompositeOperation = 'lighter';
        
        // Blend 4 nearest neighbors
        ctx.globalAlpha = w00;
        ctx.drawImage(images[p0 * nYaw + y0], 0, 0);
        
        ctx.globalAlpha = w01;
        ctx.drawImage(images[p0 * nYaw + y1], 0, 0);
        
        ctx.globalAlpha = w10;
        ctx.drawImage(images[p1 * nYaw + y0], 0, 0);
        
        ctx.globalAlpha = w11;
        ctx.drawImage(images[p1 * nYaw + y1], 0, 0);
        
        ctx.globalAlpha = 1.0;
        ctx.globalCompositeOperation = 'source-over';
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
        const yIdx = nx * (nYaw - 1);
        const pIdx = ny * (nPitch - 1);
        setFrameFloat(pIdx, yIdx);
    }});
    
    container.addEventListener('touchmove', (e) => {{
        resetIdle();
        e.preventDefault();
        const rect = container.getBoundingClientRect();
        const touch = e.touches[0];
        const nx = (touch.clientX - rect.left) / rect.width;
        const ny = (touch.clientY - rect.top) / rect.height;
        const yIdx = nx * (nYaw - 1);
        const pIdx = ny * (nPitch - 1);
        setFrameFloat(pIdx, yIdx);
    }}, {{passive: false}});
    
    // Auto wiggle
    setInterval(() => {{
        if (isIdle) {{
            wiggleT += 0.05;
            const yIdx = (nYaw-1)/2 + Math.sin(wiggleT) * (nYaw-1)/2;
            const pIdx = (nPitch-1)/2 + Math.cos(wiggleT*0.7) * (nPitch-1)/2;
            setFrameFloat(pIdx, yIdx);
        }}
    }}, 50);
    
    resetIdle();
    </script>
    """
    st.components.v1.html(html_code, height=600)
    
    # Download GIF / Dolly Zoom
    def create_orbit_gif():
        gif_frames = []
        for t in np.linspace(0, 2*np.pi, 20):
            p = int((n_pitch-1)/2 + np.sin(t)*(n_pitch-1)/2)
            y = int((n_yaw-1)/2 + np.cos(t)*(n_yaw-1)/2)
            if renderer == "Point Splatting (Original)":
                gif_frames.append(Image.fromarray(atlas[p, y]))
            else:
                # Need to grab from frames_b64 and decode
                b64 = frames_b64[p * n_yaw + y].split(",")[1]
                gif_frames.append(Image.open(io.BytesIO(base64.b64decode(b64))))
                
        buf = io.BytesIO()
        gif_frames[0].save(buf, format='GIF', save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
        return buf.getvalue()
        
    def create_dolly_zoom_gif():
        gif_frames = []
        if renderer == "Layers (MPI)":
            for dz in np.linspace(1.0, 2.0, 20):
                K_render = engine.K.copy()
                K_render[0, 0] *= dz
                K_render[1, 1] *= dz
                t = np.array([0, 0, -dz], dtype=np.float32)
                H_k = layer_homographies(engine.K, np.eye(3), t, z_k, K_render=K_render)
                frame = render_mpi(layers, H_k)
                gif_frames.append(Image.fromarray(frame))
        buf = io.BytesIO()
        if len(gif_frames) > 0:
            gif_frames[0].save(buf, format='GIF', save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
        return buf.getvalue()

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download Orbit GIF", data=create_orbit_gif(), file_name="orbit.gif", mime="image/gif")
    with c2:
        if renderer == "Layers (MPI)":
            st.download_button("Download Dolly Zoom GIF", data=create_dolly_zoom_gif(), file_name="dolly.gif", mime="image/gif")

with tab_how:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(img_array, caption="1. Original Image")
        st.latex(r"I(u, v)")
    with c2:
        d_map = engine.disparity_to_depth(disparity, 0, 1) # normalized for colormap
        st.image(depth_colormap(d_map), caption="2. Depth Map")
        if renderer == "Point Splatting (Original)":
            st.latex(r"X = Z \cdot K^{-1} [u, v, 1]^T")
        else:
            st.latex(r"Z_k = Z_{near} + (1 - c_k)(Z_{far} - Z_{near})")
    with c3:
        if renderer == "Point Splatting (Original)":
            P, _ = get_point_cloud(disparity, z_near, z_far, fov, W, H)
            ext_view = render_external_view(P, img_array.reshape(-1, 3), engine.K, H, W)
            st.image(ext_view, caption="3. 3D Point Cloud")
            st.latex(r"P' = (P - c) R^T + c + t")
        else:
            if 'layers' in locals():
                layer_mid = (layers[n_layers//2, ..., :3] * 255).astype(np.uint8)
                st.image(layer_mid, caption="3. MPI Layer Slice")
                st.latex(r"w_k(d) = \max(0, 1 - |d - c_k| \cdot L)")
    with c4:
        # decode base64
        b64 = frames_b64[(n_pitch//2) * n_yaw].split(",")[1]
        new_view = Image.open(io.BytesIO(base64.b64decode(b64)))
        st.image(new_view, caption="4. New View")
        if renderer == "Point Splatting (Original)":
            st.latex(r"[u', v', w]^T = K P'^T \\ u = u'/w")
        else:
            st.latex(r"H_k = K (R - \frac{t n^T}{Z_k}) K^{-1}")

with tab_pc:
    if renderer == "Point Splatting (Original)":
        @st.fragment
        def pc_controls():
            ext_yaw = st.slider("External Yaw", -90.0, 90.0, 55.0)
            ext_pitch = st.slider("External Pitch", -90.0, 90.0, 20.0)
            ext_view = render_external_view(P, colors, engine.K, H, W, yaw=ext_yaw, pitch=ext_pitch)
            st.image(ext_view, use_container_width=True)
        pc_controls()
    else:
        st.write("Point cloud viewer is for Point Splatting renderer.")

with tab_layers:
    if renderer == "Layers (MPI)":
        st.write("Showing a subset of MPI layers (premultiplied RGB)")
        cols = st.columns(5)
        indices = np.linspace(0, n_layers-1, 5, dtype=int)
        for i, idx in enumerate(indices):
            with cols[i]:
                st.image((layers[idx, ..., :3] * 255).astype(np.uint8), caption=f"Layer {idx} (Z={z_k[idx]:.2f})")
    else:
        st.write("Layers view is for MPI renderer.")

with tab_math:
    @st.fragment
    def math_controls():
        st.markdown("Move the sliders to see the matrices update live.")
        myaw = st.slider("Yaw", -45.0, 45.0, 15.0, key="myaw")
        mpitch = st.slider("Pitch", -45.0, 45.0, 5.0, key="mpitch")
        mroll = st.slider("Roll", -45.0, 45.0, 0.0, key="mroll")
        
        R = engine.get_rotation_matrix(np.radians(mpitch), np.radians(myaw), np.radians(mroll))
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Camera Intrinsic Matrix K**")
            st.dataframe(engine.K)
        with c2:
            st.markdown("**Rotation Matrix R**")
            st.dataframe(R)
            
        if renderer == "Layers (MPI)":
            st.markdown("**Layer Homography H_k (for nearest layer)**")
            t = np.array([base_x * (myaw / max_yaw) if max_yaw > 0 else 0,
                          base_y * (mpitch / max_pitch) if max_pitch > 0 else 0,
                          0.0], dtype=np.float32)
            H_k_matrix = layer_homographies(engine.K, R, t, z_k)
            st.dataframe(H_k_matrix[-1])
            
    math_controls()
