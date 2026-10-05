import streamlit as st
import numpy as np
from PIL import Image
import base64
import time
import io
import os
import glob
import pickle
import hashlib
import threading

from depth_estimator import DepthEstimator
from math_engine import TransformEngine
from view_atlas import orbit_angles, render_atlas, depth_colormap
from mpi_renderer import build_layers, render_atlas_mpi, calibrate_motion, layer_homographies

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

def get_hash(img_path, settings_str):
    with open(img_path, "rb") as f:
        data = f.read()
    h = hashlib.md5(data + settings_str.encode('utf-8')).hexdigest()
    return h[:12]

if "job_id" not in st.session_state:
    st.session_state.job_id = 0
if "job_status" not in st.session_state:
    st.session_state.job_status = None
if "job_progress" not in st.session_state:
    st.session_state.job_progress = 0
if "job_result" not in st.session_state:
    st.session_state.job_result = None

if "selected_image_path" not in st.session_state:
    sample_imgs = sorted(glob.glob("assets/samples/01_*.jpg"))
    st.session_state.selected_image_path = sample_imgs[0] if sample_imgs else None

global_job_results = {}
global_job_status = {}
global_job_progress = {}

def run_build_job(job_id, img_path, renderer, motion_preset_str, z_near, z_far, n_layers, sharpen_amount, is_fast_test):
    try:
        global_job_status[job_id] = "Loading image..."
        global_job_progress[job_id] = 5
        image = Image.open(img_path).convert("RGB")
        
        max_dim = 256 if is_fast_test else 1024
        if image.width > max_dim:
            ratio = max_dim / float(image.width)
            new_size = (max_dim, int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
        temp_path = "temp_resized_app.jpg"
        image.save(temp_path)
        img_array = np.array(image)
        H, W, _ = img_array.shape
        
        preset_ratio = float(motion_preset_str.split("(")[1].split(")")[0])
        settings_str = f"W{W}_H{H}_Layers{n_layers}_Preset{preset_ratio}"
        if "upload_" not in img_path and renderer == "Layers (MPI)" and sharpen_amount == 0.35 and z_near == 1.0 and z_far == 4.0 and not is_fast_test:
            h = get_hash(img_path, settings_str)
            cache_path = f"assets/samples/cache/{h}.pkl"
            if os.path.exists(cache_path):
                global_job_status[job_id] = "Loading from cache..."
                global_job_progress[job_id] = 50
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                data["img_array"] = img_array
                data["renderer"] = renderer
                global_job_results[job_id] = data
                global_job_status[job_id] = None
                return

        global_job_status[job_id] = "Estimating depth..."
        global_job_progress[job_id] = 20
        model = get_depth_model("Small")
        disparity = model.estimate_depth(temp_path, refine_depth=not is_fast_test)
        
        fov_deg = 60.0
        engine = TransformEngine(W, H, fov_deg)
        Z_map = engine.disparity_to_depth(disparity, z_near, z_far)
        P = engine.unproject_to_3d(Z_map)
        z_pivot = float(np.median(P[:, 2]))
        
        parallax_total = 2 * preset_ratio * W
        n_yaw = min(41, int(np.ceil(parallax_total / 2.5)) | 1)
        n_pitch = 3
        max_yaw = 15.0
        max_pitch = 5.0
        if is_fast_test:
            n_yaw, n_pitch = 3, 1
            max_yaw, max_pitch = 5.0, 0.0

        if renderer == "Layers (MPI)":
            global_job_status[job_id] = "Building layers..."
            global_job_progress[job_id] = 50
            layers, z_k = build_layers(img_array, disparity, z_near, z_far, n_layers)
            
            global_job_status[job_id] = "Rendering views..."
            global_job_progress[job_id] = 75
            
            baseline_x = calibrate_motion(z_near, np.max(z_k), engine.K[0, 0], W, max_yaw, preset_ratio)
            baseline_y = calibrate_motion(z_near, np.max(z_k), engine.K[1, 1], H, max_pitch, preset_ratio)
            angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)
            
            frames_b64 = render_atlas_mpi(layers, z_k, engine.K, angles, baseline_x, baseline_y, max_yaw, max_pitch, sharpen_amount, z_pivot)
            
            result = {
                "img_array": img_array, "disparity": disparity, "layers": layers, "z_k": z_k,
                "frames_b64": frames_b64, "n_yaw": n_yaw, "n_pitch": n_pitch, "max_yaw": max_yaw, "max_pitch": max_pitch,
                "z_pivot": z_pivot, "H": H, "W": W, "renderer": renderer, "baseline_x": baseline_x, "baseline_y": baseline_y
            }
        else:
            global_job_status[job_id] = "Rendering views (Point Splatting)..."
            global_job_progress[job_id] = 50
            colors = img_array.reshape(-1, 3)
            angles = orbit_angles(n_yaw=n_yaw, n_pitch=n_pitch, max_yaw=max_yaw, max_pitch=max_pitch)
            atlas = render_atlas(P, colors, engine.K, H, W, angles, Z_pivot=z_pivot, edge_tau=0.05, edge_mode='demote', s_max=2, z_near=z_near)
            
            frames_b64 = []
            for p in range(n_pitch):
                for y in range(n_yaw):
                    img = Image.fromarray(atlas[p, y])
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    frames_b64.append(f"data:image/jpeg;base64,{b64}")
                    
            result = {
                "img_array": img_array, "disparity": disparity, "P": P, "atlas": atlas,
                "frames_b64": frames_b64, "n_yaw": n_yaw, "n_pitch": n_pitch, "max_yaw": max_yaw, "max_pitch": max_pitch,
                "z_pivot": z_pivot, "H": H, "W": W, "renderer": renderer
            }

        global_job_results[job_id] = result
        global_job_status[job_id] = None
    except Exception as e:
        global_job_status[job_id] = f"Error: {str(e)}"
        global_job_progress[job_id] = 0

from streamlit.runtime.scriptrunner import add_script_run_ctx

def start_job(img_path, renderer, motion_preset_str, z_near, z_far, n_layers, sharpen_amount):
    st.session_state.job_id += 1
    st.session_state.job_result = None
    jid = st.session_state.job_id
    global_job_status[jid] = "Starting..."
    global_job_progress[jid] = 0
    global_job_results[jid] = None
    
    is_fast_test = os.environ.get("FAST_TEST") == "1"
    
    t = threading.Thread(target=run_build_job, args=(
        jid, img_path, renderer, motion_preset_str, z_near, z_far, n_layers, sharpen_amount, is_fast_test
    ))
    t.daemon = True # ensure thread doesn't block shutdown
    add_script_run_ctx(t)
    t.start()

st.title("Interactive 3D View Synthesis")
st.markdown("A single photo becomes a 3D scene you can look around in: depth estimation plus linear algebra.")

@st.fragment(run_every=1)
def show_progress():
    jid = st.session_state.job_id
    if global_job_results.get(jid) is not None and st.session_state.job_result is None:
        st.session_state.job_result = global_job_results[jid]
        st.rerun()
        
    status = global_job_status.get(jid)
    if status is not None:
        if status.startswith("Error"):
            st.error(status)
        else:
            st.progress(global_job_progress.get(jid, 0), text=status)
    else:
        # Empty placeholder so it doesn't jump
        st.empty()

show_progress()

# Settings Sidebar/Expander
with st.expander("Settings", expanded=False):
    scol1, scol2 = st.columns(2)
    with scol1:
        renderer_opt = st.radio("Renderer", ["Layers (MPI)", "Point Splatting (Original)"])
        motion_preset_opt = st.selectbox("Motion Preset", ["Subtle (0.04)", "Normal (0.07)", "Dramatic (0.11)"], index=1)
        blend_mode = st.radio("Blend Mode", ["Cross-fade", "Nearest Frame"])
    with scol2:
        z_near_opt = st.slider("Z Near", 0.5, 2.0, 1.0)
        z_far_opt = st.slider("Z Far", 2.0, 10.0, 4.0)
        n_layers_opt = st.slider("MPI Layers", 8, 32, 16)
        sharpen_amount_opt = st.slider("Unsharp Mask Amount", 0.0, 1.0, 0.35)
    
    if st.button("Apply Settings"):
        start_job(st.session_state.selected_image_path, renderer_opt, motion_preset_opt, z_near_opt, z_far_opt, n_layers_opt, sharpen_amount_opt)

if st.session_state.job_result is not None:
    res = st.session_state.job_result
    # Build JS viewer
    js_frames = str(res["frames_b64"]).replace("'", '"')
    
    html_code = f"""
    <div class="viewer-container" id="viewer-container">
        <canvas id="view-canvas" width="{res['W']}" height="{res['H']}"></canvas>
    </div>
    <script>
    const frames = {js_frames};
    const nYaw = {res['n_yaw']};
    const nPitch = {res['n_pitch']};
    const blendMode = "{blend_mode}";
    
    const canvas = document.getElementById("view-canvas");
    const ctx = canvas.getContext("2d");
    
    const images = [];
    let loadedCount = 0;
    for (let i = 0; i < frames.length; i++) {{
        const img = new Image();
        img.onload = () => {{
            loadedCount++;
            if(loadedCount === frames.length) {{
                drawFrame(0.5, 0.5);
            }}
        }};
        img.src = frames[i];
        images.push(img);
    }}
    
    let isDragging = false;
    let autoPlay = true;
    let wiggleT = 0;
    let mx = 0.5;
    let my = 0.5;
    
    function getFrames(x, y) {{
        let p = y * (nPitch - 1);
        let yw = x * (nYaw - 1);
        
        let p_idx = Math.min(Math.floor(p), nPitch - 2);
        let y_idx = Math.min(Math.floor(yw), nYaw - 2);
        
        let dp = p - p_idx;
        let dy = yw - y_idx;
        
        if (blendMode === "Nearest Frame") {{
            let final_p = Math.round(p);
            let final_y = Math.round(yw);
            return [{{img: images[final_p * nYaw + final_y], weight: 1.0}}];
        }}
        
        let idx00 = p_idx * nYaw + y_idx;
        let idx01 = p_idx * nYaw + y_idx + 1;
        let idx10 = (p_idx + 1) * nYaw + y_idx;
        let idx11 = (p_idx + 1) * nYaw + y_idx + 1;
        
        return [
            {{img: images[idx00], weight: (1-dp)*(1-dy)}},
            {{img: images[idx01], weight: (1-dp)*dy}},
            {{img: images[idx10], weight: dp*(1-dy)}},
            {{img: images[idx11], weight: dp*dy}}
        ];
    }}
    
    function drawFrame(x, y) {{
        const f = getFrames(x, y);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for(let i=0; i<f.length; i++) {{
            if(f[i].weight > 0.001) {{
                ctx.globalAlpha = f[i].weight;
                ctx.drawImage(f[i].img, 0, 0);
            }}
        }}
        ctx.globalAlpha = 1.0;
    }}
    
    canvas.addEventListener('mousedown', (e) => {{ isDragging = true; autoPlay = false; }});
    window.addEventListener('mouseup', () => {{ isDragging = false; }});
    window.addEventListener('mousemove', (e) => {{
        if (!isDragging) return;
        const rect = canvas.getBoundingClientRect();
        mx = (e.clientX - rect.left) / rect.width;
        my = (e.clientY - rect.top) / rect.height;
        mx = Math.max(0, Math.min(1, mx));
        my = Math.max(0, Math.min(1, my));
        drawFrame(mx, my);
    }});
    
    canvas.addEventListener('touchstart', (e) => {{ isDragging = true; autoPlay = false; }});
    window.addEventListener('touchend', () => {{ isDragging = false; }});
    window.addEventListener('touchmove', (e) => {{
        if (!isDragging) return;
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        mx = (e.touches[0].clientX - rect.left) / rect.width;
        my = (e.touches[0].clientY - rect.top) / rect.height;
        mx = Math.max(0, Math.min(1, mx));
        my = Math.max(0, Math.min(1, my));
        drawFrame(mx, my);
    }}, {{passive: false}});
    
    setInterval(() => {{
        if(autoPlay && loadedCount === frames.length) {{
            wiggleT += 0.05;
            mx = 0.5 + Math.sin(wiggleT) * 0.4;
            my = 0.5 + Math.cos(wiggleT * 0.7) * 0.4;
            drawFrame(mx, my);
        }}
    }}, 50);
    </script>
    """
    st.components.v1.html(html_code, height=600)

    # Downloads
    def create_orbit_gif(ext="GIF"):
        gif_frames = []
        for t in np.linspace(0, 2*np.pi, 20):
            p = int((res["n_pitch"]-1)/2 + np.sin(t)*(res["n_pitch"]-1)/2)
            y = int((res["n_yaw"]-1)/2 + np.cos(t)*(res["n_yaw"]-1)/2)
            b64 = res["frames_b64"][p * res["n_yaw"] + y].split(",")[1]
            gif_frames.append(Image.open(io.BytesIO(base64.b64decode(b64))))
                
        buf = io.BytesIO()
        gif_frames[0].save(buf, format=ext, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
        return buf.getvalue()
        
    def create_dolly_zoom_gif(ext="GIF"):
        gif_frames = []
        if res["renderer"] == "Layers (MPI)":
            engine_k = TransformEngine(res["W"], res["H"], 60.0).K
            for dz in np.linspace(1.0, 2.0, 20):
                K_render = engine_k.copy()
                K_render[0, 0] *= dz
                K_render[1, 1] *= dz
                t_dz = np.array([0, 0, -dz], dtype=np.float32)
                H_k = layer_homographies(engine_k, np.eye(3), t_dz, res["z_k"], K_render=K_render)
                frame = render_mpi(res["layers"], H_k)
                gif_frames.append(Image.fromarray(frame))
        buf = io.BytesIO()
        if len(gif_frames) > 0:
            gif_frames[0].save(buf, format=ext, save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
        return buf.getvalue()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("Download Orbit GIF", data=create_orbit_gif("GIF"), file_name="orbit.gif", mime="image/gif")
    with c2:
        if res["renderer"] == "Layers (MPI)":
            st.download_button("Download Dolly GIF", data=create_dolly_zoom_gif("GIF"), file_name="dolly.gif", mime="image/gif")
    with c3:
        st.download_button("Download Orbit WebP", data=create_orbit_gif("WEBP"), file_name="orbit.webp", mime="image/webp")
    with c4:
        if res["renderer"] == "Layers (MPI)":
            st.download_button("Download Dolly WebP", data=create_dolly_zoom_gif("WEBP"), file_name="dolly.webp", mime="image/webp")

    # Tabs
    tab_how, tab_math, tab_layers = st.tabs(["How it works", "Math", "Layers"])
    
    with tab_how:
        hc1, hc2, hc3, hc4 = st.columns(4)
        with hc1:
            st.image(res["img_array"], caption="Original")
        with hc2:
            st.image(depth_colormap(res["disparity"]), caption="Depth Map")
        with hc3:
            st.write("We estimate depth, project pixels into 3D, and render novel views.")
        
    with tab_layers:
        if res["renderer"] == "Layers (MPI)":
            st.write("Showing a subset of MPI layers (premultiplied RGB)")
            lcols = st.columns(5)
            indices = np.linspace(0, len(res["z_k"])-1, 5, dtype=int)
            for i, idx in enumerate(indices):
                with lcols[i]:
                    st.image((res["layers"][idx, ..., :3] * 255).astype(np.uint8), caption=f"Layer {idx} (Z={res['z_k'][idx]:.2f})")
    
    with tab_math:
        st.markdown("### Live View Matrices")
        myaw = st.slider("Yaw", -45.0, 45.0, 15.0, key="myaw")
        mpitch = st.slider("Pitch", -45.0, 45.0, 5.0, key="mpitch")
        mroll = st.slider("Roll", -45.0, 45.0, 0.0, key="mroll")
        
        engine_math = TransformEngine(res["W"], res["H"], 60.0)
        R = engine_math.get_rotation_matrix(np.radians(mpitch), np.radians(myaw), np.radians(mroll))
        
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Camera Intrinsic Matrix K**")
            st.dataframe(engine_math.K)
        with mc2:
            st.markdown("**Rotation Matrix R**")
            st.dataframe(R)
            st.latex(r"R R^T = \begin{bmatrix}" + " & ".join([f"{x:.2f}" for x in (R @ R.T)[0]]) + r" \\ " +
                     " & ".join([f"{x:.2f}" for x in (R @ R.T)[1]]) + r" \\ " +
                     " & ".join([f"{x:.2f}" for x in (R @ R.T)[2]]) + r"\end{bmatrix}")
            st.latex(r"\det(R) = " + f"{np.linalg.det(R):.2f}")
            
        if res["renderer"] == "Layers (MPI)":
            st.markdown("**Layer Homography H_k (for nearest layer)**")
            max_yaw_v = res["max_yaw"]
            max_pitch_v = res["max_pitch"]
            t_extra = np.array([res["baseline_x"] * (myaw / max_yaw_v) if max_yaw_v > 0 else 0,
                          res["baseline_y"] * (mpitch / max_pitch_v) if max_pitch_v > 0 else 0,
                          0.0], dtype=np.float32)
            c = np.array([0.0, 0.0, res["z_pivot"]], dtype=np.float32)
            t = c - R @ c + t_extra
            H_k_matrix = layer_homographies(engine_math.K, R, t, res["z_k"])
            st.dataframe(H_k_matrix[-1])

st.markdown("---")

# Sample Gallery at the bottom
sample_imgs = sorted(glob.glob("assets/samples/01_*.jpg"))
if len(sample_imgs) >= 3:
    c1, c2, c3 = st.columns(3)
    for i, col in enumerate([c1, c2, c3]):
        img_path = sample_imgs[i]
        with col:
            # Load and crop square
            img = Image.open(img_path).convert("RGB")
            s = min(img.width, img.height)
            img = img.crop(((img.width-s)//2, (img.height-s)//2, (img.width+s)//2, (img.height+s)//2))
            st.image(img, use_container_width=True)
            if st.button(f"Load Sample {i+1}", key=f"load_btn_{i}", use_container_width=True):
                st.session_state.selected_image_path = img_path
                start_job(img_path, "Layers (MPI)", "Normal (0.07)", 1.0, 4.0, 16, 0.35)

uploaded = st.file_uploader("Upload your own photo (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    path = f"assets/samples/upload_{uploaded.name}"
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    if st.session_state.selected_image_path != path:
        st.session_state.selected_image_path = path
        start_job(path, "Layers (MPI)", "Normal (0.07)", 1.0, 4.0, 16, 0.35)

# Initialize on first load
if st.session_state.job_result is None and st.session_state.job_status is None:
    start_job(st.session_state.selected_image_path, "Layers (MPI)", "Normal (0.07)", 1.0, 4.0, 16, 0.35)
