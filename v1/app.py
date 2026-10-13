import base64
import glob
import hashlib
import io
import os
import pickle

import numpy as np
import streamlit as st
from typing import Any
from src.core.depth_estimator import DepthEstimator
from src.core.math_engine import TransformEngine
from src.core.mpi_renderer import (
    layer_homographies,
    render_mpi,
)
from PIL import Image
from src.core.view_atlas import depth_colormap

st.set_page_config(page_title="3D View Synthesis", layout="wide")

# CSS
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_depth_model(size: str = "Small") -> Any:
    return DepthEstimator(size=size)


def get_hash(img_path: str, settings_str: str) -> str:
    with open(img_path, "rb") as f:
        data = f.read()
    h = hashlib.md5(data + settings_str.encode("utf-8")).hexdigest()
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

    # cleanup stale cache
    cache_dir = os.environ.get("CACHE_DIR", "assets/samples/cache")
    if os.path.exists(cache_dir):
        for f in glob.glob(f"{cache_dir}/*.pkl"):
            try:
                with open(f, "rb") as fp:
                    data = pickle.load(fp)
                img_path = data.get("img_path", "")
                settings_str = data.get("settings_str", "")
                if get_hash(img_path, settings_str) != os.path.basename(f).replace(".pkl", ""):
                    os.remove(f)
            except Exception:
                try:
                    os.remove(f)
                except:
                    pass

from src.pipeline import poll_job_status, submit_render_job


def start_job(img_path: str, renderer: str, motion_preset_str: str, z_near: float, z_far: float, n_layers: int, sharpen_amount: float) -> None:
    st.session_state.job_id += 1
    st.session_state.job_result = None
    jid = st.session_state.job_id
    cached = submit_render_job(jid, img_path, renderer, motion_preset_str, z_near, z_far, n_layers, sharpen_amount)
    if cached:
        st.session_state.job_result = cached

st.title("Interactive 3D View Synthesis")
st.markdown(
    "A single photo becomes a 3D scene you can look around in: depth estimation plus linear algebra."
)
if st.session_state.selected_image_path:
    st.markdown(f"**Loaded: `{os.path.basename(st.session_state.selected_image_path)}`**")

jid = st.session_state.job_id
is_running = poll_job_status(jid)['status'] is not None


@st.fragment(run_every=1 if is_running else None)
def show_progress() -> None:
    jid = st.session_state.job_id
    polled = poll_job_status(jid)
    if polled['result'] is not None and st.session_state.job_result is None:
        st.session_state.job_result = polled['result']
        st.rerun()

    status = polled['status']
    if status is not None:
        if status.startswith("Error"):
            st.error(status)
        else:
            st.progress(polled['progress'], text=status)
    else:
        # Empty placeholder so it doesn't jump
        st.empty()


show_progress()

# Settings Sidebar/Expander
with st.expander("Settings", expanded=False):
    scol1, scol2 = st.columns(2)
    with scol1:
        renderer_opt = st.radio("Renderer", ["Layers (MPI)", "Point Splatting (Original)"])
        motion_preset_opt = st.selectbox(
            "Motion Preset", ["Subtle (0.04)", "Normal (0.07)", "Dramatic (0.11)"], index=1
        )
        blend_mode = st.radio("Blend Mode", ["Cross-fade", "Nearest Frame"])
    with scol2:
        z_near_opt = st.slider("Z Near", 0.5, 2.0, 1.0)
        z_far_opt = st.slider("Z Far", 2.0, 10.0, 4.0)
        n_layers_opt = st.slider("MPI Layers", 8, 32, 16)
        sharpen_amount_opt = st.slider("Unsharp Mask Amount", 0.0, 1.0, 0.35)

    if st.button("Apply Settings"):
        start_job(
            st.session_state.selected_image_path,
            renderer_opt,
            motion_preset_opt,
            z_near_opt,
            z_far_opt,
            n_layers_opt,
            sharpen_amount_opt,
        )

if st.session_state.job_result is not None:
    res = st.session_state.job_result
    # Build JS viewer
    js_frames = str(res["frames_b64"]).replace("'", '"')

    html_code = f"""
    <div class="viewer-container" id="viewer-container" style="max-height: 65vh; overflow: hidden; display: flex; justify-content: center;">
        <canvas id="view-canvas" width="{res["W"]}" height="{res["H"]}" style="max-height: 65vh; width: auto; max-width: 100%; object-fit: contain;"></canvas>
    </div>
    <script>
    const frames_b64 = {js_frames};
    const nYaw = {res["n_yaw"]};
    const nPitch = {res["n_pitch"]};
    const blendMode = "{blend_mode}";
    // Reset Key: {st.session_state.get("reset_view_toggle", False)}
    
    const canvas = document.getElementById("view-canvas");
    const ctx = canvas.getContext("2d");
    
    let frames = new Array(frames_b64.length);
    let loadedCount = 0;
    
    async function loadFrames() {{
        for (let i = 0; i < frames_b64.length; i++) {{
            const res = await fetch(frames_b64[i]);
            const blob = await res.blob();
            const bmp = await createImageBitmap(blob);
            frames[i] = bmp;
            loadedCount++;
        }}
        needsRedraw = true;
    }}
    loadFrames();
    
    let isDragging = false;
    let autoPlay = true;
    let wiggleT = 0;
    let mx = 0.5;
    let my = 0.5;
    let needsRedraw = false;
    let lastMx = -1;
    let lastMy = -1;
    
    function getBlendFrames(x, y) {{
        let p = y * (nPitch - 1);
        let yw = x * (nYaw - 1);
        
        let p_idx = Math.min(Math.floor(p), nPitch - 2);
        let y_idx = Math.min(Math.floor(yw), nYaw - 2);
        
        if (blendMode === "Nearest Frame" || frames.length === 1) {{
            let final_p = Math.round(p);
            let final_y = Math.round(yw);
            return [{{img: frames[final_p * nYaw + final_y], weight: 1.0}}];
        }}
        
        let dp = p - p_idx;
        let dy = yw - y_idx;
        
        let arr = [
            {{img: frames[p_idx * nYaw + y_idx], weight: (1-dp)*(1-dy)}},
            {{img: frames[p_idx * nYaw + y_idx + 1], weight: (1-dp)*dy}},
            {{img: frames[(p_idx + 1) * nYaw + y_idx], weight: dp*(1-dy)}},
            {{img: frames[(p_idx + 1) * nYaw + y_idx + 1], weight: dp*dy}}
        ];
        arr.sort((a, b) => b.weight - a.weight);
        
        let totalW = arr[0].weight + arr[1].weight;
        if (totalW < 0.001) return [{{img: frames[0], weight: 1.0}}];
        
        return [
            {{img: arr[0].img, weight: 1.0}},
            {{img: arr[1].img, weight: arr[1].weight / totalW}}
        ];
    }}
    
    function drawFrame() {{
        if (loadedCount < frames_b64.length) return;
        
        if (mx === lastMx && my === lastMy && !needsRedraw) return;
        lastMx = mx;
        lastMy = my;
        needsRedraw = false;
        
        const f = getBlendFrames(mx, my);
        ctx.globalAlpha = 1.0;
        if (f[0].img) ctx.drawImage(f[0].img, 0, 0, canvas.width, canvas.height);
        
        if (f.length > 1 && f[1].img && f[1].weight > 0.001) {{
            ctx.globalAlpha = f[1].weight;
            ctx.drawImage(f[1].img, 0, 0, canvas.width, canvas.height);
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
        needsRedraw = true;
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
        needsRedraw = true;
    }}, {{passive: false}});
    
    let lastTime = 0;
    function loop(time) {{
        requestAnimationFrame(loop);
        if (document.hidden) return;
        
        let dt = time - lastTime;
        lastTime = time;
        if (dt > 100) dt = 16;
        
        if(autoPlay && loadedCount === frames_b64.length) {{
            wiggleT += 0.001 * dt;
            mx = 0.5 + Math.sin(wiggleT) * 0.4;
            my = 0.5 + Math.cos(wiggleT * 0.7) * 0.4;
            needsRedraw = true;
        }}
        
        drawFrame();
    }}
    requestAnimationFrame(loop);
    </script>
    """
    st.components.v1.html(html_code, height=600)

    # Downloads
    def create_orbit_gif(ext: str = "GIF") -> bytes:
        gif_frames = []
        for t in np.linspace(0, 2 * np.pi, 20):
            p = int((res["n_pitch"] - 1) / 2 + np.sin(t) * (res["n_pitch"] - 1) / 2)
            y = int((res["n_yaw"] - 1) / 2 + np.cos(t) * (res["n_yaw"] - 1) / 2)
            b64 = res["frames_b64"][p * res["n_yaw"] + y].split(",")[1]
            gif_frames.append(Image.open(io.BytesIO(base64.b64decode(b64))))

        buf = io.BytesIO()
        gif_frames[0].save(
            buf, format=ext, save_all=True, append_images=gif_frames[1:], duration=100, loop=0
        )
        return buf.getvalue()

    def create_dolly_zoom_gif(ext: str = "GIF") -> bytes:
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
            gif_frames[0].save(
                buf, format=ext, save_all=True, append_images=gif_frames[1:], duration=100, loop=0
            )
        return buf.getvalue()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Download Orbit GIF",
            data=create_orbit_gif("GIF"),
            file_name="orbit.gif",
            mime="image/gif",
        )
    with c2:
        if res["renderer"] == "Layers (MPI)":
            st.download_button(
                "Download Dolly GIF",
                data=create_dolly_zoom_gif("GIF"),
                file_name="dolly.gif",
                mime="image/gif",
            )
    with c3:
        st.download_button(
            "Download Orbit WebP",
            data=create_orbit_gif("WEBP"),
            file_name="orbit.webp",
            mime="image/webp",
        )
    with c4:
        if res["renderer"] == "Layers (MPI)":
            st.download_button(
                "Download Dolly WebP",
                data=create_dolly_zoom_gif("WEBP"),
                file_name="dolly.webp",
                mime="image/webp",
            )

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
            indices = np.linspace(0, len(res["z_k"]) - 1, 5, dtype=int)
            for i, idx in enumerate(indices):
                with lcols[i]:
                    st.image(
                        (res["layers"][idx, ..., :3] * 255).astype(np.uint8),
                        caption=f"Layer {idx} (Z={res['z_k'][idx]:.2f})",
                    )

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
            st.latex(
                r"R R^T = \begin{bmatrix}"
                + " & ".join([f"{x:.2f}" for x in (R @ R.T)[0]])
                + r" \\ "
                + " & ".join([f"{x:.2f}" for x in (R @ R.T)[1]])
                + r" \\ "
                + " & ".join([f"{x:.2f}" for x in (R @ R.T)[2]])
                + r"\end{bmatrix}"
            )
            st.latex(r"\det(R) = " + f"{np.linalg.det(R):.2f}")

        if res["renderer"] == "Layers (MPI)":
            st.markdown("**Layer Homography H_k (for nearest layer)**")
            max_yaw_v = res["max_yaw"]
            max_pitch_v = res["max_pitch"]
            t_extra = np.array(
                [
                    res["baseline_x"] * (myaw / max_yaw_v) if max_yaw_v > 0 else 0,
                    res["baseline_y"] * (mpitch / max_pitch_v) if max_pitch_v > 0 else 0,
                    0.0,
                ],
                dtype=np.float32,
            )
            c = np.array([0.0, 0.0, res["z_pivot"]], dtype=np.float32)
            t = c - R @ c + t_extra
            H_k_matrix = layer_homographies(engine_math.K, R, t, res["z_k"])
            st.dataframe(H_k_matrix[-1])

st.markdown("---")


def on_select_sample(img_path: str) -> None:
    st.session_state.previous_image_path = st.session_state.get("selected_image_path")
    st.session_state.selected_image_path = img_path
    start_job(img_path, "Layers (MPI)", "Normal (0.07)", 1.0, 4.0, 16, 0.35)


def on_reset_view() -> None:
    st.session_state.reset_view_toggle = not st.session_state.get("reset_view_toggle", False)


def on_prev_sample() -> None:
    prev = st.session_state.get("previous_image_path")
    if prev:
        on_select_sample(prev)


st.markdown("---")

sample_imgs = sorted(glob.glob("assets/samples/01_*.jpg"))
if len(sample_imgs) >= 3:
    c1, c2, c3 = st.columns(3)
    for i, col in enumerate([c1, c2, c3]):
        img_path = sample_imgs[i]
        with col:
            img = Image.open(img_path).convert("RGB")
            s = min(img.width, img.height)
            img = img.crop(
                (
                    (img.width - s) // 2,
                    (img.height - s) // 2,
                    (img.width + s) // 2,
                    (img.height + s) // 2,
                )
            )
            st.image(img, use_container_width=True)
            st.button(
                f"Load Sample {i + 1}",
                key=f"load_btn_{i}",
                use_container_width=True,
                on_click=on_select_sample,
                args=(img_path,),
            )

uploaded = st.file_uploader("Upload your own photo (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    path = f"assets/samples/upload_{uploaded.name}"
    with open(path, "wb") as out_f:
        out_f.write(uploaded.getbuffer())
    if st.session_state.selected_image_path != path:
        on_select_sample(path)
        st.rerun()

if st.session_state.job_result is None and not is_running:
    start_job(
        st.session_state.selected_image_path, 'Layers (MPI)', 'Normal (0.07)', 1.0, 4.0, 16, 0.35
    )
    st.rerun()





