# 05 Error Playbook

| SYMPTOM | LIKELY CAUSES | DIAGNOSTIC STEPS | FIX | PREVENTION |
|:---|:---|:---|:---|:---|
| `pip install` failures, missing wheels | Version conflicts, Windows lacking build tools | Check python version, run `pip install -v` | Use binary wheels or pure-python equivalents if possible. Ensure venv is active. | Pin exact versions in `requirements.txt`. |
| Out-of-memory on CPU/GPU | Input image too large for model, or memory leak | Check Task Manager / `nvidia-smi` | Downscale input image to a sensible size (e.g. max 518x518) before depth estimation. | Add size limit checks before passing to model. |
| Model download failure / offline | Hugging Face network block or no internet | Check HF cache `~/.cache/huggingface` | Set `HF_HUB_OFFLINE=1` if cached, or download manually. | Include instructions for offline caching if needed. |
| Extremely slow first load | PyTorch scanning large tensors, or CPU MKL threading | Profile import time | Limit `OMP_NUM_THREADS=1`. | Do not load model at import time, load lazily. |
| Output depth orientation inverted | Depth vs inverse depth confusion | Visualize depth map, check if close objects are dark or light | Invert depth via `1 / depth` or `max - depth` as needed. | Document model's depth output scale. |
| NaN or inf in depth | Division by zero in relative depth | Check input values, check `z_near` and `z_far` | Add small epsilon `1e-6` to denominator. | Validate ranges before division. |
| Singular or near-singular matrices | Matrix determinant is 0 | Print `np.linalg.det(M)` | Check camera parameters (e.g. zero FOV). | Use adjugate safely or add eps. |
| Zero or negative Z | Points behind camera | Print min(Z) | Clip Z to positive epsilon `np.clip(Z, 1e-3, None)`. | Test unprojection strictly. |
| Division by zero in perspective | Z is zero | Print Z stats | Add epsilon to Z before division. | Same as above. |
| Shape and broadcasting errors | Mismatched matrix shapes | Print `x.shape` at the crash site | Reshape to explicit `(N, 1)` or `(N, 3)` before broadcast. | Add shape tests and assertions. |
| Row-vector vs column-vector confusion | Multiplying $v R$ vs $R v$ | Check equation vs code | Standardize on $P_{new} = P R^T + t$. | Document convention. |
| Handedness and axis-sign flips | Image y-axis is down, OpenGL is up | Render a known asymmetrical object | Flip Y axis in output if needed. | Test with a known simple identity. |
| Black holes at disocclusions | Foreground moves, revealing empty background | Move camera extremely left/right | Background-biased fill or edge cutting. | Add edge-aware mesh cutting. |
| Stretched triangles | Discontinuities not cut | Render wireframe, look for long lines | Discard faces where $\Delta Z >$ threshold. | Filter indices in mesh generation. |
| z-fighting, texture seams | Precision issues in depth buffer | Move camera slightly | Increase near/far plane ratio, ensure depth is unique. | Use metric depth limits. |
| Upside-down or mirrored output | Y axis flipped in image coords | Check texture UV mapping | Flip V coordinate `v = 1.0 - v`. | Add unit test for identity pose. |
| CORS errors, port already in use | Server configuration | Check network tab in browser, `netstat` | Add CORS middleware in FastAPI, change port. | Hardcode or config specific port cleanly. |
| Payload too large | Mesh too dense, too many faces | Check network payload size | Subsample grid or use binary format instead of JSON. | Use typed arrays or binary buffer. |
| Blank page, WebGL context failure | Browser doesn't support WebGL or bad canvas | Check console errors | Fallback message, ensure THREE.js loaded. | Add error boundary in JS. |
| Memory growth, process explosion | Multiprocessing spawning uncontrollably | Check Task Manager | Guard with `if __name__ == '__main__':` on Windows. | Limit workers. |
| Tests failing due to file paths | Hardcoded paths | Run tests from different cwd | Use `Path(__file__).parent`. | Write tests robustly. |
| Unrelated files created | Scripts dumping temp files | Check `git status` | Route all temp output to `outputs/` or `cache/`. | Add golden rule. |
