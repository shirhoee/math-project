# 09 Demo Script

## The 60-Second Demo
1. **Introduction:** "Welcome to the V2 3D Photo Viewer. This system reconstructs a 3D scene from a single 2D image using monocular depth estimation and pure NumPy projective geometry."
2. **Setup:** "We start the FastAPI backend serving our static web viewer using `uvicorn src.server.main:app --reload`."
3. **First Run (Cached):** "I will select 'indoor.jpg' from the dropdown and hit 'Load Sample'. Notice it loads almost instantly (under 0.1s). The depth model output is cached and the mesh is generated on the fly."
4. **Interaction:** "As I drag my mouse, the orbit controls move the camera. You can see the `R` and `t` matrices updating in real-time in the UI panel, reflecting the current camera pose. The `K` matrix is fixed based on a 60-degree FOV."
5. **New Image (Uncached):** "Now I'll upload a new image. The system runs the Depth Anything V2 model to estimate relative depth. It completes in about 5 seconds on CPU, well within our 15s requirement. The geometry engine unprojects the depth map, cuts stretched triangles along depth discontinuities, and the viewer displays the seamless 3D result."
6. **Conclusion:** "This concludes the demonstration of Stage A, proving our vectorised math engine and performant architecture are ready for Stage B."
