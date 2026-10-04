# 60-Second Live Demo Script

1. **Open the App (0:00 - 0:15)**
   - Run `streamlit run app.py`. The app loads on the first sample image.
   - Point out that there is no debug text, just the interactive 3D viewer.
   - Wait for 2 seconds to let the auto-wiggle start, proving the image is now a 3D scene.
   - Move your mouse over the viewer to show how smoothly it tracks the cursor without any server round trips (the frames are pre-rendered).

2. **How It Works (0:15 - 0:30)**
   - Click the **"How it works"** tab.
   - Show the four images: original, depth map, point cloud, and final view.
   - Briefly point out the linear algebra formulas below each step: "This is how we go from pixels to 3D points, rotate them, and project them back."

3. **The Math Tab (0:30 - 0:45)**
   - Click the **"Math (live)"** tab.
   - Show the live matrices ($K$ and $R$).
   - Move the Yaw slider and point out the live checks: `max|R Rᵀ − I|` remains near 0 (proving $R$ is orthogonal) and `det R` is exactly 1.
   - Explain the pivot: "Rotating around the camera gives no parallax, but rotating around a pivot inside the scene gives the 3D effect."

4. **Holes and Fill (0:45 - 1:00)**
   - Mention the occlusion limitation: the back of foreground objects is unknown, leaving holes when we rotate.
   - Show how the background-biased pyramid fill covers these holes automatically (demonstrate by sliding to an extreme angle in the fallback controls).
   - Conclude.
