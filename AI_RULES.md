# AI Rules of Engagement

The AI Assistant MUST adhere strictly to the following rules when writing code or suggesting architectures for this project:

1. **Math Over Modules:** For any 3D geometry (unprojection, rotation, translation, projection), build the matrices from scratch using raw `numpy` arrays. DO NOT use libraries like `scipy.spatial.transform`, `Open3D`, `OpenGL`, or `Three.js` for the math.
2. **AI is a Sensor Only:** The Deep Learning model (e.g., MiDaS) is strictly confined to generating the Z-depth array. It MUST NOT handle any 3D rendering or transformations.
3. **NumPy Vectorization:** Python `for` loops over image pixels are strictly forbidden for performance reasons. All matrix transformations must be vectorized (e.g., multiplying an $N \times 3$ matrix by a $3 \times 3$ matrix directly). 
4. **Server-Side Rendering:** The Streamlit frontend is purely for UI. All mathematical processing (the graphics engine) happens in the Python backend. The frontend simply displays the resulting 2D image array.
5. **Mathematical Documentation:** Every core function must include a docstring explicitly detailing the linear algebra formula it represents, **including the matrix shapes for inputs and outputs** (e.g., `Input: (N, 3) @ (3, 3) -> Output: (N, 3)`).
6. **Git Version Control:** After completing every successful milestone or phase, the AI must proactively run `git add`, `git commit -m "..."`, and `git push` to maintain the version history.
