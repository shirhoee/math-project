---
trigger: model_decision
description: Apply these guidelines whenever working on the frontend, UI components, styling, or client-side React code.
---


## Frontend Architecture & Patterns

### 1. Component Design
* **Streamlit Decoupling:** Keep Streamlit UI logic completely decoupled from the heavy math and PyTorch inference logic.
* **Component Extraction:** Ensure `app.py` only handles `st.*` components (e.g., `st.file_uploader`, `st.button`) and delegates execution to a separate controller/pipeline.

### 2. Styling & UI
* **Streamlit Limitations:** Work within Streamlit's layout primitives (`st.columns`, `st.container`). Avoid injecting raw CSS or HTML unless absolutely necessary for custom rendering.
* **Responsive State:** Handle Streamlit's top-down execution model properly. Do not re-run expensive setup code unnecessarily.

### 3. State Management & Data Fetching
* **Session State:** Use `st.session_state` strictly for UI-level toggles (e.g., current view mode).
* **Caching:** Use `@st.cache_data` and `@st.cache_resource` for heavy model loading and data processing to prevent memory leaks and slow re-renders.