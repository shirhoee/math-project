import re
with open('math_engine.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update signatures
code = code.replace(
    'z_near: float = 1.0) -> tuple:',
    'z_near: float = 1.0, edge_mode: str = "demote") -> tuple:'
)

# 2. Update project_to_2d logic
old_logic_2d = """        if edge_tau is not None and Z_map_original is not None:
            dZ_dx = np.zeros_like(Z_map_original)
            dZ_dy = np.zeros_like(Z_map_original)
            dZ_dx[:, :-1] = np.abs(Z_map_original[:, 1:] - Z_map_original[:, :-1])
            dZ_dy[:-1, :] = np.abs(Z_map_original[1:, :] - Z_map_original[:-1, :])
            g = np.maximum(dZ_dx, dZ_dy) / (Z_map_original + 1e-8)
            
            valid_edges = (g < edge_tau).flatten()
            P = P[valid_edges]
            colors = colors[valid_edges]

        Z = P[:, 2]
        valid_z = Z > z_clip
        P = P[valid_z]
        colors = colors[valid_z]
        Z = Z[valid_z]
        
        uvw = P @ self.K.T
        u = uvw[:, 0] / Z
        v = uvw[:, 1] / Z
        
        return self._render_points(u, v, Z, colors, H, W, splat_gain, s_max, stride, Z_ref)"""

new_logic_2d = """        edge_flag = None
        if edge_tau is not None and Z_map_original is not None:
            dZ_dx = np.zeros_like(Z_map_original)
            dZ_dy = np.zeros_like(Z_map_original)
            dZ_dx[:, :-1] = np.abs(Z_map_original[:, 1:] - Z_map_original[:, :-1])
            dZ_dy[:-1, :] = np.abs(Z_map_original[1:, :] - Z_map_original[:-1, :])
            g = np.maximum(dZ_dx, dZ_dy) / (Z_map_original + 1e-8)
            
            mask = (g >= edge_tau).flatten()
            if edge_mode == "drop":
                valid_edges = ~mask
                P = P[valid_edges]
                colors = colors[valid_edges]
            else:
                edge_flag = mask

        Z = P[:, 2]
        valid_z = Z > z_clip
        P = P[valid_z]
        colors = colors[valid_z]
        Z = Z[valid_z]
        if edge_flag is not None:
            edge_flag = edge_flag[valid_z]
        
        uvw = P @ self.K.T
        u = uvw[:, 0] / Z
        v = uvw[:, 1] / Z
        
        return self._render_points(u, v, Z, colors, H, W, splat_gain, s_max, stride, Z_ref, edge_flag)"""

code = code.replace(old_logic_2d, new_logic_2d)

# 3. Update project_orthographic logic
old_logic_ortho = """        if edge_tau is not None and Z_map_original is not None:
            dZ_dx = np.zeros_like(Z_map_original)
            dZ_dy = np.zeros_like(Z_map_original)
            dZ_dx[:, :-1] = np.abs(Z_map_original[:, 1:] - Z_map_original[:, :-1])
            dZ_dy[:-1, :] = np.abs(Z_map_original[1:, :] - Z_map_original[:-1, :])
            g = np.maximum(dZ_dx, dZ_dy) / (Z_map_original + 1e-8)
            
            valid_edges = (g < edge_tau).flatten()
            P = P[valid_edges]
            colors = colors[valid_edges]

        Pi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        P_ortho = P @ Pi.T
        
        u = P_ortho[:, 0] * self.fx * scale + self.cx
        v = P_ortho[:, 1] * self.fy * scale + self.cy
        Z = P[:, 2] 
        
        return self._render_points(u, v, Z, colors, H, W, splat_gain, s_max, stride, Z_ref)"""

new_logic_ortho = """        edge_flag = None
        if edge_tau is not None and Z_map_original is not None:
            dZ_dx = np.zeros_like(Z_map_original)
            dZ_dy = np.zeros_like(Z_map_original)
            dZ_dx[:, :-1] = np.abs(Z_map_original[:, 1:] - Z_map_original[:, :-1])
            dZ_dy[:-1, :] = np.abs(Z_map_original[1:, :] - Z_map_original[:-1, :])
            g = np.maximum(dZ_dx, dZ_dy) / (Z_map_original + 1e-8)
            
            mask = (g >= edge_tau).flatten()
            if edge_mode == "drop":
                valid_edges = ~mask
                P = P[valid_edges]
                colors = colors[valid_edges]
            else:
                edge_flag = mask

        Pi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        P_ortho = P @ Pi.T
        
        u = P_ortho[:, 0] * self.fx * scale + self.cx
        v = P_ortho[:, 1] * self.fy * scale + self.cy
        Z = P[:, 2] 
        
        return self._render_points(u, v, Z, colors, H, W, splat_gain, s_max, stride, Z_ref, edge_flag)"""

code = code.replace(old_logic_ortho, new_logic_ortho)

# 4. Update _render_points signature and logic
old_render_sig = "stride: int, Z_ref: float) -> tuple:"
new_render_sig = "stride: int, Z_ref: float, edge_flag: np.ndarray = None) -> tuple:"
code = code.replace(old_render_sig, new_render_sig)

old_valid_logic = """        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        u_int, v_int, Z, colors = u_int[valid], v_int[valid], Z[valid], colors[valid]
        
        idx = v_int * W + u_int"""

new_valid_logic = """        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        u_int, v_int, Z, colors = u_int[valid], v_int[valid], Z[valid], colors[valid]
        if edge_flag is not None:
            edge_flag = edge_flag[valid]
        
        idx = v_int * W + u_int"""

code = code.replace(old_valid_logic, new_valid_logic)

with open('math_engine.py', 'w', encoding='utf-8') as f:
    f.write(code)
