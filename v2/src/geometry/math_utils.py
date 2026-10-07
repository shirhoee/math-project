import numpy as np

def adjugate_inverse(M: np.ndarray) -> np.ndarray:
    """
    Derivation: M^-1 = 1/det(M) * adj(M)
    Shapes: (3, 3) -> (3, 3)
    """
    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]
    
    inv = np.empty_like(M)
    inv[0, 0] = m11 * m22 - m12 * m21
    inv[0, 1] = m02 * m21 - m01 * m22
    inv[0, 2] = m01 * m12 - m02 * m11
    inv[1, 0] = m12 * m20 - m10 * m22
    inv[1, 1] = m00 * m22 - m02 * m20
    inv[1, 2] = m02 * m10 - m00 * m12
    inv[2, 0] = m10 * m21 - m11 * m20
    inv[2, 1] = m01 * m20 - m00 * m21
    inv[2, 2] = m00 * m11 - m01 * m10
    
    det = m00 * inv[0, 0] + m01 * inv[1, 0] + m02 * inv[2, 0]
    if abs(det) < 1e-12:
        det = 1e-12 if det >= 0 else -1e-12
        
    return inv / det
