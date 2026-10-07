# 02 Math Rules

1. All geometry in `src/geometry` is raw NumPy, fully vectorised. No scipy/Open3D/OpenCV there.
2. Every function has a docstring stating input and output matrix shapes and the formula.
3. Every function has a unit test.
4. 3x3 inverse via adjugate must match `np.linalg.inv` within 1e-9.
5. No behavioural change when porting from v1 unless logged in `docs/07_DECISIONS.md`.
