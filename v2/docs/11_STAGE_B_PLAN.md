# 11 Stage B Plan (Documentation Only)

**Goal:** Extend from single-view to two-view geometry.

**Steps:**
1. Feature matching between two images.
2. Compute Fundamental Matrix using the 8-point algorithm with SVD.
3. Decompose Essential Matrix for relative Camera Pose ($R$, $t$).
4. Triangulate matched points using DLT.
5. Fit scale of depth model output to the triangulated points using least squares.
6. Feed updated pose into the Geometry Engine to render seamless transition.

*Note: DO NOT IMPLEMENT IN STAGE A.*
