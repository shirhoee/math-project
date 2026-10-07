# 60-Second Demo Script

Here is sample 1. Notice the sharp edges because of our Catmull-Rom sampling.
As I move the pointer, you see 56 pixels of parallax without the UI freezing, because we precompute 123 frames in the background into an image atlas.
The app opens instantly because these atlases are aggressively cached.
Our layout dynamically calibrates the camera orbit so that no matter what motion preset you select, the motion remains visually pleasing and physically accurate.
We apply a high-quality unsharp mask right at the end to keep the display crystal clear, entirely eliminating the blurring typically seen in linear blending.
