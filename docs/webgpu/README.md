# WebGPU animation demo

The demo displays precomputed X-Humans mesh frames using a browser-side forward
implementation of the triangle-splatting renderer. Click or swipe the side
cards, or use the arrows, to switch people. The side cards play fixed-view
pre-rendered images in sync with the center timeline. Drag the center image or
use the arrow keys to rotate, and hold Shift
while dragging to pan the camera. Scroll to move the camera toward or away from
the mouse pointer; the zoom buttons use the center of the view. Reset restores
the view and zoom. The
play button and slider control the animation. Chrome or another WebGPU-capable
browser and HTTPS or localhost are required. Zoom moves the perspective camera
forward or backward and uses the real-time renderer; it does not change the
400×600 rendering resolution. Reset returns to the cached default view.

For each subject, the default view plays 70 JPEG test images, indices 0
through 69, at 24 frames per second. These images use a fixed
camera, 400×600 resolution, a white background, and CUDA's single-sample OFF
path. They are loaded and decoded incrementally for the selected subject; the
decoded images use additional browser memory. Playback waits when a required
image or mesh frame is still loading, keeping the timeline aligned with the
displayed frame. If an image cannot be
loaded, playback falls back to the mesh viewer.

When the view is rotated, the browser renders the closest of 24 precomputed
poses (test indices 0, 3, ..., 69). The motion is therefore temporally sparser
than the default-view playback. Reset returns to the corresponding full-rate
image. All mesh frames use the same
topology and learned per-face sigma and overlap parameters. Positions and RGB
colors are precomputed per pose. Lighting is baked into the colors; it is not
recomputed as the view rotates. No arbitrary-pose inference runs in the browser.

All four subjects use their completed `full_paper_20261109_v1` Full, seed 42
checkpoints (10,000 training exposures). The selected model uses the new
Lambert and self-occlusion appearance path, with lighting baked into the
per-pose colors exported to the viewer. Each subject's `metadata.json` records
its checkpoint and configuration SHA-256. The 70 preview frames and the 24
mesh frames were generated from the same checkpoint for each subject.

The WebGPU renderer implements the **single-center-sample OFF** forward path,
including triangle expansion, soft coverage and depth-ordered alpha blending.
The Release's adaptive `balanced` 1/4/16 sampling path, backward pass, and
avatar model inference are not implemented here. The rendering input uses the
checkpoint's per-face parameters and degree-0 SH colors evaluated as RGB.

The viewer uses a forward-only, single-sample OFF rendering path; the Full
models were trained and evaluated with balanced sampling. Browser checks are
limited to the explicitly tested poses and views and are not a claim of general
numerical equivalence for every input.
`renderer/` retains upstream copyright notices and license files.
