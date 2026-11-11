# Experimental WebGPU triangle splatting forward pass

This is an independent prototype of the release CUDA renderer's forward RGB compositing, not the full MSAvatar model. It uses CPU preprocessing and 16×16 tile lists sorted by view-space face-center depth; a WGSL compute pass composites the triangles affecting each pixel. The result is displayed on a canvas. No network dependency is required.

```js
import { TriangleSplatRenderer } from './renderer/triangle-splat.js';
const renderer = await TriangleSplatRenderer.create(canvas);
const stats = await renderer.render({
  vertices, // V*3 flattened world positions, Float32Array
  faces, // F*3 flattened 0-based vertex indices, Uint32Array
  colors, // V*3 flattened RGB floats, Float32Array
  weights, // V floats, Float32Array
  sigmas, // F floats or a number; omit for 0.5
  overlaps, // F floats or a number; omit for 0.7
  viewMatrix, // 16 numbers, column-major world-to-view
  projectionMatrix, // 16 numbers, column-major full world-to-clip (view included)
  background: [1, 1, 1], // RGB, defaults to white
  sampling: 'adaptive', // default; use 'single' for the original center sample
  sigmaScale: 1, // per-face multiplier in [0, 2]
  overlapScale: 1, // per-face multiplier in [0, 2]
});
// canvas.width and canvas.height determine the output pixel dimensions.
// New vertices and matrices may be supplied on each render call.
const rgba = await renderer.readPixels(); // Float32Array, row-major, RGB + (1 - T)
renderer.destroy();
```

`fixture.html` is a synthetic, two-triangle smoke fixture. Serve the parent folder on localhost (for example, `python -m http.server`) and open `renderer/fixture.html` in a WebGPU-capable browser. The fixture is deliberately not a representation of the real avatar.

The CUDA reference is `release/MSAvatar_Release/submodules/batched-diff-triangle-rasterizer/batched_forward_kernels.cuh`, preprocessing and `batchRenderKernel`. The prototype applies 3D face expansion by overlap and sigma, projects expanded vertices, constructs normalized oriented edge distances, computes phi and `phi^sigma`, interpolates vertex color using screen-space affine barycentrics of the expanded triangle, then performs front-to-back alpha blending and the source thresholds. Sorting is by face center depth per tile, as in the reference; it is not per-pixel depth sorting. The source culls on the face-center view depth (`z <= 0.002`); it does not perform near-plane polygon clipping. Other source face-culling thresholds are included.

**Sampling and limits:** the default WebGPU renderer uses adaptive pixel integration: one center sample, then 4 and (where needed) 16 independently composited subpixel samples. It uses local RGB/alpha variation and a small-triangle guard to identify pixels needing more samples. This is inspired by CUDA `balanced`, but does not implement its 20%/8% per-stage queue caps, and is not guaranteed to match CUDA output. Use `sampling: 'single'` in `render()` to restore the original center-only path. Adaptive rendering requires more CPU/GPU work and can make camera interaction slower, especially on integrated or older GPUs.

**Parameter scales:** each original checkpoint-derived face value is multiplied independently by the corresponding scale, then clamped to the viewer bounds: sigma `[0.001, 2.0]` and overlap `[0, 0.95]`. The lower sigma bound is a viewer control and extends below the model's default bound. The original values are not modified. Changed scales use the interactive renderer even at the default camera, since the cached animation was rendered with the original parameters.

The renderer outputs RGB and alpha, not normals, depth, feature maps, gradients, view-dependent SH evaluation, model inference, or pose synthesis. Input colors must already be evaluated for the intended view; for view-dependent appearance they must be updated as the camera changes. CPU tile building/sorting and uploading every frame use a correctness-oriented implementation. `bboxMargin` is an optional conservative tile-list margin, defaulting to 5 pixels.

WebGPU needs a secure context (HTTPS or localhost), a compatible browser and adapter. The caller should handle creation failure and provide a fallback. Render size and storage bindings are limited by the selected device. The renderer allocates new buffers for each frame and `readPixels` stalls to copy GPU results to CPU; do not call readback in the normal animation loop.

`python renderer/fixture_reference.py` independently computes the synthetic fixture center pixel, with the inputs rounded to FP32 and the intermediate calculation in Python double precision. One earlier center-only Chrome/WebGPU run returned `[0.43073827, 0.43044275, 0.57550883, 0.90868598]`; its largest difference from the Python result is approximately `8.6e-8`. This single synthetic pixel is a smoke check, not a CUDA parity test.

The demo’s **Sharp** control supplies constant sigma `0.001` and overlap `0` for every face. Turning it off restores the checkpoint-derived per-face arrays. Sharp playback uses the interactive renderer instead of cached preview frames.
