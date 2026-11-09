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
});
// canvas.width and canvas.height determine the output pixel dimensions.
// New vertices and matrices may be supplied on each render call.
const rgba = await renderer.readPixels(); // Float32Array, row-major, RGB + (1 - T)
renderer.destroy();
```

`fixture.html` is a synthetic, two-triangle smoke fixture. Serve the parent folder on localhost (for example, `python -m http.server`) and open `renderer/fixture.html` in a WebGPU-capable browser. The fixture is deliberately not a representation of the real avatar.

The CUDA reference is `release/MSAvatar_Release/submodules/batched-diff-triangle-rasterizer/batched_forward_kernels.cuh`, preprocessing and `batchRenderKernel`. The prototype applies 3D face expansion by overlap and sigma, projects expanded vertices, constructs normalized oriented edge distances, computes phi and `phi^sigma`, interpolates vertex color using screen-space affine barycentrics of the expanded triangle, then performs front-to-back alpha blending and the source thresholds. Sorting is by face center depth per tile, as in the reference; it is not per-pixel depth sorting. The source culls on the face-center view depth (`z <= 0.002`); it does not perform near-plane polygon clipping. Other source face-culling thresholds are included.

**Coverage and limits:** adaptive 1/4/16 sampling is OFF; one pixel-center sample is used, corresponding to the source OFF sampling path in concept. The release default `balanced` works in 4× coordinates and adaptively takes 4 or 16 samples, so this prototype is not the default balanced renderer. On three X-Humans poses and one rotated pose, 400×600 RGB differed from the matching CUDA OFF output by at most 2 8-bit levels; this is not a general equivalence guarantee. Its CPU `Number` preprocessing, GPU math, equal-depth tie handling and degenerate cases may differ from CUDA FP32; compare on identical geometry/cameras before making parity claims. It only outputs RGB and alpha, not normals, depth, feature maps, gradients, view-dependent SH evaluation, model inference, or pose synthesis. Input colors must already be evaluated for the intended view; for view-dependent appearance they must be updated as the camera changes. Frame data, learned face parameters, camera conversion, pose playback and an avatar asset are not included. CPU tile building/sorting and uploading every frame are an initial correctness-oriented implementation, with one local 400×600 fixed-pose browser benchmark (3 warmups, 10 synchronized calls averaging 108.3 ms; this includes CPU preprocessing, uploads and GPU rendering but excludes data loading and model inference). `bboxMargin` is an optional override; its default is 5, the CUDA OFF-mode bbox margin.

WebGPU needs a secure context (HTTPS or localhost), a compatible browser and adapter. The caller should handle creation failure and provide a fallback. Render size and storage bindings are limited by the selected device. The renderer allocates new buffers for each frame and `readPixels` stalls to copy GPU results to CPU; do not call readback in the normal animation loop.

`python renderer/fixture_reference.py` independently computes the synthetic fixture center pixel, with the inputs rounded to FP32 and the intermediate calculation in Python double precision. One Chrome/WebGPU run returned `[0.43073827, 0.43044275, 0.57550883, 0.90868598]`; its largest difference from the Python result is approximately `8.6e-8`. This single synthetic pixel is a smoke check, not a CUDA parity test.
