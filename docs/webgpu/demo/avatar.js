import { TriangleSplatRenderer } from '../renderer/triangle-splat.js';
import { initCarousel } from './carousel.js';

const canvas = document.getElementById('avatar');
const viewport = document.querySelector('.viewport');
const zoomIn = document.getElementById('zoom-in');
const zoomOut = document.getElementById('zoom-out');
const zoomLevel = document.getElementById('zoom-level');
const cachedCanvas = document.getElementById('cached-frame');
const cachedContext = cachedCanvas.getContext('2d');
const message = document.getElementById('message');
const play = document.getElementById('play');
const timeline = document.getElementById('timeline');
const reset = document.getElementById('reset');
let dataRoot = new URL('../data/', import.meta.url);
const state = { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0, index: 0, playing: true, dirty: true, busy: false, lastFrame: 0, pointer: null, failed: false };
let metadata, renderer, faces, weights, sigmas, overlaps, center, rightAxis, upAxis;
const frames = new Map();
let preview = null;
let nextFrameAt = 0;
let requestedIndex = 0, revision = 0, automaticRequest = false;
let cancelWait = null;
const cancelled = Symbol('cancelled');

// state.index is always the frame currently presented. requestedIndex may be
// ahead while an image or mesh is loading, but the timeline must not be.
function invalidate() {
  revision++; state.dirty = true;
  // Obsolete requests may finish in the background, but must not block a new
  // seek or camera update from displaying the user's latest selection.
  if (cancelWait) { cancelWait(); cancelWait = null; }
}
function waitForCurrent(promise) {
  return Promise.race([promise, new Promise(resolve => { cancelWait = () => resolve(cancelled); })]);
}
function resetClock() { nextFrameAt = performance.now() + 1000 / frameRate(); }

function frameCount() { return preview ? preview.count : metadata.frames.length; }
function frameRate() { return preview ? preview.fps : metadata.frame_rate; }
function poseIndex(frameIndex = state.index) {
  if (!preview) return frameIndex;
  let best = 0;
  for (let i = 1; i < metadata.frames.length; i++) {
    if (Math.abs(metadata.frames[i].original_test_index - frameIndex) < Math.abs(metadata.frames[best].original_test_index - frameIndex)) best = i;
  }
  return best;
}
function fixedView() { return Math.abs(state.yaw) < 1e-8 && Math.abs(state.pitch) < 1e-8 && Math.abs(state.zoom - 1) < 1e-8 && Math.abs(state.panX) < 1e-8 && Math.abs(state.panY) < 1e-8; }

// Decode incrementally so the first image can appear without waiting for the
// complete sequence. A missing preview falls back to the original mesh viewer.
function loadPreviewImage(info, index) {
  if (info.requests[index]) return info.requests[index];
  const promise = new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => { info.images[index] = image; resolve(image); };
    image.onerror = () => reject(new Error('Could not load the pre-rendered animation.'));
    image.src = new URL(`${info.path}${String(index).padStart(6, '0')}.${info.format || 'jpg'}`, dataRoot).href;
  });
  info.requests[index] = promise;
  return promise;
}
async function enablePreview(info) {
  if (!info || !Number.isInteger(info.count) || info.count < 1 || !(info.fps > 0) || typeof info.path !== 'string') return;
  const candidate = { ...info, images: new Array(info.count), requests: new Array(info.count) };
  try { await loadPreviewImage(candidate, 0); } catch { return; }
  // A late preview uses test-frame indices rather than sparse mesh indices.
  state.index = Math.min(candidate.count - 1, metadata.frames[state.index].original_test_index);
  requestedIndex = Math.min(candidate.count - 1, metadata.frames[requestedIndex].original_test_index);
  preview = candidate;
  cachedCanvas.width = canvas.width; cachedCanvas.height = canvas.height;
  timeline.max = String(frameCount() - 1); timeline.value = String(state.index);
  resetClock(); invalidate();
  let next = 1;
  const worker = async () => {
    while (preview === candidate && next < candidate.count) {
      const index = next++;
      try { await loadPreviewImage(candidate, index); }
      catch (error) {
        if (preview !== candidate) return;
        const displayedPose = poseIndex(state.index), requestedPose = poseIndex(requestedIndex);
        preview = null; state.index = displayedPose; requestedIndex = requestedPose;
        cachedCanvas.hidden = true;
        timeline.max = String(metadata.frames.length - 1); timeline.value = String(state.index);
        resetClock(); invalidate();
        console.warn('Pre-rendered animation unavailable; using the mesh viewer.', error);
        return;
      }
    }
  };
  await Promise.all(Array.from({length: Math.min(6, candidate.count - 1)}, worker));
}

async function readFile(path, Type) {
  const response = await fetch(new URL(path, dataRoot));
  if (!response.ok) throw new Error(`Could not load demo data (${response.status}).`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength % Type.BYTES_PER_ELEMENT !== 0) throw new Error('Demo data has an invalid length.');
  return new Type(buffer);
}

function loadFrame(index) {
  if (!frames.has(index)) {
    const entry = metadata.frames[index];
    const promise = Promise.all([readFile(entry.positions, Float32Array), readFile(entry.colors, Float32Array)])
      .then(([vertices, colors]) => {
        if (vertices.length !== metadata.vertex_count * 3 || colors.length !== vertices.length) throw new Error('Demo frame has an invalid vertex count.');
        return { vertices, colors };
      });
    frames.set(index, promise);
  }
  return frames.get(index);
}

function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (!(n > 0)) throw new Error('Invalid demo camera.');
  return v.map(x => x / n);
}

function rotateVertices(src) {
  if (Math.abs(state.yaw) < 1e-8 && Math.abs(state.pitch) < 1e-8) return src;
  const out = new Float32Array(src.length);
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const [ux, uy, uz] = upAxis, [rx, ry, rz] = rightAxis;
  for (let i = 0; i < src.length; i += 3) {
    const x = src[i]-center[0], y = src[i+1]-center[1], z = src[i+2]-center[2];
    const u = (ux*x + uy*y + uz*z) * (1-cy);
    const px = x*cy + (uy*z-uz*y)*sy + ux*u;
    const py = y*cy + (uz*x-ux*z)*sy + uy*u;
    const pz = z*cy + (ux*y-uy*x)*sy + uz*u;
    const r = (rx*px + ry*py + rz*pz) * (1-cp);
    out[i] = px*cp + (ry*pz-rz*py)*sp + rx*r + center[0];
    out[i+1] = py*cp + (rz*px-rx*pz)*sp + ry*r + center[1];
    out[i+2] = pz*cp + (rx*py-ry*px)*sp + rz*r + center[2];
  }
  return out;
}

// The stored projection is a complete world-to-clip matrix, so view and
// projection must receive the same world-space camera translation.
function cameraMatrices() {
  const view = metadata.camera.view_matrix;
  const projection = metadata.camera.projection_matrix;
  const forward = normalize([view[2], view[6], view[10]]);
  const depth = view[2] * center[0] + view[6] * center[1] + view[10] * center[2] + view[14];
  if (!(depth > 0)) throw new Error('Invalid demo camera depth.');
  const distance = depth * (1 - 1 / state.zoom);
  const translation = forward.map((v, i) => distance * v + state.panX * rightAxis[i] + state.panY * upAxis[i]);
  function shifted(matrix) {
    const result = matrix.slice();
    for (let row = 0; row < 4; row++) {
      result[12 + row] -= (matrix[row] * translation[0] + matrix[4 + row] * translation[1] + matrix[8 + row] * translation[2]);
    }
    return result;
  }
  return { viewMatrix: shifted(view), projectionMatrix: shifted(projection) };
}

// Intersect a screen ray with the plane through the subject center parallel to
// the camera image plane. Solving with the complete projection also respects
// principal-point offsets and non-square viewport scaling.
function pointOnCenterPlane(clientX, clientY) {
  const rect = viewport.getBoundingClientRect();
  const x = 2 * (clientX - rect.left) / rect.width - 1;
  const y = 2 * (clientY - rect.top) / rect.height - 1;
  const m = cameraMatrices().projectionMatrix;
  function equation(row, ndc, axis) {
    return (m[row] - ndc * m[3]) * axis[0] + (m[4 + row] - ndc * m[7]) * axis[1] + (m[8 + row] - ndc * m[11]) * axis[2];
  }
  function constant(row, ndc) {
    return equation(row, ndc, center) + m[12 + row] - ndc * m[15];
  }
  const a = equation(0, x, rightAxis), b = equation(0, x, upAxis);
  const c = equation(1, y, rightAxis), d = equation(1, y, upAxis);
  const determinant = a * d - b * c;
  if (Math.abs(determinant) < 1e-12) throw new Error('Invalid demo camera projection.');
  const u = -constant(0, x), v = -constant(1, y);
  const right = (u * d - b * v) / determinant, up = (a * v - u * c) / determinant;
  return center.map((value, i) => value + right * rightAxis[i] + up * upAxis[i]);
}
function moveCameraBetween(before, after) {
  const delta = before.map((value, i) => value - after[i]);
  state.panX += delta.reduce((sum, value, i) => sum + value * rightAxis[i], 0);
  state.panY += delta.reduce((sum, value, i) => sum + value * upAxis[i], 0);
}
function panBy(fromX, fromY, toX, toY) {
  if (fromX === toX && fromY === toY) return;
  moveCameraBetween(pointOnCenterPlane(fromX, fromY), pointOnCenterPlane(toX, toY));
  invalidate();
}

async function draw() {
  state.busy = true;
  state.dirty = false;
  const version = revision, target = requestedIndex, currentPreview = preview;
  let loadingTimer;
  const stillCurrent = () => version === revision && currentPreview === preview;
  // Keep the last displayed frame visible when a first-time fetch is slow.
  const showLoading = () => {
    loadingTimer = setTimeout(() => {
      if (!stillCurrent()) return;
      message.style.cssText = 'inset:auto 0 0;padding:6px;background:rgba(255,255,255,.85)';
      message.textContent = 'Loading frame…'; message.hidden = false;
    }, 250);
  };
  try {
    if (currentPreview && fixedView()) {
      showLoading();
      const image = await waitForCurrent(loadPreviewImage(currentPreview, target));
      if (image === cancelled || !stillCurrent()) return;
      cachedContext.drawImage(image, 0, 0);
      cachedCanvas.hidden = false;
      if (window.__avatarDebug) window.__avatarDebug.lastPreviewFrame = target;
    } else {
      const index = poseIndex(target);
      showLoading();
      const frame = await waitForCurrent(loadFrame(index));
      if (frame === cancelled || !stillCurrent()) return;
      await renderer.render({
        vertices: rotateVertices(frame.vertices), faces, colors: frame.colors, weights, sigmas, overlaps,
        ...cameraMatrices(), background: [1, 1, 1]
      });
      if (!stillCurrent()) return;
      cachedCanvas.hidden = true;
      if (window.__avatarDebug) window.__avatarDebug.lastRenderedFrame = index;
    }
    state.index = target;
    timeline.value = String(target);
    automaticRequest = false;
    // A slow frame consumes elapsed time; never catch up by skipping frames.
    nextFrameAt = Math.max(nextFrameAt + 1000 / frameRate(), performance.now());
    message.hidden = true;
  } catch (error) {
    // Preview worker switches to the mesh viewer if an image fails.
    if (currentPreview && currentPreview === preview) {
      const displayedPose = poseIndex(state.index), requestedPose = poseIndex(requestedIndex);
      preview = null; state.index = displayedPose; requestedIndex = requestedPose;
      timeline.max = String(metadata.frames.length - 1); timeline.value = String(state.index);
      resetClock(); invalidate();
      console.warn('Pre-rendered animation unavailable; using the mesh viewer.', error);
    } else if (stillCurrent()) {
      state.failed = true;
      message.style.cssText = '';
      message.hidden = false;
      message.textContent = error.message || 'The interactive demo could not be loaded.';
      play.disabled = timeline.disabled = reset.disabled = zoomIn.disabled = zoomOut.disabled = true;
      console.error(error);
    }
  } finally {
    clearTimeout(loadingTimer);
    cancelWait = null;
    if (!state.failed && !message.hidden && !stillCurrent()) message.hidden = true;
    state.busy = false;
  }
}

function tick(now) {
  if (state.failed) return;
  if (state.playing && frameCount() > 1 && !state.busy && !state.dirty && now >= nextFrameAt) {
    requestedIndex = (state.index + 1) % frameCount();
    automaticRequest = true;
    // Adjacent 24 fps timeline frames can refer to the same sparse 3D pose.
    if (!(preview && fixedView()) && poseIndex(requestedIndex) === poseIndex(state.index)) {
      state.index = requestedIndex; timeline.value = String(state.index); automaticRequest = false;
      nextFrameAt = Math.max(nextFrameAt + 1000 / frameRate(), now);
    } else invalidate();
  }
  if (state.dirty && !state.busy) void draw();
  requestAnimationFrame(tick);
}

function rotateBy(dx, dy) {
  if (!dx && !dy) return;
  state.yaw += dx * .01;
  state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + dy * .01));
  invalidate();
}
viewport.addEventListener('pointerdown', event => {
  state.pointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
  viewport.setPointerCapture(event.pointerId);
});
viewport.addEventListener('pointermove', event => {
  if (!state.pointer || state.pointer.id !== event.pointerId) return;
  if (event.shiftKey) panBy(state.pointer.x, state.pointer.y, event.clientX, event.clientY);
  else rotateBy(state.pointer.x - event.clientX, event.clientY - state.pointer.y);
  state.pointer.x = event.clientX; state.pointer.y = event.clientY;
});
for (const eventName of ['pointerup', 'pointercancel', 'lostpointercapture']) viewport.addEventListener(eventName, () => { state.pointer = null; });
canvas.addEventListener('keydown', event => {
  const delta = { ArrowLeft: [-10, 0], ArrowRight: [10, 0], ArrowUp: [0, -10], ArrowDown: [0, 10] }[event.key];
  if (delta) { event.preventDefault(); rotateBy(...delta); }
});
play.addEventListener('click', () => {
  state.playing = !state.playing;
  play.textContent = state.playing ? 'Pause' : 'Play';
  if (!state.playing && automaticRequest) {
    requestedIndex = state.index; automaticRequest = false; invalidate();
  }
  resetClock();
});
timeline.addEventListener('input', () => {
  requestedIndex = Number(timeline.value); automaticRequest = false;
  timeline.value = String(state.index);
  resetClock(); invalidate();
});
function setZoom(value, clientX, clientY) {
  const next = Math.max(0.75, Math.min(3, value));
  if (metadata && center && next !== state.zoom) {
    const rect = viewport.getBoundingClientRect();
    const x = clientX ?? rect.left + rect.width / 2;
    const y = clientY ?? rect.top + rect.height / 2;
    const before = pointOnCenterPlane(x, y);
    state.zoom = next;
    moveCameraBetween(before, pointOnCenterPlane(x, y));
  } else state.zoom = next;
  invalidate();
  zoomLevel.value = `${Math.round(state.zoom * 100)}%`;
  zoomOut.disabled = state.zoom <= 0.75;
  zoomIn.disabled = state.zoom >= 3;
}
zoomIn.addEventListener('click', () => setZoom(state.zoom * 1.2));
zoomOut.addEventListener('click', () => setZoom(state.zoom / 1.2));
viewport.addEventListener('wheel', event => {
  if (state.failed || !renderer) return;
  event.preventDefault();
  setZoom(state.zoom * Math.exp(-event.deltaY * 0.001), event.clientX, event.clientY);
}, { passive: false });
reset.addEventListener('click', () => { state.yaw = state.pitch = 0; setZoom(1); state.panX = state.panY = 0; invalidate(); });

try {
  const subjectResponse = await fetch(new URL('../data/subjects.json', import.meta.url));
  if (!subjectResponse.ok) throw new Error('Could not load the list of subjects.');
  const subjects = await subjectResponse.json();
  if (!Array.isArray(subjects) || !subjects.length || subjects.some(subject => !/^\d{5}$/.test(subject.id))) throw new Error('The subject list is invalid.');
  const requested = new URL(location.href).searchParams.get('subject');
  const selected = subjects.find(subject => subject.id === requested) || subjects[0];
  initCarousel(subjects, selected, () => ({ index: state.index, count: metadata ? frameCount() : 0 }));
  if (selected.id !== '00016') dataRoot = new URL(`../data/subjects/${selected.id}/`, import.meta.url);
  if (!navigator.gpu) throw new Error('This interactive demo needs a browser with WebGPU support.');
  const response = await fetch(new URL('metadata.json', dataRoot));
  if (!response.ok) throw new Error(`Could not load demo data (${response.status}).`);
  metadata = await response.json();
  if (!metadata.vertex_count || !metadata.face_count || !Array.isArray(metadata.frames) || !metadata.frames.length || !metadata.camera || !metadata.frame_rate) throw new Error('Demo metadata is incomplete.');
  if (metadata.camera.view_matrix.length !== 16 || metadata.camera.projection_matrix.length !== 16) throw new Error('Demo camera is invalid.');
  canvas.width = 400;
  canvas.height = Math.round(canvas.width * metadata.camera.height / metadata.camera.width);
  [faces, weights, sigmas, overlaps] = await Promise.all([
    readFile(metadata.faces, Uint32Array), readFile(metadata.weights, Float32Array),
    readFile(metadata.sigmas, Float32Array), readFile(metadata.overlaps, Float32Array)
  ]);
  if (faces.length !== metadata.face_count * 3 || weights.length !== metadata.vertex_count || sigmas.length !== metadata.face_count || overlaps.length !== metadata.face_count) throw new Error('Demo mesh data has an invalid length.');
  const first = await loadFrame(0);
  const bounds = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (let i = 0; i < first.vertices.length; i += 3) for (let k = 0; k < 3; k++) { bounds[k] = Math.min(bounds[k], first.vertices[i+k]); bounds[k+3] = Math.max(bounds[k+3], first.vertices[i+k]); }
  center = [0, 1, 2].map(k => (bounds[k] + bounds[k+3]) / 2);
  const m = metadata.camera.view_matrix;
  rightAxis = normalize([m[0], m[4], m[8]]); upAxis = normalize([m[1], m[5], m[9]]);
  renderer = await TriangleSplatRenderer.create(canvas);
  timeline.max = String(metadata.frames.length - 1);
  timeline.disabled = metadata.frames.length < 2;
  play.disabled = metadata.frames.length < 2;
  reset.disabled = false;
  setZoom(1);
  state.playing = metadata.frames.length > 1;
  if (!state.playing) play.textContent = 'Play';
  window.__avatarDebug = { state, metadata, renderer, cameraMatrices, get center() { return center; }, lastRenderedFrame: -1, lastPreviewFrame: -1, get preview() { return preview; } };
  void enablePreview(metadata.preview);
  await draw();
  if (!state.failed) { resetClock(); requestAnimationFrame(tick); }
} catch (error) {
  state.failed = true;
  message.textContent = error.message || 'The interactive demo could not be loaded.';
  console.error(error);
}
