import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const statusEl = document.getElementById('status');
function setStatus(s) { statusEl.textContent = s; }

// ---------- three.js scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1d21);
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(1.6, 1.2, 1.8);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.2));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight.position.set(2, 3, 4);
scene.add(dirLight);
scene.add(new THREE.GridHelper(2, 20, 0x333a44, 0x272c33));

const geometry = new THREE.BufferGeometry();
const material = new THREE.MeshStandardMaterial({
  color: 0x6fa8dc, metalness: 0.15, roughness: 0.55, side: THREE.DoubleSide,
});
const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);
const wire = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ wireframe: true, color: 0x9ecbff, transparent: true, opacity: 0.05 }));
scene.add(wire);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
(function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); })();

// ---------- state ----------
let info = null;
let z = [];
let zRange = { min: [], max: [] };
let ws = null;
let pending = false;      // a request is in flight
let queued = false;       // sliders moved while in flight
let lastMeta = null;

async function loadInfo() {
  info = await (await fetch('/api/info')).json();
  z = info.latent_mean.slice();
  for (let i = 0; i < info.latent_size; i++) {
    const s = Math.max(info.latent_std[i], 1e-3);
    zRange.min.push(info.latent_mean[i] - 3 * s);
    zRange.max.push(info.latent_mean[i] + 3 * s);
  }
  frameScene(info.bounds);
  buildUI();
  connect();
}

function frameScene(bounds) {
  const b = bounds || [-1, -1, -1, 1, 1, 1];
  const center = [(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2];
  const radius = 0.5 * Math.hypot(b[3] - b[0], b[4] - b[1], b[5] - b[2]);
  camera.far = Math.max(100, radius * 20);
  camera.updateProjectionMatrix();
  camera.position.set(center[0] + 0.8 * radius, center[1] + 0.6 * radius, center[2] + 0.9 * radius);
  controls.target.set(...center);
  controls.update();
}

// ---------- websocket ----------
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { setStatus('connected'); requestMesh(); };
  ws.onclose = () => { setStatus('disconnected, retrying...'); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      const m = JSON.parse(ev.data);
      if (m.error) { setStatus('error: ' + m.error); pending = false; return; }
      lastMeta = m;
      return;
    }
    applyMesh(new DataView(ev.data));
    pending = false;
    if (queued) { queued = false; requestMesh(); }
  };
}

function requestMesh(resolutionOverride) {
  if (!ws || ws.readyState !== 1) return;
  if (pending) { queued = true; return; }
  pending = true;
  const res = resolutionOverride ?? currentResolution();
  ws.send(JSON.stringify({ z, resolution: res }));
}

function currentResolution() {
  return parseInt(document.getElementById('resolution').value, 10);
}

function applyMesh(view) {
  const nv = view.getUint32(0, true);
  const nf = view.getUint32(4, true);
  let off = 8;
  const verts = new Float32Array(ev_slice(view, off, nv * 3 * 4)); off += nv * 3 * 4;
  const faces = new Int32Array(ev_slice(view, off, nf * 3 * 4)); off += nf * 3 * 4;
  const normals = new Float32Array(ev_slice(view, off, nv * 3 * 4));
  geometry.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(faces.buffer.slice(0)), 1));
  geometry.computeBoundingSphere();
  setStatus(`verts ${nv} | faces ${nf} | ${lastMeta ? lastMeta.elapsed_ms + ' ms' : ''}`);
}
function ev_slice(view, byteOffset, byteLen) {
  return view.buffer.slice(view.byteOffset + byteOffset, view.byteOffset + byteOffset + byteLen);
}

// ---------- UI ----------
function buildUI() {
  const preset = document.getElementById('preset');
  const interpA = document.getElementById('interpA');
  const interpB = document.getElementById('interpB');
  const names = info.shape_names;
  preset.innerHTML = '<option value="-1">(自定义 z)</option>';
  names.forEach((n, i) => {
    preset.insertAdjacentHTML('beforeend', `<option value="${i}">${n}</option>`);
    interpA.insertAdjacentHTML('beforeend', `<option value="${i}">${n}</option>`);
    interpB.insertAdjacentHTML('beforeend', `<option value="${i}">${n}</option>`);
  });
  if (names.length > 1) interpB.value = '1';

  preset.onchange = () => {
    const i = parseInt(preset.value, 10);
    if (i >= 0) { z = info.latents[i].slice(); syncSliders(); requestMesh(); }
  };
  document.getElementById('randomBtn').onclick = () => {
    for (let i = 0; i < z.length; i++) {
      const g = (Math.random() + Math.random() + Math.random() - 1.5) * 2; // approx normal
      z[i] = info.latent_mean[i] + g * info.latent_std[i];
    }
    preset.value = '-1';
    syncSliders(); requestMesh();
  };
  document.getElementById('resetBtn').onclick = () => {
    z = info.latent_mean.slice(); preset.value = '-1'; syncSliders(); requestMesh();
  };
  const doInterp = () => {
    const a = info.latents[parseInt(interpA.value, 10)];
    const b = info.latents[parseInt(interpB.value, 10)];
    if (!a || !b) return;
    const t = parseFloat(document.getElementById('interpAlpha').value);
    document.getElementById('interpAlphaVal').textContent = t.toFixed(2);
    z = a.map((v, i) => v * (1 - t) + b[i] * t);
    preset.value = '-1';
    syncSliders(); requestMesh();
  };
  interpA.onchange = doInterp; interpB.onchange = doInterp;
  document.getElementById('interpAlpha').oninput = doInterp;

  document.getElementById('resolution').onchange = () => requestMesh();

  buildSliders();
}

const sliderInputs = [];
function buildSliders() {
  const container = document.getElementById('latentSliders');
  const GROUP = 8;
  for (let g = 0; g < info.latent_size; g += GROUP) {
    const det = document.createElement('details');
    const end = Math.min(g + GROUP, info.latent_size);
    det.innerHTML = `<summary>z[${g}..${end - 1}]</summary>`;
    for (let i = g; i < end; i++) {
      const row = document.createElement('div');
      row.className = 'slider-row';
      const step = (zRange.max[i] - zRange.min[i]) / 200;
      row.innerHTML = `<label>z${i}</label>
        <input type="range" min="${zRange.min[i]}" max="${zRange.max[i]}" step="${step}" value="${z[i]}">
        <span class="val">${z[i].toFixed(3)}</span>`;
      const input = row.querySelector('input');
      const val = row.querySelector('.val');
      input.oninput = () => {
        z[i] = parseFloat(input.value);
        val.textContent = z[i].toFixed(3);
        document.getElementById('preset').value = '-1';
        requestMesh();
      };
      // refine at high resolution when the drag ends
      input.onchange = () => requestMesh();
      sliderInputs.push({ input, val });
      det.appendChild(row);
    }
    container.appendChild(det);
  }
}
function syncSliders() {
  sliderInputs.forEach(({ input, val }, i) => {
    input.value = z[i];
    val.textContent = z[i].toFixed(3);
  });
}

loadInfo();
