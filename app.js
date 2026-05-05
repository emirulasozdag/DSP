/* ═══════════════════ NAVIGATION ═══════════════════ */
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

function nav(page) {
  $$('.page').forEach(p => p.classList.remove('active'));
  $$('.sb-item').forEach(b => b.classList.remove('active'));
  const el = $(`#page-${page}`);
  if (el) el.classList.add('active');
  const btn = $(`.sb-item[data-page="${page}"]`);
  if (btn) btn.classList.add('active');
  closeMobile();
  // Re-sync photo panels when navigating to the aging page so the SVG
  // wireframe placeholders are hidden whenever a real photo is available.
  if (page === 'aging') syncAgingPhotoPanels();
}

$$('.sb-item[data-page]').forEach(b => {
  b.addEventListener('click', () => nav(b.dataset.page));
});

/* ═══════════════════ MOBILE MENU ═══════════════════ */
const sidebar = $('#sidebar'), overlay = $('#overlay'), menuBtn = $('#menuBtn');
function closeMobile() { sidebar.classList.remove('open'); overlay.classList.remove('show'); }
if (menuBtn) menuBtn.addEventListener('click', () => { sidebar.classList.toggle('open'); overlay.classList.toggle('show'); });
if (overlay) overlay.addEventListener('click', closeMobile);

/* ═══════════════════ TOAST ═══════════════════ */
function toast(msg, type = 'success') {
  const container = $('#toastContainer');
  if (!container) return;
  const icons = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' };
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `<div class="toast-icon">${icons[type] || 'ℹ'}</div><span>${msg}</span>`;
  container.appendChild(el);
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add('show')));
  setTimeout(() => { el.classList.add('hide'); setTimeout(() => el.remove(), 280); }, 3200);
}

/* ═══════════════════ PIPELINE STRIP ═══════════════════ */
const psOrder = ['upload', 'preprocess', 'landmarks', 'expression', 'aging', 'analysis'];
function updatePipelineStrip(page) {
  const idx = psOrder.indexOf(page);
  if (idx === -1) return;
  psOrder.forEach((step, i) => {
    const el = $(`.ps-step[data-step="${step}"]`);
    if (!el) return;
    el.classList.remove('ps-active', 'ps-done');
    const node = el.querySelector('.ps-node');
    if (i < idx) {
      el.classList.add('ps-done');
      if (node) node.textContent = '✓';
    } else {
      if (node) node.textContent = i + 1;
      if (i === idx) el.classList.add('ps-active');
    }
    const line = $(`#pline-${i + 1}`);
    if (line) line.classList.toggle('ps-done-line', i < idx);
  });
}

/* ═══════════════════ FILE UPLOAD ═══════════════════ */
const dropzone = $('#dropzone'), fileInput = $('#fileInput'), browseBtn = $('#browseBtn');
const uploadCard = $('#uploadCard'), previewCard = $('#previewCard');
const appState = {
  imageDataUrl: '',
  fileName: 'portrait_01.jpg',
  format: 'JPEG',
  sizeText: '2.4 MB',
  dimensions: '1024 × 1024',
};

const backendState = {
  landmarksLoaded: false,
  analysisLoaded: false,
  landmarks: [],
  lastLmData: null,   // cached last successful detection response
  lastExprDataUrl: '',
  lastAgingDataUrl: '',
  analysisData: null,
  analysisSpectra: null,
  spectrumMode: 'mag',
  expressionMode: 'manual',
};

function mapLandmarkMethodLabel(method, conf) {
  const m = String(method || '').toLowerCase();
  if (m === 'mediapipe') return 'MediaPipe';
  if (m === 'fallback-haar') return 'Haar CV';
  if (m === 'fallback-skin') return 'Skin Fallback';
  if (m === 'fallback-geo') return 'Geometric Fallback';
  const c = Number(conf || 0);
  return c >= 99 ? 'MediaPipe' : c >= 80 ? 'Haar CV' : 'Fallback';
}

function dataUrlToBlob(dataUrl) {
  const parts = String(dataUrl || '').split(',');
  if (parts.length < 2) return null;
  const mimeMatch = parts[0].match(/data:(.*?);base64/);
  const mime = mimeMatch ? mimeMatch[1] : 'image/jpeg';
  const binary = atob(parts[1]);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

function loadImageFromDataUrl(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not load image for transformation.'));
    img.src = dataUrl;
  });
}

async function getCurrentImageBlob() {
  const directFile = fileInput?.files?.[0];
  if (directFile) return directFile;
  if (!appState.imageDataUrl) return null;
  return dataUrlToBlob(appState.imageDataUrl);
}

function sampleLandmarks(points, targetCount = 68) {
  if (!Array.isArray(points) || points.length === 0) return [];
  if (points.length === targetCount) return points;
  const sampled = [];
  for (let i = 0; i < targetCount; i += 1) {
    const idx = Math.round((i * (points.length - 1)) / (targetCount - 1));
    sampled.push(points[idx]);
  }
  return sampled;
}

function isPointInPolygon(x, y, polygon) {
  if (!Array.isArray(polygon) || polygon.length < 3) return false;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects = ((yi > y) !== (yj > y))
      && (x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-12) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

function convexHull(points) {
  if (!Array.isArray(points) || points.length < 3) return points || [];
  const pts = points
    .map(p => ({ x: Number(p.x), y: Number(p.y) }))
    .filter(p => Number.isFinite(p.x) && Number.isFinite(p.y))
    .sort((a, b) => (a.x - b.x) || (a.y - b.y));

  if (pts.length < 3) return pts;

  const cross = (o, a, b) => ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper = [];
  for (let i = pts.length - 1; i >= 0; i -= 1) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function delaunayTriangles(nodes) {
  if (!Array.isArray(nodes) || nodes.length < 3) return [];

  const pts = nodes.map((p, i) => ({ x: Number(p.x), y: Number(p.y), idx: i }));
  const xs = pts.map(p => p.x);
  const ys = pts.map(p => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);
  const dx = maxX - minX;
  const dy = maxY - minY;
  const dMax = Math.max(dx, dy) || 1;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;

  const allPts = pts.concat([
    { x: midX - 20 * dMax, y: midY - dMax, idx: -1 },
    { x: midX, y: midY + 20 * dMax, idx: -1 },
    { x: midX + 20 * dMax, y: midY - dMax, idx: -1 },
  ]);

  const s0 = allPts.length - 3;
  const s1 = allPts.length - 2;
  const s2 = allPts.length - 1;
  let triangles = [{ a: s0, b: s1, c: s2 }];

  const inCircumcircle = (p, a, b, c) => {
    const d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if (Math.abs(d) < 1e-9) return false;

    const a2 = (a.x * a.x) + (a.y * a.y);
    const b2 = (b.x * b.x) + (b.y * b.y);
    const c2 = (c.x * c.x) + (c.y * c.y);

    const ux = (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d;
    const uy = (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d;

    const r2 = ((ux - a.x) ** 2) + ((uy - a.y) ** 2);
    const p2 = ((ux - p.x) ** 2) + ((uy - p.y) ** 2);
    return p2 <= (r2 + 1e-6);
  };

  for (let pIdx = 0; pIdx < pts.length; pIdx += 1) {
    const p = allPts[pIdx];
    const bad = [];

    triangles.forEach((t, i) => {
      if (inCircumcircle(p, allPts[t.a], allPts[t.b], allPts[t.c])) bad.push(i);
    });

    const edgeCount = new Map();
    const addEdge = (u, v) => {
      const a = Math.min(u, v);
      const b = Math.max(u, v);
      const key = `${a}:${b}`;
      edgeCount.set(key, (edgeCount.get(key) || 0) + 1);
    };

    bad.forEach(i => {
      const t = triangles[i];
      addEdge(t.a, t.b);
      addEdge(t.b, t.c);
      addEdge(t.c, t.a);
    });

    triangles = triangles.filter((_, i) => !bad.includes(i));

    edgeCount.forEach((count, key) => {
      if (count !== 1) return;
      const [a, b] = key.split(':').map(Number);
      triangles.push({ a, b, c: pIdx });
    });
  }

  const hull = convexHull(pts);
  const maxEdge = dMax * 0.35;

  const distance = (u, v) => {
    const dxv = u.x - v.x;
    const dyv = u.y - v.y;
    return Math.hypot(dxv, dyv);
  };

  return triangles
    .filter(t => t.a < pts.length && t.b < pts.length && t.c < pts.length)
    .map(t => ({ a: pts[t.a], b: pts[t.b], c: pts[t.c] }))
    .filter(t => {
      const e1 = distance(t.a, t.b);
      const e2 = distance(t.b, t.c);
      const e3 = distance(t.c, t.a);
      if (e1 > maxEdge || e2 > maxEdge || e3 > maxEdge) return false;
      if (hull.length < 3) return true;
      const cx = (t.a.x + t.b.x + t.c.x) / 3;
      const cy = (t.a.y + t.b.y + t.c.y) / 3;
      return isPointInPolygon(cx, cy, hull);
    });
}

function renderLandmarkMesh(nodes) {
  const meshG = $('#meshG');
  if (!meshG) return;

  meshG.innerHTML = '';
  if (!Array.isArray(nodes) || nodes.length < 3) return;

  const triangles = delaunayTriangles(nodes);
  triangles.forEach(tri => {
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', `${tri.a.x.toFixed(1)},${tri.a.y.toFixed(1)} ${tri.b.x.toFixed(1)},${tri.b.y.toFixed(1)} ${tri.c.x.toFixed(1)},${tri.c.y.toFixed(1)}`);
    poly.setAttribute('fill', 'none');
    poly.setAttribute('stroke', 'rgba(139,92,246,0.32)');
    poly.setAttribute('stroke-width', '.7');
    meshG.appendChild(poly);
  });

  const showMesh = Boolean($('#tgMesh')?.checked);
  meshG.style.opacity = showMesh ? '1' : '0';
}

function updateLandmarkLabels(mx, my) {
  const labelNodes = Array.from($$('#labelsG text[data-idx]'));
  labelNodes.forEach(label => {
    const idx = Number(label.getAttribute('data-idx'));
    const x = mx[idx];
    const y = my[idx];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      label.style.display = 'none';
      return;
    }
    label.style.display = '';
    label.setAttribute('x', x.toFixed(1));
    label.setAttribute('y', (y - 7).toFixed(1));
  });
}

function applyLandmarksToSvg(points68, meshPoints = points68) {
  const svg = $('#lmSvg');
  const circles = Array.from($$('#ptsG circle'));
  if (!svg || circles.length === 0) return;

  const lmCanvas = document.querySelector('.lm-canvas');
  const bgImg = document.querySelector('#lmBgImg');
  const refEl = (bgImg && bgImg.style.display !== 'none') ? bgImg : lmCanvas;
  const refRect = refEl?.getBoundingClientRect();
  const displayedImageWidth = Math.max(1, Math.round(refRect?.width || svg.clientWidth || 300));
  const displayedImageHeight = Math.max(1, Math.round(refRect?.height || svg.clientHeight || 290));

  // Keep SVG overlay pixel-space aligned with the displayed image layer.
  svg.setAttribute('width', String(displayedImageWidth));
  svg.setAttribute('height', String(displayedImageHeight));
  svg.setAttribute('viewBox', `0 0 ${displayedImageWidth} ${displayedImageHeight}`);

  const sampled = sampleLandmarks(points68, circles.length);
  backendState.landmarks = sampled;

  // Compute screen-space mapped coordinates from image-space landmarks.
  const mx = [], my = [];
  sampled.forEach((p, i) => {
    const c = circles[i];
    if (!c) return;
    const ix = Number(p.x);
    const iy = Number(p.y);

    const imageWidth = Number(p.imageWidth || displayedImageWidth);
    const imageHeight = Number(p.imageHeight || displayedImageHeight);
    mx[i] = Number.isFinite(ix) ? (ix / imageWidth) * displayedImageWidth : Number(c.getAttribute('cx'));
    my[i] = Number.isFinite(iy) ? (iy / imageHeight) * displayedImageHeight : Number(c.getAttribute('cy'));

    c.setAttribute('cx', mx[i].toFixed(2));
    c.setAttribute('cy', my[i].toFixed(2));
  });

  updateLandmarkLabels(mx, my);

  const meshNodes = [];
  (Array.isArray(meshPoints) ? meshPoints : []).forEach(p => {
    const ix = Number(p.x);
    const iy = Number(p.y);
    const imageWidth = Number(p.imageWidth || displayedImageWidth);
    const imageHeight = Number(p.imageHeight || displayedImageHeight);
    if (!Number.isFinite(ix) || !Number.isFinite(iy)) return;
    const x = (ix / imageWidth) * displayedImageWidth;
    const y = (iy / imageHeight) * displayedImageHeight;
    meshNodes.push({ x, y });
  });
  renderLandmarkMesh(meshNodes);

  // Helper: build SVG point-list string from landmark indices
  const pt = i => (mx[i] !== undefined) ? `${mx[i].toFixed(1)},${my[i].toFixed(1)}` : null;
  const pts = (...idx) => idx.map(pt).filter(Boolean).join(' ');

  // ── Rebuild connection paths ──────────────────────────────────
  // Jaw 0–16
  const jaw = $$('#connG [data-region="jaw"]')[0];
  if (jaw) jaw.setAttribute('points', pts(...Array.from({ length: 17 }, (_, i) => i)));

  // Brows: left 17–21, right 22–26
  const browEls = $$('#connG [data-region="brow"]');
  if (browEls[0]) browEls[0].setAttribute('points', pts(17, 18, 19, 20, 21));
  if (browEls[1]) browEls[1].setAttribute('points', pts(22, 23, 24, 25, 26));

  // Eyes: left 36–41, right 42–47
  const eyeEls = $$('#connG [data-region="eyes"]');
  if (eyeEls[0]) eyeEls[0].setAttribute('points', pts(36, 37, 38, 39, 40, 41));
  if (eyeEls[1]) eyeEls[1].setAttribute('points', pts(42, 43, 44, 45, 46, 47));

  // Nose: bridge polyline 27–30, bottom polyline 31–35, 2 lateral lines
  const noseEls = Array.from($$('#connG [data-region="nose"]'));
  const nosePoly = noseEls.filter(el => el.tagName.toLowerCase() !== 'line');
  const noseLines = noseEls.filter(el => el.tagName.toLowerCase() === 'line');
  if (nosePoly[0]) nosePoly[0].setAttribute('points', pts(27, 28, 29, 30));
  if (nosePoly[1]) nosePoly[1].setAttribute('points', pts(31, 32, 33, 34, 35));
  if (noseLines[0] && mx[31] !== undefined && mx[30] !== undefined) {
    noseLines[0].setAttribute('x1', mx[31].toFixed(1)); noseLines[0].setAttribute('y1', my[31].toFixed(1));
    noseLines[0].setAttribute('x2', mx[30].toFixed(1)); noseLines[0].setAttribute('y2', my[30].toFixed(1));
  }
  if (noseLines[1] && mx[35] !== undefined && mx[30] !== undefined) {
    noseLines[1].setAttribute('x1', mx[35].toFixed(1)); noseLines[1].setAttribute('y1', my[35].toFixed(1));
    noseLines[1].setAttribute('x2', mx[30].toFixed(1)); noseLines[1].setAttribute('y2', my[30].toFixed(1));
  }

  // Mouth: outer 48–59, inner 60–67
  const mouthEls = $$('#connG [data-region="mouth"]');
  if (mouthEls[0]) mouthEls[0].setAttribute('points', pts(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59));
  if (mouthEls[1]) mouthEls[1].setAttribute('points', pts(60, 61, 62, 63, 64, 65, 66, 67));
}

function applyLandmarkStatsUI(data) {
  const n = data.pointCount;
  const conf = Number(data.confidence || 85);
  const ms = Math.round(data.durationMs);
  const backendLabel = mapLandmarkMethodLabel(data.method, conf);

  if ($('#lmPtsCount')) $('#lmPtsCount').textContent = n;
  if ($('#lmLatencyVal')) $('#lmLatencyVal').textContent = ms + 'ms';
  if ($('#lmConfText')) $('#lmConfText').textContent = conf.toFixed(1);
  if ($('#lmConfBadge')) $('#lmConfBadge').textContent = conf.toFixed(1) + '%';
  if ($('#lmStatPts')) $('#lmStatPts').textContent = n;
  if ($('#lmStatConf')) $('#lmStatConf').textContent = conf.toFixed(1) + '%';
  if ($('#lmStatBackend')) $('#lmStatBackend').textContent = backendLabel;
  if ($('#lmStatusText')) $('#lmStatusText').textContent = `Detection complete · ${n}/${n} pts`;

  const circ = 251.3;
  const fillLen = ((circ * conf) / 100).toFixed(1);
  if ($('#lmAccRing')) $('#lmAccRing').setAttribute('stroke-dasharray', `${fillLen} ${circ}`);
}

async function fetchLandmarksFromBackend(force = false) {
  if (!appState.imageDataUrl) { toast('Upload an image first.', 'warning'); return; }

  // Already loaded — just re-paint the stats without hitting the network
  if (backendState.landmarksLoaded && !force) {
    if (backendState.lastLmData) applyLandmarkStatsUI(backendState.lastLmData);
    return;
  }

  // Show "detecting…" state
  if ($('#lmStatusText')) $('#lmStatusText').textContent = 'Detecting…';
  if ($('#btnRunLM')) { $('#btnRunLM').disabled = true; $('#btnRunLM').textContent = '⧗ Detecting…'; }

  try {
    const blob = await getCurrentImageBlob();
    if (!blob) return;
    const fd = new FormData();
    fd.append('image', blob, appState.fileName || 'input.jpg');
    fd.append('includeReduced68', 'true');
    fd.append('landmarkMode', 'both');
    const res = await fetch('/api/landmarks', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || 'Landmark detection failed.');

    const points68Source = (Array.isArray(data.reduced68) && data.reduced68.length)
      ? data.reduced68
      : data.landmarks;
    const meshPointsSource = (Array.isArray(data.rawLandmarks) && data.rawLandmarks.length)
      ? data.rawLandmarks
      : points68Source;

    const points68 = (points68Source || []).map(p => ({
      x: p.x,
      y: p.y,
      imageWidth: data.width,
      imageHeight: data.height,
    }));
    const meshPoints = (meshPointsSource || []).map(p => ({
      x: p.x,
      y: p.y,
      imageWidth: data.width,
      imageHeight: data.height,
    }));

    applyLandmarksToSvg(points68, meshPoints);
    backendState.landmarksLoaded = true;
    backendState.lastLmData = data;  // cache for re-entry

    applyLandmarkStatsUI(data);

    const conf = Number(data.confidence || 85);
    const ms = Math.round(data.durationMs);
    const backendLabel = mapLandmarkMethodLabel(data.method, conf);
    toast(`Landmarks: ${data.pointCount} pts · ${conf}% conf · ${ms} ms (${backendLabel})`, 'success');
  } catch (err) {
    if ($('#lmStatusText')) $('#lmStatusText').textContent = 'Detection failed.';
    toast(`Backend landmarks error: ${err.message}`, 'warning');
  } finally {
    if ($('#btnRunLM')) {
      $('#btnRunLM').disabled = false;
      $('#btnRunLM').innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Detect Landmarks';
    }
  }
}

async function buildTransformedBlobForAnalysis() {
  const sourceDataUrl = backendState.lastAgingDataUrl || backendState.lastExprDataUrl;
  if (!sourceDataUrl) return null;
  return dataUrlToBlob(sourceDataUrl);
}

function renderSpectrumImages(mode = backendState.spectrumMode || 'mag') {
  const spectra = backendState.analysisSpectra;
  if (!spectra || !spectra.original || !spectra.transformed) return;

  backendState.spectrumMode = mode;
  const key = mode === 'phase' ? 'phase' : 'magnitude';
  const boxes = Array.from(document.querySelectorAll('#page-analysis .spec-grid .spec-box'));
  if (boxes.length < 2) return;

  const applyToBox = (box, b64Data) => {
    if (!box || !b64Data) return;
    const oldSvg = box.querySelector('svg');
    if (oldSvg) oldSvg.style.display = 'none';

    let img = box.querySelector('img.spec-img-render');
    if (!img) {
      img = document.createElement('img');
      img.className = 'spec-img-render';
      img.alt = 'Spectrum image';
      img.style.cssText = 'width:100%;height:170px;object-fit:cover;border-radius:6px;border:1px solid rgba(255,255,255,.06);display:block';
      box.appendChild(img);
    }
    img.src = `data:image/jpeg;base64,${b64Data}`;
  };

  applyToBox(boxes[0], spectra.original[key]);
  applyToBox(boxes[1], spectra.transformed[key]);
}

function setAnalysisStatus(label, state = 'waiting') {
  const tag = $('#analysisStatusTag') || document.querySelector('#page-analysis .tag-done');
  if (!tag) return;
  tag.textContent = label;
  tag.classList.toggle('tag-done', state === 'done');
}

function formatMetricValue(value, digits = 2, suffix = '') {
  const n = Number(value);
  if (!Number.isFinite(n)) return '--';
  return `${n.toFixed(digits)}${suffix}`;
}

function formatCompactEnergy(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '--';
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
  return n.toFixed(2);
}

function formatScaledEnergyHtml(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '--';
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)} <small>x10^9</small>`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)} <small>x10^6</small>`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(2)} <small>x10^3</small>`;
  return n.toFixed(2);
}

function signedPercent(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '--';
  const sign = n > 0 ? '+' : '';
  return `${sign}${n.toFixed(digits)}%`;
}

function verdictClass(el, cls) {
  if (!el) return;
  el.classList.remove('mc-v-ok', 'mc-v-info', 'mc-v-changed');
  if (cls) el.classList.add(cls);
}

function setComparisonCard(index, beforeText, afterText, beforeWidth, afterWidth, verdict, vClass) {
  const card = document.querySelectorAll('#page-analysis .mc-card')[index];
  if (!card) return;
  const rows = card.querySelectorAll('.mc-bar-row');
  const beforeRow = rows[0];
  const afterRow = rows[1];
  if (beforeRow) {
    const fill = beforeRow.querySelector('.mc-fill');
    const num = beforeRow.querySelector('.mc-num');
    if (fill) fill.style.width = `${clamp(beforeWidth, 0, 100)}%`;
    if (num) num.textContent = beforeText;
  }
  if (afterRow) {
    const fill = afterRow.querySelector('.mc-fill');
    const num = afterRow.querySelector('.mc-num');
    if (fill) fill.style.width = `${clamp(afterWidth, 0, 100)}%`;
    if (num) num.textContent = afterText;
  }
  const verdictEl = card.querySelector('.mc-verdict');
  if (verdictEl) verdictEl.textContent = verdict;
  verdictClass(verdictEl, vClass);
}

function clearSpectrumImages() {
  document.querySelectorAll('#page-analysis .spec-grid .spec-box').forEach(box => {
    const img = box.querySelector('img.spec-img-render');
    if (img) img.remove();
    const oldSvg = box.querySelector('svg');
    if (oldSvg) oldSvg.style.display = '';
  });
}

function clearAnalysisMetrics(message = 'Run a transform to compute analysis.') {
  backendState.analysisLoaded = false;
  backendState.analysisData = null;
  backendState.analysisSpectra = null;
  setAnalysisStatus('Waiting', 'waiting');
  clearSpectrumImages();

  if ($('#qmMsePct')) $('#qmMsePct').textContent = '--';
  if ($('#qmPsnrPct')) $('#qmPsnrPct').textContent = '--';
  if ($('#qmSsimPct')) $('#qmSsimPct').textContent = '--';
  if ($('#qmMseVal')) $('#qmMseVal').textContent = '--';
  if ($('#qmPsnrVal')) $('#qmPsnrVal').textContent = '--';
  if ($('#qmSsimVal')) $('#qmSsimVal').textContent = '--';
  if ($('#anTblMse')) $('#anTblMse').textContent = '--';
  if ($('#anTblPsnr')) $('#anTblPsnr').textContent = '--';
  if ($('#anTblSsim')) $('#anTblSsim').textContent = '--';
  if ($('#enTotal')) $('#enTotal').textContent = '--';
  if ($('#enHF')) $('#enHF').textContent = '--';
  if ($('#enLF')) $('#enLF').textContent = '--';
  if ($('#enRatio')) $('#enRatio').textContent = '--';
  if ($('#anTblHfLf')) $('#anTblHfLf').textContent = '--';
  if ($('#anTblHf')) $('#anTblHf').textContent = '--';
  if ($('#anTblLf')) $('#anTblLf').textContent = '--';
  ['enFillTotal', 'enFillHF', 'enFillLF', 'enFillRatio'].forEach(id => {
    const el = $(`#${id}`);
    if (el) el.style.width = '0%';
  });
  setRingProgress($('#qmMseRing'), 0);
  setRingProgress($('#qmPsnrRing'), 0);
  setRingProgress($('#qmSsimRing'), 0);

  for (let i = 0; i < 6; i++) {
    setComparisonCard(i, '--', '--', 0, 0, message, 'mc-v-info');
  }
}

function applyAnalysisFromBackend(data) {
  const mse = Number(data.mse || 0);
  const psnr = Number(data.psnr || 0);
  const ssim = Number(data.ssim || 0);
  const energy = data.energy || {};
  const comparison = data.energyComparison || {};
  const origEnergy = comparison.original || {};
  const transEnergy = comparison.transformed || energy || {};
  const deltaEnergy = comparison.delta || {};
  const luminance = data.luminance || {};
  const origLum = luminance.original || {};
  const transLum = luminance.transformed || {};

  backendState.analysisData = data;
  setAnalysisStatus('Computed', 'done');

  if (data.spectrum && data.spectrum.original && data.spectrum.transformed) {
    backendState.analysisSpectra = data.spectrum;
    renderSpectrumImages(backendState.spectrumMode || 'mag');
  }

  if ($('#qmMseVal')) $('#qmMseVal').textContent = mse.toFixed(2);
  if ($('#qmPsnrVal')) $('#qmPsnrVal').textContent = `${psnr.toFixed(2)} dB`;
  if ($('#qmSsimVal')) $('#qmSsimVal').textContent = ssim.toFixed(3);
  if ($('#anTblMse')) $('#anTblMse').textContent = mse.toFixed(4);
  if ($('#anTblPsnr')) $('#anTblPsnr').textContent = psnr.toFixed(4);
  if ($('#anTblSsim')) $('#anTblSsim').textContent = ssim.toFixed(4);

  const msePercent = clamp((Math.sqrt(Math.max(mse, 0)) / 255) * 100, 0, 100);
  const psnrPercent = clamp((psnr / 40) * 100, 5, 98);
  const ssimPercent = clamp(ssim * 100, 5, 98);
  if ($('#qmMsePct')) $('#qmMsePct').textContent = `${Math.round(msePercent)}%`;
  if ($('#qmPsnrPct')) $('#qmPsnrPct').textContent = `${Math.round(psnrPercent)}%`;
  if ($('#qmSsimPct')) $('#qmSsimPct').textContent = `${Math.round(ssimPercent)}%`;

  if ($('#enTotal')) $('#enTotal').innerHTML = formatScaledEnergyHtml(transEnergy.total);
  if ($('#enHF')) $('#enHF').innerHTML = formatScaledEnergyHtml(transEnergy.high);
  if ($('#enLF')) $('#enLF').innerHTML = formatScaledEnergyHtml(transEnergy.low);
  if ($('#enRatio')) $('#enRatio').textContent = formatMetricValue(transEnergy.hfLfRatio, 3);
  if ($('#anTblHfLf')) $('#anTblHfLf').textContent = formatMetricValue(transEnergy.hfLfRatio, 6);
  if ($('#anTblHf')) $('#anTblHf').textContent = formatMetricValue(transEnergy.high, 2);
  if ($('#anTblLf')) $('#anTblLf').textContent = formatMetricValue(transEnergy.low, 2);

  const maxEnergy = Math.max(Number(origEnergy.total || 0), Number(transEnergy.total || 0), 1);
  const maxHigh = Math.max(Number(origEnergy.high || 0), Number(transEnergy.high || 0), 1);
  const maxLow = Math.max(Number(origEnergy.low || 0), Number(transEnergy.low || 0), 1);
  const ratioMax = Math.max(Number(origEnergy.hfLfRatio || 0), Number(transEnergy.hfLfRatio || 0), 0.01);

  if ($('#enFillTotal')) $('#enFillTotal').style.width = `${clamp((Number(transEnergy.total || 0) / maxEnergy) * 100, 0, 100)}%`;
  if ($('#enFillHF')) $('#enFillHF').style.width = `${clamp((Number(transEnergy.high || 0) / maxHigh) * 100, 0, 100)}%`;
  if ($('#enFillLF')) $('#enFillLF').style.width = `${clamp((Number(transEnergy.low || 0) / maxLow) * 100, 0, 100)}%`;
  if ($('#enFillRatio')) $('#enFillRatio').style.width = `${clamp((Number(transEnergy.hfLfRatio || 0) / ratioMax) * 100, 0, 100)}%`;

  setRingProgress($('#qmMseRing'), msePercent);
  setRingProgress($('#qmPsnrRing'), psnrPercent);
  setRingProgress($('#qmSsimRing'), ssimPercent);

  const origHigh = Number(origEnergy.high || 0);
  const transHigh = Number(transEnergy.high || 0);
  const origMean = Number(origLum.mean || 0);
  const transMean = Number(transLum.mean || 0);
  const origRatio = Number(origEnergy.hfLfRatio || 0);
  const transRatio = Number(transEnergy.hfLfRatio || 0);
  const hfDelta = Number(deltaEnergy.highPct || 0);
  const ratioDelta = Number(deltaEnergy.hfLfRatioPct || 0);
  const lumDelta = Number(luminance.deltaMeanPct || 0);

  setComparisonCard(0, '0.00', mse.toFixed(2), 0, msePercent, mse < 1 ? 'No visible pixel change' : 'Transformation effect measured', 'mc-v-changed');
  setComparisonCard(1, 'Original', `${psnr.toFixed(2)} dB`, 100, psnrPercent, psnr >= 30 ? 'High signal quality' : psnr >= 20 ? 'Acceptable signal quality' : 'Strong signal change detected', psnr >= 20 ? 'mc-v-ok' : 'mc-v-changed');
  setComparisonCard(2, '1.000', ssim.toFixed(3), 100, ssimPercent, ssim >= 0.85 ? 'Structure successfully preserved' : ssim >= 0.65 ? 'Moderate structural change' : 'Large structural change', ssim >= 0.85 ? 'mc-v-ok' : 'mc-v-info');
  setComparisonCard(3, formatCompactEnergy(origHigh), formatCompactEnergy(transHigh), (origHigh / maxHigh) * 100, (transHigh / maxHigh) * 100, `High-frequency change ${signedPercent(hfDelta)}`, Math.abs(hfDelta) < 5 ? 'mc-v-ok' : 'mc-v-info');
  setComparisonCard(4, origMean.toFixed(1), transMean.toFixed(1), (origMean / 255) * 100, (transMean / 255) * 100, `Mean luminance shift ${signedPercent(lumDelta)}`, Math.abs(lumDelta) <= 5 ? 'mc-v-ok' : 'mc-v-info');
  setComparisonCard(5, origRatio.toFixed(3), transRatio.toFixed(3), (origRatio / ratioMax) * 100, (transRatio / ratioMax) * 100, `HF/LF ratio change ${signedPercent(ratioDelta)}`, Math.abs(ratioDelta) < 5 ? 'mc-v-ok' : 'mc-v-info');
}

async function fetchAnalysisFromBackend(force = false) {
  if (!appState.imageDataUrl) {
    clearAnalysisMetrics('Upload an image to compute analysis.');
    return;
  }
  if (backendState.analysisLoaded && !force) return;
  try {
    if (!backendState.lastExprDataUrl && !backendState.lastAgingDataUrl) {
      clearAnalysisMetrics('Run Expression Warp or Aging first.');
      if (force) toast('Run Expression Warp or Aging first, then open Analysis.', 'info');
      return;
    }
    const originalBlob = await getCurrentImageBlob();
    const transformedBlob = await buildTransformedBlobForAnalysis();
    if (!originalBlob || !transformedBlob) {
      clearAnalysisMetrics('Could not prepare images for analysis.');
      if (force) toast('Could not prepare transformed image for analysis.', 'warning');
      return;
    }

    const fd = new FormData();
    fd.append('original', originalBlob, appState.fileName || 'original.jpg');
    fd.append('transformed', transformedBlob, 'transformed.jpg');

    const res = await fetch('/api/analysis', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || 'Analysis failed.');

    applyAnalysisFromBackend(data);
    backendState.analysisLoaded = true;
    toast(`Backend analysis updated (${data.durationMs} ms).`, 'success');
  } catch (err) {
    toast(`Backend analysis error: ${err.message}`, 'warning');
  }
}

function setPipelineImage(dataUrl, meta = {}) {
  // Fix: '' || old would keep old URL when removing. Use typeof check so empty string clears correctly.
  appState.imageDataUrl = typeof dataUrl === 'string' ? dataUrl : appState.imageDataUrl;
  appState.fileName = meta.fileName || appState.fileName;
  appState.format = meta.format || appState.format;
  appState.sizeText = meta.sizeText || appState.sizeText;
  appState.dimensions = meta.dimensions || appState.dimensions;

  const ppThumb = $('#ppThumbImg');
  const lmThumb = $('#lmThumbImg');
  const exprThumb = $('#exprThumbImg');
  [ppThumb, lmThumb, exprThumb].forEach(img => {
    if (!img) return;
    img.src = appState.imageDataUrl;
    img.style.display = appState.imageDataUrl ? 'block' : 'none';
  });

  const ppPh = $('#ppThumbPlaceholder');
  const lmPh = $('#lmThumbPlaceholder');
  const exprPh = $('#exprThumbPlaceholder');
  [ppPh, lmPh, exprPh].forEach(ph => {
    if (!ph) return;
    ph.style.display = appState.imageDataUrl ? 'none' : 'grid';
  });

  const assignText = (id, val) => {
    const el = $(id);
    if (el) el.textContent = val;
  };
  assignText('#ppFileName', appState.fileName);
  assignText('#ppDim', appState.dimensions);
  assignText('#ppFSize', appState.sizeText);
  assignText('#ppFFormat', appState.format);
  assignText('#lmFileName', appState.fileName);
  assignText('#exprFileName', appState.fileName);

  syncExpressionPhotoPanels();
  syncAgingPhotoPanels();
  syncLandmarkPhotoBg();
}

/* ── Aging page: inject photo into Before/After panels ── */
function syncAgingPhotoPanels() {
  const beforeWrap = document.querySelector('#page-aging .aging-side-card:not(.aging-side-card-active) .aging-face-wrap');
  const afterWrap = document.querySelector('#page-aging .aging-side-card.aging-side-card-active .aging-face-wrap');
  if (!beforeWrap || !afterWrap) return;

  const setup = (wrap, id) => {
    wrap.style.position = 'relative';
    let img = wrap.querySelector(`#${id}`);
    if (!img) {
      img = document.createElement('img');
      img.id = id; img.alt = '';
      // z-index:2 ensures the photo renders above the SVG wireframe (z-index:auto≈0)
      // regardless of DOM order inside the grid container.
      img.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#0b1220;border-radius:inherit;display:none;z-index:2;';
      wrap.prepend(img);
    }
    return img;
  };

  const beforeImg = setup(beforeWrap, 'agingBeforeImg');
  const afterImg = setup(afterWrap, 'agingAfterImg');
  const beforeUrl = backendState.lastExprDataUrl || appState.imageDataUrl;
  const afterUrl = backendState.lastAgingDataUrl;

  if (beforeUrl) {
    beforeImg.src = beforeUrl;
    beforeImg.style.display = 'block';
    if (afterUrl) {
      afterImg.src = afterUrl;
      afterImg.style.display = 'block';
    } else {
      afterImg.src = '';
      afterImg.style.display = 'none';
    }
    // Real photos should not carry the placeholder wireframe; it cannot align to arbitrary crops.
    $$('#page-aging .aging-face-wrap .face-svg').forEach(s => {
      s.style.display = 'none';
    });
  } else {
    beforeImg.style.display = 'none';
    afterImg.style.display = 'none';
    $$('#page-aging .aging-face-wrap .face-svg').forEach(s => { s.style.display = ''; s.style.opacity = ''; s.style.zIndex = ''; });
  }
}

/* ── Landmark page: show uploaded photo as semi-transparent canvas background ── */
function syncLandmarkPhotoBg() {
  const lmCanvas = document.querySelector('.lm-canvas');
  if (!lmCanvas) return;
  lmCanvas.style.position = 'relative';
  let img = lmCanvas.querySelector('#lmBgImg');
  if (!img) {
    img = document.createElement('img');
    img.id = 'lmBgImg'; img.alt = '';
    img.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:fill;border-radius:inherit;display:none;opacity:0.40;z-index:0;';
    lmCanvas.prepend(img);
    // Keep scan line + SVG above the photo in the same coordinate space.
    const scan = lmCanvas.querySelector('.lm-scan');
    const svg = lmCanvas.querySelector('.lm-svg');
    if (scan) {
      scan.style.position = 'absolute';
      scan.style.inset = '0';
      scan.style.zIndex = '1';
      scan.style.pointerEvents = 'none';
    }
    if (svg) {
      svg.style.position = 'absolute';
      svg.style.inset = '0';
      svg.style.width = '100%';
      svg.style.height = '100%';
      svg.style.zIndex = '2';
      svg.style.pointerEvents = 'none';
    }
  }
  const has = Boolean(appState.imageDataUrl);
  img.src = has ? appState.imageDataUrl : '';
  img.style.display = has ? 'block' : 'none';
}

function ensureExpressionPhotoElements() {
  const bodies = $$('#page-expression .comp-card .comp-body');
  if (!bodies || bodies.length < 2) return;

  // Original panel — plain <img>
  const setupImg = (body, id) => {
    if (!body) return null;
    body.style.position = 'relative';
    let img = body.querySelector(`#${id}`);
    if (!img) {
      img = document.createElement('img');
      img.id = id;
      img.alt = 'Expression preview';
      img.style.cssText = [
        'position:absolute', 'inset:0', 'width:100%', 'height:100%',
        'object-fit:cover', 'border-radius:12px', 'display:none',
      ].join(';');
      body.prepend(img);
    }
    return img;
  };

  // Transformed panel — <canvas> so we can do real pixel-level warp
  const setupCanvas = (body, id) => {
    if (!body) return null;
    body.style.position = 'relative';
    let c = body.querySelector(`#${id}`);
    if (!c) {
      c = document.createElement('canvas');
      c.id = id;
      c.style.cssText = [
        'position:absolute', 'inset:0', 'width:100%', 'height:100%',
        'border-radius:12px', 'display:none', 'object-fit:contain', 'background:#0b1220',
        'z-index:10',
      ].join(';');
      body.prepend(c);
    }
    return c;
  };

  setupImg(bodies[0], 'exprOriginalImg');
  setupCanvas(bodies[1], 'exprTransformedCanvas');
}

function renderExpressionCanvasFromDataUrl(dataUrl) {
  const canvas = document.getElementById('exprTransformedCanvas');
  if (!canvas || !dataUrl) return;
  // Ensure the canvas is visible before drawing
  if (canvas.style.display === 'none') canvas.style.display = 'block';

  const tmpImg = new Image();
  tmpImg.onload = () => {
    const rect = canvas.getBoundingClientRect();
    const W = Math.round(rect.width) || 400;
    const H = Math.round(rect.height) || 400;
    if (canvas.width !== W) canvas.width = W;
    if (canvas.height !== H) canvas.height = H;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    const scale = Math.min(W / tmpImg.naturalWidth, H / tmpImg.naturalHeight);
    const dw = tmpImg.naturalWidth * scale;
    const dh = tmpImg.naturalHeight * scale;
    const dx = (W - dw) / 2;
    const dy = (H - dh) / 2;
    ctx.fillStyle = '#0b1220';
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(tmpImg, dx, dy, dw, dh);
  };
  tmpImg.src = dataUrl;
}

function syncExpressionPhotoPanels() {
  ensureExpressionPhotoElements();

  const originalImg = $('#exprOriginalImg');
  const transformedCvs = $('#exprTransformedCanvas');
  const hasImage = Boolean(appState.imageDataUrl);
  const originalSvg = $('#page-expression .comp-card .face-svg');
  const transformedSvg = $('#exprSvg');

  if (!originalImg || !transformedCvs) return;

  if (hasImage) {
    originalImg.src = appState.imageDataUrl;
    originalImg.style.display = 'block';
    transformedCvs.style.display = 'block';
    if (originalSvg) originalSvg.style.display = 'none';
    if (transformedSvg) transformedSvg.style.display = 'none';
    renderExpressionCanvasFromDataUrl(backendState.lastExprDataUrl || appState.imageDataUrl);
    return;
  }

  originalImg.style.display = 'none';
  transformedCvs.style.display = 'none';
  if (originalSvg) originalSvg.style.display = 'block';
  if (transformedSvg) transformedSvg.style.display = 'block';
}

if (browseBtn) browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
if (dropzone) dropzone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach(ev => {
  if (dropzone) dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('dragging'); });
});
['dragleave', 'drop'].forEach(ev => {
  if (dropzone) dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('dragging'); });
});
if (dropzone) dropzone.addEventListener('drop', e => { if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
if (fileInput) fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function handleFile(file) {
  const valid = ['image/jpeg', 'image/png'];
  if (!valid.includes(file.type)) { toast('Only JPG/PNG files are accepted.', 'error'); return; }
  if (file.size > 10 * 1024 * 1024) { toast('File exceeds the 10 MB limit.', 'error'); return; }
  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      $('#pvImg').src = e.target.result;
      $('#pvName').textContent = file.name;
      $('#pvFormat').textContent = file.type.split('/')[1].toUpperCase();
      $('#pvSize').textContent = (file.size / 1024).toFixed(1) + ' KB';
      $('#pvRes').textContent = img.width + '×' + img.height;
      setPipelineImage(e.target.result, {
        fileName: file.name,
        format: file.type.split('/')[1].toUpperCase(),
        sizeText: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
        dimensions: img.width + ' × ' + img.height,
      });
      backendState.landmarksLoaded = false;
      backendState.analysisLoaded = false;
      backendState.lastExprDataUrl = '';
      backendState.lastAgingDataUrl = '';
      backendState.analysisData = null;
      backendState.analysisSpectra = null;
      uploadCard.style.display = 'none';
      previewCard.style.display = 'block';
      toast('Image loaded — ready to process!', 'success');
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

if ($('#btnRemove')) $('#btnRemove').addEventListener('click', () => {
  uploadCard.style.display = '';
  previewCard.style.display = 'none';
  fileInput.value = '';
  appState.imageDataUrl = '';
  backendState.landmarksLoaded = false;
  backendState.analysisLoaded = false;
  backendState.lastExprDataUrl = '';
  backendState.lastAgingDataUrl = '';
  backendState.analysisData = null;
  backendState.analysisSpectra = null;
  backendState.landmarks = [];
  backendState.lastLmData = null;
  setPipelineImage('', {
    fileName: 'portrait_01.jpg',
    format: 'JPEG',
    sizeText: '2.4 MB',
    dimensions: '1024 × 1024',
  });
});
if ($('#btnNext')) $('#btnNext').addEventListener('click', () => nav('preprocess'));

/* ═══════════════════ LANDMARKS ═══════════════════ */
const tgPoints = $('#tgPoints'), tgMesh = $('#tgMesh'), tgLabels = $('#tgLabels');
if (tgPoints) tgPoints.addEventListener('change', () => { $('#ptsG').style.opacity = tgPoints.checked ? 1 : 0; });
if (tgMesh) tgMesh.addEventListener('change', () => { $('#meshG').style.opacity = tgMesh.checked ? 1 : 0; });
if (tgLabels) tgLabels.addEventListener('change', () => { $('#labelsG').style.opacity = tgLabels.checked ? 1 : 0; });

/* Region filter: All / Jaw / Brow / Eyes / Nose / Mouth */
const segRegion = $('#segRegion');
if (segRegion) {
  segRegion.querySelectorAll('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      segRegion.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const val = btn.dataset.val;
      $$('#ptsG circle').forEach(c => {
        const idx = Number(c.getAttribute('data-idx'));
        const show = val === 'all'
          || (val === 'jaw' && idx <= 16)
          || (val === 'brow' && idx >= 17 && idx <= 26)
          || (val === 'nose' && idx >= 27 && idx <= 35)
          || (val === 'eyes' && idx >= 36 && idx <= 47)
          || (val === 'mouth' && idx >= 48);
        c.style.display = show ? '' : 'none';
      });
      $$('#connG [data-region]').forEach(el => {
        el.style.display = (val === 'all' || el.dataset.region === val) ? '' : 'none';
      });
      const meshG = $('#meshG');
      if (meshG) meshG.style.display = val === 'all' ? '' : 'none';
    });
  });
}

/* Re-detect button */
if ($('#btnRunLM')) {
  $('#btnRunLM').addEventListener('click', () => {
    backendState.landmarksLoaded = false;
    fetchLandmarksFromBackend(true);
  });
}

/* Model / schema toggles (visual-only — backend always uses best available) */
['#segModel', '#segSchema'].forEach(sel => {
  const grp = $(sel);
  if (!grp) return;
  grp.querySelectorAll('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      grp.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
});

/* Live coordinate display on hover */
const lmCanvasEl = document.querySelector('.lm-canvas');
if (lmCanvasEl) {
  lmCanvasEl.addEventListener('mousemove', e => {
    const rect = lmCanvasEl.getBoundingClientRect();
    const svg = $('#lmSvg');
    const vb = svg?.viewBox?.baseVal;
    const vbW = vb?.width || 300;
    const vbH = vb?.height || 290;
    const rx = ((e.clientX - rect.left) / rect.width) * vbW;
    const ry = ((e.clientY - rect.top) / rect.height) * vbH;
    const coord = $('#lmCoordDisplay');
    if (coord) coord.textContent = `x: ${rx.toFixed(0)}   y: ${ry.toFixed(0)}`;
  });
  lmCanvasEl.addEventListener('mouseleave', () => {
    const coord = $('#lmCoordDisplay');
    if (coord) coord.innerHTML = 'x: \u2014 &nbsp; y: \u2014';
  });
}

/* ═══════════════════ EXPRESSION SLIDERS ═══════════════════ */
const sliders = [
  { id: 'sSmile', valId: 'vSmile' },
  { id: 'sBrow', valId: 'vBrow' },
  { id: 'sLip', valId: 'vLip' },
  { id: 'sSlim', valId: 'vSlim' },
  { id: 'sFaceFullness', valId: 'vFaceFullness' },
  { id: 'sEyeSize', valId: 'vEyeSize' },
];
sliders.forEach(({ id, valId }) => {
  const s = $(`#${id}`), v = $(`#${valId}`);
  if (s && v) s.addEventListener('input', () => {
    v.textContent = s.value + '%';
    s.style.setProperty('--pct', s.value + '%');
    updateExprSvg();
  });
});

function updateExprSvg() {
  const smile = (+$('#sSmile')?.value || 0) / 100;
  const brow = (+$('#sBrow')?.value || 0) / 100;
  const slim = (+$('#sSlim')?.value || 0) / 100;
  const lip = (+$('#sLip')?.value || 0) / 100;

  const mouth = $('#exMouth');
  if (mouth) {
    const curve = 178 + smile * 16;
    const widen = 82 - lip * 8;
    const widenR = 118 + lip * 8;
    mouth.setAttribute('d', `M${widen} 170 Q100 ${curve} ${widenR} 170`);
  }
  const browL = $('#exBrowL'), browR = $('#exBrowR');
  if (browL) {
    const by = 102 - brow * 30;
    const cy = 95 - brow * 30;
    browL.setAttribute('d', `M72 ${by} Q80 ${cy} 92 ${by}`);
    browR.setAttribute('d', `M108 ${by} Q120 ${cy} 128 ${by}`);
  }
  const face = $('#exFace');
  if (face) {
    const rx = 72 - slim * 10;
    face.setAttribute('rx', rx);
  }

  /* FR-3.7.2: update parameter display */
  const params = $('#exprParams');
  if (params) {
    params.innerHTML =
      `<span>Smile: ${Math.round(smile * 100)}%</span>` +
      `<span>Brow: ${Math.round(brow * 100)}%</span>` +
      `<span>Lip: ${Math.round(lip * 100)}%</span>` +
      `<span>Slim: ${Math.round(slim * 100)}%</span>`;
  }

  backendState.analysisLoaded = false;
}

function setReferenceTransferReadout(data) {
  const params = $('#exprParams');
  if (!params) return;
  const moved = Number(data?.movedPoints || 0);
  const maxD = Number(data?.maxDisplacement || 0);
  const tri = Number(data?.triangleCount || 0);
  params.innerHTML =
    `<span>Ref: ${moved} pts</span>` +
    `<span>MaxD: ${Number.isFinite(maxD) ? maxD.toFixed(1) : '0.0'}</span>` +
    `<span>Tri: ${Number.isFinite(tri) ? tri : 0}</span>` +
    `<span>Mode: Ref</span>`;
}

if ($('#btnResetExpr')) $('#btnResetExpr').addEventListener('click', () => {
  sliders.forEach(({ id, valId }) => {
    const s = $(`#${id}`), v = $(`#${valId}`);
    if (s) { s.value = 0; v.textContent = '0%'; s.style.setProperty('--pct', '0%'); }
  });
  updateExprSvg();
});

/* Deformation mesh overlay toggle */
const toggleMesh = $('#toggleMesh'), meshOverlay = $('#meshOverlay');
if (toggleMesh && meshOverlay) toggleMesh.addEventListener('change', () => {
  meshOverlay.style.opacity = toggleMesh.checked ? '1' : '0';
  // Re-run warp to bake/remove the deformation grid.
  // Only fires when an image is loaded and Facial Volume is active.
  const ff = Number($('#sFaceFullness')?.value || 0);
  if (appState.imageDataUrl && ff > 0 && getExpressionMode() !== 'reference-only') {
    $('#btnApplyExpr')?.click();
  }
});

/* ═══════════════════ AGING SLIDERS ═══════════════════ */
['sAge', 'sContrast', 'sTone', 'sNoise', 'sDetail'].forEach(id => {
  const s = $(`#${id}`), v = $(`#v${id.slice(1)}`);
  if (s && v) s.addEventListener('input', () => {
    v.textContent = s.value + '%';
  });
});
const sTargetAge = $('#sTargetAge'), vTargetAge = $('#vTargetAge');
if (sTargetAge && vTargetAge) sTargetAge.addEventListener('input', () => {
  const v = +sTargetAge.value;
  vTargetAge.textContent = v === 0 ? '0' : (v > 0 ? '+' + v : v);
  updateApparentAge(v);
  updateAgeOffsetLabel(v);
});

/* ═══════════════════ SEGMENTED CONTROLS ═══════════════════ */
$$('.seg').forEach(seg => {
  seg.querySelectorAll('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      seg.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      /* FR-3.4.9: show method comparison panel when AI mode is selected */
      if (seg.id === 'ageMethod') {
        const methodCompare = $('#methodCompareCard');
        if (methodCompare) methodCompare.style.display = (btn.dataset.v === 'ai') ? 'block' : 'none';
        updateAnalysisMetrics();
      }
    });
  });
});

function syncMethodCompareVisibility() {
  const methodCompare = $('#methodCompareCard');
  const activeMethod = $('#ageMethod .seg-btn.active');
  if (!methodCompare || !activeMethod) return;
  methodCompare.style.display = (activeMethod.dataset.v === 'ai') ? 'block' : 'none';
}

/* ═══════════════════ APPARENT AGE (FR-3.8.2) ═══════════════════ */
const BASE_AGE = 28;
function updateApparentAge(offset) {
  const transformed = Math.max(1, BASE_AGE + offset);
  const el = $('#ageTransformed');
  if (el) el.textContent = transformed;
  const lbl = $('#ageOffsetLabel');
  if (lbl) {
    if (offset === 0) lbl.textContent = '0 Years';
    else if (offset > 0) lbl.textContent = `+${offset} Years`;
    else lbl.textContent = `${offset} Years`;
  }
  /* update comparison panel values */
  const freqEl = $('#ageFreqComp');
  const aiEl = $('#ageAiComp');
  if (freqEl) freqEl.textContent = transformed;
  if (aiEl) aiEl.textContent = Math.max(1, BASE_AGE + Math.round(offset * 0.92));
}
function updateAgeOffsetLabel(offset) { } /* handled inside updateApparentAge */

/* ═══════════════════ WEBCAM (FR-3.8.3) ═══════════════════ */
let webcamStream = null;
const webcamCard = $('#webcamCard'), webcamVideo = $('#webcamVideo'), webcamCanvas = $('#webcamCanvas');

/* ═══════════════════ WEBCAM LIVE MODE ═══════════════════ */
let _liveTimer = null;
let _liveProcessing = false;

function startLiveMode() {
  if (_liveTimer) return;
  const liveOutputWrap = $('#liveOutputWrap');
  const liveOutput = $('#liveOutput');
  if (liveOutputWrap) liveOutputWrap.style.display = 'block';
  const btnLive = $('#btnLiveMode');
  if (btnLive) btnLive.textContent = '⏹ Stop Live';

  _liveTimer = setInterval(async () => {
    if (_liveProcessing || !webcamVideo || !webcamCanvas || !webcamStream) return;
    const smile = Number($('#sSmile')?.value || 0);
    const brow = Number($('#sBrow')?.value || 0);
    const lip = Number($('#sLip')?.value || 0);
    const slim = Number($('#sSlim')?.value || 0);
    const doubleChin = 0;  // merged into Facial Volume (backend handles coupling)
    const faceFullness = Number($('#sFaceFullness')?.value || 0);
    const eyeSize = Number($('#sEyeSize')?.value || 0);
    if (smile === 0 && brow === 0 && lip === 0 && slim === 0 && faceFullness === 0 && eyeSize === 0) return;
    _liveProcessing = true;
    try {
      webcamCanvas.width = webcamVideo.videoWidth || 640;
      webcamCanvas.height = webcamVideo.videoHeight || 480;
      webcamCanvas.getContext('2d').drawImage(webcamVideo, 0, 0);
      const frameBlob = dataUrlToBlob(webcamCanvas.toDataURL('image/jpeg', 0.75));
      if (!frameBlob) return;
      const fd = new FormData();
      fd.append('image', frameBlob, 'live-frame.jpg');
      fd.append('smile', String(smile));
      fd.append('brow', String(brow));
      fd.append('lip', String(lip));
      fd.append('slim', String(slim));
      fd.append('face_fullness', String(faceFullness));
      fd.append('eye_size', String(eyeSize));
      fd.append('mode', 'delaunay');
      const res = await fetch('/api/transform', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.ok && data.image && liveOutput) {
        liveOutput.src = `data:image/jpeg;base64,${data.image}`;
      }
    } catch (_) { /* ignore mid-stream errors */ }
    finally { _liveProcessing = false; }
  }, 450); // ~2 fps — keeps server load acceptable
}

function stopLiveMode() {
  if (_liveTimer) { clearInterval(_liveTimer); _liveTimer = null; }
  const liveOutputWrap = $('#liveOutputWrap');
  if (liveOutputWrap) liveOutputWrap.style.display = 'none';
  const btnLive = $('#btnLiveMode');
  if (btnLive) btnLive.textContent = '⏵ Live Mode';
  _liveProcessing = false;
}

if ($('#btnLiveMode')) {
  $('#btnLiveMode').addEventListener('click', () => {
    if (_liveTimer) {
      stopLiveMode();
    } else {
      if (!webcamStream) { toast('Start webcam first.', 'warning'); return; }
      startLiveMode();
    }
  });
}

function startWebcam() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast('Webcam is not supported in this browser.', 'warning');
    return;
  }
  uploadCard.style.display = 'none';
  previewCard.style.display = 'none';
  webcamCard.style.display = 'block';
  navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false })
    .then(stream => { webcamStream = stream; webcamVideo.srcObject = stream; })
    .catch(err => {
      webcamCard.style.display = 'none';
      uploadCard.style.display = '';
      toast('Could not access webcam: ' + err.message, 'error');
    });
}
function stopWebcam() {
  stopLiveMode();
  if (webcamStream) { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
  webcamCard.style.display = 'none';
  uploadCard.style.display = '';
}
if ($('#webcamBtn')) $('#webcamBtn').addEventListener('click', e => { e.stopPropagation(); startWebcam(); });
if ($('#btnCloseWebcam')) $('#btnCloseWebcam').addEventListener('click', stopWebcam);
if ($('#btnCancelWebcam')) $('#btnCancelWebcam').addEventListener('click', stopWebcam);
if ($('#btnCapture')) $('#btnCapture').addEventListener('click', () => {
  if (!webcamVideo || !webcamCanvas) return;
  webcamCanvas.width = webcamVideo.videoWidth || 640;
  webcamCanvas.height = webcamVideo.videoHeight || 480;
  webcamCanvas.getContext('2d').drawImage(webcamVideo, 0, 0);
  const dataUrl = webcamCanvas.toDataURL('image/jpeg');
  stopWebcam();
  /* show in preview */
  $('#pvImg').src = dataUrl;
  $('#pvName').textContent = 'webcam-capture.jpg';
  $('#pvFormat').textContent = 'JPEG';
  $('#pvSize').textContent = '—';
  $('#pvRes').textContent = `${webcamCanvas.width}×${webcamCanvas.height}`;
  setPipelineImage(dataUrl, {
    fileName: 'webcam-capture.jpg',
    format: 'JPEG',
    sizeText: 'Live capture',
    dimensions: `${webcamCanvas.width} × ${webcamCanvas.height}`,
  });
  uploadCard.style.display = 'none';
  previewCard.style.display = 'block';
  backendState.landmarksLoaded = false;
  backendState.analysisLoaded = false;
  backendState.lastExprDataUrl = '';
  backendState.lastAgingDataUrl = '';
  backendState.analysisData = null;
  backendState.analysisSpectra = null;
  toast('Webcam frame captured!', 'success');
});

/* ═══════════════════ REFERENCE IMAGE (FR-3.8.1) ═══════════════════ */
const refDrop = $('#refDrop'), refInput = $('#refInput'), refPreview = $('#refPreview'), refImg = $('#refImg');
if (refDrop) {
  refDrop.addEventListener('click', () => refInput.click());
  refDrop.addEventListener('dragover', e => { e.preventDefault(); refDrop.style.borderColor = 'rgba(139,92,246,.55)'; });
  refDrop.addEventListener('dragleave', () => { refDrop.style.borderColor = ''; });
  refDrop.addEventListener('drop', e => {
    e.preventDefault(); refDrop.style.borderColor = '';
    if (e.dataTransfer.files.length) loadRefImage(e.dataTransfer.files[0]);
  });
}
if (refInput) refInput.addEventListener('change', () => { if (refInput.files.length) loadRefImage(refInput.files[0]); });
function loadRefImage(file) {
  if (!['image/jpeg', 'image/png'].includes(file.type)) { toast('Only JPG/PNG accepted.', 'error'); return; }
  const reader = new FileReader();
  reader.onload = e => {
    if (refImg) refImg.src = e.target.result;
    if (refDrop) refDrop.style.display = 'none';
    if (refPreview) refPreview.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}
if ($('#btnClearRef')) $('#btnClearRef').addEventListener('click', () => {
  if (refImg) refImg.src = '';
  if (refInput) refInput.value = '';
  if (refPreview) refPreview.style.display = 'none';
  if (refDrop) refDrop.style.display = 'flex';
});

function getExpressionMode() {
  const mode = $('#sExprMode')?.value || 'manual';
  return mode === 'reference-only' ? 'reference-only' : 'manual';
}

function syncExpressionModeUi() {
  const mode = getExpressionMode();
  backendState.expressionMode = mode;
  const isReferenceOnly = mode === 'reference-only';
  const applyBtn = $('#btnApplyExpr');
  if (applyBtn && !applyBtn.disabled) {
    applyBtn.innerHTML = isReferenceOnly
      ? '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3" /></svg> Apply Reference'
      : '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3" /></svg> Apply Warp';
  }
  $$('#page-expression .param-slider').forEach(slider => {
    slider.disabled = isReferenceOnly;
  });
}

if ($('#sExprMode')) $('#sExprMode').addEventListener('change', () => {
  syncExpressionModeUi();
  toast(getExpressionMode() === 'reference-only'
    ? 'Reference Only mode enabled. Upload a reference face, then apply.'
    : 'Manual slider mode enabled.',
    'info');
});

/* ═══════════════════ RADIO CARDS ═══════════════════ */
const radioRGB = $('#radioRGB'), radioGray = $('#radioGray');
[radioRGB, radioGray].forEach(r => {
  if (r) r.addEventListener('click', () => {
    radioRGB.classList.remove('active-radio');
    radioGray.classList.remove('active-radio');
    r.classList.add('active-radio');
  });
});

/* ═══════════════════ EXPORT CSV (landmarks demo) ═══════════════════ */
if ($('#btnExportCSV')) $('#btnExportCSV').addEventListener('click', () => {
  let rows = [];
  if (backendState.lastLmData && Array.isArray(backendState.lastLmData.landmarks) && backendState.lastLmData.landmarks.length > 0) {
    rows = backendState.lastLmData.landmarks.map((p, i) => `${p.id ?? i},${p.x},${p.y}`);
  } else {
    rows = Array.from($$('#ptsG circle')).map((c, i) => `${i},${c.getAttribute('cx')},${c.getAttribute('cy')}`);
  }
  const csv = 'index,x,y\n' + rows.join('\n');
  download('landmarks.csv', csv, 'text/csv');
  toast('Landmark CSV exported.', 'success');
});

if ($('#btnExportJSON')) $('#btnExportJSON').addEventListener('click', () => {
  let points = [];
  if (backendState.lastLmData && Array.isArray(backendState.lastLmData.landmarks) && backendState.lastLmData.landmarks.length > 0) {
    points = backendState.lastLmData.landmarks.map((p, i) => ({
      index: p.id ?? i,
      x: Number(p.x),
      y: Number(p.y),
      method: backendState.lastLmData.method || '',
    }));
  } else {
    points = Array.from($$('#ptsG circle')).map((c, i) => ({
      index: i,
      x: Number(c.getAttribute('cx')),
      y: Number(c.getAttribute('cy')),
      method: 'svg-fallback',
    }));
  }
  download('landmarks.json', JSON.stringify({ pointCount: points.length, points }, null, 2), 'application/json');
  toast('Landmark JSON exported.', 'success');
});

if ($('#btnRunPP')) $('#btnRunPP').addEventListener('click', async function () {
  if (!appState.imageDataUrl) { toast('Upload an image first.', 'warning'); return; }
  const btn = this;
  btn.disabled = true;
  btn.textContent = '⧗ Processing…';
  try {
    const blob = await getCurrentImageBlob();
    const size = parseInt($('#selectResize')?.value || '256', 10);
    const grayscale = Boolean($('#radioGray')?.classList.contains('active-radio'));
    // Extra controls — send all UI values to backend
    const confidence = Number($('#sliderConf')?.value || 85) / 100;
    const padding = Number($('#sliderPad')?.value || 20);
    const normVal = $('#segNorm .seg-btn.active')?.dataset.val || '1';
    const interpVal = $('#segInterp .seg-btn.active')?.dataset.val || 'bicubic';
    const detectorVal = $('#segDetector .seg-btn.active')?.dataset.val || 'hog';
    const fd = new FormData();
    fd.append('image', blob, appState.fileName || 'input.jpg');
    fd.append('size', String(size));
    fd.append('grayscale', String(grayscale));
    fd.append('confidence', String(confidence));
    fd.append('padding', String(padding));
    fd.append('norm', normVal);
    fd.append('interp', interpVal);
    fd.append('detector', detectorVal);
    const res = await fetch('/api/preprocess', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || 'Preprocessing failed.');

    // ── Update stat displays ──
    const sizeLabel = `${data.width}×${data.height}`;
    const colorLabel = data.grayscale ? 'Gray' : 'RGB';
    const normLabel = normVal === '255' ? '0 – 255' : normVal === 'bipolar' ? '−1 – 1' : '0 – 1';
    if ($('#ppOutSize')) $('#ppOutSize').textContent = sizeLabel;
    if ($('#ppOutNorm')) $('#ppOutNorm').textContent = normLabel;
    if ($('#ppOutColor')) $('#ppOutColor').textContent = colorLabel;
    if ($('#ppStatRes')) $('#ppStatRes').textContent = sizeLabel;
    if ($('#ppStatColor')) $('#ppStatColor').textContent = colorLabel;

    // ── Show result image in the hero face wrap ──
    if (data.saved) {
      const heroWrap = document.querySelector('#page-preprocess .hero-face-wrap');
      if (heroWrap) {
        let resultImg = heroWrap.querySelector('#ppResultImg');
        if (!resultImg) {
          resultImg = document.createElement('img');
          resultImg.id = 'ppResultImg';
          resultImg.alt = 'Preprocessed result';
          resultImg.style.cssText = [
            'position:absolute', 'inset:0', 'width:100%', 'height:100%',
            'object-fit:cover', 'border-radius:inherit', 'z-index:2',
            data.grayscale ? 'filter:grayscale(1)' : '',
          ].filter(Boolean).join(';');
          heroWrap.style.position = 'relative';
          heroWrap.appendChild(resultImg);
        } else {
          resultImg.style.filter = data.grayscale ? 'grayscale(1)' : '';
        }
        resultImg.src = '/' + data.saved + '?t=' + Date.now(); // cache-bust
      }
    }

    // ── Carry preprocessed image forward so next pipeline steps use it ──
    if (data.saved) {
      try {
        const ppUrl = '/' + data.saved + '?t=' + Date.now();
        const ppResp = await fetch(ppUrl);
        const ppBlob = await ppResp.blob();
        const ppDataUrl = await new Promise(resolve => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.readAsDataURL(ppBlob);
        });
        setPipelineImage(ppDataUrl, {
          fileName: appState.fileName,
          format: data.grayscale ? 'Grayscale' : appState.format,
          sizeText: appState.sizeText,
          dimensions: `${data.width} × ${data.height}`,
        });
        backendState.landmarksLoaded = false;
        backendState.lastExprDataUrl = '';
        backendState.lastAgingDataUrl = '';
        backendState.analysisLoaded = false;
      } catch (_ppErr) { /* non-critical: pipeline still works with original */ }
    }

    toast(`Preprocessed → ${data.width}×${data.height}${data.grayscale ? ' grayscale' : ' color'} · ${data.durationMs} ms`, 'success');
  } catch (err) {
    toast('Preprocessing error: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Pipeline';
  }
});

if ($('#btnResetPP')) $('#btnResetPP').addEventListener('click', () => {
  const conf = $('#sliderConf');
  const pad = $('#sliderPad');
  if (conf) {
    conf.value = 85;
    $('#confVal').textContent = '0.85';
  }
  if (pad) {
    pad.value = 20;
    $('#padVal').textContent = '20%';
  }
  const res = $('#selectResize');
  if (res) res.value = '256';
  const normBtns = $('#segNorm')?.querySelectorAll('.seg-btn') || [];
  normBtns.forEach(btn => btn.classList.remove('active'));
  const normDefault = $('#segNorm .seg-btn[data-val="1"]');
  if (normDefault) normDefault.classList.add('active');
  // Also reset the seg buttons for detector and interp
  ['#segDetector', '#segInterp'].forEach(sel => {
    const grp = $(sel);
    if (!grp) return;
    grp.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    grp.querySelector('.seg-btn')?.classList.add('active');
  });
  // Reset radio cards to RGB
  $('#radioRGB')?.classList.add('active-radio');
  $('#radioGray')?.classList.remove('active-radio');
  // Remove result image overlay
  document.querySelector('#page-preprocess #ppResultImg')?.remove();
  // Reset stat labels
  if ($('#ppOutSize')) $('#ppOutSize').textContent = '256×256';
  if ($('#ppOutNorm')) $('#ppOutNorm').textContent = '0 – 1';
  if ($('#ppOutColor')) $('#ppOutColor').textContent = 'RGB';
  if ($('#ppStatRes')) $('#ppStatRes').textContent = '256×256';
  if ($('#ppStatColor')) $('#ppStatColor').textContent = 'RGB';
  toast('Preprocessing controls reset.', 'info');
});

/* ═══════════════════ IMAGE EXPORTS (UI) ═══════════════════ */
function drawSvgToCanvas(svgEl, done) {
  if (!svgEl) return;
  const rect = svgEl.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width || svgEl.viewBox.baseVal.width || 800));
  const height = Math.max(1, Math.round(rect.height || svgEl.viewBox.baseVal.height || 600));
  const clone = svgEl.cloneNode(true);
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  if (!clone.getAttribute('width')) clone.setAttribute('width', String(width));
  if (!clone.getAttribute('height')) clone.setAttribute('height', String(height));

  const svgData = new XMLSerializer().serializeToString(clone);
  const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);
  const img = new Image();

  img.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0b1220';
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    URL.revokeObjectURL(url);
    done(canvas);
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    toast('Image export failed.', 'error');
  };
  img.src = url;
}

/* ═══════════════════ AGING VARIANT THUMBNAILS ═══════════════════ */
function _applyAgingVariantThumbs(variants, ageBefore, ageAfter) {
  const cols = document.querySelectorAll('#methodCompareCard .mcg-col');
  if (!cols || cols.length < 3) return;

  const injectThumb = (col, src) => {
    const body = col.querySelector('.mcg-body');
    if (!body) return;
    body.style.position = 'relative';
    let img = col.querySelector('img.mcg-thumb');
    if (!img) {
      img = document.createElement('img');
      img.className = 'mcg-thumb';
      img.alt = '';
      img.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:8px;z-index:1;';
      body.appendChild(img);
    }
    img.src = src;
  };

  // Original column
  if (appState.imageDataUrl) injectThumb(cols[0], appState.imageDataUrl);
  const origBadge = cols[0].querySelector('.age-badge strong');
  if (origBadge && ageBefore != null) origBadge.textContent = ageBefore;

  // Frequency column
  if (variants.frequency) {
    injectThumb(cols[1], `data:image/jpeg;base64,${variants.frequency}`);
    const freqBadge = document.querySelector('#ageFreqComp');
    if (freqBadge && ageAfter != null) freqBadge.textContent = ageAfter;
  }

  // AI column
  if (variants.ai) {
    injectThumb(cols[2], `data:image/jpeg;base64,${variants.ai}`);
    const aiBadge = document.querySelector('#ageAiComp');
    if (aiBadge && ageAfter != null) aiBadge.textContent = ageAfter;
  }
}

function saveSvgAsPng(svgEl, fileName) {
  drawSvgToCanvas(svgEl, canvas => {
    canvas.toBlob(blob => {
      if (!blob) {
        toast('Image export failed.', 'error');
        return;
      }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = fileName;
      a.click();
      URL.revokeObjectURL(a.href);
      toast('Image saved to downloads.', 'success');
    }, 'image/png');
  });
}

if ($('#btnSaveExpr')) $('#btnSaveExpr').addEventListener('click', () => {
  const cvs = document.getElementById('exprTransformedCanvas');
  if (cvs && cvs.style.display !== 'none' && cvs.width > 0) {
    cvs.toBlob(blob => {
      if (!blob) { toast('Export failed.', 'error'); return; }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'expression-result.png';
      a.click();
      URL.revokeObjectURL(a.href);
      toast('Expression result saved.', 'success');
    }, 'image/png');
  } else {
    saveSvgAsPng($('#exprSvg'), 'expression-result.png');
  }
});

if ($('#btnSaveAging')) $('#btnSaveAging').addEventListener('click', () => {
  if (backendState.lastAgingDataUrl) {
    const a = document.createElement('a');
    a.href = backendState.lastAgingDataUrl;
    a.download = 'aging-result.jpg';
    a.click();
    toast('Aging result saved.', 'success');
    return;
  }
  const agingSvg = $('#page-aging .aging-side-card-active .face-svg');
  saveSvgAsPng(agingSvg, 'aging-result.png');
});

if ($('#btnCompareAging')) $('#btnCompareAging').addEventListener('click', () => {
  const compareCard = $('#methodCompareCard');
  if (!compareCard) return;
  compareCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  toast('Scrolled to comparison panel.', 'info');
});

/* ═══════════════════ EXPRESSION TRANSFER ═══════════════════ */
async function getReferenceImageBlob() {
  const file = refInput?.files?.[0];
  if (file) return file;
  const refSrc = refImg?.src;
  if (!refSrc || refSrc === window.location.href) return null;
  const refResp = await fetch(refSrc);
  return refResp.blob();
}

async function runReferenceExpressionTransfer(buttonEl) {
  if (!appState.imageDataUrl) { toast('Upload a target image first.', 'warning'); return; }
  const refBlob = await getReferenceImageBlob();
  if (!refBlob) { toast('Upload a reference expression image first.', 'warning'); return; }

  const btn = buttonEl;
  const prev = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = '\u29d7 Transferring…';

  try {
    const targetBlob = await getCurrentImageBlob();
    if (!targetBlob) throw new Error('No target image available.');
    const fd = new FormData();
    fd.append('target', targetBlob, appState.fileName || 'target.jpg');
    fd.append('reference', refBlob, 'reference.jpg');
    fd.append('strength', '0.6');

    const res = await fetch('/api/expression-transfer', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || 'Expression transfer failed.');

    const outUrl = `data:image/jpeg;base64,${data.image}`;
    backendState.lastExprDataUrl = outUrl;
    backendState.lastAgingDataUrl = '';
    backendState.analysisLoaded = false;
    backendState.analysisData = null;
    backendState.analysisSpectra = null;
    renderExpressionCanvasFromDataUrl(outUrl);
    setReferenceTransferReadout(data);
    toast(`Expression transferred \u00b7 ${data.durationMs || 0} ms \u00b7 ${data.triangleCount ?? '?'} tri`, 'success');
  } catch (err) {
    toast(`Transfer error: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = prev;
    syncExpressionModeUi();
  }
}

if ($('#btnApplyTransfer')) $('#btnApplyTransfer').addEventListener('click', function () {
  runReferenceExpressionTransfer(this);
});

/* ═══════════════════ APPLY WARP FEEDBACK ═══════════════════ */
if ($('#btnApplyExpr')) $('#btnApplyExpr').addEventListener('click', async function () {
  if (!appState.imageDataUrl) { toast('Upload an image first.', 'warning'); return; }
  if (getExpressionMode() === 'reference-only') {
    await runReferenceExpressionTransfer(this);
    return;
  }

  const btn = this;
  const prevText = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = '⧗ Applying...';

  try {
    const blob = await getCurrentImageBlob();
    if (!blob) throw new Error('No input image available.');

    const fd = new FormData();
    fd.append('image', blob, appState.fileName || 'input.jpg');
    fd.append('smile', String(Number($('#sSmile')?.value || 0)));
    fd.append('brow', String(Number($('#sBrow')?.value || 0)));
    fd.append('lip', String(Number($('#sLip')?.value || 0)));
    fd.append('slim', String(Number($('#sSlim')?.value || 0)));
    fd.append('face_fullness', String(Number($('#sFaceFullness')?.value || 0)));
    fd.append('eye_size', String(Number($('#sEyeSize')?.value || 0)));
    fd.append('show_grid', ($('#toggleMesh')?.checked) ? '1' : '0');
    fd.append('mode', $('#sWarpMethod')?.value || 'delaunay');
    fd.append('interpolation', 'bicubic');
    fd.append('debug', 'false');

    const res = await fetch('/api/transform', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || !data.ok || !data.image) {
      throw new Error(data.error || 'Expression transform failed.');
    }

    const outUrl = `data:image/jpeg;base64,${data.image}`;
    backendState.lastExprDataUrl = outUrl;
    backendState.lastAgingDataUrl = '';
    backendState.analysisLoaded = false;
    backendState.analysisData = null;
    backendState.analysisSpectra = null;
    renderExpressionCanvasFromDataUrl(outUrl);
    const volumeTag = data.facialVolumeApplied ? ' · Facial Volume (Non-Linear)' : '';
    toast(`Warped · ${data.method || 'delaunay'} · ${data.usedLandmarkCount ?? '?'} pts · moved ${data.movedPoints ?? '?'}${volumeTag} · maxD ${data.maxDisplacement != null ? Number(data.maxDisplacement).toFixed(1) : '?'} · ${data.triangleCount ?? '?'} tri · ${data.durationMs || 0} ms`, 'success');
  } catch (err) {
    toast(`Expression transform error: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = prevText;
    syncExpressionModeUi();
  }
});

if ($('#btnSimulate')) $('#btnSimulate').addEventListener('click', async function () {
  if (!appState.imageDataUrl) {
    toast('Upload an image first.', 'warning');
    return;
  }

  const btn = this;
  const prev = btn.textContent;
  btn.textContent = '⧗ Processing...';
  btn.disabled = true;

  try {
    const srcBlob = await buildTransformedBlobForAnalysis() || await getCurrentImageBlob();
    if (!srcBlob) throw new Error('No source image found for aging.');

    const selectedMethod = $('#ageMethod .seg-btn.active')?.dataset.v || 'freq';
    const fd = new FormData();
    fd.append('image', srcBlob, 'aging-input.jpg');
    fd.append('mode', $('#ageMode .seg-btn.active')?.dataset.v || 'aging');
    fd.append('method', selectedMethod);
    fd.append('years', String(Number($('#sTargetAge')?.value || 10)));
    fd.append('strength', String(Number($('#sAge')?.value || 50) / 100));
    fd.append('contrast', String(Number($('#sContrast')?.value || 50) / 100));
    fd.append('tone', String(Number($('#sTone')?.value || 50) / 100));
    fd.append('noise', String(Number($('#sNoise')?.value || 0) / 100));
    fd.append('detail', String(Number($('#sDetail')?.value || 50) / 100));
    fd.append('deterministic', 'true');
    fd.append('hpf', $('#selHPF')?.value || 'gaussian_hpf');
    fd.append('lpf', $('#selLPF')?.value || 'gaussian_lpf');

    const agingEndpoint = (selectedMethod === 'ai') ? '/api/aging-ai' : '/api/aging';
    console.log('Selected aging method:', selectedMethod);
    console.log('Aging endpoint:', agingEndpoint);
    const res = await fetch(agingEndpoint, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || !data.ok || !data.image) {
      throw new Error(data.error || 'Aging simulation failed.');
    }

    backendState.lastAgingDataUrl = `data:image/jpeg;base64,${data.image}`;
    backendState.analysisLoaded = false;
    backendState.analysisData = null;
    syncAgingPhotoPanels();

    // ── Update apparent age from backend estimate ──
    if (data.apparentAgeBefore != null) {
      const elOrig = $('#ageOriginal');
      if (elOrig) elOrig.textContent = data.apparentAgeBefore;
    }
    if (data.apparentAgeAfter != null) {
      const elSim = $('#ageTransformed');
      if (elSim) elSim.textContent = data.apparentAgeAfter;
      const diff = (data.apparentAgeAfter || 0) - (data.apparentAgeBefore || 28);
      const elOffset = $('#ageOffsetLabel');
      if (elOffset) elOffset.textContent = diff >= 0 ? `+${diff} Years` : `${diff} Years`;
    }
    // ── Update method compare thumbnails ──
    if (data.variants) _applyAgingVariantThumbs(data.variants, data.apparentAgeBefore, data.apparentAgeAfter);

    toast(`Aging simulation applied (${data.durationMs || 0} ms).`, 'success');
    fetchAnalysisFromBackend(true);
  } catch (err) {
    toast(`Aging simulation error: ${err.message}`, 'error');
  } finally {
    btn.textContent = prev || 'Apply Aging Effect';
    btn.disabled = false;
  }
});

if ($('#btnAnalysisCsv')) $('#btnAnalysisCsv').addEventListener('click', () => {
  if (!backendState.analysisData) {
    toast('Run analysis before exporting CSV.', 'warning');
    return;
  }
  const rows = [
    ['metric', 'value'],
    ['MSE', ($('#qmMseVal')?.textContent || '').trim()],
    ['PSNR', ($('#qmPsnrVal')?.textContent || '').trim()],
    ['SSIM', ($('#qmSsimVal')?.textContent || '').trim()],
    ['HF/LF', ($('#enRatio')?.textContent || '').trim()],
  ];
  const csv = rows.map(r => r.join(',')).join('\n');
  download('analysis-metrics.csv', csv, 'text/csv');
  toast('Analysis CSV exported.', 'success');
});

if ($('#btnAnalysisPdf')) $('#btnAnalysisPdf').addEventListener('click', () => {
  if (!backendState.analysisData) {
    toast('Run analysis before exporting PDF.', 'warning');
    return;
  }
  const lines = [
    'FaceWarp DSP - Analysis Report',
    `Timestamp: ${new Date().toLocaleString()}`,
    '',
    `MSE: ${($('#qmMseVal')?.textContent || '').trim()}`,
    `PSNR: ${($('#qmPsnrVal')?.textContent || '').trim()}`,
    `SSIM: ${($('#qmSsimVal')?.textContent || '').trim()}`,
    `HF/LF Ratio: ${($('#enRatio')?.textContent || '').trim()}`,
  ];
  downloadSimplePdf('analysis-report.pdf', lines);
  toast('Analysis PDF exported.', 'success');
});

const specMode = $('#specMode');
if (specMode) {
  specMode.querySelectorAll('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const isPhase = btn.dataset.v === 'phase';
      backendState.spectrumMode = isPhase ? 'phase' : 'mag';
      renderSpectrumImages(backendState.spectrumMode);
      const titles = $$('#page-analysis .spec-title');
      if (titles[0]) titles[0].textContent = isPhase ? 'Original (Phase)' : 'Original';
      if (titles[1]) titles[1].textContent = isPhase ? 'Transformed (Phase)' : 'Transformed';
      toast(isPhase ? 'Phase spectrum view enabled.' : 'Magnitude spectrum view enabled.', 'info');
    });
  });
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function setRingProgress(el, percent) {
  if (!el) return;
  const circumference = 263.9;
  const p = clamp(percent, 0, 100);
  el.setAttribute('stroke-dasharray', `${(circumference * p / 100).toFixed(1)} ${circumference}`);
}

let _analysisDebounceTimer = null;

function updateAnalysisMetrics() {
  if (!appState.imageDataUrl) {
    clearAnalysisMetrics('Upload an image to compute analysis.');
    return;
  }

  // Real image loaded — always fetch from backend, never overwrite with fake values.
  if (!backendState.lastExprDataUrl && !backendState.lastAgingDataUrl) {
    clearAnalysisMetrics('Run Expression Warp or Aging first.');
    return;
  }
  backendState.analysisLoaded = false;
  clearTimeout(_analysisDebounceTimer);
  _analysisDebounceTimer = setTimeout(() => fetchAnalysisFromBackend(true), 700);
}

setPipelineImage('');
syncExpressionModeUi();
updateAnalysisMetrics();
syncMethodCompareVisibility();

function download(name, content, type) {
  const blob = new Blob([content], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function pdfEscape(text) {
  return String(text).replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)');
}

function downloadSimplePdf(fileName, lines) {
  const streamLines = (lines || []).map((line, i) => {
    const y = 800 - i * 18;
    return `BT /F1 12 Tf 50 ${y} Td (${pdfEscape(line)}) Tj ET`;
  }).join('\n');

  const obj1 = '1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n';
  const obj2 = '2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n';
  const obj3 = '3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >> endobj\n';
  const stream = `${streamLines}\n`;
  const obj4 = `4 0 obj << /Length ${stream.length} >> stream\n${stream}endstream endobj\n`;
  const obj5 = '5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n';

  const parts = ['%PDF-1.4\n', obj1, obj2, obj3, obj4, obj5];
  const offsets = [];
  let cursor = 0;
  for (const part of parts) {
    offsets.push(cursor);
    cursor += part.length;
  }

  const xrefStart = cursor;
  const xref = [
    'xref\n',
    '0 6\n',
    '0000000000 65535 f \n',
    `${String(offsets[1]).padStart(10, '0')} 00000 n \n`,
    `${String(offsets[2]).padStart(10, '0')} 00000 n \n`,
    `${String(offsets[3]).padStart(10, '0')} 00000 n \n`,
    `${String(offsets[4]).padStart(10, '0')} 00000 n \n`,
    `${String(offsets[5]).padStart(10, '0')} 00000 n \n`,
    'trailer << /Size 6 /Root 1 0 R >>\n',
    `startxref\n${xrefStart}\n%%EOF`,
  ].join('');

  const pdfContent = parts.join('') + xref;
  download(fileName, pdfContent, 'application/pdf');
}

/* ═══════════════════ BUTTON RIPPLE ═══════════════════ */
$$('.btn-primary, .btn-outline').forEach(btn => {
  btn.addEventListener('click', function (e) {
    const ripple = document.createElement('span');
    const rect = this.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    ripple.style.cssText = `
      position:absolute;border-radius:50%;pointer-events:none;
      width:${size}px;height:${size}px;
      left:${e.clientX - rect.left - size / 2}px;
      top:${e.clientY - rect.top - size / 2}px;
      background:rgba(255,255,255,.12);
      transform:scale(0);animation:rippleOut .6s ease-out forwards;
    `;
    this.appendChild(ripple);
    setTimeout(() => ripple.remove(), 650);
  });
});

/* inject ripple keyframes */
const rippleStyle = document.createElement('style');
rippleStyle.textContent = '@keyframes rippleOut{to{transform:scale(2.5);opacity:0}}';
document.head.appendChild(rippleStyle);

/* ═══════════════════ SMOOTH CARD REVEAL ═══════════════════ */
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

function initReveal() {
  document.querySelectorAll('.page.active .card, .page.active .energy-grid, .page.active .hint-cards').forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(18px)';
    el.style.transition = `opacity .5s ${i * .06}s var(--ease), transform .5s ${i * .06}s var(--ease)`;
    revealObserver.observe(el);
  });
}

/* re-init on page nav */
const origNav = nav;
nav = function (page) {
  origNav(page);
  updatePipelineStrip(page);
  if (page === 'landmarks') { fetchLandmarksFromBackend(); setTimeout(syncLandmarkPhotoBg, 60); }
  if (page === 'expression') setTimeout(syncExpressionPhotoPanels, 80);
  if (page === 'aging') setTimeout(syncAgingPhotoPanels, 60);
  if (page === 'analysis') fetchAnalysisFromBackend(true);
  window.scrollTo({ top: 0, behavior: 'smooth' });
  requestAnimationFrame(() => requestAnimationFrame(initReveal));
};

/* init on load */
requestAnimationFrame(initReveal);

/* ═══════════════════ SLIDER TRACK FILL ═══════════════════ */
function updateSliderFill(s) {
  if (s.classList.contains('slider-bipolar')) {
    const min = +s.min, max = +s.max, val = +s.value;
    const center = ((0 - min) / (max - min)) * 100;
    const pos = ((val - min) / (max - min)) * 100;
    const left = Math.min(center, pos);
    const right = Math.max(center, pos);
    const color = val >= 0 ? 'rgba(139,92,246,.5)' : 'rgba(6,182,212,.45)';
    s.style.background =
      `linear-gradient(90deg, var(--bg2) ${left}%, ${color} ${left}%, ${color} ${right}%, var(--bg2) ${right}%)`;
  } else {
    const pct = ((s.value - s.min) / (s.max - s.min)) * 100;
    s.style.background = `linear-gradient(90deg, rgba(139,92,246,.4) ${pct}%, var(--bg2) ${pct}%)`;
  }
}
$$('.slider').forEach(s => {
  updateSliderFill(s);
  s.addEventListener('input', () => updateSliderFill(s));
});
