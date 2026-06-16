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
  landmarkSchema: '68',  // '68' or '468' for dynamic schema toggle
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

function createDynamicLandmarkCircles(count) {
  const ptsG = $('#ptsG');
  if (!ptsG) return;
  ptsG.innerHTML = '';
  
  const colors = {
    jaw: '#8b5cf6', brow: '#c4b5fd', eyes: '#06b6d4',
    nose: '#f59e0b', mouth: '#f472b6',
  };
  
  for (let i = 0; i < count; i += 1) {
    let region = 'jaw';
    if (i >= 17 && i <= 26) region = 'brow';
    else if (i >= 36 && i <= 47) region = 'eyes';
    else if (i >= 27 && i <= 35) region = 'nose';
    else if (i >= 48 && i <= 67) region = 'mouth';
    
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', '150');
    circle.setAttribute('cy', '150');
    circle.setAttribute('r', '2.8');
    circle.setAttribute('fill', colors[region] || '#8b5cf6');
    circle.setAttribute('class', 'lm-pt');
    circle.setAttribute('data-idx', String(i));
    if (i !== 0 && i !== 8 && i !== 16 && i !== 25 && i !== 30) {
      circle.setAttribute('opacity', '0.82');
    }
    ptsG.appendChild(circle);
  }
}

function switchLandmarkSchema(schema) {
  if (!backendState.lastLmData) {
    toast('No landmark data cached. Run detection first.', 'warning');
    return;
  }
  
  backendState.landmarkSchema = schema;
  
  const landmarkSource = schema === '468' ? backendState.lastLmData.rawLandmarks : backendState.lastLmData.reduced68 || backendState.lastLmData.landmarks;
  const meshSource = schema === '468' ? backendState.lastLmData.rawLandmarks : backendState.lastLmData.reduced68 || backendState.lastLmData.landmarks;
  
  const targetCount = schema === '468' ? (landmarkSource?.length || 468) : 68;
  createDynamicLandmarkCircles(targetCount);
  
  const points = (landmarkSource || []).map(p => ({
    x: p.x,
    y: p.y,
    imageWidth: backendState.lastLmData.width,
    imageHeight: backendState.lastLmData.height,
  }));
  
  applyLandmarksToSvg(points, points);

  // 468-pt mode: hide dlib connection paths (they use 68-pt indices and produce wrong lines)
  // 68-pt mode: restore them
  const connG = $('#connG');
  if (connG) connG.style.display = (schema === '468') ? 'none' : '';

  const toolbar = $('#lmToolbarRight');
  if (toolbar) toolbar.textContent = `${targetCount} pts`;
  // Update Region Breakdown badge and Total count
  if ($('#lmRegionBadge')) $('#lmRegionBadge').textContent = `${targetCount} pts`;
  if ($('#lmTotalCount')) $('#lmTotalCount').textContent = targetCount;
  if ($('#lmPtsCount')) $('#lmPtsCount').textContent = targetCount;
  if ($('#lmStatPts')) $('#lmStatPts').textContent = targetCount;
  // Sync all #segSchema button states
  $$('#segSchema .seg-btn').forEach(b => b.classList.toggle('active', b.getAttribute('data-val') === schema));
  toast(`Switched to ${schema}-point schema (${targetCount} landmarks)`, 'success');
}

function applyLandmarksToSvg(points68, meshPoints = points68) {
  const svg = $('#lmSvg');
  let circles = Array.from($$('#ptsG circle'));
  if (!svg || circles.length === 0) {
    // Create at least 68 circles if none exist
    createDynamicLandmarkCircles(68);
    circles = Array.from($$('#ptsG circle'));
    if (!circles.length) return;
  }

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

  // No sampling when using exact schema; match points to circles
  const sampled = circles.length === (points68?.length || 0) ? points68 : sampleLandmarks(points68, circles.length);
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
  
  // Update schema button states
  updateSchemaButtonUI();
}

function updateSchemaButtonUI() {
  const schemaButtons = $$('#segSchema .seg-btn');
  schemaButtons.forEach(btn => {
    const val = btn.getAttribute('data-val');
    btn.classList.toggle('active', val === backendState.landmarkSchema);
  });
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

  // Match canvas aspect ratio to the photo so it scales proportionally to the
  // available column width instead of being stretched flat by object-fit:fill.
  // Landmark math uses the bg image's bounding rect, which is now correctly
  // shaped, so overlay alignment is preserved.
  if (has) {
    const probe = new Image();
    probe.onload = () => {
      const w = probe.naturalWidth;
      const h = probe.naturalHeight;
      if (w > 0 && h > 0) {
        lmCanvas.style.aspectRatio = `${w} / ${h}`;
        lmCanvas.style.minHeight = '0';
        if (backendState.landmarksLoaded && backendState.lastLmData) {
          const data = backendState.lastLmData;
          const src68 = (Array.isArray(data.reduced68) && data.reduced68.length) ? data.reduced68 : data.landmarks;
          const srcMesh = (Array.isArray(data.rawLandmarks) && data.rawLandmarks.length) ? data.rawLandmarks : src68;
          const p68 = (src68 || []).map(p => ({ x: p.x, y: p.y, imageWidth: data.width, imageHeight: data.height }));
          const pMesh = (srcMesh || []).map(p => ({ x: p.x, y: p.y, imageWidth: data.width, imageHeight: data.height }));
          applyLandmarksToSvg(p68, pMesh);
        }
      }
    };
    probe.src = appState.imageDataUrl;
  } else {
    lmCanvas.style.aspectRatio = '';
    lmCanvas.style.minHeight = '';
  }
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

/* Model toggle (visual-only) */
const segModel = $('#segModel');
if (segModel) {
  segModel.querySelectorAll('.seg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      segModel.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}
/* Schema toggle — all instances (toolbar + sidebar) — actually switches rendering */
$$('#segSchema').forEach(segSchema => {
  segSchema.querySelectorAll('.seg-btn').forEach(btn => {
    // Skip buttons that already have inline onclick handlers
    if (!btn.getAttribute('onclick')) {
      btn.addEventListener('click', () => {
        switchLandmarkSchema(btn.getAttribute('data-val'));
      });
    }
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
  { id: 'sLipTint', valId: 'vLipTint' },
  { id: 'sBlush', valId: 'vBlush' },
];
sliders.forEach(({ id, valId }) => {
  const s = $(`#${id}`), v = $(`#${valId}`);
  if (s && v) s.addEventListener('input', () => {
    v.textContent = s.value + '%';
    s.style.setProperty('--pct', s.value + '%');
    updateExprSvg();
  });
});

/* ═══════════════════ SMOOTH ACCORDION ═══════════════════ */
function initAccordions() {
  document.querySelectorAll('details.acc-section').forEach(details => {
    const summary = details.querySelector('summary');
    const body = details.querySelector('.acc-body');
    if (!summary || !body) return;

    // Prevent double-fire on repeat init
    if (summary._accBound) return;
    summary._accBound = true;

    summary.addEventListener('click', e => {
      e.preventDefault();
      if (!details.open) {
        // ── Opening ───────────────────────────────────
        details.open = true;
        body.style.overflow = 'hidden';
        body.style.maxHeight = '0px';
        body.style.opacity = '0';
        body.style.transition = 'max-height 0.32s ease-in-out, opacity 0.28s ease-in-out';
        requestAnimationFrame(() => requestAnimationFrame(() => {
          body.style.maxHeight = body.scrollHeight + 'px';
          body.style.opacity = '1';
        }));
        setTimeout(() => {
          body.style.maxHeight = '';
          body.style.overflow = '';
          body.style.transition = '';
        }, 340);
      } else {
        // ── Closing ───────────────────────────────────
        body.style.maxHeight = body.scrollHeight + 'px';
        body.style.overflow = 'hidden';
        body.style.transition = 'max-height 0.28s ease-in-out, opacity 0.22s ease-in-out';
        body.style.opacity = '1';
        requestAnimationFrame(() => requestAnimationFrame(() => {
          body.style.maxHeight = '0px';
          body.style.opacity = '0';
        }));
        setTimeout(() => {
          details.open = false;
          body.style.maxHeight = '';
          body.style.overflow = '';
          body.style.opacity = '';
          body.style.transition = '';
        }, 300);
      }
    });
  });
}
initAccordions();

/* ═══════════════════ TECHNICAL METRICS PANEL ═══════════════════ */
function updateExprMetrics(data) {
  // Reset all cards before updating
  ['mcDuration', 'mcMoved', 'mcMaxD', 'mcTri', 'mcMethod'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('has-data');
  });

  const setM = (valId, cardId, val) => {
    const el = document.getElementById(valId);
    const card = document.getElementById(cardId);
    if (el) el.textContent = val;
    if (card && val !== '—') card.classList.add('has-data');
  };
  setM('mcDurationVal', 'mcDuration', data.durationMs != null ? data.durationMs + ' ms' : '—');
  setM('mcMovedVal', 'mcMoved', data.movedPoints != null ? String(data.movedPoints) : '—');
  setM('mcMaxDVal', 'mcMaxD', data.maxDisplacement != null ? Number(data.maxDisplacement).toFixed(1) + 'px' : '—');
  setM('mcTriVal', 'mcTri', data.triangleCount != null ? String(data.triangleCount) : '—');
  setM('mcMethodVal', 'mcMethod', data.method ? data.method.toUpperCase() : '—');

  // Color Processing indicator
  const colorDot = document.getElementById('mcColorDot');
  const colorVal = document.getElementById('mcColorVal');
  const colorCard = document.getElementById('mcColorStatus');
  const colorActive = !!data.colorProcessingActive;
  if (colorVal) colorVal.textContent = colorActive ? 'Active' : 'Inactive';
  if (colorDot) colorDot.classList.toggle('color-status-dot-on', colorActive);
  if (colorCard) colorCard.classList.toggle('color-status-active', colorActive);

  const stream = document.getElementById('exprLiveStream');
  if (stream) {
    // Remove placeholder row if present
    const placeholder = stream.querySelector('.ls-row .ls-ts');
    if (placeholder && placeholder.textContent === '—') stream.innerHTML = '';
    const now = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const row = document.createElement('div');
    row.className = 'ls-row';
    const volFlag = data.facialVolumeApplied ? ' \xb7vol' : '';
    const colorFlag = data.colorProcessingActive ? ' \xb7clr' : '';
    row.innerHTML = `<span class="ls-ts">${now}</span><span class="ls-val">moved=${data.movedPoints ?? '?'} maxD=${data.maxDisplacement != null ? Number(data.maxDisplacement).toFixed(1) : '?'}px${volFlag}${colorFlag} [${(data.method || '?').slice(0, 3).toUpperCase()}]</span>`;
    stream.insertBefore(row, stream.firstChild);
    while (stream.children.length > 18) stream.removeChild(stream.lastChild);
  }
}

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
const webcamBody = $('.webcam-body');
const accessoryCanvas = $('#accessoryCanvas');
const selectedFilterEls = {
  glasses: $('#selectedGlassesName'),
  hat: $('#selectedHatName'),
  earrings: $('#selectedEarringsName'),
  necklace: $('#selectedNecklaceName'),
  lipstick: $('#selectedLipstickName'),
  blush: $('#selectedBlushName'),
};
const catalogTitle = $('#catalogTitle');
const catalogBreadcrumb = $('#catalogBreadcrumb');
const catalogMenu = $('#catalogMenu');
const catalogBackBtn = $('#catalogBackBtn');
const catalogClearBtn = $('#catalogClearBtn');
const mirrorToggleBtn = $('#mirrorToggleBtn');
const mirrorStatus = $('#mirrorStatus');
const trackingFpsEl = $('#trackingFps');
const trackingLatencyEl = $('#trackingLatency');
const trackingAlphaEl = $('#trackingAlpha');
const trackingFreshnessEl = $('#trackingFreshness');

const GLASSES_WIDTH_FACTOR_WEB = 2.45;
const HAT_WIDTH_FACTOR_WEB = 1.88;
const HAT_Y_OFFSET_FACTOR_WEB = -0.33;
const HAT_BOTTOM_ALIGNMENT_WEB = 0.55;
const FAST_SMOOTHING_ALPHA = 0.92;
const SLOW_SMOOTHING_ALPHA = 0.72;
const LANDMARK_POLL_MS = 50;
const LANDMARK_STALE_MS = 600;
const LIPSTICK_OUTER_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
const LIPSTICK_INNER_LANDMARKS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const BLUSH_LEFT_LANDMARKS = [50, 117, 118, 119, 123, 187];
const BLUSH_RIGHT_LANDMARKS = [280, 346, 347, 348, 352, 411];
const LIPSTICK_OPACITY = 0.72;
const LIPSTICK_BLUR = 2;
const BLUSH_OPACITY = 0.58;
const BLUSH_BLUR = 14;
const BLUSH_X_OFFSET = 0.045;
const BLUSH_Y_OFFSET = 0.040;
const BLUSH_SCALE = 1.45;

const FILTER_CATALOG = {
  glasses: [
    { name: 'Leopard Cat Eye', src: '/assets/glasses_1.png', image: new Image() },
    { name: 'Steampunk', src: '/assets/glasses_2.png', image: new Image() },
    { name: 'Cat Eye', src: '/assets/glasses_3.png', image: new Image() },
    { name: 'Oval', src: '/assets/glasses_4.png', image: new Image() },
    { name: 'Square', src: '/assets/glasses_5.png', image: new Image() },
    { name: 'Shield', src: '/assets/glasses_6.png', image: new Image() },
  ],
  hat: [
    { name: 'Bucket Hat', src: '/assets/hat_1.png', image: new Image() },
    { name: 'Cowboy Hat', src: '/assets/hat_2.png', image: new Image() },
    { name: 'Straw Hat', src: '/assets/hat_3.png', image: new Image() },
    { name: 'Beret', src: '/assets/hat_5.png', image: new Image() },
    { name: 'Fedora', src: '/assets/hat_6.png', image: new Image() },
  ],
  earrings: [
    { name: 'Model 1', src: '/assets/earring_1.png', image: new Image() },
    { name: 'Model 2', src: '/assets/earring_2.png', image: new Image() },
    { name: 'Model 3', src: '/assets/earring_3.png', image: new Image() },
  ],
  necklace: [
    { name: 'Model 1', src: '/assets/necklace_1.png', image: new Image() },
    { name: 'Model 2', src: '/assets/necklace_2.png', image: new Image() },
    { name: 'Model 3', src: '/assets/necklace_3.png', image: new Image() },
  ],
  lipstick: [
    { name: 'Red', color: 'rgba(190, 24, 62, 0.90)', stroke: 'rgba(255, 210, 220, 0.40)' },
    { name: 'Pink', color: 'rgba(236, 72, 153, 0.85)', stroke: 'rgba(255, 220, 240, 0.38)' },
    { name: 'Nude', color: 'rgba(178, 112, 90, 0.80)', stroke: 'rgba(255, 235, 225, 0.35)' },
    { name: 'Berry', color: 'rgba(126, 34, 98, 0.90)', stroke: 'rgba(255, 210, 240, 0.38)' },
  ],
  blush: [
    { name: 'Pink', color: 'rgba(244, 114, 182, 0.90)' },
    { name: 'Peach', color: 'rgba(251, 146, 60, 0.88)' },
    { name: 'Coral', color: 'rgba(248, 113, 113, 0.90)' },
  ],
};

const CATEGORY_TREE = {
  accessories: {
    title: 'Accessories',
    subcategories: [
      { key: 'glasses', label: 'Glasses', hint: 'Eye-locked models' },
      { key: 'hat', label: 'Hats', hint: 'Top-head placement' },
      { key: 'earrings', label: 'Earrings', hint: 'Side-face tracking' },
      { key: 'necklace', label: 'Necklace', hint: 'Lower chin / neck' },
    ],
  },
  makeup: {
    title: 'Makeup',
    subcategories: [
      { key: 'lipstick', label: 'Lipstick', hint: 'Lip landmark color' },
      { key: 'blush', label: 'Blush', hint: 'Cheek color wash' },
    ],
  },
};

const SUBCATEGORY_LABELS = {
  glasses: 'Glasses',
  hat: 'Hats',
  earrings: 'Earrings',
  necklace: 'Necklace',
  lipstick: 'Lipstick',
  blush: 'Blush',
};

const accessoryState = {
  raf: null,
  timer: null,
  processing: false,
  measurements: null,
  smoothed: null,
  fastAlpha: FAST_SMOOTHING_ALPHA,
  slowAlpha: SLOW_SMOOTHING_ALPHA,
  loadedImages: 0,
  totalImages: 17,
  requestId: 0,
  appliedRequestId: 0,
  lastLatencyMs: null,
  lastLandmarkAt: 0,
  fps: 0,
  frameCount: 0,
  fpsLastAt: performance.now(),
  lastDebugUiAt: 0,
  menuLevel: 'main',
  category: null,
  subcategory: null,
  selected: {
    glasses: null,
    hat: null,
    earrings: null,
    necklace: null,
    lipstick: null,
    blush: null,
  },
  debugMakeup: false,
  mirror: false,
};

['glasses', 'hat', 'earrings', 'necklace'].forEach(kind => {
  FILTER_CATALOG[kind].forEach(item => {
    item.image.onload = () => { accessoryState.loadedImages += 1; };
    item.image.src = item.src;
  });
});

/* ── Client-side MediaPipe FaceLandmarker (zero-latency tracking) ────── */
let _mpLandmarker = null;
let _mpState = 'idle'; // idle | loading | ready | failed
let _mpLastDetectAt = 0;
const _MP_DETECT_INTERVAL_MS = 33; // ~30 FPS detection (render stays 60 FPS)

async function _initClientLandmarker() {
  if (_mpState !== 'idle') return;
  _mpState = 'loading';
  console.log('[Webcam] Loading client-side FaceLandmarker…');
  try {
    const vision = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/vision_bundle.mjs'
    );
    const fileset = await vision.FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm'
    );
    const modelResp = await fetch('/models/face_landmarker.task');
    if (!modelResp.ok) throw new Error('Model file not found at /models/face_landmarker.task');
    const modelBuf = await modelResp.arrayBuffer();
    _mpLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer: new Uint8Array(modelBuf), delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
    _mpState = 'ready';
    // Stop server polling — client-side is now active
    if (accessoryState.timer) { clearInterval(accessoryState.timer); accessoryState.timer = null; }
    console.log('[Webcam] ✓ Client-side FaceLandmarker ready — server polling stopped');
  } catch (err) {
    console.warn('[Webcam] Client-side init failed, using server fallback:', err);
    _mpState = 'failed';
  }
}

function _detectLocal() {
  if (_mpState !== 'ready' || !_mpLandmarker) return false;
  if (!webcamVideo || !webcamVideo.videoWidth || !webcamVideo.videoHeight) return false;

  const now = performance.now();
  if (now - _mpLastDetectAt < _MP_DETECT_INTERVAL_MS) return true; // throttled — keep last result
  _mpLastDetectAt = now;

  try {
    const results = _mpLandmarker.detectForVideo(webcamVideo, now);
    if (!results.faceLandmarks || results.faceLandmarks.length === 0) return false;

    const w = webcamVideo.videoWidth;
    const h = webcamVideo.videoHeight;
    const raw = results.faceLandmarks[0];
    const points = new Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      points[i] = { id: i, x: raw[i].x * w, y: raw[i].y * h };
    }

    const meas = getAccessoryMeasurements({ rawLandmarks: points, landmarks: points });
    if (!meas) return false;

    accessoryState.lastLatencyMs = performance.now() - now;
    accessoryState.lastLandmarkAt = performance.now();
    accessoryState.measurements = meas;
    accessoryState.smoothed = smoothAccessoryMeasurements(accessoryState.smoothed, meas);
    return true;
  } catch (_) { return false; }
}

function accessoriesReady() {
  return accessoryState.loadedImages >= accessoryState.totalImages;
}

function setCatalogView(level, category = null, subcategory = null) {
  accessoryState.menuLevel = level;
  accessoryState.category = category;
  accessoryState.subcategory = subcategory;
  renderTryOnCatalog();
}

function clearTryOnFilters() {
  Object.keys(accessoryState.selected).forEach(key => { accessoryState.selected[key] = null; });
  updateSelectedFilterLabels();
  renderTryOnCatalog();
}

function updateSelectedFilterLabels() {
  Object.entries(selectedFilterEls).forEach(([key, el]) => {
    if (!el) return;
    const selectedIndex = accessoryState.selected[key];
    const item = selectedIndex !== null ? FILTER_CATALOG[key]?.[selectedIndex] : null;
    el.textContent = item ? item.name : 'Off';
  });
  if (mirrorStatus) mirrorStatus.textContent = accessoryState.mirror ? 'On' : 'Off';
  if (mirrorToggleBtn) mirrorToggleBtn.textContent = accessoryState.mirror ? 'Mirror On' : 'Mirror Off';
  const debugEl = $('#makeupDebugStatus');
  if (debugEl) debugEl.textContent = accessoryState.debugMakeup ? 'Debug anchors on' : 'Debug anchors off';
}

function updateTrackingDebug(force = false) {
  const now = performance.now();
  if (!force && now - accessoryState.lastDebugUiAt < 180) return;
  accessoryState.lastDebugUiAt = now;

  const age = accessoryState.lastLandmarkAt ? now - accessoryState.lastLandmarkAt : Infinity;
  const fresh = Number.isFinite(age) && age <= LANDMARK_STALE_MS;
  if (trackingFpsEl) trackingFpsEl.textContent = accessoryState.fps ? `${accessoryState.fps.toFixed(0)} fps` : '--';
  if (trackingLatencyEl) trackingLatencyEl.textContent = Number.isFinite(accessoryState.lastLatencyMs) ? `${Math.round(accessoryState.lastLatencyMs)} ms` : '-- ms';
  if (trackingAlphaEl) trackingAlphaEl.textContent = `${accessoryState.fastAlpha.toFixed(2)} / ${accessoryState.slowAlpha.toFixed(2)}`;
  if (trackingFreshnessEl) trackingFreshnessEl.textContent = fresh ? 'Fresh' : (accessoryState.lastLandmarkAt ? 'Stale' : 'Waiting');
}

function setMirrorMode(enabled) {
  accessoryState.mirror = Boolean(enabled);
  // Some browsers/devices deliver the webcam preview already mirrored.
  // Mirror Off is the default user-facing mode, so we counter-flip the
  // displayed video and overlay together while keeping landmark math aligned.
  if (webcamBody) webcamBody.classList.toggle('mirror-corrected', !accessoryState.mirror);
  const displayTransform = accessoryState.mirror ? 'scaleX(1)' : 'scaleX(-1)';
  if (webcamVideo) webcamVideo.style.transform = displayTransform;
  if (accessoryCanvas) accessoryCanvas.style.transform = displayTransform;
  updateSelectedFilterLabels();
}

function navCardHtml(action, title, sub) {
  return `<button class="catalog-nav-card" type="button" data-action="${action}">
    <span class="catalog-card-title">${title}</span>
    <span class="catalog-card-sub">${sub}</span>
  </button>`;
}

function modelCardHtml(kind, item, index) {
  const active = accessoryState.selected[kind] === index ? ' is-active' : '';
  return `<button class="catalog-model-card${active}" type="button" data-action="select-model" data-kind="${kind}" data-index="${index}">
    <img src="${item.src}" alt="${SUBCATEGORY_LABELS[kind]} ${index + 1}" />
    <span>${item.name}</span>
  </button>`;
}

function swatchHtml(kind, item, index) {
  const active = accessoryState.selected[kind] === index ? ' is-active' : '';
  const color = item.color.replace(/,\s*0\.\d+\)/, ', 1)');
  return `<button class="catalog-swatch${active}" type="button" data-action="select-model" data-kind="${kind}" data-index="${index}">
    <span class="catalog-swatch-dot" style="background:${color}"></span>
    <span>${item.name}</span>
  </button>`;
}

function renderTryOnCatalog() {
  if (!catalogMenu) return;

  if (catalogBackBtn) catalogBackBtn.style.visibility = accessoryState.menuLevel === 'main' ? 'hidden' : 'visible';

  if (accessoryState.menuLevel === 'main') {
    if (catalogTitle) catalogTitle.textContent = 'Categories';
    if (catalogBreadcrumb) catalogBreadcrumb.textContent = 'Main menu';
    catalogMenu.innerHTML = [
      navCardHtml('category-accessories', 'Accessories', 'Glasses, hats, earrings, necklace'),
      navCardHtml('category-makeup', 'Makeup', 'Lipstick and blush colors'),
    ].join('');
    updateSelectedFilterLabels();
    return;
  }

  if (accessoryState.menuLevel === 'subcategories') {
    const category = CATEGORY_TREE[accessoryState.category];
    if (catalogTitle) catalogTitle.textContent = category.title;
    if (catalogBreadcrumb) catalogBreadcrumb.textContent = `Main menu → ${category.title}`;
    catalogMenu.innerHTML = category.subcategories
      .map(sub => navCardHtml(`subcategory-${sub.key}`, sub.label, sub.hint))
      .join('');
    updateSelectedFilterLabels();
    return;
  }

  const kind = accessoryState.subcategory;
  const category = CATEGORY_TREE[accessoryState.category];
  if (catalogTitle) catalogTitle.textContent = SUBCATEGORY_LABELS[kind];
  if (catalogBreadcrumb) catalogBreadcrumb.textContent = `Main menu → ${category.title} → ${SUBCATEGORY_LABELS[kind]}`;
  const items = FILTER_CATALOG[kind] || [];
  const isMakeup = kind === 'lipstick' || kind === 'blush';
  catalogMenu.innerHTML = `<div class="${isMakeup ? 'catalog-swatch-grid' : 'catalog-model-grid'}">
    ${items.map((item, index) => isMakeup ? swatchHtml(kind, item, index) : modelCardHtml(kind, item, index)).join('')}
  </div>`;
  updateSelectedFilterLabels();
}

function catalogBack() {
  if (accessoryState.menuLevel === 'items') setCatalogView('subcategories', accessoryState.category, null);
  else if (accessoryState.menuLevel === 'subcategories') setCatalogView('main', null, null);
}

if (catalogMenu) {
  catalogMenu.addEventListener('click', event => {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    const action = button.dataset.action;
    if (action === 'category-accessories') setCatalogView('subcategories', 'accessories', null);
    else if (action === 'category-makeup') setCatalogView('subcategories', 'makeup', null);
    else if (action.startsWith('subcategory-')) setCatalogView('items', accessoryState.category, action.replace('subcategory-', ''));
    else if (action === 'select-model') {
      accessoryState.selected[button.dataset.kind] = Number(button.dataset.index);
      updateSelectedFilterLabels();
      renderTryOnCatalog();
    }
  });
}

if (catalogBackBtn) catalogBackBtn.addEventListener('click', catalogBack);
if (catalogClearBtn) catalogClearBtn.addEventListener('click', clearTryOnFilters);
if (mirrorToggleBtn) mirrorToggleBtn.addEventListener('click', () => setMirrorMode(!accessoryState.mirror));
setMirrorMode(false);
renderTryOnCatalog();

document.addEventListener('keydown', event => {
  if (!webcamCard || webcamCard.style.display === 'none') return;
  const tag = String(event.target?.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

  const key = event.key.toLowerCase();
  if (key === 'a') {
    setCatalogView('subcategories', 'accessories', null);
  } else if (key === 'm') {
    setCatalogView('subcategories', 'makeup', null);
  } else if (key === 'b') {
    catalogBack();
  } else if (key === '0') {
    clearTryOnFilters();
  } else if (key === 'd') {
    accessoryState.debugMakeup = !accessoryState.debugMakeup;
    updateSelectedFilterLabels();
  } else if (key === 'r') {
    setMirrorMode(!accessoryState.mirror);
  } else if (key === 'q' || event.key === 'Escape') {
    stopWebcam();
  } else {
    return;
  }
  event.preventDefault();
});

function smoothNumber(previous, current, alpha) {
  if (!Number.isFinite(previous)) return current;
  return ((1 - alpha) * previous) + (alpha * current);
}

function smoothPoint(previous, current, alpha) {
  if (!previous) return current;
  return {
    x: smoothNumber(previous.x, current.x, alpha),
    y: smoothNumber(previous.y, current.y, alpha),
  };
}

function smoothPointList(previous, current, alpha) {
  if (!Array.isArray(current)) return current;
  return current.map((point, index) => smoothPoint(previous?.[index], point, alpha));
}

function smoothAccessoryMeasurements(previous, current) {
  if (!previous) return current;
  const fast = accessoryState.fastAlpha;
  const slow = accessoryState.slowAlpha;
  return {
    eyeCenter: smoothPoint(previous.eyeCenter, current.eyeCenter, fast),
    noseBridge: smoothPoint(previous.noseBridge, current.noseBridge, fast),
    forehead: smoothPoint(previous.forehead, current.forehead, slow),
    earringLeft: smoothPoint(previous.earringLeft, current.earringLeft, fast),
    earringRight: smoothPoint(previous.earringRight, current.earringRight, fast),
    necklaceAnchor: smoothPoint(previous.necklaceAnchor, current.necklaceAnchor, slow),
    blushLeft: smoothPoint(previous.blushLeft, current.blushLeft, fast),
    blushRight: smoothPoint(previous.blushRight, current.blushRight, fast),
    blushLeftAnchors: smoothPointList(previous.blushLeftAnchors, current.blushLeftAnchors, fast),
    blushRightAnchors: smoothPointList(previous.blushRightAnchors, current.blushRightAnchors, fast),
    lipOuter: smoothPointList(previous.lipOuter, current.lipOuter, fast),
    lipInner: smoothPointList(previous.lipInner, current.lipInner, fast),
    hasPreciseMakeup: Boolean(current.hasPreciseMakeup),
    eyeDistance: smoothNumber(previous.eyeDistance, current.eyeDistance, fast),
    faceWidth: smoothNumber(previous.faceWidth, current.faceWidth, fast),
    faceHeight: smoothNumber(previous.faceHeight, current.faceHeight, slow),
    roll: smoothNumber(previous.roll, current.roll, fast),
    yaw: smoothNumber(previous.yaw, current.yaw, fast),
  };
}

function dist2d(a, b) {
  return Math.hypot(Number(a.x) - Number(b.x), Number(a.y) - Number(b.y));
}

function avgPoints(points, indices) {
  const selected = indices.map(i => points[i]).filter(Boolean);
  if (!selected.length) return null;
  const sum = selected.reduce((acc, p) => ({ x: acc.x + Number(p.x), y: acc.y + Number(p.y) }), { x: 0, y: 0 });
  return { x: sum.x / selected.length, y: sum.y / selected.length };
}

function pointList(points, indices) {
  return indices.map(i => points[i]).filter(Boolean).map(p => ({ x: Number(p.x), y: Number(p.y) }));
}

function getAccessoryMeasurements(data) {
  const points = (Array.isArray(data.rawLandmarks) && data.rawLandmarks.length >= 468)
    ? data.rawLandmarks
    : (Array.isArray(data.landmarks) ? data.landmarks : []);
  if (!points.length) return null;

  let leftEye, rightEye, noseBridge, forehead, faceWidth, faceHeight, yaw;
  let earringLeft, earringRight, necklaceAnchor, blushLeft, blushRight, blushLeftAnchors, blushRightAnchors, lipOuter, lipInner;
  let hasPreciseMakeup = false;
  if (points.length >= 468) {
    leftEye = avgPoints(points, [33, 133]);
    rightEye = avgPoints(points, [362, 263]);
    const rawEyeCenterX = (Number(leftEye.x) + Number(rightEye.x)) / 2;
    noseBridge = avgPoints(points, [168, 6, 197]);
    forehead = avgPoints(points, [10, 151, 109, 338]);
    const leftCheek = points[234];
    const rightCheek = points[454];
    const chin = points[152];
    const noseTip = points[1];
    if (!leftEye || !rightEye || !noseBridge || !forehead || !leftCheek || !rightCheek || !chin || !noseTip) return null;
    faceWidth = dist2d(leftCheek, rightCheek);
    faceHeight = dist2d(forehead, chin);
    const faceCenter = { x: (Number(leftCheek.x) + Number(rightCheek.x)) / 2, y: (Number(leftCheek.y) + Number(rightCheek.y)) / 2 };
    yaw = (Number(noseTip.x) - faceCenter.x) / Math.max(faceWidth, 1);
    earringLeft = { x: Number(leftCheek.x) - faceWidth * 0.07, y: Number(leftCheek.y) + faceHeight * 0.12 };
    earringRight = { x: Number(rightCheek.x) + faceWidth * 0.07, y: Number(rightCheek.y) + faceHeight * 0.12 };
    necklaceAnchor = { x: (Number(chin.x) * 0.20) + (rawEyeCenterX * 0.80), y: Number(chin.y) + faceHeight * 0.28 };
    blushLeftAnchors = pointList(points, BLUSH_LEFT_LANDMARKS);
    blushRightAnchors = pointList(points, BLUSH_RIGHT_LANDMARKS);
    blushLeft = avgPoints(points, BLUSH_LEFT_LANDMARKS);
    blushRight = avgPoints(points, BLUSH_RIGHT_LANDMARKS);
    lipOuter = pointList(points, LIPSTICK_OUTER_LANDMARKS);
    lipInner = pointList(points, LIPSTICK_INNER_LANDMARKS);
    hasPreciseMakeup = lipOuter.length === LIPSTICK_OUTER_LANDMARKS.length
      && lipInner.length === LIPSTICK_INNER_LANDMARKS.length
      && blushLeftAnchors.length === BLUSH_LEFT_LANDMARKS.length
      && blushRightAnchors.length === BLUSH_RIGHT_LANDMARKS.length;
  } else {
    // 68-point fallback from the backend. It is less precise than MediaPipe,
    // but it still gives stable eye, brow, and jaw locations for the demo.
    leftEye = avgPoints(points, [36, 37, 38, 39, 40, 41]);
    rightEye = avgPoints(points, [42, 43, 44, 45, 46, 47]);
    const rawEyeCenterX = (Number(leftEye.x) + Number(rightEye.x)) / 2;
    noseBridge = avgPoints(points, [27, 28]);
    const leftBrow = avgPoints(points, [17, 18, 19, 20, 21]);
    const rightBrow = avgPoints(points, [22, 23, 24, 25, 26]);
    const leftJaw = points[0];
    const rightJaw = points[16];
    const chin = points[8];
    const noseTip = points[33] || noseBridge;
    if (!leftEye || !rightEye || !noseBridge || !leftBrow || !rightBrow || !leftJaw || !rightJaw || !chin) return null;
    faceWidth = dist2d(leftJaw, rightJaw);
    const browCenter = {
      x: (Number(leftBrow.x) + Number(rightBrow.x)) / 2,
      y: (Number(leftBrow.y) + Number(rightBrow.y)) / 2,
    };
    forehead = { x: browCenter.x, y: browCenter.y - (faceWidth * 0.26) };
    faceHeight = dist2d(forehead, chin);
    const faceCenter = { x: (Number(leftJaw.x) + Number(rightJaw.x)) / 2, y: (Number(leftJaw.y) + Number(rightJaw.y)) / 2 };
    yaw = (Number(noseTip.x) - faceCenter.x) / Math.max(faceWidth, 1);
    earringLeft = { x: Number(leftJaw.x) - faceWidth * 0.02, y: Number(leftJaw.y) };
    earringRight = { x: Number(rightJaw.x) + faceWidth * 0.02, y: Number(rightJaw.y) };
    necklaceAnchor = { x: (Number(chin.x) * 0.20) + (rawEyeCenterX * 0.80), y: Number(chin.y) + faceHeight * 0.30 };
    blushLeft = avgPoints(points, [2, 3, 31]) || leftJaw;
    blushRight = avgPoints(points, [13, 14, 35]) || rightJaw;
    blushLeftAnchors = [];
    blushRightAnchors = [];
    lipOuter = pointList(points, [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]);
    lipInner = pointList(points, [60, 61, 62, 63, 64, 65, 66, 67]);
  }

  const rawEyeCenter = {
    x: (Number(leftEye.x) + Number(rightEye.x)) / 2,
    y: (Number(leftEye.y) + Number(rightEye.y)) / 2,
  };
  const eyeCenter = {
    x: (rawEyeCenter.x * 0.70) + (Number(noseBridge.x) * 0.30),
    y: (rawEyeCenter.y * 0.70) + (Number(noseBridge.y) * 0.30),
  };
  const eyeDistance = dist2d(leftEye, rightEye);
  const roll = Math.atan2(Number(rightEye.y) - Number(leftEye.y), Number(rightEye.x) - Number(leftEye.x));

  if (!Number.isFinite(eyeDistance) || eyeDistance < 12 || !Number.isFinite(faceWidth)) return null;
  return {
    eyeCenter,
    noseBridge,
    forehead,
    earringLeft,
    earringRight,
    necklaceAnchor,
    blushLeft,
    blushRight,
    blushLeftAnchors,
    blushRightAnchors,
    lipOuter,
    lipInner,
    hasPreciseMakeup,
    eyeDistance,
    faceWidth,
    faceHeight,
    roll,
    yaw,
  };
}

function drawRotatedImage(ctx, image, centerX, centerY, width, height, angle) {
  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(angle);
  ctx.drawImage(image, -width / 2, -height / 2, width, height);
  ctx.restore();
}

function drawYawWarpedImage(ctx, image, centerX, centerY, width, height, angle, yaw) {
  const yawClamped = Math.max(-0.35, Math.min(0.35, Number(yaw) || 0));
  const yawAbs = Math.abs(yawClamped);
  const visibleWidth = width * (1 - yawAbs * 0.16);
  const farSideScale = 1 - yawAbs * 0.26;
  const strips = 34;

  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.rotate(angle);

  for (let i = 0; i < strips; i += 1) {
    const t0 = i / strips;
    const t1 = (i + 1) / strips;
    const tm = (t0 + t1) / 2;
    const sourceX = image.naturalWidth * t0;
    const sourceW = Math.ceil(image.naturalWidth / strips);
    const destX = (-visibleWidth / 2) + (visibleWidth * t0);
    const destW = Math.ceil(visibleWidth / strips) + 1;
    const farAmount = yawClamped >= 0 ? tm : (1 - tm);
    const stripScale = 1 - ((1 - farSideScale) * farAmount);
    const stripH = height * stripScale;
    const ySkew = (tm - 0.5) * yawClamped * height * 0.16;
    ctx.drawImage(image, sourceX, 0, sourceW, image.naturalHeight, destX, -stripH / 2 + ySkew, destW, stripH);
  }

  ctx.restore();
}

function drawFilledLandmarkShape(ctx, points, fill, stroke = null) {
  if (!Array.isArray(points) || points.length < 3) return;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.25;
    ctx.stroke();
  }
  ctx.restore();
}

function drawSoftPolygonMask(ctx, points, color, opacity, blurPx, stroke = null) {
  if (!Array.isArray(points) || points.length < 3) return;
  const layer = document.createElement('canvas');
  layer.width = ctx.canvas.width;
  layer.height = ctx.canvas.height;
  const layerCtx = layer.getContext('2d');
  layerCtx.save();
  layerCtx.filter = `blur(${blurPx}px)`;
  layerCtx.globalAlpha = opacity;
  drawFilledLandmarkShape(layerCtx, points, color, stroke);
  layerCtx.restore();
  ctx.drawImage(layer, 0, 0);
}

function drawLipstickMask(ctx, outerPoints, innerPoints, color, opacity, blurPx, stroke = null) {
  if (!Array.isArray(outerPoints) || outerPoints.length < 3) return;
  const layer = document.createElement('canvas');
  layer.width = ctx.canvas.width;
  layer.height = ctx.canvas.height;
  const layerCtx = layer.getContext('2d');

  layerCtx.save();
  layerCtx.globalAlpha = opacity;
  drawFilledLandmarkShape(layerCtx, outerPoints, color, null);
  if (Array.isArray(innerPoints) && innerPoints.length >= 3) {
    // Cut the inner mouth opening out of the lipstick mask so color stays on the lips.
    layerCtx.globalCompositeOperation = 'destination-out';
    layerCtx.globalAlpha = 1;
    drawFilledLandmarkShape(layerCtx, innerPoints, 'rgba(0, 0, 0, 1)', null);
  }
  layerCtx.restore();

  const blurred = document.createElement('canvas');
  blurred.width = ctx.canvas.width;
  blurred.height = ctx.canvas.height;
  const blurredCtx = blurred.getContext('2d');
  blurredCtx.save();
  blurredCtx.filter = `blur(${blurPx}px)`;
  blurredCtx.drawImage(layer, 0, 0);
  blurredCtx.restore();
  ctx.drawImage(blurred, 0, 0);

  if (stroke) {
    ctx.save();
    ctx.globalAlpha = opacity * 0.35;
    drawFilledLandmarkShape(ctx, outerPoints, 'rgba(0, 0, 0, 0)', stroke);
    ctx.restore();
  }
}

function rgbaWithAlpha(color, alpha) {
  const match = String(color).match(/rgba?\(([^)]+)\)/);
  if (!match) return color;
  const parts = match[1].split(',').map(part => part.trim());
  return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
}

function drawBlush(ctx, center, faceWidth, color) {
  if (!center) return;
  const radiusX = faceWidth * 0.105 * BLUSH_SCALE;
  const radiusY = faceWidth * 0.065 * BLUSH_SCALE;
  const layer = document.createElement('canvas');
  layer.width = ctx.canvas.width;
  layer.height = ctx.canvas.height;
  const layerCtx = layer.getContext('2d');
  layerCtx.save();
  layerCtx.filter = `blur(${BLUSH_BLUR}px)`;
  layerCtx.globalAlpha = BLUSH_OPACITY;
  layerCtx.translate(center.x, center.y);
  layerCtx.scale(radiusX, radiusY);
  const gradient = layerCtx.createRadialGradient(0, 0, 0, 0, 0, 1);
  gradient.addColorStop(0, rgbaWithAlpha(color, 0.95));
  gradient.addColorStop(0.48, rgbaWithAlpha(color, 0.36));
  gradient.addColorStop(1, rgbaWithAlpha(color, 0));
  layerCtx.fillStyle = gradient;
  layerCtx.beginPath();
  layerCtx.arc(0, 0, 1, 0, Math.PI * 2);
  layerCtx.fill();
  layerCtx.restore();
  ctx.drawImage(layer, 0, 0);
}

function drawMakeupDebug(ctx, measurements) {
  if (!accessoryState.debugMakeup || !measurements) return;
  const drawPoints = (points, color) => {
    if (!Array.isArray(points)) return;
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    points.forEach(point => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
    if (points.length > 2) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
      ctx.closePath();
      ctx.stroke();
    }
    ctx.restore();
  };
  drawPoints(measurements.lipOuter, 'rgba(255, 64, 128, 0.95)');
  drawPoints(measurements.lipInner, 'rgba(255, 210, 80, 0.95)');
  drawPoints(measurements.blushLeftAnchors, 'rgba(80, 220, 255, 0.95)');
  drawPoints(measurements.blushRightAnchors, 'rgba(80, 220, 255, 0.95)');
}

function drawAccessoriesOnContext(ctx, measurements) {
  if (!measurements) return;

  const lipstickIndex = accessoryState.selected.lipstick;
  if (lipstickIndex !== null && measurements.hasPreciseMakeup) {
    const lipstick = FILTER_CATALOG.lipstick[lipstickIndex];
    const yawAbs = Math.abs(measurements.yaw || 0);
    const lipstickOpacity = LIPSTICK_OPACITY * (1 - (yawAbs * 0.40));
    const lipstickBlur = LIPSTICK_BLUR + (yawAbs * 4);
    drawLipstickMask(ctx, measurements.lipOuter, measurements.lipInner, lipstick.color, lipstickOpacity, lipstickBlur, lipstick.stroke);
  }

  const blushIndex = accessoryState.selected.blush;
  if (blushIndex !== null && measurements.hasPreciseMakeup) {
    const blush = FILTER_CATALOG.blush[blushIndex];
    const leftBlushCenter = {
      x: measurements.blushLeft.x - (measurements.faceWidth * BLUSH_X_OFFSET),
      y: measurements.blushLeft.y + (measurements.faceWidth * BLUSH_Y_OFFSET),
    };
    const rightBlushCenter = {
      x: measurements.blushRight.x + (measurements.faceWidth * BLUSH_X_OFFSET),
      y: measurements.blushRight.y + (measurements.faceWidth * BLUSH_Y_OFFSET),
    };
    drawBlush(ctx, leftBlushCenter, measurements.faceWidth, blush.color);
    drawBlush(ctx, rightBlushCenter, measurements.faceWidth, blush.color);
  }

  const glassesIndex = accessoryState.selected.glasses;
  if (glassesIndex !== null && accessoriesReady()) {
    const glasses = FILTER_CATALOG.glasses[glassesIndex].image;
    const glassesW = measurements.eyeDistance * GLASSES_WIDTH_FACTOR_WEB;
    const glassesH = glassesW * (glasses.naturalHeight / glasses.naturalWidth);
    const yawShift = (Number(measurements.yaw) || 0) * measurements.eyeDistance * 0.16;
    const anchorX = measurements.eyeCenter.x + yawShift;
    const anchorY = measurements.eyeCenter.y + (glassesH * 0.04);
    drawYawWarpedImage(ctx, glasses, anchorX, anchorY, glassesW, glassesH, measurements.roll, measurements.yaw);
  }

  const hatIndex = accessoryState.selected.hat;
  if (hatIndex !== null && accessoriesReady()) {
    const hat = FILTER_CATALOG.hat[hatIndex].image;
    const hatW = measurements.faceWidth * HAT_WIDTH_FACTOR_WEB;
    const hatH = hatW * (hat.naturalHeight / hat.naturalWidth);
    const hatY = measurements.forehead.y
      - (hatH * HAT_BOTTOM_ALIGNMENT_WEB)
      + (measurements.faceHeight * HAT_Y_OFFSET_FACTOR_WEB);
    const hatYawShift = (Number(measurements.yaw) || 0) * measurements.faceWidth * 0.26;
    const hatX = measurements.forehead.x + hatYawShift;
    drawYawWarpedImage(ctx, hat, hatX, hatY + (hatH / 2), hatW, hatH, measurements.roll, measurements.yaw);
  }

  const earringsIndex = accessoryState.selected.earrings;
  if (earringsIndex !== null && accessoriesReady()) {
    const earrings = FILTER_CATALOG.earrings[earringsIndex].image;
    const earringH = measurements.faceHeight * 0.24;
    const earringW = earringH * (earrings.naturalWidth / earrings.naturalHeight);
    const yaw = measurements.yaw || 0;
    
    let leftOpacity = 1.0;
    let rightOpacity = 1.0;
    if (yaw > 0) {
      rightOpacity = Math.max(0, 1 - (yaw * 3.8));
    } else {
      leftOpacity = Math.max(0, 1 - (Math.abs(yaw) * 3.8));
    }

    if (leftOpacity > 0.05) {
      ctx.save();
      ctx.globalAlpha = leftOpacity;
      drawRotatedImage(ctx, earrings, measurements.earringLeft.x, measurements.earringLeft.y, earringW, earringH, measurements.roll * 0.45);
      ctx.restore();
    }
    if (rightOpacity > 0.05) {
      ctx.save();
      ctx.globalAlpha = rightOpacity;
      drawRotatedImage(ctx, earrings, measurements.earringRight.x, measurements.earringRight.y, earringW, earringH, measurements.roll * 0.45);
      ctx.restore();
    }
  }

  const necklaceIndex = accessoryState.selected.necklace;
  if (necklaceIndex !== null && accessoriesReady()) {
    const necklace = FILTER_CATALOG.necklace[necklaceIndex].image;
    const necklaceW = measurements.faceWidth * 1.62;
    const necklaceH = necklaceW * (necklace.naturalHeight / necklace.naturalWidth);
    drawRotatedImage(ctx, necklace, measurements.necklaceAnchor.x, measurements.necklaceAnchor.y, necklaceW, necklaceH, measurements.roll * 0.18);
  }

  drawMakeupDebug(ctx, measurements);
}

function drawAccessoryOverlay() {
  if (!accessoryCanvas || !webcamVideo) return;
  const width = webcamVideo.videoWidth || 640;
  const height = webcamVideo.videoHeight || 480;
  if (accessoryCanvas.width !== width) accessoryCanvas.width = width;
  if (accessoryCanvas.height !== height) accessoryCanvas.height = height;

  // Client-side detection: runs at ~30 FPS, throttled inside _detectLocal()
  if (_mpState === 'ready') _detectLocal();

  const now = performance.now();
  accessoryState.frameCount += 1;
  if (now - accessoryState.fpsLastAt >= 500) {
    accessoryState.fps = (accessoryState.frameCount * 1000) / (now - accessoryState.fpsLastAt);
    accessoryState.frameCount = 0;
    accessoryState.fpsLastAt = now;
  }

  const ctx = accessoryCanvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  drawAccessoriesOnContext(ctx, accessoryState.smoothed);
  updateTrackingDebug();
  accessoryState.raf = requestAnimationFrame(drawAccessoryOverlay);
}

function drawVisibleWebcamFrame(ctx, width, height) {
  if (!webcamVideo) return;
  if (accessoryState.mirror) {
    ctx.drawImage(webcamVideo, 0, 0, width, height);
    drawAccessoriesOnContext(ctx, accessoryState.smoothed);
    return;
  }

  ctx.save();
  ctx.translate(width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(webcamVideo, 0, 0, width, height);
  drawAccessoriesOnContext(ctx, accessoryState.smoothed);
  ctx.restore();
}

async function updateAccessoryLandmarks() {
  // Skip server round-trip when client-side detection is active
  if (_mpState === 'ready') return;
  if (accessoryState.processing || !webcamStream || !webcamVideo || !webcamCanvas) {
    updateTrackingDebug();
    return;
  }
  if (!webcamVideo.videoWidth || !webcamVideo.videoHeight) return;

  accessoryState.processing = true;
  const requestId = accessoryState.requestId + 1;
  accessoryState.requestId = requestId;
  const startedAt = performance.now();
  try {
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    webcamCanvas.getContext('2d').drawImage(webcamVideo, 0, 0);
    const frameBlob = dataUrlToBlob(webcamCanvas.toDataURL('image/jpeg', 0.56));
    if (!frameBlob) return;

    const fd = new FormData();
    fd.append('image', frameBlob, 'accessory-frame.jpg');
    fd.append('includeReduced68', 'true');
    fd.append('landmarkMode', 'both');
    fd.append('saveOverlay', 'false');
    const res = await fetch('/api/landmarks', { method: 'POST', body: fd });
    const data = await res.json();
    if (requestId < accessoryState.appliedRequestId) return;
    if (!res.ok || !data.ok) return;

    const measurements = getAccessoryMeasurements(data);
    if (!measurements) return;
    accessoryState.appliedRequestId = requestId;
    accessoryState.lastLatencyMs = performance.now() - startedAt;
    accessoryState.lastLandmarkAt = performance.now();
    accessoryState.measurements = measurements;
    accessoryState.smoothed = smoothAccessoryMeasurements(accessoryState.smoothed, measurements);
    updateTrackingDebug(true);
  } catch (_) {
    // Keep the last good smoothed measurements during short detection failures.
  } finally {
    accessoryState.processing = false;
  }
}

function startAccessoryFilter() {
  if (!accessoryCanvas) return;
  if (!accessoryState.raf) accessoryState.raf = requestAnimationFrame(drawAccessoryOverlay);
  // Begin client-side FaceLandmarker init (async, non-blocking)
  if (_mpState === 'idle') _initClientLandmarker();
  // Server polling as fallback while client-side loads (or if it fails)
  if (_mpState !== 'ready' && !accessoryState.timer) {
    updateAccessoryLandmarks();
    accessoryState.timer = setInterval(updateAccessoryLandmarks, LANDMARK_POLL_MS);
  }
}

function stopAccessoryFilter() {
  if (accessoryState.timer) { clearInterval(accessoryState.timer); accessoryState.timer = null; }
  if (accessoryState.raf) { cancelAnimationFrame(accessoryState.raf); accessoryState.raf = null; }
  accessoryState.processing = false;
  accessoryState.measurements = null;
  accessoryState.smoothed = null;
  accessoryState.lastLatencyMs = null;
  accessoryState.lastLandmarkAt = 0;
  _mpLastDetectAt = 0;
  updateTrackingDebug(true);
  if (accessoryCanvas) {
    const ctx = accessoryCanvas.getContext('2d');
    ctx.clearRect(0, 0, accessoryCanvas.width || 1, accessoryCanvas.height || 1);
  }
}

/* ═══════════════════ WEBCAM ═══════════════════ */

function startWebcam() {
  // Begin client-side FaceLandmarker init early for faster startup
  if (_mpState === 'idle') _initClientLandmarker();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast('Webcam is not supported in this browser.', 'warning');
    return;
  }
  if (webcamCard && webcamCard.parentElement !== document.body) {
    document.body.appendChild(webcamCard);
  }
  document.body.classList.add('tryon-open');
  uploadCard.style.display = 'none';
  previewCard.style.display = 'none';
  webcamCard.style.display = 'flex';
  if (webcamStream) {
    startAccessoryFilter();
    return;
  }
  navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false })
    .then(stream => {
      webcamStream = stream;
      webcamVideo.srcObject = stream;
      webcamVideo.onloadedmetadata = () => startAccessoryFilter();
      startAccessoryFilter();
    })
    .catch(err => {
      webcamCard.style.display = 'none';
      document.body.classList.remove('tryon-open');
      uploadCard.style.display = '';
      toast('Could not access webcam: ' + err.message, 'error');
    });
}
function stopWebcam() {
  stopAccessoryFilter();
  if (webcamStream) { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
  document.body.classList.remove('tryon-open');
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
  const captureCtx = webcamCanvas.getContext('2d');
  drawVisibleWebcamFrame(captureCtx, webcamCanvas.width, webcamCanvas.height);
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
    fd.append('lip_tint', String(Number($('#sLipTint')?.value || 0)));
    fd.append('blush', String(Number($('#sBlush')?.value || 0)));
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
    updateExprMetrics(data);
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
    const yearsSigned = Number($('#sTargetAge')?.value || 10);
    const agingMode = yearsSigned < 0 ? 'deaging' : 'aging';
    const fd = new FormData();
    fd.append('image', srcBlob, 'aging-input.jpg');
    fd.append('mode', agingMode);
    fd.append('method', selectedMethod);
    fd.append('years', String(Math.abs(yearsSigned)));
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
