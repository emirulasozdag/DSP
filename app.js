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

function setPipelineImage(dataUrl, meta = {}) {
  appState.imageDataUrl = dataUrl || appState.imageDataUrl;
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

/* ═══════════════════ EXPRESSION SLIDERS ═══════════════════ */
const sliders = [
  { id: 'sSmile', valId: 'vSmile' },
  { id: 'sBrow', valId: 'vBrow' },
  { id: 'sLip', valId: 'vLip' },
  { id: 'sSlim', valId: 'vSlim' },
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
    const by = 102 - brow * 14;
    const cy = 95 - brow * 14;
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

  updateAnalysisMetrics();
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
});

/* ═══════════════════ AGING SLIDERS ═══════════════════ */
['sAge', 'sContrast', 'sTone', 'sNoise', 'sDetail'].forEach(id => {
  const s = $(`#${id}`), v = $(`#v${id.slice(1)}`);
  if (s && v) s.addEventListener('input', () => {
    v.textContent = s.value + '%';
    updateAnalysisMetrics();
  });
});
const sTargetAge = $('#sTargetAge'), vTargetAge = $('#vTargetAge');
if (sTargetAge && vTargetAge) sTargetAge.addEventListener('input', () => {
  const v = +sTargetAge.value;
  vTargetAge.textContent = v === 0 ? '0' : (v > 0 ? '+' + v : v);
  updateApparentAge(v);
  updateAgeOffsetLabel(v);
  updateAnalysisMetrics();
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
    refImg.src = e.target.result;
    refDrop.style.display = 'none';
    refPreview.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}
if ($('#btnClearRef')) $('#btnClearRef').addEventListener('click', () => {
  refImg.src = '';
  refInput.value = '';
  refPreview.style.display = 'none';
  refDrop.style.display = 'flex';
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
  const pts = Array.from($$('#ptsG circle')).map((c, i) => `${i},${c.getAttribute('cx')},${c.getAttribute('cy')}`);
  const csv = 'index,x,y\n' + pts.join('\n');
  download('landmarks.csv', csv, 'text/csv');
});

if ($('#btnExportJSON')) $('#btnExportJSON').addEventListener('click', () => {
  const points = Array.from($$('#ptsG circle')).map((c, i) => ({
    index: i,
    x: Number(c.getAttribute('cx')),
    y: Number(c.getAttribute('cy')),
  }));
  download('landmarks.json', JSON.stringify({ pointCount: points.length, points }, null, 2), 'application/json');
  toast('Landmark JSON exported.', 'success');
});

if ($('#btnRunPP')) $('#btnRunPP').addEventListener('click', () => {
  const conf = Number($('#sliderConf')?.value || 85) / 100;
  const pad = Number($('#sliderPad')?.value || 20);
  toast(`Preprocessing completed (conf ${conf.toFixed(2)}, pad ${pad}%).`, 'success');
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
  saveSvgAsPng($('#exprSvg'), 'expression-result.png');
});

if ($('#btnSaveAging')) $('#btnSaveAging').addEventListener('click', () => {
  const agingSvg = $('#page-aging .aging-side-card-active .face-svg');
  saveSvgAsPng(agingSvg, 'aging-result.png');
});

if ($('#btnCompareAging')) $('#btnCompareAging').addEventListener('click', () => {
  const compareCard = $('#methodCompareCard');
  if (!compareCard) return;
  compareCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  toast('Scrolled to comparison panel.', 'info');
});

/* ═══════════════════ APPLY WARP FEEDBACK ═══════════════════ */
if ($('#btnApplyExpr')) $('#btnApplyExpr').addEventListener('click', function () {
  this.textContent = '✓ Warp Applied';
  this.style.background = 'linear-gradient(135deg,#10b981,#059669)';
  toast('Expression warp applied!', 'success');
  setTimeout(() => { this.textContent = '▶ Apply Warp'; this.style.background = ''; }, 1500);
});

if ($('#btnSimulate')) $('#btnSimulate').addEventListener('click', function () {
  const span = this.querySelector('span') || this;
  span.textContent = '⧗ Processing...';
  this.disabled = true;
  toast('Aging simulation applied!', 'success');
  setTimeout(() => {
    span.textContent = 'Apply Aging Effect';
    this.disabled = false;
    updateAnalysisMetrics();
  }, 1600);
});

if ($('#btnAnalysisCsv')) $('#btnAnalysisCsv').addEventListener('click', () => {
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

function updateAnalysisMetrics() {
  const exprAverage = (
    Number($('#sSmile')?.value || 0) +
    Number($('#sBrow')?.value || 0) +
    Number($('#sLip')?.value || 0) +
    Number($('#sSlim')?.value || 0)
  ) / 4 / 100;
  const ageShift = Math.abs(Number($('#sTargetAge')?.value || 0)) / 50;
  const ageStrength = Number($('#sAge')?.value || 50) / 100;
  const intensity = clamp(0.12 + exprAverage * 0.38 + ageShift * 0.26 + ageStrength * 0.24, 0, 1);

  const mse = 40 + intensity * 170;
  const psnr = 36 - intensity * 12;
  const ssim = 0.95 - intensity * 0.24;

  const hf = 3.0 + intensity * 4.8;
  const lf = 9.4 - intensity * 1.7;
  const ratio = hf / (lf * 10);
  const total = lf + hf / 10;

  const msePct = clamp((mse - 30) / 2, 5, 98);
  const psnrPct = clamp(((psnr - 18) / 18) * 100, 5, 98);
  const ssimPct = clamp(ssim * 100, 5, 98);

  if ($('#qmMsePct')) $('#qmMsePct').textContent = `${Math.round(msePct)}%`;
  if ($('#qmPsnrPct')) $('#qmPsnrPct').textContent = `${Math.round(psnrPct)}%`;
  if ($('#qmSsimPct')) $('#qmSsimPct').textContent = `${Math.round(ssimPct)}%`;

  if ($('#qmMseVal')) $('#qmMseVal').textContent = mse.toFixed(2);
  if ($('#qmPsnrVal')) $('#qmPsnrVal').textContent = `${psnr.toFixed(2)} dB`;
  if ($('#qmSsimVal')) $('#qmSsimVal').textContent = ssim.toFixed(3);

  if ($('#enTotal')) $('#enTotal').innerHTML = `${total.toFixed(2)} <small>×10⁷</small>`;
  if ($('#enHF')) $('#enHF').innerHTML = `${hf.toFixed(1)} <small>×10⁶</small>`;
  if ($('#enLF')) $('#enLF').innerHTML = `${lf.toFixed(1)} <small>×10⁷</small>`;
  if ($('#enRatio')) $('#enRatio').textContent = ratio.toFixed(3);

  if ($('#enFillTotal')) $('#enFillTotal').style.width = `${clamp(total / 10 * 100, 20, 100)}%`;
  if ($('#enFillHF')) $('#enFillHF').style.width = `${clamp(hf / 8 * 100, 10, 100)}%`;
  if ($('#enFillLF')) $('#enFillLF').style.width = `${clamp(lf / 10 * 100, 20, 100)}%`;
  if ($('#enFillRatio')) $('#enFillRatio').style.width = `${clamp(ratio / 0.12 * 100, 5, 100)}%`;

  setRingProgress($('#qmMseRing'), msePct);
  setRingProgress($('#qmPsnrRing'), psnrPct);
  setRingProgress($('#qmSsimRing'), ssimPct);
}

setPipelineImage('');
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
