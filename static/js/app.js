/* ═══════════════════════════════════════
   FaceTransform DSP — Application Logic
   ═══════════════════════════════════════ */

// ── State ──
const S = {
    origB64: null,       // original preprocessed image base64
    origFile: null,      // original raw file for preview
    landmarks: [],
    transformed: null,   // last transformed image b64
    metricsHistory: []
};

// ── DOM helpers ──
const $ = id => document.getElementById(id);
const show = (el, cls='show') => el.classList.add(cls);
const hide = (el, cls='show') => el.classList.remove(cls);

// ── Loader ──
function loader(on, txt) {
    const el = $('loader');
    if (on) { $('loaderTxt').textContent = txt || 'Processing…'; el.classList.add('show'); }
    else el.classList.remove('show');
}

// ── Toast ──
function toast(msg, type='') {
    const el = $('toast');
    el.className = 'toast ' + type;
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.remove('show'), 3000);
}

// ═══════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════
function goTab(tabName) {
    // Activate nav button
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    const btn = document.querySelector(`.nav-item[data-tab="${tabName}"]`);
    if (btn) btn.classList.add('active');

    // Show page
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const page = $('page-' + tabName);
    if (page) page.classList.add('active');

    // Initialize page content
    if (tabName === 'landmarks' && S.origB64) initLandmarks();
    if (tabName === 'expression' && S.origB64) initExpression();
    if (tabName === 'aging' && S.origB64) initAging();
    if (tabName === 'frequency' && S.origB64) initFrequency();
    if (tabName === 'metrics') initMetrics();
}

function enableNav() {
    document.querySelectorAll('.nav-item').forEach(b => b.removeAttribute('disabled'));
}

// ═══════════════════════════════════════
// UPLOAD (FR-3.1)
// ═══════════════════════════════════════
function initUpload() {
    const zone = $('dropZone');
    const input = $('fileIn');

    // Drag & drop
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    // Click to browse
    input.addEventListener('change', () => { if (input.files.length) handleFile(input.files[0]); });
}

async function handleFile(file) {
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['jpg','jpeg','png'].includes(ext)) return toast('Only JPG / PNG images allowed', 'error');
    if (file.size > 16 * 1024 * 1024) return toast('File too large (max 16 MB)', 'error');

    // Show original preview immediately
    S.origFile = URL.createObjectURL(file);
    $('prevOrig').src = S.origFile;

    loader(true, 'Uploading & preprocessing…');
    const fd = new FormData();
    fd.append('image', file);

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Upload failed');

        S.origB64 = data.image;
        S.landmarks = data.landmarks || [];

        $('prevProc').src = 'data:image/png;base64,' + data.image;
        $('infoBar').innerHTML =
            `<span><strong>${data.count}</strong> landmarks detected</span>` +
            `<span>Size: <strong>512×512</strong></span>` +
            `<span>Format: <strong>PNG</strong></span>`;

        $('dropInner').classList.add('hidden');
        $('dropResult').classList.remove('hidden');
        enableNav();
        toast('Image processed successfully', 'success');
    } catch (err) {
        toast(err.message, 'error');
    } finally {
        loader(false);
    }
}

function resetUpload() {
    S.origB64 = null;
    S.origFile = null;
    S.landmarks = [];
    S.transformed = null;
    S.metricsHistory = [];

    $('dropInner').classList.remove('hidden');
    $('dropResult').classList.add('hidden');
    $('fileIn').value = '';
    $('prevOrig').src = '';
    $('prevProc').src = '';

    // Disable nav items except upload
    document.querySelectorAll('.nav-item').forEach(b => {
        if (b.dataset.tab !== 'upload') b.setAttribute('disabled', '');
    });
    goTab('upload');
}

// ═══════════════════════════════════════
// LANDMARKS (FR-3.2)
// ═══════════════════════════════════════
async function initLandmarks() {
    $('lmCount').textContent = S.landmarks.length;
    const showPts = $('lmToggle').checked;
    await updateLandmarkImage(showPts);
}

async function toggleLM() {
    const showPts = $('lmToggle').checked;
    await updateLandmarkImage(showPts);
}

async function updateLandmarkImage(showPts) {
    if (!S.origB64) return;
    loader(true, 'Rendering landmarks…');
    try {
        const res = await fetch('/api/landmarks_overlay', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: S.origB64, landmarks: S.landmarks, show: showPts })
        });
        const data = await res.json();
        $('lmImg').src = 'data:image/png;base64,' + data.image;
    } catch (err) {
        toast('Landmark overlay failed', 'error');
    } finally {
        loader(false);
    }
}

async function exportLM() {
    if (!S.landmarks.length) return toast('No landmarks to export', 'error');
    try {
        const res = await fetch('/api/export_landmarks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ landmarks: S.landmarks })
        });
        const blob = await res.blob();
        downloadBlob(blob, 'landmarks.csv');
        toast('Landmarks exported', 'success');
    } catch (err) {
        toast('Export failed', 'error');
    }
}

// ═══════════════════════════════════════
// EXPRESSION (FR-3.3)
// ═══════════════════════════════════════
function initExpression() {
    $('exOrig').src = 'data:image/png;base64,' + S.origB64;
    if (S.transformed) $('exRes').src = 'data:image/png;base64,' + S.transformed;
}

async function doExpr(type) {
    if (!S.origB64 || !S.landmarks.length) return toast('Upload an image first', 'error');
    const slider = $('r_' + type);
    const intensity = slider ? slider.value : 50;

    loader(true, `Applying ${type}…`);
    try {
        const res = await fetch('/api/expression', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: S.origB64, landmarks: S.landmarks, type, intensity })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Expression failed');

        S.transformed = data.result;
        $('exRes').src = 'data:image/png;base64,' + data.result;

        if (data.metrics) {
            $('exMetrics').innerHTML = formatMiniMetrics(data.metrics);
            addMetric(`Expression: ${type} (${intensity}%)`, data.metrics);
        }
        toast(`${type} applied`, 'success');
    } catch (err) {
        toast(err.message, 'error');
    } finally {
        loader(false);
    }
}

// ═══════════════════════════════════════
// AGING (FR-3.4)
// ═══════════════════════════════════════
function initAging() {
    $('agOrig').src = 'data:image/png;base64,' + S.origB64;
}

async function doAge(mode) {
    if (!S.origB64) return toast('Upload an image first', 'error');
    const slider = mode === 'aging' ? $('r_aging') : $('r_deaging');
    const intensity = slider ? slider.value : 50;

    loader(true, `Applying ${mode}…`);
    try {
        const res = await fetch('/api/aging', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: S.origB64, intensity, mode })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Aging failed');

        S.transformed = data.result;
        $('agRes').src = 'data:image/png;base64,' + data.result;

        if (data.metrics) {
            $('agMetrics').innerHTML = formatMiniMetrics(data.metrics);
            addMetric(`${mode === 'aging' ? 'Aging' : 'De-Aging'} (${intensity}%)`, data.metrics);
        }
        toast(`${mode} applied`, 'success');
    } catch (err) {
        toast(err.message, 'error');
    } finally {
        loader(false);
    }
}

// ═══════════════════════════════════════
// FREQUENCY (FR-3.5)
// ═══════════════════════════════════════
function initFrequency() {
    // Show original spectrum if available
    if (!S.origB64) return;
    doFreq('original');
}

async function doFreq(source) {
    if (!S.origB64) return toast('Upload an image first', 'error');

    loader(true, 'Computing FFT spectrum…');
    try {
        const body = { image: S.origB64 };
        if (source !== 'original' && S.transformed) {
            body.transformed = S.transformed;
        }

        const res = await fetch('/api/frequency', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Frequency analysis failed');

        // Original spectrum
        $('specOrig').src = 'data:image/png;base64,' + data.spectrum;
        $('eOrig').innerHTML = formatEnergy('Original', data);

        // Transformed spectrum
        if (data.t_spectrum) {
            $('specTrans').src = 'data:image/png;base64,' + data.t_spectrum;
            $('eTrans').innerHTML = formatEnergy('Transformed', {
                total: data.t_total, low: data.t_low, high: data.t_high,
                ratio: data.t_ratio, low_pct: data.t_low_pct, high_pct: data.t_high_pct
            });

            // Comparison table
            $('freqTable').innerHTML = buildFreqComparison(data);
        } else {
            $('specTrans').src = '';
            $('eTrans').innerHTML = '<span class="muted">Apply a transform first</span>';
        }

        // Switch to frequency page if called from expression/aging
        if (source === 'expression' || source === 'aging') goTab('frequency');

        toast('Spectrum computed', 'success');
    } catch (err) {
        toast(err.message, 'error');
    } finally {
        loader(false);
    }
}

function formatEnergy(label, d) {
    return `<span><strong>Low-freq:</strong> ${d.low_pct.toFixed(1)}%</span>` +
           `<span><strong>High-freq:</strong> ${d.high_pct.toFixed(1)}%</span>` +
           `<span><strong>L/H Ratio:</strong> ${d.ratio.toFixed(2)}</span>` +
           `<span><strong>Total Energy:</strong> ${d.total.toExponential(2)}</span>`;
}

function buildFreqComparison(d) {
    return `<table class="m-table">
        <thead><tr><th>Metric</th><th>Original</th><th>Transformed</th><th>Change</th></tr></thead>
        <tbody>
            <tr><td>Low-Freq %</td><td>${d.low_pct.toFixed(1)}%</td><td>${d.t_low_pct.toFixed(1)}%</td>
                <td>${(d.t_low_pct - d.low_pct).toFixed(2)}%</td></tr>
            <tr><td>High-Freq %</td><td>${d.high_pct.toFixed(1)}%</td><td>${d.t_high_pct.toFixed(1)}%</td>
                <td>${(d.t_high_pct - d.high_pct).toFixed(2)}%</td></tr>
            <tr><td>L/H Ratio</td><td>${d.ratio.toFixed(2)}</td><td>${d.t_ratio.toFixed(2)}</td>
                <td>${(d.t_ratio - d.ratio).toFixed(2)}</td></tr>
            <tr><td>Total Energy</td><td>${d.total.toExponential(2)}</td><td>${d.t_total.toExponential(2)}</td>
                <td>${((d.t_total - d.total) / d.total * 100).toFixed(1)}%</td></tr>
        </tbody></table>`;
}

// ═══════════════════════════════════════
// METRICS (FR-3.6)
// ═══════════════════════════════════════
function formatMiniMetrics(m) {
    return `<span>MSE: <strong>${m.mse}</strong></span>` +
           `<span>PSNR: <strong>${m.psnr} dB</strong></span>` +
           `<span>SSIM: <strong>${m.ssim}</strong></span>`;
}

function addMetric(name, m) {
    S.metricsHistory.push({ name, mse: m.mse, psnr: m.psnr, ssim: m.ssim });
}

function initMetrics() {
    const tbody = $('mBody');
    if (S.metricsHistory.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="muted center">No data yet — apply transformations first</td></tr>';
        return;
    }
    tbody.innerHTML = S.metricsHistory.map(r =>
        `<tr><td>${escapeHtml(r.name)}</td><td>${r.mse}</td><td>${r.psnr} dB</td><td>${r.ssim}</td></tr>`
    ).join('');
}

async function exportCSV() {
    if (S.metricsHistory.length === 0) return toast('No metrics to export', 'error');
    try {
        const res = await fetch('/api/export_metrics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rows: S.metricsHistory })
        });
        const blob = await res.blob();
        downloadBlob(blob, 'metrics.csv');
        toast('Metrics exported', 'success');
    } catch (err) {
        toast('Export failed', 'error');
    }
}

function exportPDF() {
    window.print();
}

// ═══════════════════════════════════════
// IMAGE SAVE
// ═══════════════════════════════════════
function saveImg(imgId) {
    const img = $(imgId);
    if (!img || !img.src || !img.src.startsWith('data:')) return toast('No image to save', 'error');
    const a = document.createElement('a');
    a.href = img.src;
    a.download = imgId + '_result.png';
    a.click();
    toast('Image saved', 'success');
}

// ═══════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════
function downloadBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ═══════════════════════════════════════
// SLIDER VALUE DISPLAY
// ═══════════════════════════════════════
function initSliders() {
    document.querySelectorAll('.range').forEach(slider => {
        const id = slider.id;
        if (!id) return;
        const valId = 'v_' + id.replace('r_', '');
        const valEl = $(valId);
        if (!valEl) return;

        slider.addEventListener('input', () => {
            valEl.textContent = slider.value + '%';
        });
    });
}

// ═══════════════════════════════════════
// INIT
// ═══════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initSliders();

    // Sidebar navigation click handlers
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.disabled) return;
            goTab(btn.dataset.tab);
        });
    });
});
