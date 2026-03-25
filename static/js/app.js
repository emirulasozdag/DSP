// ═══════════ STATE ═══════════
let state = {
    originalImage: null,    // base64 of preprocessed image
    landmarks: [],
    transformedImage: null,
    lastTransformType: null,
    metricsHistory: []
};

// ═══════════ TAB NAVIGATION ═══════════
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.disabled) return;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
});

function enableTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.disabled = false);
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.remove('active');
        if (b.dataset.tab === tabName) b.classList.add('active');
    });
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
}

// ═══════════ LOADING & TOAST ═══════════
function showLoading(text) {
    document.getElementById('loadingText').textContent = text || 'Processing...';
    document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

function showToast(message, type) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast ' + (type || 'success') + ' show';
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// ═══════════ FILE UPLOAD (FR-3.1) ═══════════
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileUpload(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
});

function handleFileUpload(file) {
    // FR-3.1.1: Validate format
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showToast('Invalid format! Only JPG and PNG are supported.', 'error');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showToast('File too large! Maximum 16MB allowed.', 'error');
        return;
    }

    // Show original preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewOriginal').src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Upload to server
    showLoading('Uploading and preprocessing image...');
    const formData = new FormData();
    formData.append('image', file);

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        // Store state
        state.originalImage = data.original;
        state.landmarks = data.landmarks;

        // FR-3.1.3: Display processed image
        document.getElementById('previewProcessed').src = 'data:image/png;base64,' + data.original;
        document.getElementById('uploadPlaceholder').style.display = 'none';
        document.getElementById('uploadPreview').style.display = 'block';

        // Show info
        document.getElementById('uploadInfo').innerHTML = `
            <span><i class="fas fa-check-circle"></i> Face Detected: ${data.face_detected ? 'Yes' : 'No'}</span>
            <span><i class="fas fa-check-circle"></i> Landmarks: ${data.landmark_count} points</span>
            <span><i class="fas fa-check-circle"></i> Resized: ${data.image_size[0]}×${data.image_size[1]}</span>
            <span><i class="fas fa-check-circle"></i> Format Valid</span>
        `;

        enableTabs();
        showToast('Image uploaded and preprocessed successfully!');
    })
    .catch(err => {
        hideLoading();
        showToast('Upload failed: ' + err.message, 'error');
    });
}

function resetUpload() {
    state = { originalImage: null, landmarks: [], transformedImage: null, lastTransformType: null, metricsHistory: [] };
    document.getElementById('uploadPlaceholder').style.display = 'block';
    document.getElementById('uploadPreview').style.display = 'none';
    document.getElementById('fileInput').value = '';
    document.querySelectorAll('.tab-btn').forEach((btn, i) => {
        if (i > 0) btn.disabled = true;
    });
    switchTab('upload');
}

function proceedToLandmarks() {
    updateLandmarkView();
    switchTab('landmarks');
}

// ═══════════ LANDMARKS (FR-3.2) ═══════════
function updateLandmarkView() {
    if (!state.originalImage) return;

    showLoading('Rendering landmarks...');
    const showLandmarks = document.getElementById('toggleLandmarks').checked;

    fetch('/api/landmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: state.originalImage,
            landmarks: state.landmarks,
            show: showLandmarks
        })
    })
    .then(res => res.json())
    .then(data => {
        hideLoading();
        document.getElementById('landmarkImage').src = 'data:image/png;base64,' + data.image;
        document.getElementById('landmarkCount').textContent = state.landmarks.length;

        // Set up expression and aging originals
        document.getElementById('exprOriginal').src = 'data:image/png;base64,' + state.originalImage;
        document.getElementById('agingOriginal').src = 'data:image/png;base64,' + state.originalImage;
    })
    .catch(err => {
        hideLoading();
        showToast('Failed to render landmarks', 'error');
    });
}

function toggleLandmarkVisibility() {
    updateLandmarkView();
}

function exportLandmarks() {
    showLoading('Exporting landmarks...');
    fetch('/api/export_landmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks: state.landmarks })
    })
    .then(res => res.blob())
    .then(blob => {
        hideLoading();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'landmarks.csv';
        a.click();
        URL.revokeObjectURL(url);
        showToast('Landmarks exported as CSV');
    })
    .catch(err => {
        hideLoading();
        showToast('Export failed', 'error');
    });
}

// ═══════════ EXPRESSION MANIPULATION (FR-3.3) ═══════════

// Slider value display
['smile', 'eyebrow', 'lips', 'slim'].forEach(type => {
    const slider = document.getElementById(type + 'Slider');
    const display = document.getElementById(type + 'Value');
    slider.addEventListener('input', () => {
        display.textContent = slider.value + '%';
    });
});

function applyTransform(type) {
    if (!state.originalImage || !state.landmarks.length) {
        showToast('Please upload an image first', 'error');
        return;
    }

    const sliderId = type + 'Slider';
    const intensity = parseInt(document.getElementById(sliderId).value);

    showLoading(`Applying ${type} transformation...`);

    fetch('/api/transform', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: state.originalImage,
            landmarks: state.landmarks,
            type: type,
            intensity: intensity
        })
    })
    .then(res => res.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        state.transformedImage = data.transformed;
        state.lastTransformType = type;

        // FR-3.7.1: Side-by-side display
        document.getElementById('exprOriginal').src = 'data:image/png;base64,' + state.originalImage;
        document.getElementById('exprTransformed').src = 'data:image/png;base64,' + data.transformed;

        // Show metrics
        displayMiniMetrics('exprMetrics', data.metrics);
        addMetricsToHistory(getTransformName(type), data.metrics);

        showToast(`${getTransformName(type)} applied successfully!`);
    })
    .catch(err => {
        hideLoading();
        showToast('Transform failed: ' + err.message, 'error');
    });
}

function getTransformName(type) {
    const names = {
        'smile': 'Smile Enhancement',
        'eyebrow': 'Eyebrow Raising',
        'lips': 'Lip Widening',
        'slim': 'Face Slimming',
        'aging': 'Aging Simulation',
        'deaging': 'De-Aging Simulation'
    };
    return names[type] || type;
}

// ═══════════ AGING/DE-AGING (FR-3.4) ═══════════

['aging', 'deaging'].forEach(type => {
    const slider = document.getElementById(type + 'Slider');
    const display = document.getElementById(type + 'Value');
    slider.addEventListener('input', () => {
        display.textContent = slider.value + '%';
    });
});

function applyAging(type) {
    if (!state.originalImage) {
        showToast('Please upload an image first', 'error');
        return;
    }

    const sliderId = type + 'Slider';
    const intensity = parseInt(document.getElementById(sliderId).value);
    const mode = type === 'deaging' ? 'deaging' : 'aging';

    showLoading(`Applying ${type === 'deaging' ? 'de-aging' : 'aging'} simulation...`);

    fetch('/api/aging', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: state.originalImage,
            mode: mode,
            intensity: intensity
        })
    })
    .then(res => res.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        state.transformedImage = data.transformed;
        state.lastTransformType = type;

        document.getElementById('agingOriginal').src = 'data:image/png;base64,' + state.originalImage;
        document.getElementById('agingTransformed').src = 'data:image/png;base64,' + data.transformed;

        displayMiniMetrics('agingMetrics', data.metrics);
        addMetricsToHistory(getTransformName(type), data.metrics);

        showToast(`${getTransformName(type)} applied successfully!`);
    })
    .catch(err => {
        hideLoading();
        showToast('Aging transform failed: ' + err.message, 'error');
    });
}

// ═══════════ FREQUENCY ANALYSIS (FR-3.5) ═══════════

function analyzeFrequency(source) {
    if (!state.originalImage) {
        showToast('Please upload an image first', 'error');
        return;
    }

    showLoading('Computing frequency analysis...');

    const payload = { image: state.originalImage };
    if (state.transformedImage) {
        payload.transformed = state.transformedImage;
    }

    fetch('/api/frequency', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        hideLoading();

        // Original spectrum
        document.getElementById('origSpectrum').src = 'data:image/png;base64,' + data.spectrum_image;
        document.getElementById('origEnergyStats').innerHTML = `
            <div class="energy-stat"><span class="label">Total Energy</span><span class="value">${formatEnergy(data.total_energy)}</span></div>
            <div class="energy-stat"><span class="label">Low-Freq Energy</span><span class="value">${formatEnergy(data.low_energy)}</span></div>
            <div class="energy-stat"><span class="label">High-Freq Energy</span><span class="value">${formatEnergy(data.high_energy)}</span></div>
            <div class="energy-stat"><span class="label">LF/HF Ratio</span><span class="value">${data.lf_hf_ratio.toFixed(4)}</span></div>
        `;

        // Transformed spectrum
        if (data.transformed_spectrum) {
            document.getElementById('transSpectrum').src = 'data:image/png;base64,' + data.transformed_spectrum;
            document.getElementById('transEnergyStats').innerHTML = `
                <div class="energy-stat"><span class="label">Total Energy</span><span class="value">${formatEnergy(data.transformed_energy)}</span></div>
                <div class="energy-stat"><span class="label">Low-Freq Energy</span><span class="value">${formatEnergy(data.transformed_low_energy)}</span></div>
                <div class="energy-stat"><span class="label">High-Freq Energy</span><span class="value">${formatEnergy(data.transformed_high_energy)}</span></div>
                <div class="energy-stat"><span class="label">LF/HF Ratio</span><span class="value">${data.transformed_lf_hf_ratio.toFixed(4)}</span></div>
            `;

            // FR-3.5.6: Comparative analysis table
            document.getElementById('freqComparisonTable').innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Original</th>
                            <th>Transformed</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Total Spectral Energy</td>
                            <td>${formatEnergy(data.total_energy)}</td>
                            <td>${formatEnergy(data.transformed_energy)}</td>
                            <td>${formatChange(data.total_energy, data.transformed_energy)}</td>
                        </tr>
                        <tr>
                            <td>Low-Frequency Energy</td>
                            <td>${data.low_pct.toFixed(2)}%</td>
                            <td>${data.transformed_low_pct.toFixed(2)}%</td>
                            <td>${(data.transformed_low_pct - data.low_pct).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>High-Frequency Energy</td>
                            <td>${data.high_pct.toFixed(2)}%</td>
                            <td>${data.transformed_high_pct.toFixed(2)}%</td>
                            <td>${(data.transformed_high_pct - data.high_pct).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>LF/HF Ratio</td>
                            <td>${data.lf_hf_ratio.toFixed(4)}</td>
                            <td>${data.transformed_lf_hf_ratio.toFixed(4)}</td>
                            <td>${(data.transformed_lf_hf_ratio - data.lf_hf_ratio).toFixed(4)}</td>
                        </tr>
                    </tbody>
                </table>
            `;
        }

        switchTab('frequency');
        showToast('Frequency analysis complete');
    })
    .catch(err => {
        hideLoading();
        showToast('Frequency analysis failed', 'error');
    });
}

function formatEnergy(val) {
    if (val > 1e12) return (val / 1e12).toFixed(2) + 'T';
    if (val > 1e9) return (val / 1e9).toFixed(2) + 'G';
    if (val > 1e6) return (val / 1e6).toFixed(2) + 'M';
    if (val > 1e3) return (val / 1e3).toFixed(2) + 'K';
    return val.toFixed(2);
}

function formatChange(orig, trans) {
    const pct = ((trans - orig) / (orig + 1e-10)) * 100;
    const sign = pct >= 0 ? '+' : '';
    return sign + pct.toFixed(2) + '%';
}

// ═══════════ METRICS (FR-3.6) ═══════════

function displayMiniMetrics(containerId, metrics) {
    document.getElementById(containerId).innerHTML = `
        <div class="metric-item">
            <span class="metric-label">MSE</span>
            <span class="metric-value">${metrics.mse.toFixed(2)}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">PSNR</span>
            <span class="metric-value">${metrics.psnr.toFixed(2)} dB</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">SSIM</span>
            <span class="metric-value">${metrics.ssim.toFixed(4)}</span>
        </div>
    `;
}

function addMetricsToHistory(name, metrics) {
    // Remove if already exists
    state.metricsHistory = state.metricsHistory.filter(m => m.name !== name);
    state.metricsHistory.push({ name, ...metrics });
    updateMetricsTable();
}

function updateMetricsTable() {
    const tbody = document.getElementById('metricsBody');
    if (state.metricsHistory.length === 0) {
        tbody.innerHTML = '<tr class="placeholder-row"><td colspan="4">No transformations applied yet.</td></tr>';
        return;
    }

    tbody.innerHTML = state.metricsHistory.map(m => `
        <tr>
            <td><strong>${m.name}</strong></td>
            <td>${m.mse.toFixed(4)}</td>
            <td>${m.psnr.toFixed(4)}</td>
            <td>${m.ssim.toFixed(4)}</td>
        </tr>
    `).join('');
}

// FR-3.6.5: Export metrics as CSV
function exportMetricsCSV() {
    if (state.metricsHistory.length === 0) {
        showToast('No metrics to export', 'error');
        return;
    }

    fetch('/api/export_metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metrics: state.metricsHistory })
    })
    .then(res => res.blob())
    .then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'metrics_report.csv';
        a.click();
        URL.revokeObjectURL(url);
        showToast('Metrics exported as CSV');
    })
    .catch(() => showToast('Export failed', 'error'));
}

// FR-3.6.5: Export metrics as PDF (simple HTML-based)
function exportMetricsPDF() {
    if (state.metricsHistory.length === 0) {
        showToast('No metrics to export', 'error');
        return;
    }

    const printWindow = window.open('', '_blank');
    const rows = state.metricsHistory.map(m =>
        `<tr><td>${m.name}</td><td>${m.mse.toFixed(4)}</td><td>${m.psnr.toFixed(4)}</td><td>${m.ssim.toFixed(4)}</td></tr>`
    ).join('');

    printWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>DSP Metrics Report</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 2rem; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 2rem; }
                table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
                th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
                th { background: #f5f5f5; font-weight: bold; }
                .footer { margin-top: 2rem; color: #999; font-size: 0.8rem; }
            </style>
        </head>
        <body>
            <h1>Facial Image Transformation - Evaluation Report</h1>
            <p>CENG 384 - DSP Project | Ankara Science University</p>
            <h2>Quantitative Metrics</h2>
            <table>
                <thead>
                    <tr><th>Transformation</th><th>MSE</th><th>PSNR (dB)</th><th>SSIM</th></tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
            <div class="footer">
                <p>Generated: ${new Date().toLocaleString()}</p>
                <p>MSE: Mean Squared Error | PSNR: Peak Signal-to-Noise Ratio | SSIM: Structural Similarity Index</p>
            </div>
        </body>
        </html>
    `);
    printWindow.document.close();
    printWindow.print();
}

// ═══════════ SAVE IMAGE (FR-3.7.3) ═══════════

function saveImage(imgElementId) {
    const img = document.getElementById(imgElementId);
    if (!img.src || img.src === '') {
        showToast('No image to save', 'error');
        return;
    }

    // Download via link
    const a = document.createElement('a');
    a.href = img.src;
    a.download = 'transformed_face.png';
    a.click();
    showToast('Image saved');
}
