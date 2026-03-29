/* ============================================================
   DSP Face Lab — Main JavaScript
   Navigation, Upload Zone, Sliders, Toggles, Interactions
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initMobileMenu();
    initUploadZone();
    initSliders();
    initToggles();
    initEnergyBars();
});

/* ---------- Navbar scroll effect & active state ---------- */
function initNavbar() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;

    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 30);
    });

    // Set active link based on current page
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-links a').forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPage || (currentPage === '' && href === 'index.html')) {
            link.classList.add('active');
        }
    });
}

/* ---------- Mobile hamburger menu ---------- */
function initMobileMenu() {
    const toggle = document.querySelector('.nav-toggle');
    const links = document.querySelector('.nav-links');
    if (!toggle || !links) return;

    toggle.addEventListener('click', () => {
        links.classList.toggle('open');
        toggle.classList.toggle('active');
    });

    // Close on link click
    links.querySelectorAll('a').forEach(a => {
        a.addEventListener('click', () => {
            links.classList.remove('open');
            toggle.classList.remove('active');
        });
    });
}

/* ---------- Upload Zone — Drag & Drop visual feedback ---------- */
function initUploadZone() {
    const zone = document.querySelector('.upload-zone');
    if (!zone) return;

    const fileInput = zone.querySelector('input[type="file"]');
    const previewSection = document.querySelector('.upload-preview');
    const previewImg = document.querySelector('.preview-image img');
    const previewPlaceholder = document.querySelector('.preview-image .placeholder-face');
    const fileName = document.querySelector('.file-name');
    const fileRes = document.querySelector('.file-resolution');
    const fileSize = document.querySelector('.file-size');
    const validationStatus = document.querySelector('.validation-status');
    const proceedBtn = document.querySelector('.btn-proceed');

    // Drag events
    ['dragenter', 'dragover'].forEach(evt => {
        zone.addEventListener(evt, e => {
            e.preventDefault();
            zone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        zone.addEventListener(evt, e => {
            e.preventDefault();
            zone.classList.remove('dragover');
        });
    });

    // Click to open file dialog
    zone.addEventListener('click', () => {
        if (fileInput) fileInput.click();
    });

    // Handle drop
    zone.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    // Handle file input change
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
        });
    }

    function handleFile(file) {
        const validTypes = ['image/jpeg', 'image/png'];
        const isValid = validTypes.includes(file.type);

        // Show preview section
        if (previewSection) previewSection.classList.remove('hidden');

        // Update file info
        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = `Size: ${(file.size / 1024).toFixed(1)} KB`;

        // Show image preview
        if (isValid) {
            const reader = new FileReader();
            reader.onload = e => {
                if (previewImg) {
                    previewImg.src = e.target.result;
                    previewImg.classList.remove('hidden');
                }
                if (previewPlaceholder) previewPlaceholder.classList.add('hidden');

                // Get resolution
                const img = new Image();
                img.onload = () => {
                    if (fileRes) fileRes.textContent = `Resolution: ${img.width} × ${img.height}`;
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        // Validation status
        if (validationStatus) {
            if (isValid) {
                validationStatus.className = 'validation-status valid';
                validationStatus.innerHTML = '✓ Valid format';
            } else {
                validationStatus.className = 'validation-status invalid';
                validationStatus.innerHTML = '✗ Unsupported format';
            }
        }

        // Enable/disable proceed button
        if (proceedBtn) {
            if (isValid) {
                proceedBtn.classList.remove('disabled');
            } else {
                proceedBtn.classList.add('disabled');
            }
        }
    }
}

/* ---------- Sliders — live value display ---------- */
function initSliders() {
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        const valueDisplay = slider.closest('.slider-group')?.querySelector('.slider-value');
        if (!valueDisplay) return;

        const update = () => {
            const min = parseFloat(slider.min) || 0;
            const max = parseFloat(slider.max) || 100;
            const val = parseFloat(slider.value);

            // If the slider has a data-unit attribute, use that
            const unit = slider.dataset.unit || '%';

            if (unit === 'years') {
                valueDisplay.textContent = `${val > 0 ? '+' : ''}${val} yrs`;
            } else {
                valueDisplay.textContent = `${val}${unit}`;
            }

            // Update CSS custom property for fill color
            const percent = ((val - min) / (max - min)) * 100;
            slider.style.background = `linear-gradient(90deg, var(--neon-cyan) 0%, var(--neon-cyan) ${percent}%, rgba(255,255,255,0.08) ${percent}%)`;
        };

        slider.addEventListener('input', update);
        update(); // init
    });
}

/* ---------- Toggle switches ---------- */
function initToggles() {
    document.querySelectorAll('.toggle-switch input').forEach(toggle => {
        toggle.addEventListener('change', () => {
            const target = toggle.dataset.target;
            if (!target) return;

            const el = document.querySelector(target);
            if (el) {
                el.classList.toggle('visible', toggle.checked);
            }
        });
    });
}

/* ---------- Energy Bars — animate on scroll ---------- */
function initEnergyBars() {
    const bars = document.querySelectorAll('.energy-bar-fill');
    if (bars.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const fill = entry.target;
                const width = fill.dataset.width || '0%';
                fill.style.width = width;
                observer.unobserve(fill);
            }
        });
    }, { threshold: 0.3 });

    bars.forEach(bar => {
        bar.style.width = '0%';
        observer.observe(bar);
    });
}

/* ---------- Reset All (Transform page) ---------- */
function resetAll() {
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        // For age slider, default is 0; for others, default is 0 or as set
        slider.value = slider.defaultValue || 0;
        slider.dispatchEvent(new Event('input'));
    });

    document.querySelectorAll('.toggle-switch input').forEach(toggle => {
        toggle.checked = false;
        toggle.dispatchEvent(new Event('change'));
    });
}
