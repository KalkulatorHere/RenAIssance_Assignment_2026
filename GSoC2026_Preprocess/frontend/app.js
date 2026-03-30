/**
 * app.js — OCR Preprocess Studio
 * ────────────────────────────────────────────────────────────
 * Clean module pattern: State → API → UI helpers → Event wiring
 * No build step; runs directly in browser against the FastAPI backend.
 */

'use strict';

/* ════════════════════════════════════════════════════════════
   STATE
════════════════════════════════════════════════════════════ */
const State = {
  sessionId:     null,
  pageCount:     0,
  currentPage:   0,        // 0-indexed
  savedPages:    new Set(), // page indices saved to disk
  previewCache:  new Map(), // page_idx → object URL (original JPEG)
  activeJobId:   null,
  pollTimer:     null,
  debounceTimer: null,

  defaults: {
    deskew_range:    10.0,
    bg_sigma:        80.0,
    denoise_h:       6.0,
    binarize_window: 51,
    binarize_k:      0.2,
    morph_kernel:    2,
  },
};

/* ════════════════════════════════════════════════════════════
   API LAYER
════════════════════════════════════════════════════════════ */
const API_BASE = '';   // same origin

const API = {
  async loadPdf(file, dpi) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('dpi', dpi);
    const r = await fetch(`${API_BASE}/load-pdf`, { method: 'POST', body: fd });
    return _checkJson(r);
  },

  originalUrl(pageIdx) {
    return `${API_BASE}/page/${pageIdx}/original?session_id=${encodeURIComponent(State.sessionId)}&t=${Date.now()}`;
  },

  async previewPage(pageIdx, params) {
    const r = await fetch(`${API_BASE}/page/${pageIdx}/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: State.sessionId, params }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(e.detail || r.statusText);
    }
    return r.blob();
  },

  async savePage(pageIdx, params, outputDir) {
    const r = await fetch(`${API_BASE}/save-page`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: State.sessionId, page_idx: pageIdx, params, output_dir: outputDir }),
    });
    return _checkJson(r);
  },

  async processAll(params, outputDir) {
    const r = await fetch(`${API_BASE}/process-all`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: State.sessionId, params, output_dir: outputDir }),
    });
    return _checkJson(r);
  },

  async jobStatus(jobId) {
    const r = await fetch(`${API_BASE}/status/${jobId}`);
    return _checkJson(r);
  },

  async exportPdf(outputDir, pageIndices) {
    const body = { session_id: State.sessionId, output_dir: outputDir };
    if (pageIndices !== null) body.page_indices = pageIndices;
    const r = await fetch(`${API_BASE}/export-pdf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(e.detail || r.statusText);
    }
    return r.blob();
  },
};

async function _checkJson(r) {
  const data = await r.json().catch(() => ({ detail: r.statusText }));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

/* ════════════════════════════════════════════════════════════
   UI — DOM REFERENCES
════════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);

const El = {
  fileInput:       $('pdf-file-input'),
  pdfInfo:         $('pdf-info'),
  pdfName:         $('pdf-name'),
  pdfPages:        $('pdf-pages'),
  dpiInput:        $('dpi-input'),
  pageLabel:       $('page-label'),
  btnPrev:         $('btn-prev-page'),
  btnNext:         $('btn-next-page'),
  thumbStrip:      $('thumb-strip'),
  btnPreview:      $('btn-preview'),
  btnSavePage:     $('btn-save-page'),
  btnProcessAll:   $('btn-process-all'),
  btnExportPdf:    $('btn-export-pdf'),
  btnReset:        $('btn-reset'),
  outputDir:       $('output-dir'),
  imgOriginal:     $('img-original'),
  imgProcessed:    $('img-processed'),
  placeholderOrig: $('placeholder-orig'),
  placeholderProc: $('placeholder-proc'),
  spinnerOrig:     $('spinner-orig'),
  spinnerProc:     $('spinner-proc'),
  origPageLabel:   $('orig-page-label'),
  procPageLabel:   $('proc-page-label'),
  progressWrap:    $('progress-wrap'),
  progressBar:     $('progress-bar'),
  progressText:    $('progress-text'),
  statusDot:       document.querySelector('.status-dot'),
  statusText:      $('status-text'),
  toastContainer:  $('toast-container'),
  savedPagesBar:   $('saved-pages-bar'),
  savedChips:      $('saved-chips'),
};

/* Parameter slider ↔ number input bindings */
const PARAM_PAIRS = [
  ['p-deskew-range',    'n-deskew-range',    'deskew_range'],
  ['p-bg-sigma',        'n-bg-sigma',        'bg_sigma'],
  ['p-denoise-h',       'n-denoise-h',       'denoise_h'],
  ['p-binarize-window', 'n-binarize-window', 'binarize_window'],
  ['p-binarize-k',      'n-binarize-k',      'binarize_k'],
  ['p-morph-kernel',    'n-morph-kernel',    'morph_kernel'],
];

/* ════════════════════════════════════════════════════════════
   UI — STATUS / TOAST
════════════════════════════════════════════════════════════ */
function setStatus(text, mode = 'idle') {
  El.statusText.textContent = text;
  El.statusDot.className = `status-dot ${mode}`;
}

function toast(msg, type = 'info', duration = 4000) {
  const t = document.createElement('div');
  t.className = `toast toast--${type}`;
  t.textContent = msg;
  El.toastContainer.appendChild(t);
  requestAnimationFrame(() => { requestAnimationFrame(() => t.classList.add('show')); });
  setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => t.remove(), 250);
  }, duration);
}

/* ════════════════════════════════════════════════════════════
   UI — PARAMS
════════════════════════════════════════════════════════════ */
function getParams() {
  const p = {};
  for (const [sliderId, , key] of PARAM_PAIRS) {
    p[key] = parseFloat($(sliderId).value);
  }
  return p;
}

function syncSliderToNumber(sliderId, numId) {
  const s = $(sliderId), n = $(numId);
  s.addEventListener('input', () => { n.value = s.value; schedulePreview(); });
  n.addEventListener('change', () => {
    s.value = n.value;
    schedulePreview();
  });
}

function resetParams() {
  const d = State.defaults;
  for (const [sliderId, numId, key] of PARAM_PAIRS) {
    $(sliderId).value = d[key];
    $(numId).value    = d[key];
  }
  schedulePreview();
}

/* ════════════════════════════════════════════════════════════
   UI — PAGE SELECTION
════════════════════════════════════════════════════════════ */
function selectPage(idx) {
  if (!State.sessionId || idx < 0 || idx >= State.pageCount) return;
  State.currentPage = idx;
  El.pageLabel.textContent = `Page ${idx + 1} / ${State.pageCount}`;
  El.origPageLabel.textContent = `Page ${idx + 1}`;
  El.procPageLabel.textContent = `Page ${idx + 1}`;
  El.btnPrev.disabled = idx === 0;
  El.btnNext.disabled = idx === State.pageCount - 1;
  updateThumbActive(idx);
  loadOriginalPage(idx);
  clearProcessedPanel();
}

function updateThumbActive(idx) {
  document.querySelectorAll('.thumb').forEach((t, i) => {
    t.classList.toggle('active', i === idx);
  });
}

/* ════════════════════════════════════════════════════════════
   UI — THUMBNAIL STRIP
════════════════════════════════════════════════════════════ */
function buildThumbStrip() {
  El.thumbStrip.innerHTML = '';
  for (let i = 0; i < State.pageCount; i++) {
    const wrap = document.createElement('div');
    wrap.className = 'thumb-wrap';

    const img = document.createElement('img');
    img.className = 'thumb';
    img.alt = `Page ${i + 1}`;
    img.loading = 'lazy';
    img.dataset.page = i;
    img.addEventListener('click', () => selectPage(i));

    const num = document.createElement('div');
    num.className = 'thumb-num';
    num.textContent = i + 1;

    const dot = document.createElement('div');
    dot.className = 'thumb-saved-dot';
    dot.id = `saved-dot-${i}`;

    wrap.appendChild(img);
    wrap.appendChild(num);
    wrap.appendChild(dot);
    El.thumbStrip.appendChild(wrap);

    // Load thumbnail lazily via IntersectionObserver
    _lazyLoadThumb(img, i);
  }
}

const _thumbObserver = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) {
      const img = e.target;
      const pageIdx = parseInt(img.dataset.page, 10);
      if (!img.src) {
        img.src = API.originalUrl(pageIdx);
      }
      _thumbObserver.unobserve(img);
    }
  }
}, { rootMargin: '100px' });

function _lazyLoadThumb(img, pageIdx) {
  _thumbObserver.observe(img);
}

/* ════════════════════════════════════════════════════════════
   UI — ORIGINAL PANEL
════════════════════════════════════════════════════════════ */
function loadOriginalPage(pageIdx) {
  showSpinner(El.spinnerOrig, true);
  El.imgOriginal.classList.add('hidden');
  El.placeholderOrig.classList.add('hidden');

  // Revoke old object URL if any
  if (State.previewCache.has(pageIdx)) {
    El.imgOriginal.src = State.previewCache.get(pageIdx);
    showSpinner(El.spinnerOrig, false);
    El.imgOriginal.classList.remove('hidden');
    return;
  }

  const url = API.originalUrl(pageIdx);
  El.imgOriginal.onload = () => {
    State.previewCache.set(pageIdx, url);
    showSpinner(El.spinnerOrig, false);
    El.imgOriginal.classList.remove('hidden');
  };
  El.imgOriginal.onerror = () => {
    showSpinner(El.spinnerOrig, false);
    El.placeholderOrig.classList.remove('hidden');
    toast(`Could not load page ${pageIdx + 1}`, 'error');
  };
  El.imgOriginal.src = url;
}

/* ════════════════════════════════════════════════════════════
   UI — PROCESSED PANEL
════════════════════════════════════════════════════════════ */
function clearProcessedPanel() {
  El.imgProcessed.src = '';
  El.imgProcessed.classList.add('hidden');
  El.placeholderProc.classList.remove('hidden');
  showSpinner(El.spinnerProc, false);
}

function showProcessedImage(blob) {
  const url = URL.createObjectURL(blob);
  El.imgProcessed.onload = () => {
    showSpinner(El.spinnerProc, false);
    El.imgProcessed.classList.remove('hidden');
    El.placeholderProc.classList.add('hidden');
    URL.revokeObjectURL(url);   // revoke after display
  };
  El.imgProcessed.src = url;
}

function showSpinner(el, show) {
  el.classList.toggle('hidden', !show);
}

/* ════════════════════════════════════════════════════════════
   UI — SAVED PAGES BAR
════════════════════════════════════════════════════════════ */
function markPageSaved(pageIdx) {
  State.savedPages.add(pageIdx);

  // Show green dot on thumbnail
  const dot = $(`saved-dot-${pageIdx}`);
  if (dot) dot.classList.add('visible');

  // Rebuild chips
  El.savedPagesBar.classList.remove('hidden');
  El.savedChips.innerHTML = '';
  for (const idx of [...State.savedPages].sort((a, b) => a - b)) {
    const chip = document.createElement('span');
    chip.className = 'saved-chip';
    chip.textContent = `Page ${idx + 1}`;
    El.savedChips.appendChild(chip);
  }

  // Enable export
  El.btnExportPdf.disabled = false;
}

/* ════════════════════════════════════════════════════════════
   UI — PROGRESS BAR
════════════════════════════════════════════════════════════ */
function showProgress(show) {
  El.progressWrap.classList.toggle('hidden', !show);
}
function setProgress(done, total) {
  const pct = total > 0 ? (done / total) * 100 : 0;
  El.progressBar.style.setProperty('--progress', `${pct}%`);
  El.progressText.textContent = `${done} / ${total}`;
}

/* ════════════════════════════════════════════════════════════
   BUTTON ENABLE/DISABLE
════════════════════════════════════════════════════════════ */
function setPageButtonsEnabled(enabled) {
  El.btnPreview.disabled      = !enabled;
  El.btnSavePage.disabled     = !enabled;
  El.btnProcessAll.disabled   = !enabled;
}

/* ════════════════════════════════════════════════════════════
   DEBOUNCED LIVE PREVIEW
════════════════════════════════════════════════════════════ */
function schedulePreview() {
  if (!State.sessionId) return;
  clearTimeout(State.debounceTimer);
  State.debounceTimer = setTimeout(runPreview, 320);
}

async function runPreview() {
  if (!State.sessionId) return;
  const pageIdx = State.currentPage;
  const params  = getParams();

  showSpinner(El.spinnerProc, true);
  El.imgProcessed.classList.add('hidden');
  setStatus(`Processing page ${pageIdx + 1}…`, 'active');

  try {
    const blob = await API.previewPage(pageIdx, params);
    showProcessedImage(blob);
    setStatus(`Page ${pageIdx + 1} previewed`, 'success');
  } catch (err) {
    showSpinner(El.spinnerProc, false);
    El.placeholderProc.classList.remove('hidden');
    setStatus('Preview failed', 'error');
    toast(`Preview error: ${err.message}`, 'error');
  }
}

/* ════════════════════════════════════════════════════════════
   BATCH JOB POLLING
════════════════════════════════════════════════════════════ */
function startPolling(jobId, total) {
  State.activeJobId = jobId;
  showProgress(true);
  setProgress(0, total);

  State.pollTimer = setInterval(async () => {
    try {
      const s = await API.jobStatus(jobId);
      setProgress(s.done, s.total);
      setStatus(`Processing… ${s.done}/${s.total} pages`, 'active');

      if (s.finished) {
        clearInterval(State.pollTimer);
        State.pollTimer = null;
        State.activeJobId = null;

        // Mark all pages saved
        for (let i = 0; i < s.total; i++) markPageSaved(i);

        // Refresh thumbnail strip (all pages now processed)
        buildThumbStrip();
        selectPage(State.currentPage);

        setStatus(`All ${s.total} pages processed`, 'success');
        toast(`All ${s.total} pages processed successfully!`, 'success');

        if (s.errors.length) {
          toast(`${s.errors.length} page(s) had errors — check the console.`, 'error');
          console.warn('Batch errors:', s.errors);
        }

        showProgress(false);
        setPageButtonsEnabled(true);
        El.btnProcessAll.disabled = false;
      }
    } catch (err) {
      clearInterval(State.pollTimer);
      State.pollTimer = null;
      setStatus('Batch job polling failed', 'error');
      toast(`Status error: ${err.message}`, 'error');
      showProgress(false);
    }
  }, 800);
}

/* ════════════════════════════════════════════════════════════
   EVENT WIRING
════════════════════════════════════════════════════════════ */

// ── File picker ────────────────────────────────────────────
El.fileInput.addEventListener('change', async () => {
  const file = El.fileInput.files[0];
  if (!file) return;

  const dpi = parseInt(El.dpiInput.value, 10) || 300;

  setStatus('Loading PDF…', 'active');
  El.btnPreview.disabled    = true;
  El.btnSavePage.disabled   = true;
  El.btnProcessAll.disabled = true;
  El.btnExportPdf.disabled  = true;
  El.thumbStrip.innerHTML   = '<div class="thumb-strip__empty">Loading…</div>';

  try {
    const data = await API.loadPdf(file, dpi);

    // Reset session state
    State.sessionId   = data.session_id;
    State.pageCount   = data.page_count;
    State.currentPage = 0;
    State.savedPages  = new Set();
    State.previewCache.clear();
    El.savedPagesBar.classList.add('hidden');
    El.savedChips.innerHTML = '';
    El.btnExportPdf.disabled = true;

    El.pdfName.textContent  = file.name;
    El.pdfPages.textContent = `${data.page_count} page${data.page_count !== 1 ? 's' : ''} · ${dpi} DPI`;
    El.pdfInfo.classList.remove('hidden');

    buildThumbStrip();
    setPageButtonsEnabled(true);
    selectPage(0);

    setStatus(`Loaded "${file.name}" — ${data.page_count} pages`, 'success');
    toast(`PDF loaded: ${data.page_count} pages at ${dpi} DPI`, 'success');
  } catch (err) {
    setStatus('Failed to load PDF', 'error');
    toast(`Load error: ${err.message}`, 'error');
    El.thumbStrip.innerHTML = '<div class="thumb-strip__empty">Load failed</div>';
  }
});

// ── Page navigation ────────────────────────────────────────
El.btnPrev.addEventListener('click', () => selectPage(State.currentPage - 1));
El.btnNext.addEventListener('click', () => selectPage(State.currentPage + 1));

// ── Keyboard shortcuts ─────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft')  selectPage(State.currentPage - 1);
  if (e.key === 'ArrowRight') selectPage(State.currentPage + 1);
  if (e.key === 'Enter' && State.sessionId) runPreview();
});

// ── Param sliders ↔ numbers ────────────────────────────────
for (const [sliderId, numId] of PARAM_PAIRS) {
  syncSliderToNumber(sliderId, numId);
}

// ── Reset ──────────────────────────────────────────────────
El.btnReset.addEventListener('click', resetParams);

// ── Preview page ───────────────────────────────────────────
El.btnPreview.addEventListener('click', runPreview);

// ── Save current page ──────────────────────────────────────
El.btnSavePage.addEventListener('click', async () => {
  if (!State.sessionId) return;
  const pageIdx   = State.currentPage;
  const params    = getParams();
  const outputDir = El.outputDir.value.trim() || 'ocr_output';

  setStatus(`Saving page ${pageIdx + 1}…`, 'active');
  El.btnSavePage.disabled = true;

  try {
    const data = await API.savePage(pageIdx, params, outputDir);
    markPageSaved(pageIdx);
    setStatus(`Page ${pageIdx + 1} saved`, 'success');
    toast(`Page ${pageIdx + 1} saved → ${data.path}`, 'success');
  } catch (err) {
    setStatus('Save failed', 'error');
    toast(`Save error: ${err.message}`, 'error');
  } finally {
    El.btnSavePage.disabled = false;
  }
});

// ── Process entire PDF ─────────────────────────────────────
El.btnProcessAll.addEventListener('click', async () => {
  if (!State.sessionId) return;
  const params    = getParams();
  const outputDir = El.outputDir.value.trim() || 'ocr_output';

  setStatus('Starting batch processing…', 'active');
  setPageButtonsEnabled(false);
  El.btnProcessAll.disabled = true;

  try {
    const data = await API.processAll(params, outputDir);
    toast(`Batch started — ${data.total} pages. Processing…`, 'info');
    startPolling(data.job_id, data.total);
  } catch (err) {
    setStatus('Batch start failed', 'error');
    toast(`Batch error: ${err.message}`, 'error');
    setPageButtonsEnabled(true);
    El.btnProcessAll.disabled = false;
  }
});

// ── Export as PDF ──────────────────────────────────────────
El.btnExportPdf.addEventListener('click', async () => {
  if (!State.sessionId || State.savedPages.size === 0) return;
  const outputDir    = El.outputDir.value.trim() || 'ocr_output';
  const pageIndices  = [...State.savedPages].sort((a, b) => a - b);

  setStatus('Exporting PDF…', 'active');
  El.btnExportPdf.disabled = true;

  try {
    const blob = await API.exportPdf(outputDir, pageIndices);
    // Trigger browser download
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'cleaned.pdf';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 2000);

    setStatus('PDF exported', 'success');
    toast(`cleaned.pdf downloaded (${pageIndices.length} pages)`, 'success');
  } catch (err) {
    setStatus('Export failed', 'error');
    toast(`Export error: ${err.message}`, 'error');
  } finally {
    El.btnExportPdf.disabled = State.savedPages.size === 0;
  }
});
