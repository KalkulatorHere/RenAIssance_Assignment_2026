# OCR Preprocess Studio

A polished, full-featured web UI for preprocessing scanned PDF pages for OCR.
Built on top of the existing `ocr_preprocess (1).py` pipeline.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you already have PyMuPDF, OpenCV, scikit-image, and NumPy installed
> (from the pipeline), you only need to add FastAPI and uvicorn:
> ```bash
> pip install fastapi "uvicorn[standard]" python-multipart
> ```

### 2. Run the server

```bash
python run.py
```

The app opens automatically in your browser at `http://127.0.0.1:8000`.

Optional flags:

| Flag | Description |
|---|---|
| `--host 0.0.0.0` | Listen on all interfaces (LAN access) |
| `--port 9000` | Use a different port |
| `--no-browser` | Don't auto-open the browser |
| `--reload` | Dev mode: hot-reload on file changes |

---

## Project Structure

```
GSoC2026_Preprocess/
├── ocr_preprocess (1).py      ← original pipeline (untouched)
├── backend/
│   ├── __init__.py
│   ├── app.py                 ← FastAPI server
│   └── preprocess_core.py     ← pipeline importer + helpers
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── requirements.txt
├── run.py
└── README.md
```

---

## Architecture

### Data flow

```
Browser ──upload PDF──► POST /load-pdf
                              │  pdf_to_images() → BGR arrays cached in memory
                              │  returns session_id + page_count
         ◄──────────────session_id──────────────────────────────────────────

Browser ──select page─► GET /page/{n}/original
                              │  encode cached raw array → JPEG
         ◄──────────────JPEG (original panel)───────────────────────────────

Browser ──adjust param─► POST /page/{n}/preview   (debounced 320 ms)
                              │  preprocess_page() on ONE cached array
                              │  encode result → JPEG
         ◄──────────────JPEG (processed panel)──────────────────────────────

Browser ──Save Page───► POST /save-page
                              │  writes PNG to output_dir/page_NNNN.png
         ◄──────────────{path}───────────────────────────────────────────────

Browser ──Process All─► POST /process-all → {job_id}
                              │  BatchJob runs in thread pool
        ──poll─────────► GET /status/{job_id}  (every 800 ms)
         ◄──────────────{done, total, finished}──────────────────────────────

Browser ──Export PDF──► POST /export-pdf
                              │  images_to_pdf() on saved PNGs
         ◄──────────────cleaned.pdf (download)──────────────────────────────
```

### Key design decisions

- **Session cache** — `pdf_to_images()` runs once when the PDF loads. All raw BGR
  arrays stay in memory. Only one page is re-processed on every parameter change,
  keeping preview latency low.
- **Thread pool** — All CPU-bound work (image processing, encoding) runs in
  `ThreadPoolExecutor`, leaving the asyncio event loop free.
- **Debounce** — Parameter slider changes are debounced by 320 ms before triggering
  a preview request, so rapid slider drags don't flood the server.
- **Thumbnail strip** — Thumbnails are lazy-loaded via `IntersectionObserver`.
  Only thumbnails in the viewport are fetched.
- **No duplication** — `preprocess_core.py` imports the original pipeline file via
  `importlib.util` (handles the odd filename). No preprocessing logic is copied.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| DPI | 300 | Render resolution when loading the PDF |
| Deskew range (°) | 10 | Max absolute skew angle to correct |
| BG blur sigma | 80 | Gaussian sigma for illumination normalisation |
| Denoise strength (h) | 6 | NLM filter strength (5–10 = light) |
| Sauvola window (px) | 51 | Local threshold neighbourhood size (must be odd) |
| Sauvola k | 0.2 | Sauvola sensitivity constant (0.1–0.5) |
| Morph kernel (px) | 2 | Structuring element size for opening cleanup |

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `←` / `→` | Previous / next page |
| `Enter` | Preview current page |
