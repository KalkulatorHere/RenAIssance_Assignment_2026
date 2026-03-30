"""
backend/app.py
──────────────
FastAPI application for the OCR preprocessing UI.

Routes
──────
POST /load-pdf                – upload a PDF; cache pages in memory
GET  /page/{n}/original       – return JPEG of the raw cached page
POST /page/{n}/preview        – run pipeline on ONE page; return JPEG
POST /save-page               – save ONE processed page to disk
POST /process-all             – queue batch job for all pages
GET  /status/{job_id}         – poll batch-job progress
POST /export-pdf              – pack saved pages into a PDF; return file
DELETE /session/{session_id}  – free memory (optional cleanup)
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.preprocess_core import (
    DEFAULT_PARAMS,
    encode_jpeg,
    encode_png,
    images_to_pdf,
    pdf_to_images,
    preprocess_page,
)

# ───────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="OCR Preprocess UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound preprocessing (avoids blocking the event loop)
_pool = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))


# ═══════════════════════════════════════════════════════════════════════════════
# In-memory session store
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Session:
    pdf_path: str                              # original PDF (temp file)
    raw_pages: list[np.ndarray]               # BGR arrays, one per page
    processed: dict[int, np.ndarray] = field(default_factory=dict)
    saved_paths: dict[int, str] = field(default_factory=dict)


@dataclass
class BatchJob:
    total: int
    done: int = 0
    errors: list[str] = field(default_factory=list)
    finished: bool = False


# Global stores (single-process; sufficient for a local desktop tool)
_sessions: dict[str, Session] = {}
_jobs: dict[str, BatchJob] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_session(session_id: str) -> Session:
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return s


def _coerce_params(raw: dict) -> dict:
    """Merge caller params over defaults, applying type coercion."""
    p = dict(DEFAULT_PARAMS)
    p.update(raw)
    p["dpi"] = int(p["dpi"])
    p["binarize_window"] = int(p["binarize_window"])
    p["morph_kernel"] = int(p["morph_kernel"])
    p["deskew_range"] = float(p["deskew_range"])
    p["bg_sigma"] = float(p["bg_sigma"])
    p["denoise_h"] = float(p["denoise_h"])
    p["binarize_k"] = float(p["binarize_k"])
    # binarize_window must be odd
    if p["binarize_window"] % 2 == 0:
        p["binarize_window"] += 1
    return p


def _process_one(raw: np.ndarray, params: dict) -> np.ndarray:
    """Run the full preprocessing pipeline on a single BGR page (blocking)."""
    return preprocess_page(
        raw,
        deskew_range=params["deskew_range"],
        bg_sigma=params["bg_sigma"],
        denoise_h=params["denoise_h"],
        binarize_window=params["binarize_window"],
        binarize_k=params["binarize_k"],
        morph_kernel=params["morph_kernel"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Routes – PDF loading
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/load-pdf")
async def load_pdf(file: UploadFile = File(...), dpi: int = Form(300)):
    """
    Accept a PDF upload, render all pages to memory at *dpi*, return session_id.
    The raw page images are cached so subsequent calls only re-process one page.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Write upload to a temp file (pdf_to_images needs a path)
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        raw_pages: list[np.ndarray] = await loop.run_in_executor(
            _pool, lambda: pdf_to_images(tmp_path, dpi=dpi)
        )
    except Exception as exc:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    session_id = str(uuid.uuid4())
    _sessions[session_id] = Session(pdf_path=tmp_path, raw_pages=raw_pages)

    return {"session_id": session_id, "page_count": len(raw_pages), "dpi": dpi}


# ═══════════════════════════════════════════════════════════════════════════════
# Routes – Page preview (original + processed)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/page/{page_idx}/original")
async def get_original(page_idx: int, session_id: str):
    """Return the raw (unprocessed) page image as JPEG."""
    s = _get_session(session_id)
    if page_idx < 0 or page_idx >= len(s.raw_pages):
        raise HTTPException(status_code=404, detail="Page index out of range.")
    
    loop = asyncio.get_event_loop()
    jpeg = await loop.run_in_executor(_pool, lambda: encode_jpeg(s.raw_pages[page_idx], quality=80))
    return Response(content=jpeg, media_type="image/jpeg")


@app.post("/page/{page_idx}/preview")
async def preview_page(page_idx: int, body: dict):
    """
    Run the preprocessing pipeline on one page with the given params.
    Returns the result as JPEG.  Does NOT save to disk.
    """
    session_id = body.get("session_id", "")
    s = _get_session(session_id)
    if page_idx < 0 or page_idx >= len(s.raw_pages):
        raise HTTPException(status_code=404, detail="Page index out of range.")

    params = _coerce_params(body.get("params", {}))
    raw = s.raw_pages[page_idx]

    loop = asyncio.get_event_loop()
    try:
        result: np.ndarray = await loop.run_in_executor(_pool, lambda: _process_one(raw, params))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    # Cache the processed result in the session
    s.processed[page_idx] = result

    jpeg = await loop.run_in_executor(_pool, lambda: encode_jpeg(result, quality=85))
    return Response(content=jpeg, media_type="image/jpeg")


# ═══════════════════════════════════════════════════════════════════════════════
# Routes – Save
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/save-page")
async def save_page(body: dict):
    """
    Save the currently processed page (or re-process if needed) to *output_dir*.
    """
    session_id = body.get("session_id", "")
    page_idx = int(body.get("page_idx", 0))
    output_dir = body.get("output_dir", "ocr_output")
    params = _coerce_params(body.get("params", {}))

    s = _get_session(session_id)
    if page_idx < 0 or page_idx >= len(s.raw_pages):
        raise HTTPException(status_code=404, detail="Page index out of range.")

    loop = asyncio.get_event_loop()

    # Use cached result if params haven't changed, otherwise re-process
    result = s.processed.get(page_idx)
    if result is None:
        try:
            result = await loop.run_in_executor(_pool, lambda: _process_one(s.raw_pages[page_idx], params))
            s.processed[page_idx] = result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"page_{page_idx + 1:04d}.png")
    
    await loop.run_in_executor(_pool, lambda: cv2.imwrite(out_path, result))
    s.saved_paths[page_idx] = out_path

    return {"path": os.path.abspath(out_path), "page_idx": page_idx}


# ═══════════════════════════════════════════════════════════════════════════════
# Routes – Batch processing
# ═══════════════════════════════════════════════════════════════════════════════

def _batch_worker(session_id: str, params: dict, output_dir: str, job_id: str):
    """Blocking batch-process loop – runs in thread pool."""
    s = _sessions[session_id]
    job = _jobs[job_id]
    os.makedirs(output_dir, exist_ok=True)

    for idx, raw in enumerate(s.raw_pages):
        try:
            result = _process_one(raw, params)
            s.processed[idx] = result
            out_path = os.path.join(output_dir, f"page_{idx + 1:04d}.png")
            cv2.imwrite(out_path, result)
            s.saved_paths[idx] = out_path
        except Exception:
            job.errors.append(f"Page {idx + 1}: {traceback.format_exc(limit=2)}")
        finally:
            job.done += 1

    job.finished = True


@app.post("/process-all")
async def process_all(body: dict, background_tasks: BackgroundTasks):
    """
    Kick off batch processing of all pages.
    Returns a job_id to poll via GET /status/{job_id}.
    """
    session_id = body.get("session_id", "")
    output_dir = body.get("output_dir", "ocr_output")
    params = _coerce_params(body.get("params", {}))

    s = _get_session(session_id)
    job_id = str(uuid.uuid4())
    _jobs[job_id] = BatchJob(total=len(s.raw_pages))

    # Run in thread pool via background task
    loop = asyncio.get_event_loop()
    background_tasks.add_task(
        loop.run_in_executor,
        _pool,
        lambda: _batch_worker(session_id, params, output_dir, job_id),
    )

    return {"job_id": job_id, "total": len(s.raw_pages)}


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    """Return batch job progress."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job_id,
        "total": job.total,
        "done": job.done,
        "finished": job.finished,
        "errors": job.errors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Routes – PDF export
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/export-pdf")
async def export_pdf(body: dict):
    """
    Pack the saved page images (specified by *page_indices* or all saved) into
    a single PDF and return it as a downloadable file.
    """
    session_id = body.get("session_id", "")
    output_dir = body.get("output_dir", "ocr_output")
    page_indices: list[int] | None = body.get("page_indices")  # None = all saved

    s = _get_session(session_id)

    if page_indices is None:
        page_indices = sorted(s.saved_paths.keys())

    if not page_indices:
        raise HTTPException(
            status_code=400,
            detail="No processed pages found. Save at least one page first.",
        )

    # Collect paths in page order
    missing = [i for i in page_indices if i not in s.saved_paths]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Pages not yet saved: {[i + 1 for i in missing]}. "
                   "Run save-page or process-all first.",
        )

    paths = [s.saved_paths[i] for i in sorted(page_indices)]

    os.makedirs(output_dir, exist_ok=True)
    out_pdf = os.path.join(output_dir, "cleaned.pdf")

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(_pool, lambda: images_to_pdf(paths, out_pdf))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        path=out_pdf,
        media_type="application/pdf",
        filename="cleaned.pdf",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Routes – Session cleanup
# ═══════════════════════════════════════════════════════════════════════════════

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Free cached page images and delete the temp PDF."""
    s = _sessions.pop(session_id, None)
    if s:
        try:
            os.unlink(s.pdf_path)
        except OSError:
            pass
    return {"deleted": session_id}


# ═══════════════════════════════════════════════════════════════════════════════
# Static files – serve the frontend
# ═══════════════════════════════════════════════════════════════════════════════

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
