"""
backend/preprocess_core.py
──────────────────────────
Thin re-export layer that imports the pipeline from the original source file
whose name contains spaces and parentheses.  All pipeline code lives there;
nothing is duplicated here.
"""

import importlib.util
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent.parent          # project root
_PIPELINE_PATH = _HERE / "ocr_preprocess (1).py"

if not _PIPELINE_PATH.exists():
    raise FileNotFoundError(
        f"Pipeline source not found: {_PIPELINE_PATH}\n"
        "Make sure ocr_preprocess (1).py is in the project root."
    )

_spec = importlib.util.spec_from_file_location("ocr_preprocess", _PIPELINE_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules["ocr_preprocess"] = _module
_spec.loader.exec_module(_module)

# Public re-exports
pdf_to_images: callable        = _module.pdf_to_images
preprocess_page: callable      = _module.preprocess_page
images_to_pdf: callable        = _module.images_to_pdf


def encode_jpeg(bgr: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR ndarray to JPEG bytes for HTTP response."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def encode_png(bgr: np.ndarray) -> bytes:
    """Encode a BGR ndarray to PNG bytes (lossless)."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode PNG failed")
    return buf.tobytes()




DEFAULT_PARAMS = {
    "dpi":             300,
    "deskew_range":    10.0,
    "bg_sigma":        80.0,
    "denoise_h":       6.0,
    "binarize_window": 51,
    "binarize_k":      0.2,
    "morph_kernel":    2,
}
