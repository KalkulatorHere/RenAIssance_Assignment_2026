import os
from pathlib import Path

from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

DEFAULT_OUTPUT_FOLDER = BASE_DIR / "outputs" / "processing"
DEFAULT_LINE_SEGMENTS_FOLDER = BASE_DIR / "outputs" / "line_segments"
DEFAULT_INFERENCE_FOLDER = BASE_DIR / "outputs" / "inferences"

# ── Gemini API ───────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Core toggles ──────────────────────────────────────────────────────────────
SPLIT_PAGE = True
USE_GPU = True
SAVE_OUTPUTS = True

# ── Ablation flags ────────────────────────────────────────────────────────────
USE_LLM_CORRECTION = True
USE_VLM_VERIFICATION = True

# ── Model paths ───────────────────────────────────────────────────────────────
# Placeholders for local model weights and config paths.
TEXTLINE_MODEL_PATH = os.getenv("TEXTLINE_MODEL_PATH", str(BASE_DIR / "models" / "maskrcnn_model.pth"))
TROCR_SPANISH_MODEL = os.getenv("TROCR_SPANISH_MODEL", "qantev/trocr-large-spanish")

LLM_MODEL = "gemini-2.5-flash-lite"
VLM_MODEL = "gemini-2.5-flash-lite"

# ── Processing parameters ─────────────────────────────────────────────────────
DPI = 200
AREA_THRESHOLD_PERCENT = 12.5
SCORE_THRESHOLD = 0.5
SPLIT_RATIO = 0.5
MIN_ASPECT_RATIO = 1.25

# ── Auto-Validation ───────────────────────────────────────────────────────────
_KEY_IS_SET = GEMINI_API_KEY not in ("", "YOUR_GEMINI_API_KEY")

if not _KEY_IS_SET:
    import logging
    logging.warning("GEMINI_API_KEY not set — LLM/VLM stages will be disabled.")
    USE_LLM_CORRECTION = False
    USE_VLM_VERIFICATION = False

if USE_VLM_VERIFICATION and not USE_LLM_CORRECTION:
    USE_VLM_VERIFICATION = False
