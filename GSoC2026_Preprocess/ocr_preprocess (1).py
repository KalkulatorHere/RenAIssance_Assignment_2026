"""
ocr_preprocess.py
─────────────────
Pipeline to preprocess scanned PDF pages for OCR.

Steps per page:
  1. PDF → PIL images at ~300 DPI          (pdf_to_images)
  2. Deskew via Hough-line angle detection  (deskew)
  3. Normalize uneven background lighting   (normalize_background)
  4. Edge-preserving denoising              (denoise)
  5. Adaptive binarization (Sauvola)        (adaptive_binarize)
  6. Light morphological cleanup            (morphological_cleanup)
  7. Save result to disk                    (process_pdf)

Dependencies:
    pip install pymupdf opencv-python-headless numpy scikit-image Pillow
    Optional faster fallback: pip install pdf2image  +  apt install poppler-utils
"""

import os
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola

# ── Primary: PyMuPDF (no system dependencies) ─────────────────────────────────
try:
    import fitz  # PyMuPDF — pip install pymupdf
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False

# ── Fallback: pdf2image (requires poppler system binaries) ────────────────────
# pdf2image imports fine even when poppler is missing, but every call then
# raises PDFInfoNotInstalledError.  We probe for the binary at import time
# so we can skip it cleanly rather than crash at runtime.
try:
    import shutil
    from pdf2image import convert_from_path
    _PDF2IMAGE_AVAILABLE = shutil.which("pdfinfo") is not None
except ImportError:
    _PDF2IMAGE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PDF → images
# ═══════════════════════════════════════════════════════════════════════════════

def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[np.ndarray]:
    """
    Convert every page of a PDF to a NumPy BGR image array at *dpi* resolution.

    Tries pdf2image (poppler) first; falls back to PyMuPDF if unavailable.

    Parameters
    ----------
    pdf_path : str
        Path to the input PDF file.
    dpi : int
        Render resolution. 300 is the OCR-friendly sweet spot.

    Returns
    -------
    list[np.ndarray]
        One BGR uint8 array per page.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    images: list[np.ndarray] = []

    if _PYMUPDF_AVAILABLE:
        # PyMuPDF: pure-Python, no system binaries required — preferred on
        # Kaggle, Colab, and any environment where poppler is unavailable.
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0          # PyMuPDF's base resolution is 72 DPI
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            bgr = cv2.cvtColor(
                np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3),
                cv2.COLOR_RGB2BGR,
            )
            images.append(bgr)
        doc.close()

    elif _PDF2IMAGE_AVAILABLE:
        # pdf2image: faster for large PDFs but needs poppler system binaries.
        # Only reached when pdfinfo was found on PATH at import time.
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
        for pil_img in pil_pages:
            bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            images.append(bgr)

    else:
        raise ImportError(
            "No PDF renderer available. Install PyMuPDF (recommended):\n"
            "  pip install pymupdf\n\n"
            "Or install pdf2image + poppler:\n"
            "  pip install pdf2image\n"
            "  apt install poppler-utils   # Debian/Ubuntu/Kaggle\n"
            "  brew install poppler        # macOS"
        )

    return images


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Deskew
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_skew_angle(gray: np.ndarray, angle_range: float = 10.0) -> float:
    """
    Estimate document skew using probabilistic Hough transform on Canny edges.

    Only angles within ±*angle_range* degrees are considered, which prevents
    false detections on page content (e.g. vertical text or diagonal rules).

    Returns the median angle of detected near-horizontal lines, in degrees.
    """
    # Work on a downscaled copy for speed
    scale = 0.5
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Canny edges
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Probabilistic Hough: find line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=small.shape[1] // 6,   # at least 1/6 of page width
        maxLineGap=20,
    )

    if lines is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Keep only near-horizontal lines
        if abs(angle) <= angle_range:
            angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


def deskew(image: np.ndarray, angle_range: float = 10.0) -> np.ndarray:
    """
    Rotate *image* to correct skew detected in its grayscale projection.

    The full page (including two-page spreads) is rotated as a unit —
    no region detection or cropping is performed.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.
    angle_range : float
        Maximum absolute skew angle to correct (degrees). Larger values
        risk misidentifying intentional page rotation.

    Returns
    -------
    np.ndarray
        Deskewed BGR image (same size; white fill for rotation padding).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = _detect_skew_angle(gray, angle_range=angle_range)

    if abs(angle) < 0.1:          # negligible — skip rotation
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),   # white background
    )
    return rotated


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Background normalisation (illumination correction)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_background(
    image: np.ndarray,
    blur_ksize: int = 0,
    blur_sigma: float = 80.0,
) -> np.ndarray:
    """
    Remove uneven background illumination (e.g. scanner light gradients,
    shadow from bound book spine) while keeping text contrast intact.

    Method: estimate the background via a very large Gaussian blur, then
    divide the original by the background and rescale to [0, 255].

    Parameters
    ----------
    image : np.ndarray
        BGR input image.
    blur_ksize : int
        Kernel size for the background blur. 0 = derived from *blur_sigma*.
        Should be large enough to span typical illumination variation.
    blur_sigma : float
        Standard deviation for the background blur (pixels). Default 80
        handles gradients spanning dozens of millimetres at 300 DPI.

    Returns
    -------
    np.ndarray
        Illumination-normalised BGR image.
    """
    # Work in float32 to preserve precision during division
    img_f = image.astype(np.float32)

    # Per-channel background estimate
    ksize = blur_ksize if blur_ksize > 0 else 0   # cv2 accepts 0 → auto from sigma
    background = cv2.GaussianBlur(img_f, (ksize, ksize), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # Avoid division by zero; 1.0 floor is safe for uint8 content
    background = np.clip(background, 1.0, None)

    # Divide and re-centre around mid-grey (128) so the result doesn't clip
    normalised = (img_f / background) * 128.0
    normalised = np.clip(normalised, 0, 255).astype(np.uint8)
    return normalised


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Denoising (edge-preserving)
# ═══════════════════════════════════════════════════════════════════════════════

def denoise(
    image: np.ndarray,
    h: float = 6,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """
    Apply light edge-preserving denoising via OpenCV's Non-Local Means.

    NLM preserves text strokes and fine details far better than a simple
    Gaussian blur, making it well-suited for scanned documents.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.
    h : float
        Filter strength. Values 5–10 give light denoising; higher = more
        aggressive blurring of fine detail.
    template_window_size : int
        Size (px) of the patch used to compare pixels. Must be odd.
    search_window_size : int
        Size (px) of the neighbourhood searched for similar patches. Must be odd.

    Returns
    -------
    np.ndarray
        Denoised BGR image.
    """
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=h,
        hColor=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Adaptive binarisation (Sauvola)
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_binarize(
    image: np.ndarray,
    window_size: int = 51,
    k: float = 0.2,
    use_sauvola: bool = True,
) -> np.ndarray:
    """
    Convert a (possibly colour) image to a binary (black-on-white) image.

    Sauvola thresholding is preferred: it computes a local threshold based on
    the mean and standard deviation of each neighbourhood, which handles
    variable contrast and faint text better than global or simple adaptive
    methods.

    Falls back to OpenCV's adaptive Gaussian threshold if *use_sauvola* is
    False or if scikit-image is unavailable.

    Parameters
    ----------
    image : np.ndarray
        BGR or grayscale input image.
    window_size : int
        Local neighbourhood size in pixels (must be odd). Larger values smooth
        out larger illumination gradients; smaller values track fine local
        contrast. 51 px at 300 DPI ≈ 4 mm.
    k : float
        Sauvola sensitivity constant. Typical range 0.1–0.5; higher = lighter
        (more pixels become black).
    use_sauvola : bool
        If True, use Sauvola (scikit-image). If False, use OpenCV adaptive
        Gaussian. Useful as a fallback if scikit-image is unavailable.

    Returns
    -------
    np.ndarray
        Binary uint8 image (0 = black / text, 255 = white / background).
    """
    # Ensure grayscale input
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if use_sauvola:
        # window_size must be odd
        ws = window_size if window_size % 2 == 1 else window_size + 1
        thresh_map = threshold_sauvola(gray, window_size=ws, k=k)
        binary = (gray > thresh_map).astype(np.uint8) * 255
    else:
        # OpenCV adaptive Gaussian: blockSize must be odd and > 1
        ws = window_size if window_size % 2 == 1 else window_size + 1
        binary = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=ws,
            C=10,
        )

    return binary


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Morphological cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def morphological_cleanup(
    binary: np.ndarray,
    kernel_size: int = 2,
) -> np.ndarray:
    """
    Apply a single opening pass (erosion → dilation) to remove isolated
    noise pixels without affecting character strokes.

    A very small kernel is intentional: larger kernels risk breaking thin
    strokes or joining nearby characters.

    Parameters
    ----------
    binary : np.ndarray
        Binary uint8 image (0 = text, 255 = background).
    kernel_size : int
        Side length of the square structuring element (pixels). 1–3 recommended.

    Returns
    -------
    np.ndarray
        Cleaned binary image.
    """
    if kernel_size < 1:
        return binary

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    # Opening removes small dark specks on the white background
    # We invert so "text" is white, apply opening, then invert back
    inverted = cv2.bitwise_not(binary)
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(opened)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Per-page orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_page(
    image: np.ndarray,
    deskew_range: float = 10.0,
    bg_sigma: float = 80.0,
    denoise_h: float = 6,
    binarize_window: int = 51,
    binarize_k: float = 0.2,
    morph_kernel: int = 2,
    debug_dir: str | None = None,
    page_idx: int = 0,
) -> np.ndarray:
    """
    Run the full preprocessing pipeline on a single page image.

    Steps (in order):
      deskew → normalize_background → denoise → adaptive_binarize
              → morphological_cleanup

    Parameters
    ----------
    image : np.ndarray
        Raw BGR page image from pdf_to_images.
    deskew_range, bg_sigma, denoise_h, binarize_window, binarize_k,
    morph_kernel : pipeline parameters forwarded to each step.
    debug_dir : str or None
        If given, intermediate images are saved here for inspection.
    page_idx : int
        Page number used for debug file names.

    Returns
    -------
    np.ndarray
        Final binary preprocessed image ready for OCR.
    """

    def _save_debug(img: np.ndarray, stage: str) -> None:
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            fname = os.path.join(debug_dir, f"p{page_idx:04d}_{stage}.png")
            cv2.imwrite(fname, img)

    _save_debug(image, "0_raw")

    # Step 1 – Deskew
    deskewed = deskew(image, angle_range=deskew_range)
    _save_debug(deskewed, "1_deskewed")

    # Step 2 – Background normalisation
    normalised = normalize_background(deskewed, blur_sigma=bg_sigma)
    _save_debug(normalised, "2_normalised")

    # Step 3 – Denoising
    denoised = denoise(normalised, h=denoise_h)
    _save_debug(denoised, "3_denoised")

    # Step 4 – Binarisation
    binary = adaptive_binarize(denoised, window_size=binarize_window, k=binarize_k)
    _save_debug(binary, "4_binary")

    # Step 5 – Morphological cleanup
    cleaned = morphological_cleanup(binary, kernel_size=morph_kernel)
    _save_debug(cleaned, "5_cleaned")

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Full-PDF pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    deskew_range: float = 10.0,
    bg_sigma: float = 80.0,
    denoise_h: float = 6,
    binarize_window: int = 51,
    binarize_k: float = 0.2,
    morph_kernel: int = 2,
    debug: bool = False,
    image_format: str = "png",
) -> list[str]:
    """
    Convert a scanned PDF to cleaned, OCR-ready images.

    Parameters
    ----------
    pdf_path : str
        Path to the input PDF.
    output_dir : str
        Directory where processed images are saved (created if absent).
    dpi : int
        Render DPI for rasterisation.
    deskew_range : float
        Max absolute skew angle to correct (degrees).
    bg_sigma : float
        Blur sigma for background normalisation.
    denoise_h : float
        NLM denoising filter strength.
    binarize_window : int
        Sauvola neighbourhood size (odd pixels).
    binarize_k : float
        Sauvola k constant.
    morph_kernel : int
        Morphological cleanup kernel size.
    debug : bool
        If True, intermediate stage images are saved alongside outputs.
    image_format : str
        Output image format: "png" (lossless, preferred) or "tiff".

    Returns
    -------
    list[str]
        Paths of all successfully saved output images.
    """
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug") if debug else None

    print(f"[process_pdf] Reading PDF: {pdf_path}")
    raw_pages = pdf_to_images(pdf_path, dpi=dpi)
    print(f"[process_pdf] {len(raw_pages)} page(s) found — DPI={dpi}")

    saved_paths: list[str] = []

    for idx, page_img in enumerate(raw_pages):
        try:
            print(f"  Processing page {idx + 1}/{len(raw_pages)} …", end=" ", flush=True)
            result = preprocess_page(
                page_img,
                deskew_range=deskew_range,
                bg_sigma=bg_sigma,
                denoise_h=denoise_h,
                binarize_window=binarize_window,
                binarize_k=binarize_k,
                morph_kernel=morph_kernel,
                debug_dir=debug_dir,
                page_idx=idx,
            )
            out_path = os.path.join(output_dir, f"page_{idx + 1:04d}.{image_format}")
            cv2.imwrite(out_path, result)
            saved_paths.append(out_path)
            print(f"saved → {out_path}")
        except Exception:
            # Log the error but continue processing remaining pages
            print(f"ERROR — skipping page {idx + 1}")
            traceback.print_exc()

    print(f"[process_pdf] Done. {len(saved_paths)}/{len(raw_pages)} pages saved to '{output_dir}'.")
    return saved_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Main example
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Minimal usage example. Edit the paths below and run:

        python ocr_preprocess.py
    """
    # ── Configure these ────────────────────────────────────────────────────────
    PDF_PATH   = "scanned_document.pdf"   # ← your input PDF
    OUTPUT_DIR = "ocr_output"             # ← where cleaned images land
    DEBUG_MODE = False                    # ← True to save intermediate stages
    # ──────────────────────────────────────────────────────────────────────────

    saved = process_pdf(
        pdf_path       = PDF_PATH,
        output_dir     = OUTPUT_DIR,
        dpi            = 300,
        deskew_range   = 10.0,   # correct skew up to ±10°
        bg_sigma       = 80.0,   # background blur radius (px)
        denoise_h      = 6,      # NLM filter strength
        binarize_window= 51,     # Sauvola window (px, odd)
        binarize_k     = 0.2,    # Sauvola k constant
        morph_kernel   = 2,      # morphology kernel (px)
        debug          = DEBUG_MODE,
        image_format   = "png",  # or "tiff" for lossless 16-bit tools
    )

    # Pack all cleaned page images into a single PDF ready for the next stage
    OUTPUT_PDF = os.path.join(OUTPUT_DIR, "cleaned.pdf")
    if saved:
        images_to_pdf(saved, OUTPUT_PDF)
        print(f"\n✓ Cleaned PDF ready for next OCR pipeline → {OUTPUT_PDF}")
    else:
        print("\n⚠ No pages were processed — PDF not created.")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Save cleaned images → PDF
# ═══════════════════════════════════════════════════════════════════════════════

def images_to_pdf(image_paths: list, output_pdf: str) -> str:
    """
    Pack a list of preprocessed images into a single PDF using PyMuPDF.

    Each image becomes one page sized exactly to its pixel dimensions —
    no resampling, no quality loss.  The output PDF is ready to feed into
    any downstream OCR pipeline (Tesseract, AWS Textract, Azure DI, etc.).

    Parameters
    ----------
    image_paths : list[str]
        Ordered list of image file paths (one page each).
        Typically the list returned by process_pdf().
    output_pdf : str
        Destination path for the output PDF (created or overwritten).

    Returns
    -------
    str
        Absolute path of the written PDF.
    """
    if not image_paths:
        raise ValueError("image_paths is empty — nothing to pack into a PDF.")

    if not _PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is required for images_to_pdf.  pip install pymupdf")

    doc = fitz.open()  # new blank PDF

    for img_path in image_paths:
        if not os.path.isfile(img_path):
            print(f"  [images_to_pdf] WARNING: skipping missing file — {img_path}")
            continue

        # Open image just to read its dimensions (no full decode needed)
        tmp = fitz.open(img_path)
        rect = tmp[0].rect          # bounding rect in pts (1 pt = 1 px at 72 DPI)
        tmp.close()

        # New page exactly the same size as the source image
        page = doc.new_page(width=rect.width, height=rect.height)

        # Embed image so it fills the entire page
        page.insert_image(rect, filename=img_path)

    out_dir = os.path.dirname(os.path.abspath(output_pdf))
    os.makedirs(out_dir, exist_ok=True)

    # garbage=4  → deduplicate & compress xrefs; deflate=True → compress streams
    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()

    abs_path = os.path.abspath(output_pdf)
    print(f"[images_to_pdf] {len(image_paths)} page(s) saved → {abs_path}")
    return abs_path
