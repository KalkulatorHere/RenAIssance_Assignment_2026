import cv2
import numpy as np
import fitz  # PyMuPDF

def pdf_to_images(pdf_path: str, dpi: int) -> list[np.ndarray]:
    """Convert a PDF document to a list of OpenCV BGR images."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix  = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img  = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        images.append(img)
    doc.close()
    return images

def load_page(pdf_path: str, page_num: int, dpi: int) -> np.ndarray:
    """Load a specific page of a PDF as an OpenCV BGR image."""
    doc  = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix  = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    img  = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    doc.close()
    return img
