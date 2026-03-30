import cv2
import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TROCREngine:
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Loading TrOCR model from {model_path} onto {self.device}...")
        
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_path, local_files_only=False)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=False,
                local_files_only=False,
            )
            self.model.to(self.device)
        except OSError as e:
            raise RuntimeError(f"TrOCR model failed to load from '{model_path}'. Error: {e}") from e

    def process_textlines(self, cropped_textlines, reading_order_info):
        ocr_results = []
        for idx, (crop, ro) in enumerate(zip(cropped_textlines, reading_order_info)):
            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pv = self.processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values.to(self.device)
                with torch.no_grad():
                    ids = self.model.generate(pv)
                text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            except Exception as e:
                print(f"OCR inference failed for crop {idx}: {e}")
                text = ""
            ocr_results.append({**ro, 'text': text, 'confidence': 1.0 if text else 0.0})
            
        return sorted(ocr_results, key=lambda x: x['reading_order_index'])

def run_ocr(engine: TROCREngine, crops: list, ro_info: list):
    """Run standard OCR sequence given valid crops."""
    return engine.process_textlines(crops, ro_info)

def assemble_page_text(ocr_results: list, boxes: np.ndarray, scores: np.ndarray) -> tuple:
    """Combine individual line crops into full page text using reading order."""
    ocr_map = {r["reading_order_index"]: r for r in ocr_results}
    segments = []
    for idx, (bbox, score) in enumerate(zip(boxes, scores)):
        ocr = ocr_map.get(idx, {"text": "", "confidence": 0.0})
        x1, y1, x2, y2 = map(int, bbox)
        segments.append({
            "line_index": idx, "bbox": [x1, y1, x2, y2], "score": float(score),
            "text": ocr["text"], "confidence": float(ocr["confidence"]),
        })
    segments.sort(key=lambda s: s["line_index"])
    full_text = "\n".join(s["text"] for s in segments if s["text"].strip())
    return segments, full_text
