import os
import json
import logging
import numpy as np
from typing import List, Dict

from config import (
    SPLIT_PAGE, USE_LLM_CORRECTION, USE_VLM_VERIFICATION, 
    LLM_MODEL, VLM_MODEL, GEMINI_API_KEY, SAVE_OUTPUTS,
    DEFAULT_INFERENCE_FOLDER
)
from src.data_loading import pdf_to_images
from src.preprocessing import TextlineExtractor, detect_text_regions
from src.inference import TROCREngine, run_ocr, assemble_page_text
from src.postprocessing import correct_with_llm, extract_last_n_lines, verify_with_vlm, merge_vlm_feedback

def _make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    return obj

def _process_single_image(extractor: TextlineExtractor, engine: TROCREngine, image: np.ndarray, image_name: str) -> dict:
    boxes, scores, ro_info, crops, padding = detect_text_regions(extractor, image)
    if len(boxes) == 0: 
        return {"success": False, "error": "No textlines detected", "full_text": ""}
    
    ocr_results = run_ocr(engine, crops, ro_info)
    segments, full_text = assemble_page_text(ocr_results, boxes, scores)
    
    return {
        "success": True, 
        "line_segments": segments, 
        "full_text": full_text, 
        "boxes": boxes, 
        "scores": scores, 
        "padding": padding
    }

def _process_page_image(extractor: TextlineExtractor, engine: TROCREngine, image: np.ndarray, image_name: str) -> dict:
    if SPLIT_PAGE and extractor.should_split(image):
        left, right, split_x = extractor.split_image(image)
        lr = _process_single_image(extractor, engine, left,  f"{image_name}_left")
        rr = _process_single_image(extractor, engine, right, f"{image_name}_right")
        
        if not (lr["success"] and rr["success"]): 
            return {"success": False, "error": "Split failed", "full_text": ""}
        
        offset = len(lr.get("line_segments", []))
        for seg in rr.get("line_segments", []): 
            seg["line_index"] += offset
            
        combined = sorted(lr.get("line_segments", []) + rr.get("line_segments", []), key=lambda s: s["line_index"])
        text = "\n".join(filter(None, [lr.get("full_text", ""), rr.get("full_text", "")]))
        return {"success": True, "line_segments": combined, "full_text": text}
        
    return _process_single_image(extractor, engine, image, image_name)

def run_pipeline(pdf_path: str,
                 extractor: TextlineExtractor,
                 engine: TROCREngine,
                 dpi: int = 200,
                 use_llm: bool = USE_LLM_CORRECTION,
                 use_vlm: bool = USE_VLM_VERIFICATION,
                 vlm_model: str = VLM_MODEL,
                 max_pages: int = None,
                 output_folder: str = str(DEFAULT_INFERENCE_FOLDER)) -> List[Dict]:
                 
    logging.info(f"Loading PDF: {pdf_path}")
    images = pdf_to_images(pdf_path, dpi=dpi)
    if max_pages: 
        images = images[:max_pages]
        
    os.makedirs(output_folder, exist_ok=True)
    all_results, llm_context = [], ""
    
    for i, image in enumerate(images):
        logging.info(f"Processing Page {i+1}/{len(images)}...")
        page_id = f"page_{i+1}"
        
        result = _process_page_image(extractor, engine, image, page_id)
        result.update({"page_id": page_id, "page_num": i+1, "image": image})
        
        if not result["success"]:
            logging.error(f"  OCR failed: {result.get('error')}")
            all_results.append(result)
            continue
            
        logging.info(f"  OCR extracted {len(result['full_text'])} chars")
        
        if use_llm:
            corrected, status = correct_with_llm(result["full_text"], GEMINI_API_KEY, LLM_MODEL, context=llm_context)
            result.update({"corrected_text": corrected, "llm_status": status})
            llm_context = extract_last_n_lines(corrected)
            logging.info(f"  LLM correction status: {status}")
        else:
            result.update({"corrected_text": result["full_text"], "llm_status": "skipped"})
            
        if use_vlm and use_llm:
            vlm = verify_with_vlm(image, result["corrected_text"], GEMINI_API_KEY, vlm_model)
            final_text = merge_vlm_feedback(result["corrected_text"], vlm)
            result.update({"vlm_result": vlm, "final_text": final_text})
            logging.info(f"  VLM verification confidence: {vlm.get('confidence')}, flags: {len(vlm.get('flagged_spans', []))}")
        else:
            result.update({"vlm_result": None, "final_text": result["corrected_text"]})
            
        if SAVE_OUTPUTS:
            pd = {k: _make_serializable(v) for k, v in result.items() if k != "image"}
            out_file = os.path.join(output_folder, f"{page_id}_result.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(pd, f, ensure_ascii=False, indent=2)
            logging.info(f"  Saved page results to {out_file}")
            
        all_results.append(result)
        
    return all_results
