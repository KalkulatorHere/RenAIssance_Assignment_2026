import io
import time
import re
import json
import cv2
from PIL import Image
import google.generativeai as genai

def _correction_prompt(text: str, context: str = "") -> str:
    ctx = f"Previous page context (last 2 lines):\n{context}\n\n" if context else ""
    return (
        "Correct the following historical Spanish OCR text while PRESERVING ORIGINAL GRAMMAR AND STYLE.\n"
        "Only fix orthographic errors, punctuation, and obvious OCR mistakes.\n"
        f"{ctx}Text to correct:\n{text}\n\nReturn ONLY the corrected text, no comments."
    )

def correct_with_llm(ocr_text: str, api_key: str, model_name: str, context: str = "", max_retries: int = 3):
    if not ocr_text.strip(): return ocr_text, "skipped_empty"
    genai.configure(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            resp = genai.GenerativeModel(model_name).generate_content(_correction_prompt(ocr_text, context))
            if resp.text: return resp.text.strip(), "success"
        except Exception as e:
            print(f"  LLM attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return ocr_text, "max_retries_exceeded"

def extract_last_n_lines(text: str, n: int = 2):
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:]) if lines else ""

_VLM_PROMPT = """You are a historical document verification assistant.
You will be given a scan of a historical Spanish manuscript page and its OCR-corrected text.
Verify alignment between image and text. Do NOT rewrite the text. 
Flag words or short spans that look wrong and suggest corrections for those specific parts. 
Return EXACTLY this JSON: { "flagged_spans": ["suspicious"], "suggested_corrections": {"suspicious": "correction"}, "confidence": "high" | "medium" | "low", "notes": "Brief reasoning" }
"""

def verify_with_vlm(page_image, corrected_text: str, api_key: str, model_name: str, max_retries=3):
    empty = {"flagged_spans": [], "suggested_corrections": {}, "confidence": "low", "notes": "err", "_status": "api_error"}
    genai.configure(api_key=api_key)
    
    try:
        rgb = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        img_part = {"mime_type": "image/png", "data": buf.getvalue()}
    except Exception as e:
        print(f"Image preparation for VLM failed: {e}")
        return empty
    
    msg = f"Verify this text:\n\n{corrected_text}\n\nReturn JSON."
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name=model_name, system_instruction=_VLM_PROMPT)
            raw = model.generate_content([img_part, msg]).text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
            res = json.loads(raw)
            return {
                "flagged_spans": res.get("flagged_spans", []),
                "suggested_corrections": res.get("suggested_corrections", {}),
                "confidence": res.get("confidence", "medium"),
                "notes": res.get("notes", ""),
                "_status": "success",
            }
        except Exception as e:
            print(f"VLM attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return empty

def merge_vlm_feedback(corrected_text: str, vlm_result: dict) -> str:
    if vlm_result.get("_status") != "success": 
        return corrected_text
    
    text = corrected_text
    for span, repl in vlm_result.get("suggested_corrections", {}).items():
        if span and repl and span != repl:
            try:
                text = re.sub(re.escape(span), repl, text)
            except re.error:
                pass
    return text
