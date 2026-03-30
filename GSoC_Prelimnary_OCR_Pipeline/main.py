import argparse
import logging
from pathlib import Path

from config import TEXTLINE_MODEL_PATH, TROCR_SPANISH_MODEL, DPI, USE_GPU
from src.preprocessing import TextlineExtractor
from src.inference import TROCREngine
from src.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def output_summary(results):
    print("\n" + "="*50)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f"{'Page':<6} {'OCR chars':<12} {'LLM chars':<12} {'Conf':<6} {'Status'}")
    print("─" * 50)
    for r in results:
        vlm = r.get("vlm_result") or {}
        print(f"{r.get('page_num','?'):<6} "
              f"{len(r.get('full_text','')):<12} "
              f"{len(r.get('corrected_text','')):<12} "
              f"{vlm.get('confidence','—'):<6} "
              f"{'✅' if r.get('success') else '❌'}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular Historical OCR Pipeline")
    parser.add_argument("pdf_path", type=str, help="Path to input PDF document")
    parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages to process")
    parser.add_argument("--dpi", type=int, default=DPI, help="DPI for PDF rendering")
    args = parser.parse_args()

    pdf_file = Path(args.pdf_path)
    if not pdf_file.exists():
        logging.error(f"Input file not found: {pdf_file}")
        exit(1)

    logging.info("Initializing models...")
    extractor = TextlineExtractor(model_path=TEXTLINE_MODEL_PATH)
    engine = TROCREngine(model_path=TROCR_SPANISH_MODEL, use_gpu=USE_GPU)

    logging.info("Starting pipeline execution...")
    try:
        results = run_pipeline(
            pdf_path=str(pdf_file),
            extractor=extractor,
            engine=engine,
            dpi=args.dpi,
            max_pages=args.max_pages
        )
        output_summary(results)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
