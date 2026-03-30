# Historical OCR Pipeline

A modular, robust, and cleanly structured pipeline for extracting historical Spanish text from PDF documents using Detectron2 (for text region cropping), TrOCR (for handwritten/historical text generation), and Gemini APIs (for post-correction and VLM verification).

## 🗂️ Project Structure
```text
pipeline_modular/
├── config.py              # Configuration values, Model Paths, Hyperparameters
├── main.py                # Pipeline execution entrypoint
├── requirements.txt       # Dependencies
└── src/
    ├── data_loading.py    # PyMuPDF bindings to convert PDF to images
    ├── inference.py       # TrOCR execution wrappers
    ├── pipeline.py        # Orchestrator executing the complete pipeline flow
    ├── postprocessing.py  # Gemini LLM spelling correction and VLM verification
    ├── preprocessing.py   # Detectron2 crop logic and box alignment algorithms
    └── utils.py           # Text metrics computation (CER, WER, BLEU)
```

## ⚙️ Setup and Installation

1. **Virtual Environment**: 
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Requirements**:
```bash
pip install -r requirements.txt
```
> **Note:** `detectron2` is extremely specific with its PyTorch dependencies. Ensure you have the matching CUDA PyTorch versions installed if compiling from source fails.

3. **Configure Environment Variables**:
The code will look for environment variables before falling back to local defaults defined in `config.py`.
```bash
# Set your Gemini API key for correction and verification stages
export GEMINI_API_KEY="YOUR_KEY_HERE"

# Ensure you have your Detectron2 weights accessible 
# (By default it assumes a `./models/maskrcnn_model.pth` or checks the env var)
export TEXTLINE_MODEL_PATH="/path/to/your/maskrcnn_model.pth"
```

## 🚀 Usage

Execute `main.py` against any target PDF document:

```bash
python main.py /path/to/my/historical_document.pdf --max_pages 5 --dpi 200
```

### Outputs
The pipeline automatically provisions outputs in the `outputs/inferences` folder relative to the script's working directory. Each processed page will generate a JSON artifact mapping the predicted text regions to their LLM corrected content.
