# AI-Powered Text Summarization and Report Generation System

A local Flask dashboard that turns one pasted source or one TXT/PDF upload into short and detailed academic summaries, points, keywords, conclusions, revision questions, and approximate coverage checks. Text and uploaded files are mutually exclusive. Downloads use the exact server-side report returned for display, rather than report contents sent back by the browser.

## Requirements

Python 3.9 or newer. PyMuPDF reads selectable PDF text; scanned PDFs require OCR before use. Uploaded files are limited to 10 MB and pasted text to 200,000 characters.

## Windows setup

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest
python app.py
```

Open <http://127.0.0.1:5000>. To stop the server, press Ctrl+C. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again.

## Notes

The optional Transformers integration is disabled by default, so first startup needs no model download or API key. To use a model already stored on your computer, install `transformers` and a compatible PyTorch build, then set `SUMMARIZER_MODEL_PATH` to that local model directory before starting Flask. The app uses local-only model loading and falls back if the package or model is unavailable. Main points and questions are derived from source sentences. Accuracy rows and coverage percentages are approximate lexical checks; the displayed count shows how many extracted source points match the summary, and these checks do not establish factual correctness. Scanned PDFs need OCR before upload.
