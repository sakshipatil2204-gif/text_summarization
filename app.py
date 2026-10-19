"""Flask application for generating local academic summary reports."""
from io import BytesIO
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import pymupdf as fitz
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from summarizer import SAMPLE_ARTICLE, create_report, normalize_text

ROOT = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {"txt", "pdf"}
MAX_TEXT_CHARS = 200_000
MAX_STORED_REPORTS = 50
_reports = OrderedDict()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/sample")
def sample():
    return jsonify({"title": "Artificial Intelligence in Education", "text": SAMPLE_ARTICLE})


def get_source_text():
    """Read exactly one input source: pasted text or one supported upload."""
    text = normalize_text(request.form.get("text", ""))
    uploaded = request.files.get("file")
    has_file = bool(uploaded and uploaded.filename)
    if text and has_file:
        raise ValueError("Choose pasted text or an uploaded file, not both.")
    if has_file:
        filename = secure_filename(uploaded.filename)
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported file type. Upload a TXT or PDF file.")
        raw = uploaded.read(app.config["MAX_CONTENT_LENGTH"] + 1)
        if len(raw) > app.config["MAX_CONTENT_LENGTH"]:
            raise ValueError("The uploaded file exceeds the 10 MB limit.")
        if suffix == "txt":
            try:
                text = raw.decode("utf-8-sig", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValueError("This TXT file is not valid UTF-8 text.") from exc
        else:
            try:
                with fitz.open(stream=raw, filetype="pdf") as document:
                    text = "\n".join(page.get_text("text") for page in document)
            except Exception as exc:
                raise ValueError("This PDF could not be read. Please check the file and try again.") from exc
            if not text:
                raise ValueError("No selectable text was found. Scanned PDFs need OCR before upload.")
    text = normalize_text(text)
    if not text:
        raise ValueError("The selected document is empty. Paste text or upload a non-empty TXT/PDF file.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("The document is too long. The maximum is 200,000 characters.")
    return text


@app.post("/summarize")
def summarize():
    try:
        text = get_source_text()
        title = normalize_text(request.form.get("title", "Untitled document"))[:160] or "Untitled document"
        length = request.form.get("length", "short")
        if length not in {"short", "detailed"}:
            raise ValueError("Choose either a short or detailed summary.")
        report = create_report(text, title, length)
        report_id = uuid4().hex
        report["report_id"] = report_id
        _reports[report_id] = report["report_text"]
        while len(_reports) > MAX_STORED_REPORTS:
            _reports.popitem(last=False)
        return jsonify(report)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/download")
def download():
    payload = request.get_json(silent=True) or {}
    report_id = payload.get("report_id", "")
    content = _reports.get(report_id) if isinstance(report_id, str) else None
    if not content:
        return jsonify({"error": "Generate a report before downloading."}), 400
    title = secure_filename(str(payload.get("title", "summary_report"))) or "summary_report"
    return send_file(BytesIO(content.encode("utf-8")), mimetype="text/plain",
                     as_attachment=True, download_name=f"{Path(title).stem}_report.txt")


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify({"error": "The uploaded file exceeds the 10 MB limit."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
