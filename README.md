# Local AI Document Search

A local Python browser app that searches PDF and TXT documents by meaning and shows cited source passages. No paid APIs. This is a retrieval demo, not a generative answer system.

## Windows setup

Install Python 3.14, download and extract this repository, and open Command Prompt in its folder.

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8003 and keep the command window open. Ctrl+C stops the server.
The first run downloads BAAI/bge-small-en-v1.5 through FastEmbed. Document processing and search run locally. Initial software and model downloads require internet access.

## Documents

An original fictional handbook is included. Put your own UTF-8 TXT files or text-based PDFs in documents, then restart the app to rebuild the index. Do not commit confidential documents. Scanned PDFs require OCR; encrypted PDFs are skipped. The model is intended for English text.

## Limitations and validation

A user ran this app on Windows with 8 GB RAM and Python 3.14 and retrieved the correct password-reset passage from the fictional handbook. This is a smoke test, not an accuracy benchmark. Similarity scores are not confidence percentages. Search always returns the five closest available passages, even when no passage answers the question. Review sources before drawing conclusions. Fixed overlapping word chunks may cross section boundaries. Example questions in the sample handbook may rank highly without providing answers.

The server binds only to 127.0.0.1 and is intended for personal local use, not internet hosting. There is no authentication or production deployment configuration.

## Example queries

- How long does a password reset link last?
- Are renewal payments refundable?
- What is the maximum attachment size?
- Who can change a team member's role?

The sample handbook is fictional and includes explicit permission to publish and reuse it. Third-party packages and model weights have their own licenses.
