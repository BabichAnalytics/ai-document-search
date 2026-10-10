"""Local document search. Run with: python app.py"""
from pathlib import Path
import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "BAAI/bge-small-en-v1.5"
PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Document Search</title>
<style>body{font:17px system-ui;background:#eef2f7;color:#17243a;max-width:850px;margin:50px auto;padding:20px}
input{width:70%;padding:14px;font:inherit}button{padding:14px;font:inherit;cursor:pointer}
article{background:white;padding:22px;margin:18px 0;border-radius:12px}
pre{white-space:pre-wrap;font:inherit;line-height:1.6}small{color:#59677b}</style>
<h1>Local Document Search</h1><p>Search by meaning. Results are source passages, not generated answers.</p>
<form id="f"><input id="q" required maxlength="1000" placeholder="How long does a reset link last?">
<button>Search</button></form><p id="status"></p><main id="results"></main>
<script>
f.onsubmit=async e=>{e.preventDefault();statusText('Searching…');results.replaceChildren();
try{let r=await fetch('/search?q='+encodeURIComponent(q.value));let d=await r.json();
if(!r.ok)throw Error(d.error);statusText(d.length?'Closest passages — check the source before drawing conclusions.':'No results.');
for(let x of d){let a=document.createElement('article'),h=document.createElement('h3'),s=document.createElement('small'),p=document.createElement('pre');
h.textContent=x.source;s.textContent=x.location+' · similarity '+x.score.toFixed(3);p.textContent=x.text;a.append(h,s,p);results.append(a)}}
catch(e){statusText(e.message)}};
function statusText(t){document.getElementById('status').textContent=t}
</script></html>"""

def chunks(text, limit=180):
    words = text.split()
    for start in range(0, len(words), limit - 30):
        part = words[start:start + limit]
        if part:
            yield " ".join(part)
        if start + limit >= len(words):
            break

def read_documents():
    from pypdf import PdfReader
    docs = ROOT / "documents"
    docs.mkdir(exist_ok=True)
    rows = []
    for path in sorted(docs.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".txt", ".pdf"):
            continue
        try:
            if path.suffix.lower() == ".pdf":
                reader = PdfReader(path)
                if reader.is_encrypted:
                    print("Skipping encrypted PDF:", path.name)
                    continue
                pages = [(f"Page {n}", p.extract_text() or "") for n, p in enumerate(reader.pages, 1)]
            else:
                pages = [("Text file", path.read_text(encoding="utf-8-sig"))]
            count = 0
            for location, text in pages:
                for n, part in enumerate(chunks(text), 1):
                    rows.append((path.name, f"{location}, passage {n}", part))
                    count += 1
            if not count:
                print("No extractable text:", path.name)
        except Exception as exc:
            print("Could not read", path.name, ":", exc)
    return rows

def main():
    import numpy as np
    from fastembed import TextEmbedding
    rows = read_documents()
    if not rows:
        raise SystemExit("Add a text-based PDF or UTF-8 TXT file to the documents folder, then run again.")
    print("Loading local search model...")
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(ROOT / "models"), threads=2)
    vectors = list(model.embed([r[2] for r in rows], batch_size=8))
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    with sqlite3.connect(ROOT / "search.sqlite3") as db:
        db.execute("CREATE TABLE IF NOT EXISTS passages (source TEXT, location TEXT, text TEXT, vector BLOB)")
        db.execute("DELETE FROM passages")
        db.executemany("INSERT INTO passages VALUES (?,?,?,?)",
                       [(*r, v.tobytes()) for r, v in zip(rows, matrix)])
    print(f"Indexed {len(rows)} passages. Index rebuilds on startup.")
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = PAGE.encode()
                mime = "text/html; charset=utf-8"
                code = 200
            elif parsed.path == "/search":
                query = parse_qs(parsed.query).get("q", [""])[0].strip()
                if not query or len(query) > 1000:
                    result, code = {"error": "Enter a query of 1–1000 characters."}, 400
                else:
                    try:
                        vec = np.asarray(next(model.query_embed([query])), dtype=np.float32)
                        vec /= max(float(np.linalg.norm(vec)), 1e-12)
                        scores = matrix @ vec
                        indices = np.argsort(scores)[::-1][:5]
                        result = [dict(source=rows[i][0], location=rows[i][1],
                                       text=rows[i][2], score=float(scores[i])) for i in indices]
                        code = 200
                    except Exception as exc:
                        print("Search error:", exc)
                        result, code = {"error": "Search failed; check the command window."}, 500
                data, mime = json.dumps(result).encode(), "application/json; charset=utf-8"
            else:
                data, mime, code = b"Not found", "text/plain", 404
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
    # Single request at a time keeps local inference memory usage predictable.
    from http.server import HTTPServer
    server = HTTPServer(("127.0.0.1", 8003), Handler)
    print("Open http://127.0.0.1:8003 — Ctrl+C stops the app.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
