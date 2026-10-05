# Inspect AI — Beta

A practical PDF Analyzer beta with:
- PDF upload
- Page-aware text extraction with PyMuPDF
- Chunking and TF-IDF retrieval
- Q&A with page citations
- Local Ollama LLM support when available
- Extractive fallback when no LLM is configured
- Document summary
- Multi-document comparison
- Document statistics
- Clean responsive UI

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000

Optional local LLM:
```bash
ollama pull gemma3:4b
```
Then set `OLLAMA_MODEL=gemma3:4b` if needed.
