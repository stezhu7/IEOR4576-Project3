import logging
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import UploadResponse, AnalyzeResponse
from app.tools.ingest_tool import ingest_document
from app.agents.orchestrator import analyze_contract
from app.tools.report_tool import generate_report

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Clause — AI Contract Review")
app.mount("/static", StaticFiles(directory="static"), name="static")

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")

_doc_store: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload", response_model=UploadResponse)
async def upload_contract(file: UploadFile = File(...)):
    """
    Step 1: Upload and ingest a contract PDF.
    Parses, chunks, embeds, and stores in ChromaDB.
    Returns doc_id for subsequent /analyze call.
    """
    if not file.filename.lower().endswith((".pdf",)):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    max_size = 10 * 1024 * 1024  # 10 MB
    file_bytes = await file.read()
    if len(file_bytes) > max_size:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB).")

    log.info("upload: received %s (%d bytes)", file.filename, len(file_bytes))

    try:
        ingested = ingest_document(file_bytes, file.filename)
    except Exception as e:
        log.exception("upload: ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    _doc_store[ingested.doc_id] = {
        "filename": ingested.filename,
        "doc_type": ingested.doc_type,
        "summary":  ingested.summary,
    }

    return UploadResponse(
        doc_id=ingested.doc_id,
        filename=ingested.filename,
        doc_type=ingested.doc_type,
        chunk_count=ingested.chunk_count,
        message=f"Successfully ingested {ingested.chunk_count} chunks. Ready for analysis.",
    )


def _cleanup_document(doc_id: str) -> None:
    """
    Delete contract text from disk after analysis — privacy protection.
    Removes: full-text JSON, ChromaDB embeddings for this doc.
    The generated PDF report is kept (user needs to download it).
    """
    import json
    from app.tools.ingest_tool import get_collection

    store_dir = Path(__file__).resolve().parent.parent / "data" / "documents"
    json_path = store_dir / f"{doc_id}.json"
    if json_path.exists():
        json_path.unlink()
        log.info("cleanup: deleted %s", json_path)

    try:
        collection = get_collection()
        collection.delete(where={"doc_id": doc_id})
        log.info("cleanup: deleted ChromaDB entries for doc_id=%s", doc_id)
    except Exception as e:
        log.warning("cleanup: ChromaDB delete failed: %s", e)

    _doc_store.pop(doc_id, None)
    log.info("cleanup: doc_id=%s fully purged", doc_id)


@app.post("/analyze/{doc_id}", response_model=AnalyzeResponse)
async def analyze(doc_id: str):
    """
    Steps 2 + 3: Run parallel agent analysis and return structured report.
    Triggers Risk, Gap, and Negotiation agents in parallel, then Critic.
    Also generates a downloadable PDF artifact.
    Contract text files are deleted from disk after analysis (privacy).
    """
    if doc_id not in _doc_store:
        raise HTTPException(status_code=404, detail="Document not found. Upload it first.")

    meta = _doc_store[doc_id]
    log.info("analyze: starting for doc_id=%s", doc_id)

    try:
        report = await analyze_contract(
            doc_id=doc_id,
            filename=meta["filename"],
            doc_type=meta["doc_type"],
        )
    except Exception as e:
        log.exception("analyze: pipeline failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    try:
        report_path = generate_report(report)
        report_filename = Path(report_path).name
    except Exception as e:
        log.error("analyze: report generation failed: %s", e)
        report_filename = None

    _cleanup_document(doc_id)

    return AnalyzeResponse(
        doc_id=doc_id,
        filename=meta["filename"],
        final_risk_score=report.verdict.final_risk_score,
        risk_label=report.verdict.risk_label,
        executive_summary=report.verdict.executive_summary,
        top_three_actions=report.verdict.top_three_actions,
        risky_clauses=[c.model_dump() for c in report.risk.risky_clauses],
        missing_terms=[t.model_dump() for t in report.gaps.missing_terms],
        suggestions=[s.model_dump() for s in report.negotiation.suggestions],
        report_path=f"/artifacts/{report_filename}" if report_filename else None,
        confidence=report.verdict.confidence,
    )


@app.get("/documents")
def list_documents():
    """List all ingested documents in this session."""
    return {"documents": [
        {"doc_id": k, **v} for k, v in _doc_store.items()
    ]}