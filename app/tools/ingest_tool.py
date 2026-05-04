from __future__ import annotations
import logging
import os
import re
import uuid
from pathlib import Path

import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types

from app.schemas import IngestedDocument

log = logging.getLogger(__name__)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "europe-west1")
MODEL    = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-004"

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

CHROMA_DIR = str(Path(__file__).resolve().parents[2] / "data" / "chroma")
_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
COLLECTION_NAME = "contracts"


def get_collection() -> chromadb.Collection:
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )



def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text.strip()}")
    return "\n\n".join(pages), len(reader.pages)



def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    log.info("chunk_text: %d words → %d chunks", len(words), len(chunks))
    return chunks



def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    batch_size = 5
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            result = _client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
            )
            for emb in result.embeddings:
                embeddings.append(emb.values)
        except Exception as e:
            log.error("embed_texts: batch %d failed: %s", i, e)
            # fallback: zero vector (won't break pipeline)
            for _ in batch:
                embeddings.append([0.0] * 768)
    return embeddings



CLASSIFY_SYSTEM = """
You are a contract classification expert. Given the first 600 words of a document,
determine if it is a legal contract or agreement, and classify it.

First decide: is this actually a legal contract, agreement, or financial instrument?
- YES if: it contains parties, obligations, terms, signatures, legal language
- NO if: it is a resume, CV, report, article, invoice, presentation, or other non-contract document

Return ONLY a JSON object with no markdown:
{
  "is_contract": true,
  "doc_type": "freelance_contract" | "terms_sheet" | "nda" | "employment" | "sales_contract" | "service_agreement" | "unknown",
  "summary": "2-sentence description"
}
"""


def classify_document(text_preview: str) -> tuple[str, str, bool]:
    import json, re as _re
    preview = " ".join(text_preview.split()[:600])
    try:
        resp = _client.models.generate_content(
            model=MODEL,
            contents=preview,
            config=types.GenerateContentConfig(
                system_instruction=CLASSIFY_SYSTEM,
                temperature=0.0,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = (resp.text or "").strip()
        raw = _re.sub(r"^```(?:json)?", "", raw, flags=_re.I).strip()
        raw = _re.sub(r"```$", "", raw).strip()
        data = json.loads(raw)
        is_contract = data.get("is_contract", True)
        return data.get("doc_type", "unknown"), data.get("summary", ""), is_contract
    except Exception as e:
        log.error("classify_document: failed: %s", e)
        return "unknown", "Contract document uploaded for analysis.", True



def ingest_document(file_bytes: bytes, filename: str) -> IngestedDocument:
    doc_id = str(uuid.uuid4())
    log.info("ingest_document: %s → doc_id=%s", filename, doc_id)

    full_text, page_count = extract_text_from_pdf(file_bytes)
    if not full_text.strip():
        raise ValueError(f"No text could be extracted from {filename}")
    log.info("ingest_document: extracted %d chars, %d pages", len(full_text), page_count)

    doc_type, summary, is_contract = classify_document(full_text)
    if not is_contract:
        raise ValueError(
            f"'{filename}' does not appear to be a contract or legal agreement. "
            "Please upload a contract, NDA, terms sheet, or employment agreement."
        )

    chunks = chunk_text(full_text)

    log.info("ingest_document: embedding %d chunks...", len(chunks))
    embeddings = embed_texts(chunks)

    collection = get_collection()
    ids        = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas  = [
        {
            "doc_id":   doc_id,
            "filename": filename,
            "doc_type": doc_type,
            "chunk_idx": i,
            "page_count": page_count,
        }
        for i in range(len(chunks))
    ]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    log.info("ingest_document: stored %d chunks in ChromaDB", len(chunks))

    # 6. Also store full text for direct retrieval
    _store_full_text(doc_id, full_text, filename, doc_type)

    return IngestedDocument(
        doc_id=doc_id,
        filename=filename,
        doc_type=doc_type,
        page_count=page_count,
        chunk_count=len(chunks),
        summary=summary,
    )


def _store_full_text(doc_id: str, text: str, filename: str, doc_type: str) -> None:
    import json
    store_dir = Path(__file__).resolve().parents[2] / "data" / "documents"
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / f"{doc_id}.json"
    path.write_text(json.dumps({
        "doc_id":   doc_id,
        "filename": filename,
        "doc_type": doc_type,
        "text":     text,
    }), encoding="utf-8")
    log.info("_store_full_text: saved to %s", path)


def get_full_text(doc_id: str) -> str:
    import json
    store_dir = Path(__file__).resolve().parents[2] / "data" / "documents"
    path = store_dir / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No stored document for doc_id={doc_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["text"]