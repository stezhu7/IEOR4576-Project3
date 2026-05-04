
from __future__ import annotations
import logging
import os

from app.tools.ingest_tool import get_collection, embed_texts, EMBED_MODEL

log = logging.getLogger(__name__)


def retrieve_clauses(
    doc_id: str,
    query: str,
    n_results: int = 6,
) -> list[dict]:
    """
    Retrieve the top-N most relevant chunks from a specific document.

    Args:
        doc_id:    The document UUID to search within
        query:     Natural language query (e.g. "non-compete clause duration")
        n_results: Number of chunks to return

    Returns:
        List of dicts: {text, chunk_idx, relevance_score}
    """
    log.info("retrieve_clauses: doc_id=%s, query=%r, n=%d", doc_id, query, n_results)

    query_embedding = embed_texts([query])[0]

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            chunks.append({
                "text":            doc,
                "chunk_idx":       meta.get("chunk_idx", i),
                "relevance_score": round(1 - dist, 4),  # cosine similarity
            })

    log.info("retrieve_clauses: returned %d chunks", len(chunks))
    return chunks


def retrieve_all_chunks(doc_id: str) -> list[str]:
    collection = get_collection()
    results = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
    )

    if not results["documents"]:
        return []

    # Sort by chunk_idx
    paired = list(zip(results["metadatas"], results["documents"]))
    paired.sort(key=lambda x: x[0].get("chunk_idx", 0))
    chunks = [doc for _, doc in paired]
    log.info("retrieve_all_chunks: %d chunks for doc_id=%s", len(chunks), doc_id)
    return chunks


def format_chunks_for_prompt(chunks: list[dict], max_chars: int = 4000, min_relevance: float = 0.3) -> str:
    relevant = [c for c in chunks if c["relevance_score"] >= min_relevance]
    if not relevant:
        return "[No relevant clauses found in this document for the queried topics.]"

    lines = []
    total = 0
    for i, chunk in enumerate(relevant):
        snippet = f"[Chunk {chunk['chunk_idx']+1} | relevance={chunk['relevance_score']}]\n{chunk['text']}"
        if total + len(snippet) > max_chars:
            lines.append(f"[... {len(relevant) - i} more chunks truncated ...]")
            break
        lines.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(lines)