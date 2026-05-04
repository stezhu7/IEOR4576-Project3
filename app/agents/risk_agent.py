"""
agents/risk_agent.py — Risk Analyzer Agent

Queries RAG for risky clause types (non-compete, IP assignment,
indemnification, liability) and returns structured RiskAnalysis.
"""

from __future__ import annotations
import json
import logging
import os
import re

from google import genai
from google.genai import types

from app.schemas import RiskAnalysis, RiskClause
from app.tools.rag_tool import retrieve_clauses, format_chunks_for_prompt
from app.tools.ingest_tool import get_full_text

log = logging.getLogger(__name__)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL    = "gemini-2.5-flash"

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

RISK_QUERIES = [
    "non-compete clause duration scope",
    "intellectual property ownership work product",
    "indemnification liability damages",
    "termination without cause penalty",
    "governing law jurisdiction",
    "unlimited liability exposure",
    "non-solicitation restriction",
    "arbitration waiver jury trial",
]

RISK_SYSTEM = """
You are a contract risk analyst specialising in freelance and commercial contracts.
Your job is to protect the contractor or borrower (the weaker party).

Analyse the provided contract excerpts and identify risky clauses.

Focus on:
- Non-compete clauses (duration, geographic scope, breadth)
- IP/work product assignment (does contractor keep any rights?)
- Indemnification and liability (is the contractor exposed to unlimited liability?)
- Termination clauses (can client terminate immediately with no pay?)
- Governing law (unfavourable jurisdiction?)
- One-sided arbitration / waiver of jury trial
- Loan terms: interest rate, prepayment penalties, recourse provisions

For each risky clause found, extract the verbatim text (max 80 words).

Return ONLY a JSON object with no markdown fencing:
{
  "risky_clauses": [
    {
      "clause_title": "Non-Compete",
      "severity": "high",
      "original_text": "...",
      "explanation": "...",
      "page_hint": "Section 9"
    }
  ],
  "overall_risk_score": 7,
  "top_concern": "The unlimited global non-compete clause..."
}

Severity guide:
- high: could cause significant financial or legal harm
- medium: unfavourable but manageable
- low: minor imbalance, worth noting
"""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


async def run_risk_agent(doc_id: str) -> RiskAnalysis:
    """Run the risk analyzer on a contract. Called in parallel by orchestrator."""
    import asyncio

    log.info("risk_agent: starting for doc_id=%s", doc_id)

    # RAG: retrieve risk-relevant chunks
    all_chunks = []
    for query in RISK_QUERIES:
        chunks = retrieve_clauses(doc_id, query, n_results=3)
        all_chunks.extend(chunks)

    # Deduplicate by chunk_idx
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        if c["chunk_idx"] not in seen:
            seen.add(c["chunk_idx"])
            unique_chunks.append(c)

    # Sort by relevance, take top 10
    unique_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_chunks = unique_chunks[:10]

    context = format_chunks_for_prompt(top_chunks, max_chars=5000)

    prompt = f"Analyse this contract for risks:\n\n{context}"

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=RISK_SYSTEM,
                temperature=0.1,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    )

    raw = (response.text or "").strip()
    log.info("risk_agent: raw output length=%d", len(raw))

    try:
        data = _parse_json(raw)
        return RiskAnalysis(**data)
    except Exception as exc:
        log.error("risk_agent: parse failed: %s", exc)
        return RiskAnalysis(
            risky_clauses=[],
            overall_risk_score=5,
            top_concern=f"Analysis failed: {exc}",
        )