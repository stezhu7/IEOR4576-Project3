"""
agents/negotiation_agent.py — Negotiation Agent

Takes risky clauses found by the risk agent and rewrites them
with more balanced language. Gives the user concrete text to propose.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re

from google import genai
from google.genai import types

from app.schemas import NegotiationAnalysis, NegotiationSuggestion
from app.tools.rag_tool import retrieve_clauses, format_chunks_for_prompt

log = logging.getLogger(__name__)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL    = "gemini-2.5-flash"

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

NEGOTIATION_QUERIES = [
    "non-compete clause",
    "payment terms compensation",
    "intellectual property work product ownership",
    "termination clause notice period",
    "indemnification liability",
    "confidentiality obligations duration",
    "interest rate loan repayment",
    "prepayment penalty",
]

NEGOTIATION_SYSTEM = """
You are a negotiation coach for freelancers and small business owners.
Your job is to rewrite one-sided contract clauses into fairer, balanced language.

CRITICAL RULE: Only suggest changes for clauses that ACTUALLY EXIST in the provided text.
- If a clause has blank fields (e.g. "for a period of __________ months"), flag it as incomplete
  but do NOT invent specific values like "24 months" or "global scope".
- Your original_text must be a verbatim quote from the provided contract excerpts.
- Never fabricate clause language that isn't in the document.
- If a clause is blank/incomplete, your suggestion should be to fill it in with fair terms,
  not to rewrite text that doesn't exist yet.

For each problematic clause you find:
1. Quote the ACTUAL text from the contract (verbatim)
2. Identify what makes it unfair or incomplete
3. Provide replacement or completion language that is professional and balanced

Principles for fair language:
- Non-compete: limit to 6 months max, specific industry, specific geography
- IP: contractor retains portfolio rights; client gets exclusive use license
- Termination: client must give 14 days notice; kill fee of 25% for cancellation after start
- Liability: cap at total contract value (not unlimited)
- Confidentiality: limit to 2 years after contract end (not perpetual)
- Payment: add 1.5%/month late payment interest
- Loan: cure period of 30 days before default declared

Return ONLY a JSON object with no markdown fencing:
{
  "suggestions": [
    {
      "clause_title": "Non-Compete",
      "issue": "Duration is blank — client could fill in any duration",
      "original_text": "for a period of __________ months",
      "suggested_text": "for a period of six (6) months following termination, limited to direct competitors within the same city",
      "benefit_to_user": "Fixes the blank to a reasonable 6-month local restriction..."
    }
  ],
  "negotiation_priority": ["Non-Compete", "IP Assignment", "Termination"],
  "leverage_note": "This appears to be a standard template contract with room for negotiation..."
}
"""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    # Attempt to repair truncated JSON by closing open structures
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find last complete suggestion object and close the JSON
        last_complete = text.rfind('},')
        if last_complete > 0:
            truncated = text[:last_complete + 1]
            # Close suggestions array and add required fields
            repaired = truncated + '], "negotiation_priority": [], "leverage_note": "Analysis truncated — see individual suggestions above."}'
            try:
                return json.loads(repaired)
            except Exception:
                pass
        raise


async def run_negotiation_agent(doc_id: str) -> NegotiationAnalysis:
    """Run the negotiation advisor. Called in parallel by orchestrator."""
    log.info("negotiation_agent: starting for doc_id=%s", doc_id)

    # RAG: retrieve negotiation-relevant chunks
    all_chunks = []
    for query in NEGOTIATION_QUERIES:
        chunks = retrieve_clauses(doc_id, query, n_results=2)
        all_chunks.extend(chunks)

    seen = set()
    unique_chunks = []
    for c in all_chunks:
        if c["chunk_idx"] not in seen:
            seen.add(c["chunk_idx"])
            unique_chunks.append(c)

    unique_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_chunks = unique_chunks[:5]  # Reduced from 8 to keep output manageable
    context = format_chunks_for_prompt(top_chunks, max_chars=3000)  # Reduced from 4500

    prompt = f"Suggest up to 4 negotiation improvements for this contract (keep each suggestion concise):\n\n{context}"

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=NEGOTIATION_SYSTEM,
                temperature=0.2,
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    )

    raw = (response.text or "").strip()
    log.info("negotiation_agent: raw output length=%d", len(raw))

    try:
        data = _parse_json(raw)
        return NegotiationAnalysis(**data)
    except Exception as exc:
        log.error("negotiation_agent: parse failed: %s", exc)
        return NegotiationAnalysis(
            suggestions=[],
            negotiation_priority=[],
            leverage_note=f"Negotiation analysis failed: {exc}",
        )