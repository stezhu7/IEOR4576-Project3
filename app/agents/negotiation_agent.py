"""
agents/negotiation_agent.py — Negotiation Agent

Uses the FULL contract text (same approach as gap_agent) instead of RAG.
RAG only retrieves fragments, causing hallucination on well-written contracts.
Full text gives the model accurate context to judge what's actually unfair.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re

from google import genai
from google.genai import types

from app.schemas import NegotiationAnalysis
from app.tools.ingest_tool import get_full_text

log = logging.getLogger(__name__)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL    = "gemini-2.5-flash"

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

NEGOTIATION_SYSTEM = """
You are a negotiation coach. You review the FULL contract text and suggest improvements
ONLY for clauses that are genuinely one-sided or unfair for this type of contract.

STRICT RULES:
1. First identify the contract type. Apply appropriate standards for that type.
2. Only flag clauses that are ACTUALLY present AND genuinely unfair.
3. original_text MUST be verbatim from the contract — copy exactly.
4. If a clause is already balanced and appropriate for its contract type, do NOT suggest changes.
5. Return empty suggestions [] if the contract is already well-balanced.
   A short or empty list is a GOOD outcome for a fair contract.
6. Do NOT suggest removing or weakening standard commercial protections just because
   they favour one party — both parties' standard protections are legitimate.

CONTRACT-TYPE STANDARDS:

FREELANCE / SERVICE CONTRACTS — suggest improvements if:
- Non-compete: over 12 months → suggest 6 months max
- IP: contractor loses ALL rights including portfolio → add portfolio retention
- Termination: less than 14 days notice OR no kill fee → improve both
- Liability: fully uncapped for contractor → suggest cap at contract value
- Confidentiality: perpetual → suggest 2-year limit

COMMERCIAL SUPPLY AGREEMENTS — suggest improvements if:
- Governing law: completely absent in cross-border contract → suggest adding
- Liability cap: excludes substitute goods cost AND based only on 12-month payments
  with no floor → suggest a reasonable floor amount
- Termination no-liability: applies even to non-compliant termination (not just compliant)
  → narrow to compliant termination only
- Arbitration: no governing substantive law specified → suggest adding governing law
DO NOT suggest changes to:
  - Late payment interest tied to published rates (Prime Rate, SOFR, etc.) — commercially normal
  - Arbitration in neutral third country — not inherently unfair
  - Prevailing-party attorneys' fees — standard commercial practice
  - Liability caps that are industry-standard (12-month rolling cap with bodily injury carve-out)
  - Most-favored pricing absence — negotiating point, not a fairness issue

LOAN / TERMS SHEETS — suggest improvements if:
- No cure period → suggest 30 days
- Prepayment penalty in first 12 months with no carve-out → suggest carve-out for refinancing

For each genuine issue:
{
  "clause_title": "...",
  "issue": "specific reason this is unfair for this contract type",
  "original_text": "EXACT verbatim quote from the contract",
  "suggested_text": "improved replacement — must be realistic for this contract type",
  "benefit_to_user": "how this change protects the user without being unreasonable"
}

Return ONLY a JSON object with no markdown:
{
  "suggestions": [],
  "negotiation_priority": [],
  "leverage_note": "Overall assessment: is this a standard contract for its type? What is the negotiating position?"
}
"""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        last_complete = text.rfind('},')
        if last_complete > 0:
            repaired = text[:last_complete + 1] + '], "negotiation_priority": [], "leverage_note": "Analysis truncated."}'
            try:
                return json.loads(repaired)
            except Exception:
                pass
        raise


async def run_negotiation_agent(doc_id: str) -> NegotiationAnalysis:
    """Run negotiation advisor using full contract text. Called in parallel by orchestrator."""
    log.info("negotiation_agent: starting for doc_id=%s", doc_id)

    # Use full text — same as gap_agent
    # Full text gives accurate context; RAG fragments cause hallucination
    try:
        full_text = get_full_text(doc_id)
        words = full_text.split()
        if len(words) > 4000:
            full_text = " ".join(words[:4000]) + "\n[... document truncated ...]"
    except Exception as e:
        log.error("negotiation_agent: could not load full text: %s", e)
        full_text = "Document text unavailable."

    prompt = (
        "Review this contract and suggest improvements ONLY for clauses that are "
        "genuinely unfair. If the contract is already balanced, return empty suggestions.\n\n"
        f"{full_text}"
    )

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=NEGOTIATION_SYSTEM,
                temperature=0.1,
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