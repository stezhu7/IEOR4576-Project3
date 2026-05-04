"""
agents/gap_agent.py — Gap Detector Agent

Checks what standard contract terms are MISSING.
A freelancer may not notice what's absent — this agent does.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re

from google import genai
from google.genai import types

from app.schemas import GapAnalysis, MissingTerm
from app.tools.ingest_tool import get_full_text

log = logging.getLogger(__name__)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL    = "gemini-2.5-flash"

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

GAP_SYSTEM = """
You are a contract completeness expert. You review contracts to find what is MISSING —
standard clauses that protect the contractor or weaker party but are absent.

Standard clauses to check for (by contract type):

Freelance contracts should have:
- Clear payment schedule and late payment penalties
- Scope of work / change order process
- Revision limits
- Kill fee (partial payment if client cancels)
- Portfolio / credit rights (can contractor show the work?)
- Force majeure
- Dispute resolution process
- Governing law
- Limitation of liability cap

Terms sheets / loan agreements should have:
- Default and cure period
- Events of default definition
- Cross-default provisions
- Representations and warranties
- Conditions precedent to funding
- Clear maturity date and repayment schedule

For each missing term, provide a short example of standard language they could add.

Return ONLY a JSON object with no markdown fencing:
{
  "missing_terms": [
    {
      "term_name": "Kill fee",
      "importance": "critical",
      "why_needed": "...",
      "standard_language": "If Client cancels the project after work has begun..."
    }
  ],
  "completeness_score": 4,
  "summary": "This contract is missing several key protections for the contractor..."
}

Importance:
- critical: serious financial or legal exposure without it
- recommended: standard practice, should be included
- optional: nice to have
"""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


async def run_gap_agent(doc_id: str) -> GapAnalysis:
    """Run the gap detector. Called in parallel by orchestrator."""
    log.info("gap_agent: starting for doc_id=%s", doc_id)

    # Use full text (gap detection needs the whole picture)
    try:
        full_text = get_full_text(doc_id)
        # Truncate to ~4000 words to stay in context
        words = full_text.split()
        if len(words) > 4000:
            full_text = " ".join(words[:4000]) + "\n[... document truncated ...]"
    except Exception as e:
        log.error("gap_agent: could not load full text: %s", e)
        full_text = "Document text unavailable."

    prompt = f"Check this contract for missing standard terms:\n\n{full_text}"

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GAP_SYSTEM,
                temperature=0.1,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    )

    raw = (response.text or "").strip()
    log.info("gap_agent: raw output length=%d", len(raw))

    try:
        data = _parse_json(raw)
        return GapAnalysis(**data)
    except Exception as exc:
        log.error("gap_agent: parse failed: %s", exc)
        return GapAnalysis(
            missing_terms=[],
            completeness_score=5,
            summary=f"Gap analysis failed: {exc}",
        )