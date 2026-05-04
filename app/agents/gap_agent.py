"""
agents/gap_agent.py — Gap Detector Agent

Checks what standard contract terms are MISSING.
Contract-type aware: different standards for freelance vs commercial supply vs loan.
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
You are a contract completeness expert. You review contracts to find what is GENUINELY MISSING —
not merely what could be added as a preference, but what creates real legal or financial exposure.

CRITICAL CALIBRATION RULES:
1. First identify the contract type (freelance, supply agreement, loan/terms sheet, NDA, employment).
   Apply the appropriate standard for THAT contract type — do not apply freelance standards to
   commercial supply agreements, or loan agreement standards to service contracts.
2. Only flag terms as "critical" if their absence creates serious, immediate legal or financial exposure.
3. If a term exists but is implemented differently from best practice, do NOT flag it as missing.
   (e.g. if cure period exists as 90 days but you prefer 30, that is not a gap — it exists)
4. Do NOT flag the absence of "nice to have" terms as critical or recommended.
5. Do NOT flag the absence of preferential terms (most-favored pricing, cross-default in simple
   bilateral contracts) as gaps — these are negotiating points, not standard requirements.

Standards by contract type:

FREELANCE / SERVICE CONTRACT gaps to check:
- Payment schedule (how/when contractor gets paid) — critical if completely absent
- Kill fee or partial payment on early termination — critical if absent
- Governing law — critical if absent
- Scope definition (what is in/out of scope) — critical if vague or absent
- IP ownership clarity — critical if ambiguous
- Portfolio/credit rights for contractor — recommended if absent
- Revision limits — optional

COMMERCIAL SUPPLY AGREEMENT gaps to check:
- Governing law (substantive law, not just arbitration venue) — high if absent
- Product specifications reference — critical if absent
- Delivery terms and acceptance criteria — critical if absent
- Material breach definition (even implicit) — check if cure mechanism exists before flagging
- Force majeure — recommended if absent
- Warranty disclaimer — check if present
DO NOT flag as gaps in supply agreements:
  - Cross-default provisions (optional in simple bilateral supply contracts)
  - Conditions precedent to performance (unless contract is silent on performance obligations)
  - Most-favored pricing / best customer terms (negotiating point, not a gap)
  - Detailed events of default list (if material breach + cure already present)

LOAN / TERMS SHEET gaps to check:
- Events of default with cure periods — critical if absent
- Maturity date and repayment schedule — critical if absent
- Interest rate (fixed or referenced rate) — critical if absent
- Governing law — critical if absent
- Representations and warranties — recommended
- Conditions precedent to funding — recommended

NDA gaps to check:
- Definition of confidential information — critical if absent
- Duration of confidentiality obligation — critical if perpetual with no end date
- Permitted disclosures / exceptions — recommended
- Return of materials on termination — recommended

For each genuinely missing term, explain concisely why its absence matters for THIS contract.

Return ONLY a JSON object with no markdown fencing:
{
  "missing_terms": [
    {
      "term_name": "Governing law",
      "importance": "critical",
      "why_needed": "The contract specifies AAA arbitration in the UK but does not state which country's substantive law governs interpretation. This gap could cause a costly preliminary dispute about applicable law.",
      "standard_language": "This Agreement shall be governed by and construed in accordance with the laws of [agreed jurisdiction], excluding its conflict-of-laws rules."
    }
  ],
  "completeness_score": 6,
  "summary": "Overall assessment of completeness appropriate to this contract type."
}

Importance:
- critical: serious financial or legal exposure from this gap
- recommended: standard practice for this contract type, should be included
- optional: minor improvement only
"""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


async def run_gap_agent(doc_id: str) -> GapAnalysis:
    """Run the gap detector. Called in parallel by orchestrator."""
    log.info("gap_agent: starting for doc_id=%s", doc_id)

    try:
        full_text = get_full_text(doc_id)
        words = full_text.split()
        if len(words) > 4000:
            full_text = " ".join(words[:4000]) + "\n[... document truncated ...]"
    except Exception as e:
        log.error("gap_agent: could not load full text: %s", e)
        full_text = "Document text unavailable."

    prompt = (
        "First identify the contract type, then check for genuinely missing standard terms "
        "appropriate for that contract type. Do not flag terms that already exist.\n\n"
        f"{full_text}"
    )

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