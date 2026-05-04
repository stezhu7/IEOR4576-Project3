from __future__ import annotations
import json
import logging
import os
import re

from google import genai
from google.genai import types

from app.schemas import RiskAnalysis, RiskClause
from app.tools.ingest_tool import get_full_text
from app.tools.ingest_tool import get_full_text

log = logging.getLogger(__name__)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "europe-west1")
MODEL    = "gemini-2.5-flash"

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

RISK_SYSTEM = """
You are a contract risk analyst. Your job is to identify clauses that create genuine legal
or financial exposure for the weaker or reviewing party.

STRICT RULES:
1. Only flag clauses that are ACTUALLY present and genuinely risky.
2. original_text MUST be verbatim from the contract — copy exactly.
3. Be calibrated by contract type — commercial supply agreements, loan agreements, and
   freelance contracts have different risk standards.
4. Do NOT flag clauses that are standard practice for their contract type.
5. overall_risk_score calibration:
   - 0: well-drafted contract, no meaningful risks
   - 1-2: one or two minor concerns, generally safe to sign
   - 3-5: genuine issues worth negotiating before signing
   - 6-7: significant risks, careful review required
   - 8-10: do not sign without legal counsel
   IMPORTANT: A contract with only one medium-severity issue scores 1-2, NOT 3+.
   Reserve 6+ for contracts with multiple genuine issues.

CONTRACT-TYPE SPECIFIC STANDARDS:

FREELANCE / SERVICE contracts — flag if:
- Non-compete: over 12 months duration OR global/unlimited scope → high
- IP: contractor loses ALL rights including portfolio use → high
- Liability: unlimited, uncapped contractor exposure → high
- Liability cap ONE-SIDED (contractor capped but client not) → medium
- Termination: no notice period AND no kill fee → high
- Confidentiality: perpetual with no end date → medium

DO NOT flag in freelance contracts:
- Mutual liability cap (both parties equally capped at contract value) — balanced and fair
- Mutual consequential damages exclusion — standard in all contract types
- Insurance requirements — normal commercial practice
- Kill fee already present — do not flag termination as risky

COMMERCIAL SUPPLY AGREEMENTS — flag if:
- Liability cap: excludes cost of substitute goods AND caps at 12-month payments → high
  (but note: 12-month cap + bodily injury carve-out is commercially common → medium)
- Termination no-liability: broad exclusion of ALL damages including from non-compliant
  termination → high. But if clause only excludes damages from COMPLIANT termination
  (i.e. party followed all notice/cure requirements) → medium, as this is commercially normal
- Governing law: completely absent in a cross-border contract → high
- Warranty disclaimer: implied warranties fully disclaimed with no product warranty → high
  (but: disclaimer alongside explicit product conformance warranty → medium)
- Arbitration: mandatory arbitration with no governing law specified → medium

LOAN / TERMS SHEETS — flag if:
- No cure period before default → high
- Interest rate above market or compounding unexpectedly → high
- Full personal recourse with no carve-outs → high
- Prepayment penalty in first 12 months → medium

DO NOT flag as risks (any contract type):
- Mutual liability caps (both parties equally capped) — balanced and fair
- Mutual consequential damages exclusions — commercially standard
- Most-favored pricing / best customer status absence — not a risk
- Cross-default in simple bilateral contracts — not standard
- Late payment interest tied to published rates like Prime Rate — commercially normal
- Arbitration site in neutral third country — not inherently risky
- Prevailing-party attorneys' fees — standard commercial practice

Return ONLY a JSON object with no markdown fencing:
{
  "risky_clauses": [
    {
      "clause_title": "Governing Law — Missing",
      "severity": "high",
      "original_text": "exact verbatim text from contract, or N/A if clause is absent",
      "explanation": "why this is risky for this specific contract",
      "page_hint": "Section 13 / Appendix A"
    }
  ],
  "overall_risk_score": 6,
  "top_concern": "The most serious risk in one sentence"
}

Severity guide:
- high: significant financial or legal exposure
- medium: unfavourable but manageable, worth negotiating
- low: minor imbalance, worth noting
"""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


async def run_risk_agent(doc_id: str) -> RiskAnalysis:
    import asyncio

    log.info("risk_agent: starting for doc_id=%s", doc_id)

    try:
        full_text = get_full_text(doc_id)
        words = full_text.split()
        if len(words) > 4000:
            full_text = " ".join(words[:4000]) + "\n[... document truncated ...]"
    except Exception as e:
        log.error("risk_agent: could not load full text: %s", e)
        full_text = "Document text unavailable."

    prompt = f"Analyse this contract for genuine risks to the contractor:\n\n{full_text}"

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