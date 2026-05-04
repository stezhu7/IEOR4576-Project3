"""
agents/orchestrator.py — Orchestrator + Critic

Responsibilities:
1. Fan out to Risk, Gap, and Negotiation agents in parallel (asyncio.gather)
2. Pass combined findings to the Critic agent
3. Assemble and return the final ContractReport
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re

from google import genai
from google.genai import types

from app.schemas import (
    ContractReport, CriticVerdict,
    RiskAnalysis, GapAnalysis, NegotiationAnalysis,
)
from app.agents.risk_agent        import run_risk_agent
from app.agents.gap_agent         import run_gap_agent
from app.agents.negotiation_agent import run_negotiation_agent

log = logging.getLogger(__name__)

PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
MODEL    = "gemini-2.5-flash"

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)


# ── Critic ────────────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """
You are a senior contract attorney reviewing a junior analyst's work.
You receive findings from three specialist agents:
- Risk analyzer (risky clauses and severity)
- Gap detector (missing standard terms)
- Negotiation advisor (suggested improvements)

Your job:
1. Review the findings for consistency and accuracy
2. Assign a final risk score (0-10) — be calibrated:
   - 0: well-drafted contract, no meaningful risks
   - 1-2: one or two minor concerns, generally safe to sign
   - 3-5: genuine issues worth negotiating before signing
   - 6-7: significant risks, careful review required
   - 8-10: do not sign without legal counsel
   IMPORTANT: One medium-severity issue = 1-2, NOT 3+. Reserve 6+ for multiple genuine issues.
3. Write a plain-English executive summary (3-4 sentences, no jargon)
4. Give exactly 3 concrete action items

Return ONLY a JSON object with no markdown:
{
  "final_risk_score": 5,
  "risk_label": "moderate",
  "executive_summary": "...",
  "top_three_actions": [
    "Request that the non-compete be limited to 6 months...",
    "Add a kill fee clause...",
    "Clarify payment terms..."
  ],
  "confidence": "high"
}

risk_label must be: "low" (0-2), "moderate" (3-5), "high" (6-7), "critical" (8-10)
confidence: "high" if 5+ risk issues found, "medium" if 2-4, "low" if 0-1
"""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


async def run_critic(
    risk: RiskAnalysis,
    gaps: GapAnalysis,
    negotiation: NegotiationAnalysis,
) -> CriticVerdict:
    """Critic reviews all three agent outputs and produces a final verdict."""
    log.info("critic: reviewing findings")

    findings_summary = json.dumps({
        "risk_analysis": {
            "overall_risk_score": risk.overall_risk_score,
            "top_concern":        risk.top_concern,
            "risky_clauses": [
                {
                    "clause_title": c.clause_title,
                    "severity":     c.severity,
                    "explanation":  c.explanation,
                }
                for c in risk.risky_clauses
            ],
        },
        "gap_analysis": {
            "completeness_score": gaps.completeness_score,
            "summary":            gaps.summary,
            "missing_terms": [
                {
                    "term_name":   t.term_name,
                    "importance":  t.importance,
                    "why_needed":  t.why_needed,
                }
                for t in gaps.missing_terms
            ],
        },
        "negotiation": {
            "negotiation_priority": negotiation.negotiation_priority,
            "leverage_note":        negotiation.leverage_note,
            "suggestion_count":     len(negotiation.suggestions),
        },
    }, indent=2)

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _client.models.generate_content(
            model=MODEL,
            contents=f"Review these contract analysis findings:\n\n{findings_summary}",
            config=types.GenerateContentConfig(
                system_instruction=CRITIC_SYSTEM,
                temperature=0.0,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    )

    raw = (response.text or "").strip()
    log.info("critic: raw output length=%d", len(raw))

    try:
        data = _parse_json(raw)
        return CriticVerdict(**data)
    except Exception as exc:
        log.error("critic: parse failed: %s", exc)
        # Derive a reasonable verdict from the risk score
        score = risk.overall_risk_score
        label = "low" if score <= 2 else "moderate" if score <= 5 else "high" if score <= 7 else "critical"
        return CriticVerdict(
            final_risk_score=score,
            risk_label=label,
            executive_summary=risk.top_concern,
            top_three_actions=[
                "Review the identified risky clauses carefully",
                "Consider adding missing standard terms",
                "Consult a lawyer for high-severity clauses",
            ],
            confidence="low",
        )


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def analyze_contract(
    doc_id: str,
    filename: str,
    doc_type: str,
) -> ContractReport:
    """
    Full analysis pipeline:
    1. Fan out to 3 specialist agents IN PARALLEL
    2. Critic reviews combined findings
    3. Return ContractReport
    """
    log.info("orchestrator: analyzing doc_id=%s (%s)", doc_id, filename)

    # Step 1: Parallel fan-out (grab-bag: Parallel Execution)
    log.info("orchestrator: launching 3 agents in parallel")
    risk, gaps, negotiation = await asyncio.gather(
        run_risk_agent(doc_id),
        run_gap_agent(doc_id),
        run_negotiation_agent(doc_id),
        return_exceptions=True,
    )

    # Handle any agent failures gracefully
    if isinstance(risk, Exception):
        log.error("orchestrator: risk_agent failed: %s", risk)
        risk = RiskAnalysis(risky_clauses=[], overall_risk_score=5,
                            top_concern="Risk analysis failed.")
    if isinstance(gaps, Exception):
        log.error("orchestrator: gap_agent failed: %s", gaps)
        gaps = GapAnalysis(missing_terms=[], completeness_score=5,
                           summary="Gap analysis failed.")
    if isinstance(negotiation, Exception):
        log.error("orchestrator: negotiation_agent failed: %s", negotiation)
        negotiation = NegotiationAnalysis(suggestions=[], negotiation_priority=[],
                                          leverage_note="Negotiation analysis failed.")

    log.info("orchestrator: all agents complete, running critic")

    # Step 2: Critic reviews combined findings
    verdict = await run_critic(risk, gaps, negotiation)

    log.info("orchestrator: critic verdict — score=%d label=%s",
             verdict.final_risk_score, verdict.risk_label)

    return ContractReport(
        doc_id=doc_id,
        filename=filename,
        doc_type=doc_type,
        risk=risk,
        gaps=gaps,
        negotiation=negotiation,
        verdict=verdict,
    )