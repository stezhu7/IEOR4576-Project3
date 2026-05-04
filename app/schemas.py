"""
schemas.py — Pydantic structured output contracts for every agent boundary.

Every agent emits one of these; the orchestrator validates before passing downstream.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestedDocument(BaseModel):
    doc_id: str                     = Field(description="UUID for this document in ChromaDB")
    filename: str
    doc_type: Literal["freelance_contract", "terms_sheet", "nda", "employment", "sales_contract", "service_agreement", "unknown"]
    page_count: int
    chunk_count: int
    summary: str                    = Field(description="2-sentence description of the document")


# ── Risk Analyzer ─────────────────────────────────────────────────────────────

class RiskClause(BaseModel):
    clause_title: str               = Field(description="Name of the clause, e.g. 'Non-Compete'")
    severity: Literal["high", "medium", "low"]
    original_text: str              = Field(description="Verbatim excerpt from the contract (≤80 words)")
    explanation: str                = Field(description="Why this clause is risky for the user")
    page_hint: Optional[str]        = Field(None, description="Section number or page reference if available")


class RiskAnalysis(BaseModel):
    risky_clauses: list[RiskClause]
    overall_risk_score: int         = Field(ge=0, le=10, description="0=very safe, 10=extremely risky")
    top_concern: str                = Field(description="Single most important risk in one sentence")


# ── Gap Detector ──────────────────────────────────────────────────────────────

class MissingTerm(BaseModel):
    term_name: str                  = Field(description="e.g. 'Payment schedule', 'Dispute resolution'")
    importance: Literal["critical", "recommended", "optional"]
    why_needed: str                 = Field(description="Why a freelancer should care about this missing term")
    standard_language: str          = Field(description="Example clause text they could add (≤60 words)")


class GapAnalysis(BaseModel):
    missing_terms: list[MissingTerm]
    completeness_score: int         = Field(ge=0, le=10, description="0=very incomplete, 10=comprehensive")
    summary: str                    = Field(description="Overall assessment of contract completeness")


# ── Negotiation Agent ─────────────────────────────────────────────────────────

class NegotiationSuggestion(BaseModel):
    clause_title: str
    issue: str                      = Field(description="What's wrong or one-sided about the current language")
    original_text: str              = Field(description="Current problematic text (≤60 words)")
    suggested_text: str             = Field(description="Improved replacement language (≤80 words)")
    benefit_to_user: str            = Field(description="How this change protects the user")


class NegotiationAnalysis(BaseModel):
    suggestions: list[NegotiationSuggestion]
    negotiation_priority: list[str] = Field(
        description="Ordered list of clause titles to negotiate first (most impactful first)"
    )
    leverage_note: str              = Field(
        description="Brief note on negotiating position — is this a standard or unusual contract?"
    )


# ── Critic ────────────────────────────────────────────────────────────────────

class CriticVerdict(BaseModel):
    final_risk_score: int           = Field(ge=0, le=10)
    risk_label: Literal["low", "moderate", "high", "critical"]
    executive_summary: str          = Field(description="3-4 sentence plain-English summary for a non-lawyer")
    top_three_actions: list[str]    = Field(description="Exactly 3 concrete actions the user should take")
    confidence: Literal["high", "medium", "low"]


# ── Full report (assembled from all agents) ───────────────────────────────────

class ContractReport(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    risk:        RiskAnalysis
    gaps:        GapAnalysis
    negotiation: NegotiationAnalysis
    verdict:     CriticVerdict


# ── API shapes ────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    doc_id:     str
    filename:   str
    doc_type:   str
    chunk_count: int
    message:    str


class AnalyzeResponse(BaseModel):
    doc_id:          str
    filename:        str
    final_risk_score: int
    risk_label:      str
    executive_summary: str
    top_three_actions: list[str]
    risky_clauses:   list[dict]
    missing_terms:   list[dict]
    suggestions:     list[dict]
    report_path:     Optional[str] = None
    confidence:      str