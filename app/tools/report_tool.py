from __future__ import annotations
import logging
import os
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

from app.schemas import ContractReport

log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RED    = colors.HexColor("#E24B4A")
AMBER  = colors.HexColor("#BA7517")
GREEN  = colors.HexColor("#1D9E75")
BLUE   = colors.HexColor("#185FA5")
LIGHT  = colors.HexColor("#F1EFE8")
DARK   = colors.HexColor("#2C2C2A")

SEVERITY_COLOR = {"high": RED, "medium": AMBER, "low": GREEN}
RISK_LABEL_COLOR = {"low": GREEN, "moderate": AMBER, "high": RED, "critical": RED}
IMPORTANCE_COLOR = {"critical": RED, "recommended": AMBER, "optional": GREEN}


def _styles():
    styles = getSampleStyleSheet()
    custom = {
        "Title":    ParagraphStyle("Title",    fontSize=20, textColor=DARK,
                                    spaceAfter=6, fontName="Helvetica-Bold"),
        "H2":       ParagraphStyle("H2",       fontSize=13, textColor=BLUE,
                                    spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold"),
        "Body":     ParagraphStyle("Body",     fontSize=10, textColor=DARK,
                                    spaceAfter=4,  leading=14),
        "Small":    ParagraphStyle("Small",    fontSize=8.5, textColor=colors.HexColor("#5F5E5A"),
                                    spaceAfter=2,  leading=12),
        "Bold":     ParagraphStyle("Bold",     fontSize=10, textColor=DARK,
                                    fontName="Helvetica-Bold"),
        "Clause":   ParagraphStyle("Clause",   fontSize=9, textColor=colors.HexColor("#444441"),
                                    leading=13,    fontName="Helvetica-Oblique"),
    }
    return custom


def generate_report(report: ContractReport) -> str:
    """
    Generate a PDF report and save to artifacts/.
    Returns the file path.
    """
    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in report.filename[:30])
    path     = str(ARTIFACTS_DIR / f"{ts}_{safe_name}_clause_report.pdf")

    doc = SimpleDocTemplate(
        path,
        pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch,   bottomMargin=0.75*inch,
    )

    s = _styles()
    story = []

    story.append(Paragraph("Clause — Contract Analysis Report", s["Title"]))
    story.append(Paragraph(
        f"<b>File:</b> {report.filename} &nbsp;|&nbsp; "
        f"<b>Type:</b> {report.doc_type.replace('_', ' ').title()} &nbsp;|&nbsp; "
        f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        s["Small"]
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    v     = report.verdict
    score = v.final_risk_score
    label = v.risk_label.upper()
    score_color = RISK_LABEL_COLOR.get(v.risk_label, AMBER)

    score_table = Table(
        [[
            Paragraph(f"<b>Risk Score</b>", s["Bold"]),
            Paragraph(f"<font color='{score_color.hexval()}'><b>{score}/10 — {label}</b></font>",
                      s["Bold"]),
            Paragraph(f"<b>Confidence:</b> {v.confidence}", s["Small"]),
        ]],
        colWidths=[1.5*inch, 2.5*inch, 3*inch],
    )
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT]),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.HexColor("#B4B2A9")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Executive Summary", s["H2"]))
    story.append(Paragraph(v.executive_summary, s["Body"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Top 3 Actions", s["H2"]))
    for i, action in enumerate(v.top_three_actions, 1):
        story.append(Paragraph(f"{i}. {action}", s["Body"]))
    story.append(Spacer(1, 6))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(
        f"Risk Analysis  <font color='gray'>(score: {report.risk.overall_risk_score}/10)</font>",
        s["H2"]
    ))

    if report.risk.risky_clauses:
        for clause in report.risk.risky_clauses:
            sc = SEVERITY_COLOR.get(clause.severity, AMBER)
            block = [
                Paragraph(
                    f"<font color='{sc.hexval()}'><b>[{clause.severity.upper()}]</b></font> "
                    f"<b>{clause.clause_title}</b>"
                    + (f" <font color='gray'>— {clause.page_hint}</font>" if clause.page_hint else ""),
                    s["Bold"]
                ),
                Paragraph(clause.explanation, s["Body"]),
                Paragraph(f'"{clause.original_text}"', s["Clause"]),
                Spacer(1, 4),
            ]
            story.append(KeepTogether(block))
    else:
        story.append(Paragraph("No significant risk clauses identified.", s["Body"]))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(
        f"Gap Analysis  <font color='gray'>(completeness: {report.gaps.completeness_score}/10)</font>",
        s["H2"]
    ))
    story.append(Paragraph(report.gaps.summary, s["Body"]))

    if report.gaps.missing_terms:
        story.append(Spacer(1, 4))
        for term in report.gaps.missing_terms:
            ic = IMPORTANCE_COLOR.get(term.importance, AMBER)
            block = [
                Paragraph(
                    f"<font color='{ic.hexval()}'><b>[{term.importance.upper()}]</b></font> "
                    f"<b>{term.term_name}</b>",
                    s["Bold"]
                ),
                Paragraph(term.why_needed, s["Body"]),
                Paragraph(f'Suggested language: "{term.standard_language}"', s["Clause"]),
                Spacer(1, 4),
            ]
            story.append(KeepTogether(block))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph("Negotiation Suggestions", s["H2"]))

    if report.negotiation.leverage_note:
        story.append(Paragraph(f"<i>{report.negotiation.leverage_note}</i>", s["Small"]))
        story.append(Spacer(1, 4))

    if report.negotiation.negotiation_priority:
        story.append(Paragraph(
            "<b>Priority order:</b> " + " → ".join(report.negotiation.negotiation_priority),
            s["Body"]
        ))
        story.append(Spacer(1, 4))

    if report.negotiation.suggestions:
        for sug in report.negotiation.suggestions:
            block = [
                Paragraph(f"<b>{sug.clause_title}</b> — {sug.issue}", s["Bold"]),
                Paragraph(f'Current: "{sug.original_text}"', s["Clause"]),
                Paragraph(f'<b>Suggested:</b> "{sug.suggested_text}"', s["Body"]),
                Paragraph(f"<font color='{GREEN.hexval()}'>{sug.benefit_to_user}</font>",
                          s["Small"]),
                Spacer(1, 6),
            ]
            story.append(KeepTogether(block))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(
        "⚠ Disclaimer: This report is generated by an AI system and is for informational purposes only. "
        "It does not constitute legal advice. Always consult a qualified attorney before signing any contract.",
        s["Small"]
    ))

    doc.build(story)
    log.info("generate_report: saved to %s", path)
    return path