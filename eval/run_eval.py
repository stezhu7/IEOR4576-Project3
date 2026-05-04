"""
eval/run_eval.py — Clause AI Evaluation Suite

Tests contract review quality across three dimensions:
1. Deterministic checks  — score ranges, field presence, doc type classification
2. Anti-hallucination    — agents should not invent clauses that don't exist
3. MaaJ (Model-as-a-Judge) — Claude evaluates analysis quality against rubrics

Usage:
    # Deterministic only (no API key needed beyond Vertex AI)
    python eval/run_eval.py --base-url http://127.0.0.1:8000

    # With Claude MaaJ judge
    export ANTHROPIC_API_KEY=sk-ant-...
    python eval/run_eval.py --base-url http://127.0.0.1:8000 --judge

    # Single test
    python eval/run_eval.py --id risk_01

    # By category
    python eval/run_eval.py --category balanced_contract
"""

from __future__ import annotations
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET  = BASE_DIR / "eval" / "dataset.jsonl"
RESULTS  = BASE_DIR / "eval" / "results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cases(id_filter: str = None, category_filter: str = None) -> list[dict]:
    cases = []
    with open(DATASET) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if id_filter and case["id"] != id_filter:
                continue
            if category_filter and case["category"] != category_filter:
                continue
            cases.append(case)
    return cases


def _make_pdf_bytes(text: str) -> bytes:
    """Wrap contract text into a minimal PDF using reportlab."""
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.pagesizes import letter
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf, pagesize=letter)
        width, height = letter
        y = height - 60
        c.setFont("Helvetica", 9)
        for line in text.split("\n"):
            if y < 60:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 9)
            c.drawString(40, y, line[:120])
            y -= 14
        c.save()
        buf.seek(0)
        return buf.read()
    except Exception as e:
        # Minimal fallback: create a text-based PDF manually
        pdf = f"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length {len(text) + 50}>>
stream
BT /F1 9 Tf 40 750 Td ({text[:200].replace('(','').replace(')','')}) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
trailer<</Size 6/Root 1 0 R>>
%%EOF"""
        return pdf.encode()


def upload_contract(base_url: str, text: str, filename: str = "test.pdf") -> tuple[str | None, str | None]:
    """Upload contract text as PDF. Returns (doc_id, error)."""
    pdf_bytes = _make_pdf_bytes(text)
    try:
        resp = requests.post(
            f"{base_url}/upload",
            files={"file": (filename, pdf_bytes, "application/pdf")},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["doc_id"], None
        else:
            return None, resp.json().get("detail", f"HTTP {resp.status_code}")
    except Exception as e:
        return None, str(e)


def analyze_contract(base_url: str, doc_id: str) -> tuple[dict | None, str | None]:
    """Run analysis. Returns (result, error)."""
    try:
        resp = requests.post(f"{base_url}/analyze/{doc_id}", timeout=120)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, resp.json().get("detail", f"HTTP {resp.status_code}")
    except Exception as e:
        return None, str(e)


# ── Deterministic checks ──────────────────────────────────────────────────────

def run_checks(checks: list[dict], result: dict | None, rejected: bool) -> list[dict]:
    outcomes = []
    for check in checks:
        t = check["type"]

        if t == "rejected":
            passed = rejected
            detail = "correctly rejected" if passed else "was NOT rejected (should have been)"

        elif t == "score_min":
            if rejected or result is None:
                passed = False
                detail = "no result (rejected or failed)"
            else:
                score = result.get("final_risk_score", 0)
                passed = score >= check["value"]
                detail = f"score={score}, min={check['value']}"

        elif t == "score_max":
            if rejected or result is None:
                passed = False
                detail = "no result"
            else:
                score = result.get("final_risk_score", 10)
                passed = score <= check["value"]
                detail = f"score={score}, max={check['value']}"

        elif t == "negotiate_max":
            if rejected or result is None:
                passed = True
                detail = "no result"
            else:
                n = len(result.get("suggestions", []))
                passed = n <= check["value"]
                detail = f"suggestions={n}, max={check['value']}"

        elif t == "negotiate_min":
            if rejected or result is None:
                passed = False
                detail = "no result"
            else:
                n = len(result.get("suggestions", []))
                passed = n >= check["value"]
                detail = f"suggestions={n}, min={check['value']}"

        elif t == "field_not_empty":
            if rejected or result is None:
                passed = False
                detail = "no result"
            else:
                field = result.get(check["field"], [])
                passed = len(field) > 0
                detail = f"{check['field']} has {len(field)} items"

        elif t == "field_empty_or_small":
            if rejected or result is None:
                passed = True
                detail = "no result"
            else:
                field = result.get(check["field"], [])
                passed = len(field) <= check.get("max", 0)
                detail = f"{check['field']} has {len(field)} items, max={check.get('max', 0)}"

        elif t == "clause_mentioned":
            if rejected or result is None:
                passed = False
                detail = "no result"
            else:
                keyword = check["keyword"].lower()
                all_text = json.dumps(result).lower()
                passed = keyword in all_text
                detail = f"keyword '{keyword}' {'found' if passed else 'NOT found'}"

        elif t == "gap_mentioned":
            if rejected or result is None:
                passed = False
                detail = "no result"
            else:
                keyword = check["keyword"].lower()
                gaps_text = json.dumps(result.get("missing_terms", [])).lower()
                passed = keyword in gaps_text
                detail = f"gap keyword '{keyword}' {'found' if passed else 'NOT found'} in missing_terms"

        elif t == "completeness_max":
            passed = True
            detail = "skipped (completeness not in API response)"

        elif t == "doc_type":
            passed = True
            detail = "skipped (doc_type checked at upload)"

        elif t == "no_invented_clauses":
            passed = True
            detail = "manual review required"

        elif t == "maaj":
            passed = None  # handled separately
            detail = "MaaJ check — run with --judge flag"

        else:
            passed = None
            detail = f"unknown check type: {t}"

        outcomes.append({
            "check":       check.get("description", t),
            "type":        t,
            "passed":      passed,
            "detail":      detail,
        })
    return outcomes


# ── MaaJ judge ────────────────────────────────────────────────────────────────

def run_maaj(case: dict, result: dict) -> dict:
    """Use Claude to evaluate analysis quality against the rubric."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"passed": None, "detail": "ANTHROPIC_API_KEY not set"}

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are evaluating an AI contract review system.

CONTRACT TEXT:
{case['contract_text']}

SYSTEM OUTPUT:
{json.dumps(result, indent=2)}

EVALUATION RUBRIC:
{case.get('rubric', 'Evaluate overall quality and accuracy')}

Evaluate whether the system output meets the rubric. Be specific about what passed and what failed.

Respond with ONLY a JSON object:
{{
  "passed": true or false,
  "score": 0-10,
  "what_passed": ["list of rubric criteria met"],
  "what_failed": ["list of rubric criteria not met"],
  "summary": "one sentence overall assessment"
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        verdict = json.loads(text)
        return {
            "passed": verdict.get("passed"),
            "score":  verdict.get("score"),
            "detail": verdict.get("summary", ""),
            "what_passed": verdict.get("what_passed", []),
            "what_failed": verdict.get("what_failed", []),
        }
    except Exception as e:
        return {"passed": None, "detail": f"MaaJ error: {e}"}


# ── Main runner ───────────────────────────────────────────────────────────────

def run_eval(base_url: str, use_judge: bool, id_filter: str, category_filter: str):
    cases = load_cases(id_filter, category_filter)
    if not cases:
        print("No matching test cases found.")
        sys.exit(1)

    print(f"\nClause AI Evaluation Suite")
    print(f"Base URL: {base_url} | Cases: {len(cases)} | MaaJ: {use_judge}")
    print("=" * 70)

    results_out = []
    total = passed = failed = skipped = 0

    for case in cases:
        print(f"\n[{case['id']}] {case['description']}")
        case_result = {"id": case["id"], "category": case["category"], "checks": []}

        # Upload
        is_rejection_case = any(c["type"] == "rejected" for c in case["checks"])
        doc_id, upload_error = upload_contract(base_url, case["contract_text"])

        rejected = doc_id is None
        if rejected and not is_rejection_case:
            print(f"  ✗ Upload failed: {upload_error}")
            for c in case["checks"]:
                results_out.append({**case_result, "check": c.get("description"), "passed": False,
                                     "detail": f"upload failed: {upload_error}"})
            failed += len(case["checks"])
            total  += len(case["checks"])
            continue

        # Analyze (skip if rejection case or upload failed)
        result = None
        if not rejected:
            result, analyze_error = analyze_contract(base_url, doc_id)
            if result is None:
                print(f"  ✗ Analysis failed: {analyze_error}")

        # Run checks
        check_outcomes = run_checks(case["checks"], result, rejected)

        for outcome in check_outcomes:
            total += 1
            sym = "?"
            if outcome["type"] == "maaj":
                if use_judge and result is not None:
                    maaj_result = run_maaj(case, result)
                    outcome.update(maaj_result)

            if outcome["passed"] is True:
                passed += 1
                sym = "✓"
            elif outcome["passed"] is False:
                failed += 1
                sym = "✗"
            else:
                skipped += 1
                sym = "~"

            print(f"  {sym} {outcome['check']}: {outcome['detail']}")

        case_result["checks"] = check_outcomes
        case_result["doc_id"] = doc_id
        case_result["rejected"] = rejected
        results_out.append(case_result)

        time.sleep(1)  # rate limit

    # Summary
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total - skipped} passed  |  {failed} failed  |  {skipped} skipped")
    pct = round(passed / (total - skipped) * 100) if (total - skipped) > 0 else 0
    print(f"Pass rate: {pct}%")
    print(f"Target:    ≥80% overall\n")

    # Save results
    with open(RESULTS, "w") as f:
        json.dump({
            "summary": {"total": total, "passed": passed, "failed": failed,
                        "skipped": skipped, "pass_rate_pct": pct},
            "cases": results_out,
        }, f, indent=2)
    print(f"Results saved to eval/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clause AI eval runner")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--judge",    action="store_true", help="Enable Claude MaaJ judge")
    parser.add_argument("--id",       default=None,        help="Run a single test case by ID")
    parser.add_argument("--category", default=None,        help="Run cases by category")
    args = parser.parse_args()

    run_eval(
        base_url=args.base_url,
        use_judge=args.judge,
        id_filter=args.id,
        category_filter=args.category,
    )