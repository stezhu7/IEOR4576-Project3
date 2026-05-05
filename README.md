# Clause — AI Contract Review Agent

An AI-powered contract review agent that protects freelancers, small businesses, and commercial buyers before they sign. Upload any contract PDF — freelance agreement, supply agreement, NDA, terms sheet — and receive a structured risk analysis, gap detection, negotiation suggestions, and a downloadable PDF report in under 30 seconds.

---

## Live Demo

**Deployed URL:** `https://ieor4576-project3-git-117951089771.europe-west1.run.app`

---

## What It Does

Clause runs three specialist AI agents **in parallel**, then a critic agent reviews their combined findings:

1. **Risk Analyzer** — identifies dangerous clauses with verbatim citations and severity ratings
2. **Gap Detector** — finds missing standard terms appropriate for the contract type
3. **Negotiation Advisor** — suggests concrete replacement language for unfair clauses
4. **Critic** — synthesises all findings into a calibrated risk score (0–10) and three priority actions

The system is **contract-type aware**: it applies different standards to freelance contracts, commercial supply agreements, loan/terms sheets, and NDAs. A mutual liability cap in a supply agreement is not flagged the same way as an uncapped one-sided indemnity in a freelance contract.

---

## Running Locally

### Prerequisites
- Python 3.13
- [uv](https://github.com/astral-sh/uv)
- Google Cloud project with Vertex AI API enabled
- Authenticated: `gcloud auth application-default login`

### Setup

```bash
git clone https://github.com/stezhu7/IEOR4576-Project3
cd clause

# Install dependencies
uv sync

# Set environment variables (PowerShell)
$env:GOOGLE_CLOUD_PROJECT = "my-project-4576-project3"
$env:GOOGLE_CLOUD_REGION  = "europe-west1"
$env:PYTHONPATH           = "."

# Activate venv and start
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --reload-dir app

# Or with uv run (no activation needed)
uv run uvicorn app.main:app --reload --reload-dir app
```

Open `http://127.0.0.1:8000` and upload a contract PDF.

**Diagnostic endpoint:** `http://127.0.0.1:8000/health`

---

## Architecture

```
User uploads PDF
      │
      ▼
POST /upload
  app/tools/ingest_tool.py → ingest_document()
  ├── extract_text_from_pdf()   pypdf text extraction
  ├── classify_document()       Gemini: contract type + is_contract check
  ├── chunk_text()              600-word overlapping chunks
  ├── embed_texts()             Vertex AI text-embedding-004
  └── ChromaDB store            persistent vector index

POST /analyze/{doc_id}
  app/agents/orchestrator.py → analyze_contract()
  │
  └── asyncio.gather() ── parallel fan-out ──────────────────────┐
        │                                                         │
        ├── app/agents/risk_agent.py → run_risk_agent()          │
        │   Full contract text → Gemini → RiskAnalysis           │
        │                                                         │
        ├── app/agents/gap_agent.py → run_gap_agent()            │
        │   Full contract text → Gemini → GapAnalysis            │
        │                                                         │
        └── app/agents/negotiation_agent.py → run_negotiation_agent()
            Full contract text → Gemini → NegotiationAnalysis   ─┘
                    │
                    ▼
      app/agents/orchestrator.py → run_critic()
      Combined findings → Gemini → CriticVerdict
                    │
                    ▼
      app/tools/report_tool.py → generate_report()
      ReportLab → PDF artifact saved to artifacts/
                    │
                    ▼
      JSON response → frontend (risk score, clauses, suggestions, download link)
```

---

## Class Concepts Implemented

### 1. RAG — Retrieval-Augmented Generation
**File:** `app/tools/ingest_tool.py` → `embed_texts()`, `ingest_document()`
**File:** `app/tools/rag_tool.py` → `retrieve_clauses()`, `format_chunks_for_prompt()`

Contract PDFs are parsed, chunked into 600-word overlapping windows, and embedded using Vertex AI `text-embedding-004`. Embeddings are stored in a persistent ChromaDB vector index. The `retrieve_clauses()` function performs semantic search within a specific document to find the most relevant sections for a given query (e.g. "non-compete clause duration scope"). A `min_relevance=0.3` threshold filters out low-similarity results to prevent hallucination.

**Design decision:** After testing, all three analysis agents switched from RAG-retrieved fragments to full-text analysis. Single-document RAG caused fragment-based hallucination — the model received partial clauses without surrounding context and filled in the rest from training data. Full-text analysis eliminated this. The RAG pipeline (embed + ChromaDB store) still runs at ingestion time; `retrieve_clauses()` in `rag_tool.py` remains available for multi-document or cross-contract retrieval use cases.

### 2. Multi-Agent Pattern — Orchestrator + Parallel Fan-out + Generator-Critic
**File:** `app/agents/orchestrator.py` → `analyze_contract()`, `run_critic()`
**File:** `app/agents/risk_agent.py` → `run_risk_agent()`
**File:** `app/agents/gap_agent.py` → `run_gap_agent()`
**File:** `app/agents/negotiation_agent.py` → `run_negotiation_agent()`

Four distinct agents each with a separate system prompt and responsibility:
- **Risk Agent**: identifies present risky clauses, calibrated by contract type
- **Gap Agent**: identifies missing standard terms, calibrated by contract type
- **Negotiation Agent**: rewrites unfair clauses with concrete replacement language
- **Critic Agent**: generator-critic pattern — reviews all three outputs independently and produces a final calibrated verdict

The orchestrator uses `asyncio.gather()` to run Risk, Gap, and Negotiation agents **truly in parallel**, then passes their combined structured outputs to the Critic. This follows the fan-out + generator-critic multi-agent pattern.

### 3. Structured Output
**File:** `app/schemas.py`

11 Pydantic models enforce structured output at every agent boundary:

| Schema | Used at |
|---|---|
| `IngestedDocument` | After PDF ingestion |
| `RiskClause`, `RiskAnalysis` | Risk agent output |
| `MissingTerm`, `GapAnalysis` | Gap agent output |
| `NegotiationSuggestion`, `NegotiationAnalysis` | Negotiation agent output |
| `CriticVerdict` | Critic agent output |
| `ContractReport` | Full assembled report |
| `UploadResponse`, `AnalyzeResponse` | API responses to frontend |

Every agent prompt instructs the model to return a specific JSON structure. The orchestrator validates all outputs against Pydantic schemas before passing to the next stage, preventing hallucinated or malformed data from propagating downstream.

### 4. Parallel Execution
**File:** `app/agents/orchestrator.py` → `analyze_contract()` line with `asyncio.gather()`

The three specialist agents run concurrently using `asyncio.gather()`. Each agent makes independent Gemini API calls without waiting for the others. This reduces total analysis time from ~45 seconds (sequential) to ~15 seconds (parallel). `return_exceptions=True` ensures one agent failure does not crash the pipeline — the orchestrator handles partial failures gracefully and substitutes empty-but-valid responses.

### 5. Artifacts
**File:** `app/tools/report_tool.py` → `generate_report()`

Every analysis produces a timestamped PDF report written to the `artifacts/` directory using ReportLab. The report includes: risk score banner, executive summary, top 3 actions, full risky clause list with verbatim citations, missing terms with suggested language, and negotiation suggestions with before/after clause text. The report path is returned in the API response and served via `/artifacts/` static mount for download.

### 6. Agent Framework — Google genai (Vertex AI)
**File:** `app/agents/orchestrator.py`, `app/agents/risk_agent.py`, `app/agents/gap_agent.py`, `app/agents/negotiation_agent.py`

All agents use `google.genai.Client` with Vertex AI backend. Each agent has a distinct `system_instruction` defining its role, contract-type-specific evaluation standards, and strict anti-hallucination rules (verbatim-only quotes, empty output preferred over fabricated findings). `thinking_config=ThinkingConfig(thinking_budget=0)` disables Gemini 2.5 Flash's thinking mode for consistent JSON output.

### 7. Evaluation — Dataset + Model-as-a-Judge
**File:** `eval/dataset.jsonl` — 12 test cases across 6 categories
**File:** `eval/run_eval.py` → `run_eval()`, `run_maaj()`

The evaluation suite covers: risk detection accuracy, gap detection, negotiation quality, contract type classification, non-contract rejection, and anti-hallucination. Each case defines deterministic checks (score ranges, field presence, keyword detection) that run without a judge, plus MaaJ rubrics for quality cases.

The MaaJ judge (`run_maaj()`) sends the contract text, system output, and a human-written rubric to `claude-sonnet-4-6` — a different model family from the Gemini generator — to evaluate whether findings are accurate, grounded in the actual contract text, and correctly calibrated by contract type. This directly implements the Model-as-a-Judge pattern from the Evaluation lecture (Feb 09).

```bash
# Deterministic only
python eval/run_eval.py --base-url http://127.0.0.1:8000

# With Claude MaaJ judge
export ANTHROPIC_API_KEY=sk-ant-...
python eval/run_eval.py --base-url http://127.0.0.1:8000 --judge

# Single case
python eval/run_eval.py --id risk_01

# By category
python eval/run_eval.py --category balanced_contract
```

Target: ≥80% overall, 100% on non-contract rejection, 100% on anti-hallucination.

---

## Project Structure

```
clause/
├── app/
│   ├── main.py                  # FastAPI: /upload /analyze/{id} /health /documents
│   ├── schemas.py               # 11 Pydantic structured output models
│   ├── agents/
│   │   ├── orchestrator.py      # asyncio.gather fan-out + critic pattern
│   │   ├── risk_agent.py        # Identifies risky clauses (full text)
│   │   ├── gap_agent.py         # Identifies missing terms (full text)
│   │   └── negotiation_agent.py # Suggests improved clause language (full text)
│   └── tools/
│       ├── ingest_tool.py       # PDF parse → chunk → embed → ChromaDB
│       ├── rag_tool.py          # Semantic search with relevance filtering
│       └── report_tool.py       # ReportLab PDF artifact generator
├── data/
│   ├── chroma/                  # ChromaDB vector store (auto-created)
│   ├── documents/               # Full-text JSON store (auto-created)
│   └── templates/               # Sample contracts for testing
├── eval/
│   ├── dataset.jsonl            # Test cases (risk detection, OOS rejection, safety)
│   └── run_eval.py              # Eval runner: deterministic checks + Claude MaaJ judge
├── static/
│   └── index.html               # Upload UI + risk dashboard
├── artifacts/                   # Generated PDF reports (auto-created)
├── Dockerfile
├── pyproject.toml
├── .gitignore
├── uv.lock
└── README.md
```

---

## Supported Contract Types

| Type | Classification | Standards applied |
|---|---|---|
| Freelance / service contract | `freelance_contract` | Non-compete, IP, kill fee, portfolio rights |
| Commercial supply agreement | `sales_contract` | Liability cap, governing law, warranty, termination |
| Loan / terms sheet | `terms_sheet` | Default/cure, interest, prepayment, representations |
| NDA | `nda` | Confidentiality duration, permitted disclosures |
| Employment agreement | `employment` | At-will, non-compete, IP assignment |
| Other commercial | `service_agreement`, `unknown` | Best-effort analysis |

Non-contract documents (resumes, reports, presentations) are rejected at ingestion with a clear error message.

---

## Privacy & Data Handling

Clause is designed for **transient processing** — contract text is never stored beyond the duration of a single analysis session.

| Stage | What happens | Retention |
|---|---|---|
| Upload | PDF parsed to text, chunked, embedded | Deleted from disk after analysis completes |
| Analysis | Full text sent to Vertex AI (Gemini) for agent calls | Google's [data processing terms](https://cloud.google.com/terms/data-processing-terms) apply |
| ChromaDB | Contract embeddings stored locally during session | Deleted from ChromaDB after analysis completes |
| PDF report | Generated report saved to `artifacts/` | Persists for download; cleared on container restart |

**After `/analyze/{doc_id}` returns:** the full-text JSON file and all ChromaDB embeddings for that document are deleted from disk (`_cleanup_document()` in `app/main.py`).

**Vertex AI data processing:** Contract text is sent to Google Vertex AI for Gemini inference. Under Google Cloud's data processing terms, customer data is not used to train Google's models. See [Google Cloud data governance](https://cloud.google.com/terms/data-processing-terms).

**Recommendation for sensitive contracts:** For highly confidential documents (M&A, litigation-related), consider self-hosting this application within your own GCP project to ensure data never leaves your infrastructure.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Frontend UI |
| `/health` | GET | Health check |
| `/upload` | POST | Ingest a contract PDF (multipart/form-data, field: `file`) |
| `/analyze/{doc_id}` | POST | Run full analysis on an ingested document |
| `/documents` | GET | List all ingested documents in the current session |
| `/artifacts/{filename}` | GET | Download a generated PDF report |